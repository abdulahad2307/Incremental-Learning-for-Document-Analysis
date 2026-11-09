"""Open-set recognition along the CIL steps: does a model keep rejecting the classes it has not learned yet, and do
the EVM methods keep that ability better than the classifier's own scores?

In CIL, the classes of later steps are real unknowns at every earlier step. For every run and step 0..K-1 (step K
has no unknowns left) it scores the known test documents against the unknown ones with two protocols:
  all     knowns = the test documents of all classes learned so far, unknowns = those of all later classes (as
          build_results.py, which averages it over the steps)
  fixed   knowns = the 11 base classes, unknown = the last class of the order (budget) at every step: the same
          documents throughout, so the curve over the steps shows the change of the open-set ability itself
Scores (higher = known): msp (max softmax), energy (logsumexp of the logits), maxlogit, evm (the run's EVM inclusion
probability as saved, EVM methods), ood (the method's OOD detector, EVM+OOD). Metrics: AUROC, FPR95, OSCR
(utils/metrics/openset_metrics.py). Step 0 is the shared base model (no EVM).

--features <drift root> adds, on the drift features (300 test documents per class; --feature_seeds, default 42), the
EVM scores
that the saved probabilities cannot give:
  evm_uncapped   the run's own trained EVM (its saved open-set state), scored with the uncapped log inclusion
                 probability: the saved probabilities underflow to a common floor far from all extreme vectors,
                 which ties those documents and distorts AUROC / FPR95 (see tools/eval/posthoc_evt.py)
  evm_posthoc    an EVM fitted after the step on the step's own joint features of 200 training documents of every
                 known class (tail 0.3, 50 extreme vectors per class) - for every method, also without EVM training:
                 separates "EVM as a scorer" from "EVM in the training"
  msp_sub        msp on the same documents (the reference for these two)

Outputs in --out: openset_steps.csv (per run x seed x step x protocol x score), openset_steps_summary.csv (mean and
std over seeds), openset_steps.md, openset_steps_<backbone>.png.

Example:
  python tools/eval/openset_steps.py --run_root $RUN_ROOT --features $RUN_ROOT/features/drift --out results/openset
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.metrics.openset_metrics import auroc, fpr_at_tpr, oscr  # noqa: E402
from utils.metrics.runs import base_predictions, find_runs, load_cil_run, run_label  # noqa: E402
from utils.evm.evm_state import evm_state_path  # noqa: E402

JOINT = {"eaml": "fused", "llmv3": "cls", "llmv3hf": "cls"}
FIXED_UNKNOWN = "budget"


def logit_scores(set_):
    z = np.asarray(set_["logits"], dtype=np.float64)
    m = z.max(axis=1, keepdims=True)
    lse = (m + np.log(np.exp(z - m).sum(axis=1, keepdims=True)))[:, 0]
    out = {"msp": np.exp(m[:, 0] - lse), "energy": lse, "maxlogit": m[:, 0]}
    if "evm_prob" in set_:
        out["evm"] = np.asarray(set_["evm_prob"], dtype=np.float64).max(axis=1)
    if "ood_score" in set_:
        out["ood"] = np.asarray(set_["ood_score"], dtype=np.float64)
    return out


def metrics(sk, su, correct):
    return {"auroc": auroc(sk, su), "fpr95": fpr_at_tpr(sk, su), "oscr": oscr(sk, correct, su)}


def protocols(true_known, true_unknown, base_classes):
    """{protocol: (known mask, unknown mask)}."""
    out = {"all": (np.ones(len(true_known), bool), np.ones(len(true_unknown), bool))}
    fu = true_unknown == FIXED_UNKNOWN
    if fu.any():
        out["fixed"] = (np.isin(true_known, base_classes), fu)
    return out


def step_rows(pred, base_classes):
    sets = pred["sets"]
    if "unseen" not in sets:
        return []
    k, u = sets["seen"], sets["unseen"]
    out_classes = np.asarray(pred["output_classes"], dtype=object)
    tk = np.asarray(k["class_names"], dtype=object)[np.asarray(k["y_true"], dtype=int)]
    tu = np.asarray(u["class_names"], dtype=object)[np.asarray(u["y_true"], dtype=int)]
    correct = tk == out_classes[np.asarray(k["pred"], dtype=int)]
    sk, su = logit_scores(k), logit_scores(u)
    rows = []
    for proto, (mk, mu) in protocols(tk, tu, base_classes).items():
        for score in sk:
            if score in su:
                rows.append(dict(protocol=proto, score=score, **metrics(sk[score][mk], su[score][mu], correct[mk])))
    return rows


# ------------------------------------------------------------------ EVM scores on the drift features
def uncapped_log_proba(evm_models, feats, device):
    """(N, C) uncapped log inclusion probabilities -min_i (d / scale_i) ^ shape_i of saved (point, scale, shape)
    extreme vectors (EVMClassifier.weibull_models / IncrementalEVM.class_evs), columns in dict order."""
    x = torch.as_tensor(feats, dtype=torch.float32, device=device)
    cols = []
    for evs in evm_models.values():
        p = torch.tensor(np.stack([np.asarray(e[0], dtype=np.float32) for e in evs]), device=device)
        sc = torch.tensor([float(e[1]) + 1e-8 for e in evs], dtype=torch.float32, device=device)
        sh = torch.tensor([float(e[2]) + 1e-8 for e in evs], dtype=torch.float32, device=device)
        log_t = sh * (torch.log(torch.cdist(x, p) + 1e-12) - torch.log(sc))       # log (d / scale) ^ shape
        cols.append(-torch.exp(log_t.min(dim=1).values))
    return torch.stack(cols, dim=1).cpu().numpy().astype(np.float64)


def trained_evm(checkpoint):
    path = evm_state_path(checkpoint)
    if not path or not os.path.exists(path):
        return None
    st = torch.load(path, map_location="cpu", weights_only=False).get("evm") or {}
    return st.get("weibull_models") or st.get("class_evs") or None


def feature_rows(root, backbone, run, seed, base_classes, device):
    from utils.evm.mevm_classifier import MultivariateEVM
    joint, rows = JOINT[backbone], []
    files = {int(re.search(r"step(\d+)\.pt$", f).group(1)): f
             for f in glob.glob(os.path.join(root, backbone, run, *([] if seed == 42 else [f"seed{seed}"]),
                                             "step*.pt"))}
    files[0] = os.path.join(root, backbone, "step0.pt")
    for k, f in sorted(files.items()):
        if not os.path.exists(f):
            continue
        d = torch.load(f, map_location="cpu", weights_only=False)
        S, known = d["splits"], list(d["known_classes"])
        if "test_unseen" not in S:
            continue
        tr_y = np.asarray(S["train"]["labels"], dtype=object)
        tr_f = np.asarray(S["train"]["features"][joint], dtype=np.float32)
        kf = np.asarray(S["test_seen"]["features"][joint], dtype=np.float32)
        uf = np.asarray(S["test_unseen"]["features"][joint], dtype=np.float32)
        tk = np.asarray(S["test_seen"]["labels"], dtype=object)
        tu = np.asarray(S["test_unseen"]["labels"], dtype=object)
        logits = {s: np.asarray(S[s]["logits"], dtype=np.float64) for s in ("test_seen", "test_unseen")}
        correct = np.asarray(known, dtype=object)[logits["test_seen"].argmax(1)] == tk
        softmax_max = lambda z: 1.0 / np.exp(z - z.max(1, keepdims=True)).sum(1)
        scores = {"msp_sub": (softmax_max(logits["test_seen"]), softmax_max(logits["test_unseen"]), correct)}
        mevm = MultivariateEVM(tailsize=0.3, max_fit_samples=50)
        mevm.fit({c: {joint: tr_f[tr_y == c]} for c in known})
        t = lambda x: {joint: torch.as_tensor(x, device=device)}
        cols = [list(mevm.weibull_models).index(c) for c in known]
        pk, pu = (mevm.log_proba_tensor(t(x)).numpy()[:, cols] for x in (kf, uf))
        scores["evm_posthoc"] = (pk.max(1), pu.max(1), np.asarray(known, dtype=object)[pk.argmax(1)] == tk)
        evs = trained_evm(d["meta"]["checkpoint"]) if k > 0 else None
        if evs and len(np.asarray(next(iter(evs.values()))[0][0]).ravel()) == kf.shape[1]:
            names = [str(c) for c in evs]
            pk, pu = uncapped_log_proba(evs, kf, device), uncapped_log_proba(evs, uf, device)
            scores["evm_uncapped"] = (pk.max(1), pu.max(1), np.asarray(names, dtype=object)[pk.argmax(1)] == tk)
        for proto, (mk, mu) in protocols(tk, tu, base_classes).items():
            for s, (a, b, ok) in scores.items():
                rows.append(dict(backbone=backbone, run=run, seed=seed, step=k, protocol=proto, score=s,
                                 **metrics(a[mk], b[mu], ok[mk])))
        print(f"  features {backbone}/{run}/seed{seed} step {k}: " + ", ".join(
            f"{r['score']} {r['auroc']:.3f}" for r in rows[-len(scores):]), flush=True)
    return rows


# ------------------------------------------------------------------ report
def summarise(df):
    g = df.groupby(["backbone", "run", "method", "step", "protocol", "score"])
    out = g[["auroc", "fpr95", "oscr"]].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.seed.nunique()
    return out.reset_index()


def markdown(summ):
    pc = lambda m, s: "-" if pd.isna(m) else (f"{100 * m:.1f}" + ("" if pd.isna(s) else f"±{100 * s:.1f}"))
    lines = ["# Open-set recognition along the CIL steps", "",
             "AUROC in % (mean ± std over seeds where >1). all = all not-yet-learned classes are the unknowns; "
             f"fixed = base classes vs {FIXED_UNKNOWN} at every step. *_sub / evm_posthoc / evm_uncapped: drift "
             "features (300 test documents per class).", ""]
    for (b, proto), g in summ.groupby(["backbone", "protocol"]):
        steps = sorted(g.step.unique())
        lines += [f"## {b}, protocol {proto}", "", "| run | score | " + " | ".join(f"step {s}" for s in steps)
                  + " | mean |", "|" + "---|" * (3 + len(steps))]
        for (run, score), h in g.groupby(["run", "score"]):
            v = h.set_index("step")
            cells = [pc(v.auroc_mean.get(s), v.auroc_std.get(s)) for s in steps]
            lines.append(f"| {run} | {score} | " + " | ".join(cells) + f" | {pc(h.auroc_mean.mean(), np.nan)} |")
        lines.append("")
    return "\n".join(lines)


def figures(summ, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for b, g in summ.groupby("backbone"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
        for ax, proto in zip(axes, ["all", "fixed"]):
            h = g[g.protocol == proto]
            for (run, score), x in h.groupby(["run", "score"]):
                if score not in ("msp", "energy", "evm", "evm_uncapped", "evm_posthoc", "ood"):
                    continue
                if score in ("energy", "evm_posthoc") and not run.startswith(("noevm", "finetune", "evm_std")):
                    continue
                x = x.sort_values("step")
                ax.plot(x.step, 100 * x.auroc_mean, marker="o", ms=3, label=f"{run} / {score}",
                        ls="-" if score.startswith(("evm", "ood")) else "--")
            ax.set_title(f"{b}: {proto}")
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("AUROC (%)")
        axes[1].legend(fontsize=6, ncol=2, loc="lower left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"openset_steps_{b}.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--backbones", nargs="+", default=["eaml", "llmv3hf"])
    ap.add_argument("--runs", nargs="+", default=None, help="run folders (default: all)")
    ap.add_argument("--features", default=None, help="drift feature root: adds evm_uncapped / evm_posthoc")
    ap.add_argument("--feature_runs", nargs="+", default=["finetune_std", "noevm_std", "noevm_kd", "evm_std",
                                                          "evm_kd", "ievm_std", "evmood_std"])
    ap.add_argument("--feature_seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default="results/openset")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, methods = [], {}
    for b in args.backbones:
        base = base_predictions(args.run_root, b, "CIL")
        if base is None:
            print(f"[skip] {b}: no step-0 predictions")
            continue
        base_classes = list(base["output_classes"])
        for (bb, run, seed), files in sorted(find_runs(args.run_root, "CIL").items()):
            if bb != b or (args.runs and run not in args.runs):
                continue
            steps, _ = load_cil_run(files, base)
            if len(steps) < 2:
                continue
            methods[(b, run)] = run_label(steps[-1])[0]
            for k, s in enumerate(steps):
                rows += [dict(backbone=b, run=run, seed=seed, step=k, **r) for r in step_rows(s, base_classes)]
            print(f"{b}/{run}/seed{seed}: steps 0-{len(steps) - 1}", flush=True)
        if args.features:
            for seed in args.feature_seeds:
                for run in args.feature_runs:
                    rows += feature_rows(args.features, b, run, seed, base_classes, device)
    if not rows:
        sys.exit("no runs found")
    df = pd.DataFrame(rows)
    df["method"] = [methods.get((b, r), r) for b, r in zip(df.backbone, df.run)]
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "openset_steps.csv"), index=False)
    summ = summarise(df)
    summ.to_csv(os.path.join(args.out, "openset_steps_summary.csv"), index=False)
    with open(os.path.join(args.out, "openset_steps.md"), "w") as f:
        f.write(markdown(summ))
    figures(summ, args.out)
    print(f"-> {args.out}/openset_steps.md")


if __name__ == "__main__":
    main()
