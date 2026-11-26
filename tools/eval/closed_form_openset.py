"""Open-set recognition of the closed-form classifier on frozen base features: can it reject the classes it has not
learned yet, as well as the incremental learners do?

Same classifier as tools/eval/closed_form_classifier.py (frozen base model, running sums G and C, ridge readout
W = (G + lambda I)^-1 C, lambda chosen at every step on held-out documents of the seen classes, final sums on all
training documents). Same open-set protocols as tools/eval/openset_steps.py, at steps 0..K-1:
  all     knowns = the test documents of all classes learned so far, unknowns = those of all later classes
  fixed   knowns = the 11 base classes, unknown = budget (the last class of the order) at every step
Scores (higher = known):
  ridge / ranpac   maxlogit, energy (logsumexp of the outputs), msp (max softmax of the outputs)
  ncm              cosine similarity to the nearest class mean (class means from the same sums)
  evm              an EVM fitted on the frozen joint features of --evm_train training documents of every known class
                   (tail 0.3, 50 extreme vectors per class, as evm_posthoc in openset_steps.py), uncapped log
                   inclusion probability; it is the EVM that a frozen backbone would add for rejection
Metrics: AUROC, FPR95, OSCR (utils/metrics/openset_metrics.py); OSCR uses the score's own classifier (ridge / ranpac /
ncm / evm) for "correct". The step-0 rows are all on the base model, so they can be compared with step 0 of the
learners.

Outputs in --out: closed_form_openset.csv (per backbone x method x projection seed x step x protocol x score),
closed_form_openset_summary.csv (mean and std over projection seeds), closed_form_openset.md

Example:
  python tools/eval/closed_form_openset.py --base '$RUN_ROOT/features/{backbone}/base_full_cil.pt' \\
      --out $RUN_ROOT/analysis/closed_form_openset
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.metrics.openset_metrics import auroc, fpr_at_tpr, oscr  # noqa: E402
from tools.eval.closed_form_classifier import (BASE_CLASSES, JOINT, ORDER, ClosedForm, best_lambda,  # noqa: E402
                                               holdout, load_base, make_phi, subsample)

FIXED_UNKNOWN = ORDER[-1]


def output_scores(z):
    z = z.astype(np.float64)
    m = z.max(1, keepdims=True)
    lse = (m + np.log(np.exp(z - m).sum(1, keepdims=True)))[:, 0]
    return {"maxlogit": m[:, 0], "energy": lse, "msp": np.exp(m[:, 0] - lse)}


def outputs(cf, x, lam, phi, chunk=4096):
    W = cf.weights(lam)
    return np.concatenate([phi(x[i:i + chunk]) @ W for i in range(0, len(x), chunk)])


def protocol_rows(key, scores, yte, known):
    """scores: {name: (score per test document, predicted label per test document)} on the documents of xte."""
    k, u = np.isin(yte, known), ~np.isin(yte, known)
    if not u.any():
        return []
    protos = {"all": (k, u)}
    if FIXED_UNKNOWN not in known:
        protos["fixed"] = (np.isin(yte, BASE_CLASSES), yte == FIXED_UNKNOWN)
    rows = []
    for proto, (mk, mu) in protos.items():
        for name, (s, pred) in scores.items():
            rows.append(dict(key, protocol=proto, score=name, auroc=auroc(s[mk], s[mu]), fpr95=fpr_at_tpr(s[mk], s[mu]),
                             oscr=oscr(s[mk], pred[mk] == yte[mk], s[mu])))
    return rows


def evm_scores(xtr_raw, ytr, xte_raw, known, n_fit, joint, device, seed=0):
    from utils.evm.mevm_classifier import MultivariateEVM
    idx = subsample(ytr, n_fit, seed)
    tr_x, tr_y = xtr_raw[idx], ytr[idx]
    evm = MultivariateEVM(tailsize=0.3, max_fit_samples=50)
    evm.fit({c: {joint: tr_x[tr_y == c]} for c in known})
    cols = [list(evm.weibull_models).index(c) for c in known]
    lp = np.concatenate([evm.log_proba_tensor({joint: torch.as_tensor(xte_raw[i:i + 4096], device=device)})
                         .cpu().numpy()[:, cols] for i in range(0, len(xte_raw), 4096)])
    return lp.max(1), np.asarray(known, dtype=object)[lp.argmax(1)]


def run_backbone(backbone, base_path, proj_dim, proj_seeds, evm_train, device):
    joint = JOINT[backbone]
    xtr_raw, ytr, xte_raw, yte = load_base(base_path, joint)
    keep = np.isin(ytr, BASE_CLASSES)
    mu, sd = xtr_raw[keep].mean(0), xtr_raw[keep].std(0) + 1e-6
    xtr, xte = (xtr_raw - mu) / sd, (xte_raw - mu) / sd
    val = holdout(ytr)
    steps = [BASE_CLASSES] + [[c] for c in ORDER[:-1]]                 # step K has no unknowns left
    rows = []
    for kind, seeds in (("ridge", [0]), ("ranpac", proj_seeds)):
        for seed in seeds:
            phi = make_phi(kind, xtr.shape[1], proj_dim, seed)
            dim = phi(xtr[:1]).shape[1]
            fit, full, known = ClosedForm(dim), ClosedForm(dim), []
            for step, new in enumerate(steps):
                for c in new:
                    fit.add_class_x(c, xtr[(ytr == c) & ~val], phi)
                    full.add_class_x(c, xtr[ytr == c], phi)
                    known.append(c)
                v = val & np.isin(ytr, known)
                lam = best_lambda(fit, xtr[v], ytr[v], phi)
                z = outputs(full, xte, lam, phi)
                pred = np.asarray(known, dtype=object)[z.argmax(1)]
                sc = {s: (val_, pred) for s, val_ in output_scores(z).items()}
                new_rows = protocol_rows(dict(backbone=backbone, method=kind, seed=seed, step=step, lam=lam), sc, yte,
                                         known)
                rows += new_rows
                print(f"  {backbone} {kind} seed {seed} step {step}: lambda {lam:g}, AUROC all " + ", ".join(
                    f"{r['score']} {r['auroc']:.3f}" for r in new_rows if r["protocol"] == "all"), flush=True)
    nrm = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    known, means = [], {}
    for step, new in enumerate(steps):
        for c in new:
            means[c] = xtr[ytr == c].mean(0)
            known.append(c)
        sim = nrm(xte) @ nrm(np.stack([means[c] for c in known])).T
        sc = {"cosine": (sim.max(1), np.asarray(known, dtype=object)[sim.argmax(1)])}
        sc["evm"] = evm_scores(xtr_raw, ytr, xte_raw, known, evm_train, joint, device)
        new_rows = [dict(r, method="ncm" if r["score"] == "cosine" else "evm") for r in protocol_rows(
            dict(backbone=backbone, method="", seed=0, step=step, lam=np.nan), sc, yte, known)]
        rows += new_rows
        print(f"  {backbone} ncm / evm step {step}: " + ", ".join(
            f"{r['method']} {r['protocol']} {r['auroc']:.3f}" for r in new_rows), flush=True)
    return rows


def markdown(s):
    pc = lambda m, sd: "-" if pd.isna(m) else f"{100 * m:.1f}" + ("" if pd.isna(sd) or sd == 0 else f"±{100 * sd:.1f}")
    lines = ["# Open-set recognition of the closed-form classifier (frozen base features)", "",
             "AUROC in % (ranpac: mean ± std over projection seeds); FPR95 and OSCR in the csv. all = all "
             f"not-yet-learned classes are the unknowns; fixed = base classes vs {FIXED_UNKNOWN} at every step.", ""]
    for (b, proto), g in s.groupby(["backbone", "protocol"]):
        steps = sorted(g.step.unique())
        lines += [f"## {b}, protocol {proto}", "", "| method | score | " + " | ".join(f"step {k}" for k in steps)
                  + " | mean |", "|" + "---|" * (3 + len(steps))]
        for (m, sc), h in g.groupby(["method", "score"], sort=False):
            v = h.set_index("step")
            lines.append(f"| {m} | {sc} | " + " | ".join(pc(v.auroc_mean.get(k), v.auroc_std.get(k)) for k in steps)
                         + f" | {pc(h.auroc_mean.mean(), np.nan)} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base-model feature file, {backbone} is replaced")
    ap.add_argument("--backbones", nargs="+", default=["eaml", "llmv3hf"])
    ap.add_argument("--proj_dim", type=int, default=5000)
    ap.add_argument("--proj_seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--evm_train", type=int, default=200, help="training documents per class for the EVM fit")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for b in a.backbones:
        rows += run_backbone(b, a.base.replace("{backbone}", b), a.proj_dim, a.proj_seeds, a.evm_train, device)
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "closed_form_openset.csv"), index=False)
    keys = ["backbone", "method", "step", "protocol", "score"]
    s = df.groupby(keys, sort=False)[["auroc", "fpr95", "oscr"]].agg(["mean", "std"])
    s.columns = [f"{m}_{f}" for m, f in s.columns]
    s = s.reset_index()
    s.to_csv(os.path.join(a.out, "closed_form_openset_summary.csv"), index=False)
    md = markdown(s)
    with open(os.path.join(a.out, "closed_form_openset.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
