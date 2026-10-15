"""Second, pre-registered post-hoc round of the Multivariate EVM: modality weights (JOURNAL_EXTENSION_PLAN.md,
items 11 and 14) on the cached features of tools/eval/extract_step_features.py.

Variants (all on the same extreme vectors: tail size 0.3, 50 per class, per-modality Weibulls as in MEVM):
  A  weighted MEVM       -log Psi_i = || (w_j t_ij)_j ||_theta,  class score  -min_i ||.||_theta
                         grid: w on the simplex (step 0.1), theta in {estimated per class, 1, 2, 4, 8}
  B  class-level fusion  log Psi_c = sum_j w_j log max_i Psi_ij  (one EVM per modality, scores combined per class)
                         grid: w on the simplex (step 0.1)
Both contain the classic EVM on the joint feature as a grid point (all weight on EAML "fused" / LayoutLMv3 "cls"),
so the selection can only fall back to it. Scores are uncapped log inclusion probabilities, as in posthoc_evt.py.

Protocol (agreed before the run):
  - the test documents (known and unknown classes) are split 50/50 by a hash of their document ID: a selection half
    and a report half;
  - on the selection half, the configuration (variant A or B, weights, theta) with the highest open-set AUROC among
    those with closed-set accuracy >= EVM's - 1 point is chosen, per feature file;
  - on the report half, the rule of the first round is applied once: GO if, on every file, the chosen configuration
    has AUROC >= EVM's + 2 points and accuracy >= EVM's - 1 point.

Example:
  python tools/eval/posthoc_mevm_variants.py $RUN_ROOT/features/eaml/*.pt $RUN_ROOT/features/llmv3hf/*.pt \
      --out $RUN_ROOT/features/posthoc_variants
"""
import argparse
import hashlib
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.eval.posthoc_evt import AUROC_GAIN, ACC_LOSS, JOINT, evaluate, per_class  # noqa: E402
from utils.evm.mevm_classifier import MultivariateEVM  # noqa: E402

THETAS = ["est", 1.0, 2.0, 4.0, 8.0]
STEP = 0.1


def simplex(m, step=STEP):
    """All weight vectors with m entries, multiples of `step`, summing to 1."""
    n = int(round(1 / step))
    return [np.array(c, dtype=np.float64) / n for c in itertools.product(range(n + 1), repeat=m) if sum(c) == n]


def halves(split):
    """Selection (0) / report (1) half of every document, from a hash of its ID."""
    return np.array([int(hashlib.md5(d.encode()).hexdigest(), 16) % 2 for d in split["doc_id"]])


def subset(split, mask):
    return {"features": {m: np.asarray(v)[mask] for m, v in split["features"].items()},
            "labels": [l for l, k in zip(split["labels"], mask) if k]}


def log_energies(mevm, split, classes):
    """{class: (N, M, m) log t_ij} for one half (float32 tensors on CPU)."""
    feats = {m: torch.as_tensor(np.asarray(v, dtype=np.float32)) for m, v in split["features"].items()}
    with torch.no_grad():
        return {c: mevm.log_energies(feats, c) for c in classes}


def score_a(log_t, w, theta):
    """Variant A: (N, C) log scores -min_i || (w_j t_ij) ||_theta."""
    log_w = torch.log(torch.as_tensor(w, dtype=torch.float32))              # log 0 = -inf drops a modality
    cols = []
    for c, lt in log_t.items():
        th = float(theta[c]) if isinstance(theta, dict) else float(theta)
        z = torch.logsumexp(th * (lt + log_w), dim=-1) / th                    # log || w t_i ||_theta: (N, M)
        cols.append(-torch.exp(z).min(dim=1).values)
    return torch.stack(cols, dim=1).numpy()


def per_modality(log_t):
    """{class: (N, m) -min_i t_ij}: the log class score of each single-modality EVM."""
    return {c: -torch.exp(lt).min(dim=1).values for c, lt in log_t.items()}


def score_b(pm, w):
    """Variant B: (N, C) log scores sum_j w_j log max_i Psi_ij."""
    w = torch.as_tensor(w, dtype=torch.float32)
    return torch.stack([(s * w).sum(dim=1) for s in pm.values()], dim=1).numpy()


def run_file(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    classes, backbone = data["known_classes"], data["meta"]["backbone"]
    S = data["splits"]
    modalities = list(S["train"]["features"])
    joint = modalities.index(JOINT[backbone])
    mevm = MultivariateEVM(tailsize=0.3, max_fit_samples=50).fit(per_class(S["train"], classes, modalities))
    print(f"{path}  ({backbone}, modalities {modalities}, mean theta {np.mean(list(mevm.theta.values())):.2f})",
          flush=True)

    cache = {}
    for h in (0, 1):
        seen, unseen = (subset(S[k], halves(S[k]) == h) for k in ("test_seen", "test_unseen"))
        lt_s, lt_u = log_energies(mevm, seen, classes), log_energies(mevm, unseen, classes)
        cache[h] = (seen["labels"], lt_s, lt_u, per_modality(lt_s), per_modality(lt_u))

    def result(h, cfg):
        labels, lt_s, lt_u, pm_s, pm_u = cache[h]
        variant, w, theta = cfg
        if variant == "A":
            th = mevm.theta if theta == "est" else theta
            ks, us = score_a(lt_s, w, th), score_a(lt_u, w, th)
        else:
            ks, us = score_b(pm_s, w), score_b(pm_u, w)
        return evaluate(ks, us, labels, classes)

    one_hot = np.eye(len(modalities))[joint]
    evm_cfg, mevm_cfg = ("B", one_hot, None), ("A", np.ones(len(modalities)) / len(modalities), "est")
    grid = [("A", w, t) for w in simplex(len(modalities)) for t in THETAS] + \
           [("B", w, None) for w in simplex(len(modalities))]
    evm_sel = result(0, evm_cfg)
    sel = [(cfg, result(0, cfg)) for cfg in grid]
    print(f"  {len(grid)} configurations scored on the selection half", flush=True)

    def best(cands):
        ok = [x for x in cands if x[1]["acc"] >= evm_sel["acc"] - ACC_LOSS] or cands
        return max(ok, key=lambda x: x[1]["auroc"])[0]

    chosen = {"chosen": best(sel), "A": best([x for x in sel if x[0][0] == "A"]),
              "B": best([x for x in sel if x[0][0] == "B"])}
    named = {"EVM": evm_cfg, "MEVM (equal weights)": mevm_cfg, "best A": chosen["A"], "best B": chosen["B"],
             "chosen": chosen["chosen"]}
    desc = lambda cfg: {"variant": cfg[0], "weights": dict(zip(modalities, np.round(cfg[1], 2).tolist())),
                        "theta": cfg[2]}
    rows = {}
    for name, cfg in named.items():
        rows[name] = {"config": desc(cfg), "selection": result(0, cfg), "report": result(1, cfg)}
        r, s = rows[name]["report"], rows[name]["selection"]
        print(f"  {name:22s} {cfg[0]} w={np.round(cfg[1], 2).tolist()} theta={cfg[2]}  "
              f"sel acc {s['acc']:.4f} auroc {s['auroc']:.4f} | report acc {r['acc']:.4f} auroc {r['auroc']:.4f}",
              flush=True)
    d_auroc = rows["chosen"]["report"]["auroc"] - rows["EVM"]["report"]["auroc"]
    d_acc = rows["chosen"]["report"]["acc"] - rows["EVM"]["report"]["acc"]
    return {"file": path, "backbone": backbone, "llmv3_model": data["meta"].get("llmv3_model", ""),
            "checkpoint": data["meta"]["checkpoint"], "modalities": modalities, "rows": rows,
            "delta_auroc": d_auroc, "delta_acc": d_acc,
            "passes_rule": bool(d_auroc >= AUROC_GAIN and d_acc >= -ACC_LOSS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", nargs="+", help="feature files of extract_step_features.py")
    ap.add_argument("--out", default="results/mevm_variants")
    args = ap.parse_args()
    reports = [run_file(f) for f in args.features]
    go = all(r["passes_rule"] for r in reports)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "posthoc_mevm_variants.json"), "w") as f:
        json.dump({"rule": {"auroc_gain": AUROC_GAIN, "acc_loss": ACC_LOSS}, "go": go, "reports": reports}, f,
                  indent=1, default=str)
    lines = ["# MEVM variants (weighted MEVM, class-level fusion): post-hoc go / no-go", "",
             f"Configuration chosen on the selection half; rule judged on the report half: AUROC >= EVM + "
             f"{AUROC_GAIN * 100:.0f} points and accuracy >= EVM - {ACC_LOSS * 100:.0f} point on every file. "
             f"**Decision: {'GO' if go else 'NO-GO'}**", ""]
    for r in reports:
        lines += [f"## {r['backbone']}{' (' + r['llmv3_model'] + ')' if r['llmv3_model'] else ''}: "
                  f"{os.path.basename(r['checkpoint'])}", "",
                  f"Chosen vs EVM (report half): AUROC {r['delta_auroc'] * 100:+.2f}, accuracy "
                  f"{r['delta_acc'] * 100:+.2f} -> {'passes' if r['passes_rule'] else 'fails'}", "",
                  "| Model | Variant | Weights | θ | Sel. acc | Sel. AUROC | Report acc | Report AUROC | Report FPR95 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for name, row in r["rows"].items():
            c, s, p = row["config"], row["selection"], row["report"]
            w = ", ".join(f"{k} {v:.1f}" for k, v in c["weights"].items())
            lines.append(f"| {name} | {c['variant']} | {w} | {c['theta'] if c['theta'] is not None else '–'} | "
                         f"{s['acc'] * 100:.2f} | {s['auroc'] * 100:.2f} | {p['acc'] * 100:.2f} | "
                         f"{p['auroc'] * 100:.2f} | {p['fpr95'] * 100:.2f} |")
        lines.append("")
    with open(os.path.join(args.out, "posthoc_mevm_variants.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nDecision: {'GO' if go else 'NO-GO'}  ->  {args.out}/posthoc_mevm_variants.md")


if __name__ == "__main__":
    main()
