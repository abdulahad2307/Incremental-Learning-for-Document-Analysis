"""Build the result tables of all IL runs from their saved predictions (no run is repeated).

For every CIL run (base model = step 0, then steps 1..K) and every DIL run it computes the IL metrics
(utils/metrics/il_metrics.py), the open-set metrics (utils/metrics/openset_metrics.py), a bootstrap confidence interval
of the final accuracy and an exact McNemar test against "No EVM" with the same backbone, strategy, seed and number of
steps (utils/metrics/stats.py). Runs that differ only in the seed are aggregated as mean +- std.

Outputs in --out_dir:
  cil_runs.csv, dil_runs.csv          one row per run (seed)
  cil_summary.csv, dil_summary.csv    mean / std over seeds per backbone x method x strategy (x steps for CIL)
  cil_accuracy_matrices.json          accuracy matrix A[k][j] of every CIL run
  results.md                          the tables for the paper (accuracies in %)

Example:
  python tools/eval/build_results.py --out_dir results/journal
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.metrics.il_metrics import cil_summary, dil_summary, named  # noqa: E402
from utils.metrics.openset_metrics import SCORES, openset_summary, scores, auroc  # noqa: E402
from utils.metrics.runs import base_predictions, find_runs, load_cil_run, run_label  # noqa: E402
from utils.metrics.stats import bootstrap_ci, mcnemar, mean_std, paired  # noqa: E402
from utils.eval.predictions import load_predictions  # noqa: E402

REFERENCE = "No EVM"
KEYS = ["backbone", "method", "strategy"]


def correctness(prediction, set_name):
    true, pred = named(prediction["sets"][set_name], prediction["output_classes"])
    return list(prediction["sets"][set_name]["doc_id"]), true == pred


def add_mcnemar(rows, correct, keys, sets):
    """p-value of every run against the REFERENCE run with the same `keys` (and seed), per set in `sets`."""
    for r in rows:
        ref = next((o for o in rows if o["method"] == REFERENCE and o["seed"] == r["seed"]
                    and all(o[k] == r[k] for k in keys)), None)
        for s in sets:
            if ref is None or r["method"] == REFERENCE:
                r[f"p_vs_noevm_{s}"] = float("nan")
                continue
            a, b = paired(*correct[id(r)][s], *correct[id(ref)][s])
            r[f"p_vs_noevm_{s}"] = mcnemar(a, b)[2]


def cil_rows(run_root, backbones):
    rows, matrices, correct = [], {}, {}
    bases = {}
    for (backbone, method_dir, seed), files in sorted(find_runs(run_root, "CIL").items()):
        if backbones and backbone not in backbones:
            continue
        if backbone not in bases:
            bases[backbone] = base_predictions(run_root, backbone, "CIL")
        if bases[backbone] is None:
            print(f"[skip] no step-0 predictions for {backbone} (scripts/eval/run_base_predictions.sh)")
            continue
        steps, complete = load_cil_run(files, bases[backbone])
        if len(steps) < 2:
            print(f"[skip] {backbone}/{method_dir}/seed{seed}: no step 1")
            continue
        summary, A, tasks = cil_summary(steps)
        method, strategy = run_label(steps[-1])
        row = {"backbone": backbone, "method": method, "strategy": strategy, "seed": seed, "run": method_dir,
               "complete": complete, **summary}
        ids, ok = correctness(steps[-1], "seen")
        row["acc_final_ci_low"], row["acc_final_ci_high"] = bootstrap_ci(ok)
        open_set = [openset_summary(s) for s in steps[1:]]               # steps of the method (step 0 = base model)
        for key in sorted({k for o in open_set for k in o}):
            vals = [o[key] for o in open_set if key in o]
            row[key] = float(np.mean(vals))
        row["open_set_steps"] = sum(1 for o in open_set if o)
        rows.append(row)
        correct[id(row)] = {"seen": (ids, ok)}
        matrices[f"{backbone}/{method_dir}/seed{seed}"] = {"tasks": tasks, "A": np.round(A, 6).tolist()}
    add_mcnemar(rows, correct, ["backbone", "strategy", "steps"], ["seen"])
    return rows, matrices


def dil_rows(run_root, backbones):
    rows, correct, bases = [], {}, {}
    for (backbone, method_dir, seed), files in sorted(find_runs(run_root, "DIL").items()):
        if backbones and backbone not in backbones:
            continue
        if backbone not in bases:
            bases[backbone] = base_predictions(run_root, backbone, "DIL")
        if bases[backbone] is None:
            print(f"[skip] no step-0 predictions for {backbone} (scripts/eval/run_base_predictions.sh)")
            continue
        adapted = load_predictions(files[0])
        method, strategy = run_label(adapted)
        row = {"backbone": backbone, "method": method, "strategy": strategy, "seed": seed, "run": method_dir,
               **dil_summary(bases[backbone], adapted)}
        for kind in SCORES:          # domain-shift detection: RVL-CDIP = known, Tobacco-3482 = unknown
            k, u = scores(adapted["sets"]["rvl"], kind), scores(adapted["sets"]["tobacco"], kind)
            if k is not None and u is not None:
                row[f"shift_{kind}_auroc"] = auroc(k, u)
        rows.append(row)
        correct[id(row)] = {d: correctness(adapted, d) for d in ("rvl", "tobacco")}
    add_mcnemar(rows, correct, ["backbone", "strategy"], ["rvl", "tobacco"])
    return rows


def summarise(rows, group):
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in group + ["seed", "run", "complete"] and df[c].dtype != object]
    out = []
    for key, g in df.groupby(group, dropna=False, sort=True):
        rec = dict(zip(group, key if isinstance(key, tuple) else (key,)))
        rec["seeds"] = len(g)
        for c in metric_cols:
            rec[c], rec[f"{c}_std"] = mean_std(g[c].tolist())
        out.append(rec)
    return pd.DataFrame(out)


def fmt(row, col, pct=True):
    m, s = row.get(col, float("nan")), row.get(f"{col}_std", float("nan"))
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "–"
    f = 100 if pct else 1
    return f"{m * f:.1f} ± {s * f:.1f}" if not np.isnan(s) else f"{m * f:.1f}"


def markdown(cil, dil):
    lines = ["# Results", "", "Accuracies and open-set scores in %; mean ± std over seeds (std only with >1 seed).", ""]
    if not cil.empty:
        cols = [("acc_final", "Final acc"), ("avg_inc_acc", "Avg. inc. acc"), ("forgetting", "Forgetting"),
                ("bwt", "BWT"), ("g_il_final", "G_IL"), ("acc_old_final", "Old classes"),
                ("acc_new_final", "New class"), ("msp_auroc", "AUROC (MSP)"), ("evm_auroc", "AUROC (EVM)"),
                ("ood_auroc", "AUROC (OOD)"), ("msp_oscr", "OSCR (MSP)")]
        for backbone, g in cil.groupby("backbone"):
            lines += [f"## CIL: {backbone}", "",
                      "| Method | Strategy | Steps | Seeds | " + " | ".join(n for _, n in cols) + " |",
                      "|" + "---|" * (4 + len(cols))]
            for _, r in g.sort_values(["strategy", "method", "steps"]).iterrows():
                lines.append(f"| {r['method']} | {r['strategy']} | {int(r['steps'])} | {int(r['seeds'])} | "
                             + " | ".join(fmt(r, c) for c, _ in cols) + " |")
            lines.append("")
    if not dil.empty:
        cols = [("acc_rvl_before", "RVL before"), ("acc_rvl_after", "RVL after"), ("forgetting_rvl", "RVL forgetting"),
                ("acc_tobacco_after", "Tobacco after"), ("acc_mean_after", "Mean after"),
                ("macro_f1_rvl_after", "RVL macro-F1"), ("macro_f1_tobacco_after", "Tobacco macro-F1"),
                ("shift_msp_auroc", "Shift AUROC (MSP)"), ("shift_ood_auroc", "Shift AUROC (OOD)")]
        for backbone, g in dil.groupby("backbone"):
            lines += [f"## DIL (RVL-CDIP → Tobacco-3482): {backbone}", "",
                      "| Method | Strategy | Seeds | " + " | ".join(n for _, n in cols) + " |",
                      "|" + "---|" * (3 + len(cols))]
            for _, r in g.sort_values(["strategy", "method"]).iterrows():
                lines.append(f"| {r['method']} | {r['strategy']} | {int(r['seeds'])} | "
                             + " | ".join(fmt(r, c) for c, _ in cols) + " |")
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_root", default=os.environ.get("RUN_ROOT", "/home/woody/iwi5/iwi5280h/il_runs"))
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--backbone", nargs="*", default=None, help="eaml / llmv3 / llmv3hf (default: all)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cil, matrices = cil_rows(args.run_root, args.backbone)
    dil = dil_rows(args.run_root, args.backbone)
    cil_sum = summarise(cil, KEYS + ["steps"])
    dil_sum = summarise(dil, KEYS)
    pd.DataFrame(cil).to_csv(os.path.join(args.out_dir, "cil_runs.csv"), index=False)
    pd.DataFrame(dil).to_csv(os.path.join(args.out_dir, "dil_runs.csv"), index=False)
    cil_sum.to_csv(os.path.join(args.out_dir, "cil_summary.csv"), index=False)
    dil_sum.to_csv(os.path.join(args.out_dir, "dil_summary.csv"), index=False)
    with open(os.path.join(args.out_dir, "cil_accuracy_matrices.json"), "w") as f:
        json.dump(matrices, f, indent=1)
    with open(os.path.join(args.out_dir, "results.md"), "w") as f:
        f.write(markdown(cil_sum, dil_sum))
    print(f"{len(cil)} CIL runs, {len(dil)} DIL runs -> {args.out_dir}")


if __name__ == "__main__":
    main()
