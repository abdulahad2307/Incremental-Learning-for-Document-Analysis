"""Summarise the shared IL results table written by utils/run_log.py.

Examples:
  python tools/eval/il_results_table.py                       # final results, latest run per configuration
  python tools/eval/il_results_table.py --all-runs            # final results of every run
  python tools/eval/il_results_table.py --phase epoch --run-id <id>   # training curve of one run
  python tools/eval/il_results_table.py --backbone eaml --setting CIL --out results/eaml_cil.md
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.run_log import _table_path  # noqa: E402

CONFIG = ["backbone", "setting", "method", "strategy", "bias_correction", "new_class"]
METRICS = ["acc", "precision", "recall", "f1", "acc_rvl", "acc_tob", "gil_base", "gil_prev",
           "evm_known_acc", "evm_unknown_rej", "ood_auroc", "ood_fpr95"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=_table_path())
    ap.add_argument("--phase", default="final", help="final | epoch | step_test | open_set | ood | all")
    ap.add_argument("--backbone")
    ap.add_argument("--setting")
    ap.add_argument("--method")
    ap.add_argument("--run-id")
    ap.add_argument("--all-runs", action="store_true", help="Keep every run instead of the latest per configuration")
    ap.add_argument("--out", help="Write to .csv or .md instead of printing")
    args = ap.parse_args()

    df = pd.read_csv(args.table, dtype={"slurm_job_id": str, "new_class": str})
    for col in ("backbone", "setting", "method", "run_id"):
        val = getattr(args, col.replace("-", "_"), None)
        if val:
            df = df[df[col] == val]
    if args.phase != "all":
        df = df[df["phase"] == args.phase]

    if not args.all_runs and not args.run_id:
        # a configuration re-run replaces the earlier result
        last_run = df.groupby(CONFIG, dropna=False)["run_id"].transform("last")
        df = df[df["run_id"] == last_run]

    df = df.dropna(axis=1, how="all")
    lead = [c for c in ["timestamp", "run_id"] + CONFIG + ["step", "epoch", "split", "phase"] if c in df]
    rest = [c for c in df.columns if c not in lead and c not in ("slurm_job_id", "script", "extra")]
    if args.phase in ("final", "step_test", "open_set", "ood"):
        rest = [c for c in METRICS if c in df] + ["extra"] * ("extra" in df)
    df = df[lead + rest]

    if args.out and args.out.endswith(".md"):
        cells = df.astype(object).where(df.notna(), "").astype(str)
        with open(args.out, "w") as f:
            f.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
            for row in cells.itertuples(index=False):
                f.write("| " + " | ".join(row) + " |\n")
    elif args.out:
        df.to_csv(args.out, index=False)
    else:
        with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 250,
                               "display.max_colwidth", 60):
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
