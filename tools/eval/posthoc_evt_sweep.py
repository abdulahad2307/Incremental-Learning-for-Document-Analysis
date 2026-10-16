"""Post-hoc EVT hyperparameter sweep on cached features (JOURNAL_EXTENSION_PLAN.md, item 18): how much the EVM results
depend on the settings every IL run uses (tail size 0.3, cover threshold 0.7, 50 extreme-vector candidates per class,
Euclidean distance).

On the joint feature of a feature file of tools/eval/extract_step_features.py (EAML "fused", LayoutLMv3 "cls"):
  tailsize          0.1, 0.2, 0.3, 0.5, 0.7 (fraction of the negatives), each with Euclidean and cosine distance
  max_fit_samples   20, 50, 100, all cached training documents (extreme-vector candidates per class)
  cover_threshold   0.5 ... 0.9: accuracy with rejection on the knowns and rejection rate of the unknowns, for the
                    default and the cosine EVM (the threshold does not change accuracy or AUROC)
One factor is varied at a time from the defaults (tail size also crossed with the distance). Metrics: closed-set
accuracy, open-set AUROC, FPR95 and OSCR, the classes not learned yet being the unknowns. Scores are the uncapped log
inclusion probabilities (MultivariateEVM with one modality, which is exactly EVMClassifier).

Cosine distance is the Euclidean distance between L2-normalised features: ||a - b||^2 = 2 (1 - cos(a, b)) is a power
transform of the cosine distance, and the Weibull family is closed under power transforms, so the fitted EVM is the
same up to its parameterisation.

This is a sensitivity analysis on the full test sets, not tuning: the IL runs keep the defaults.

Example (one file at a time on the login node; several files in one run were killed there):
  python tools/eval/posthoc_evt_sweep.py $RUN_ROOT/features/eaml/base_cil.pt --out $RUN_ROOT/features/sweep/eaml_base
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.eval.posthoc_evt import JOINT, evaluate  # noqa: E402
from utils.evm.mevm_classifier import MultivariateEVM  # noqa: E402

DEFAULT = {"tailsize": 0.3, "max_fit_samples": 50, "distance": "euclidean"}
TAILSIZES = [0.1, 0.2, 0.3, 0.5, 0.7]
FIT_SAMPLES = [20, 50, 100, None]                     # None: all cached training documents of the class
DISTANCES = ["euclidean", "cosine"]
COVER = [0.5, 0.6, 0.7, 0.8, 0.9]


def normalise(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def configurations():
    """Defaults, tail size x distance, then max_fit_samples; without duplicates, defaults first."""
    out = [dict(DEFAULT)]
    out += [{**DEFAULT, "tailsize": t, "distance": d} for d in DISTANCES for t in TAILSIZES]
    out += [{**DEFAULT, "max_fit_samples": m} for m in FIT_SAMPLES]
    seen, unique = set(), []
    for c in out:
        key = tuple(sorted(c.items(), key=lambda kv: kv[0]))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def with_rejection(log_seen, log_unseen, labels, classes, thresholds=COVER):
    """Per cover threshold: accuracy on the knowns when a document below the threshold counts as rejected (wrong),
    and the fraction of unknowns rejected."""
    p_s, p_u = np.exp(log_seen), np.exp(log_unseen)
    pred = np.asarray(classes, dtype=object)[p_s.argmax(axis=1)]
    correct = pred == np.asarray(labels, dtype=object)
    return {f"{t:.1f}": {"known_acc": float(np.mean(correct & (p_s.max(axis=1) >= t))),
                         "unknown_rejection": float(np.mean(p_u.max(axis=1) < t))} for t in thresholds}


def run_file(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    classes, backbone = data["known_classes"], data["meta"]["backbone"]
    S, joint = data["splits"], JOINT[data["meta"]["backbone"]]
    labels_tr = np.asarray(S["train"]["labels"], dtype=object)
    X = {"train": np.asarray(S["train"]["features"][joint], dtype=np.float32),
         "seen": np.asarray(S["test_seen"]["features"][joint], dtype=np.float32),
         "unseen": np.asarray(S["test_unseen"]["features"][joint], dtype=np.float32)}
    n_train = {c: int((labels_tr == c).sum()) for c in classes}
    print(f"{path}  ({backbone}, {joint}, {len(classes)} known / {len(data['unknown_classes'])} unknown, "
          f"{min(n_train.values())}-{max(n_train.values())} training documents per class)", flush=True)
    rows = []
    for cfg in configurations():
        f = {k: (normalise(v) if cfg["distance"] == "cosine" else v) for k, v in X.items()}
        evm = MultivariateEVM(tailsize=cfg["tailsize"], max_fit_samples=cfg["max_fit_samples"])
        evm.fit({c: {joint: f["train"][labels_tr == c]} for c in classes})
        cols = [list(evm.weibull_models).index(c) for c in classes]
        ls = evm.log_proba_tensor({joint: torch.as_tensor(f["seen"])}).numpy()[:, cols]
        lu = evm.log_proba_tensor({joint: torch.as_tensor(f["unseen"])}).numpy()[:, cols]
        row = {**cfg, **evaluate(ls, lu, S["test_seen"]["labels"], classes)}
        if cfg["tailsize"] == DEFAULT["tailsize"] and cfg["max_fit_samples"] == DEFAULT["max_fit_samples"]:
            row["cover"] = with_rejection(ls, lu, S["test_seen"]["labels"], classes)
        rows.append(row)
        print(f"  tail {cfg['tailsize']:.1f}  fit {str(cfg['max_fit_samples']):>4s}  {cfg['distance']:9s}  "
              f"acc {row['acc']:.4f}  auroc {row['auroc']:.4f}  fpr95 {row['fpr95']:.4f}  oscr {row['oscr']:.4f}",
              flush=True)
    return {"file": path, "backbone": backbone, "llmv3_model": data["meta"].get("llmv3_model", ""),
            "checkpoint": data["meta"]["checkpoint"], "feature": joint, "train_per_class": n_train, "rows": rows}


def markdown(report):
    pct = lambda v: f"{v * 100:.2f}"
    is_default = lambda r: all(r[k] == v for k, v in DEFAULT.items())
    fit = lambda m: "all" if m is None else str(m)
    lines = [f"## {report['backbone']}{' (' + report['llmv3_model'] + ')' if report['llmv3_model'] else ''}: "
             f"{os.path.basename(report['checkpoint'])} ({report['feature']})", "",
             "| Tail size | Fit samples | Distance | Acc | AUROC | FPR95 | OSCR |", "|---|---|---|---|---|---|---|"]
    for r in report["rows"]:
        b = "**" if is_default(r) else ""
        lines.append(f"| {b}{r['tailsize']:.1f}{b} | {b}{fit(r['max_fit_samples'])}{b} | {b}{r['distance']}{b} | "
                     f"{pct(r['acc'])} | {pct(r['auroc'])} | {pct(r['fpr95'])} | {pct(r['oscr'])} |")
    lines += ["", "Cover threshold (tail size 0.3, 50 fit samples): known accuracy with rejection / unknown rejection",
              "", "| Distance | " + " | ".join(COVER_LABELS) + " |", "|---|" + "---|" * len(COVER)]
    for r in report["rows"]:
        if "cover" in r:
            lines.append(f"| {r['distance']} | " + " | ".join(
                f"{pct(r['cover'][t]['known_acc'])} / {pct(r['cover'][t]['unknown_rejection'])}"
                for t in COVER_LABELS) + " |")
    return lines + [""]


COVER_LABELS = [f"{t:.1f}" for t in COVER]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", nargs="+", help="feature files of extract_step_features.py")
    ap.add_argument("--out", default="results/evt_sweep")
    args = ap.parse_args()
    reports = [run_file(f) for f in args.features]
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "evt_sweep.json"), "w") as f:
        json.dump({"default": DEFAULT, "reports": reports}, f, indent=1)
    lines = ["# Post-hoc EVT sweep", "", "Bold: the setting of all IL runs. Unknowns: the classes not learned yet.", ""]
    for r in reports:
        lines += markdown(r)
    with open(os.path.join(args.out, "evt_sweep.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"-> {args.out}/evt_sweep.md")


if __name__ == "__main__":
    main()
