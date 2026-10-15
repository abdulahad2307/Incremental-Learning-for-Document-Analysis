"""Post-hoc go/no-go check of the Multivariate EVM (JOURNAL_EXTENSION_PLAN.md, item B.11): fit EVM and MEVM on cached
per-modality features (tools/eval/extract_step_features.py) and compare them on the same test documents.

Models (all fitted on the same "train" documents: tail size 0.3, 50 extreme vectors per class):
  EVM            classic EVM on the joint feature (EAML "fused", LayoutLMv3 "cls") - as used in the IL methods
  MEVM           all modalities, theta per class from Kendall's tau
  MEVM theta=1   all modalities, product of the modality inclusions (ablation)
  EVM <modality> classic EVM on one modality only (what each modality carries alone)
  Softmax        the classifier's maximum softmax probability (reference)
Metrics: closed-set accuracy on the known test documents (argmax of the inclusion probabilities), open-set AUROC,
FPR95 and OSCR with the not-yet-learned classes as unknowns (utils/metrics/openset_metrics.py). The EVM scores are
the uncapped log inclusion probabilities (MultivariateEVM.log_proba_tensor; with one modality it is exactly
EVMClassifier): the probabilities themselves underflow to a common floor far from all extreme vectors, which ties
those documents and distorts accuracy and FPR95.

Go / no-go rule (agreed before the comparison, issue "MEVM core and post-hoc go/no-go"):
  GO if, on every feature file given (both backbones), MEVM's open-set AUROC is >= EVM's + 2 points and its
  closed-set accuracy is >= EVM's - 1 point; otherwise MEVM becomes an ablation. --auroc_gain / --acc_loss change the
  thresholds; --rejudge <posthoc_evt.json> applies them to a previous comparison without refitting.

Example:
  python tools/eval/posthoc_evt.py $RUN_ROOT/features/eaml/*.pt $RUN_ROOT/features/llmv3hf/*.pt --out results/mevm_go
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.evm.mevm_classifier import MultivariateEVM  # noqa: E402
from utils.metrics.openset_metrics import auroc, fpr_at_tpr, oscr  # noqa: E402

AUROC_GAIN, ACC_LOSS = 0.02, 0.01      # the agreed go / no-go rule (defaults of --auroc_gain / --acc_loss)
JOINT = {"eaml": "fused", "llmv3": "cls"}


def per_class(split, classes, modalities):
    labels = np.asarray(split["labels"], dtype=object)
    return {c: {m: np.asarray(split["features"][m][labels == c], dtype=np.float32) for m in modalities}
            for c in classes}


def tensors(split, modalities):
    return {m: torch.as_tensor(np.asarray(split["features"][m], dtype=np.float32)) for m in modalities}


def evaluate(probs_seen, probs_unseen, labels_seen, classes):
    """Closed-set accuracy and open-set metrics of (N, C) scores (columns = classes, higher = more likely)."""
    pred = np.asarray(classes, dtype=object)[probs_seen.argmax(axis=1)]
    correct = pred == np.asarray(labels_seen, dtype=object)
    k, u = probs_seen.max(axis=1), probs_unseen.max(axis=1)
    return {"acc": float(correct.mean()), "auroc": auroc(k, u), "fpr95": fpr_at_tpr(k, u), "oscr": oscr(k, correct, u)}


def compare(path, auroc_gain, acc_loss):
    data = torch.load(path, map_location="cpu", weights_only=False)
    classes, backbone = data["known_classes"], data["meta"]["backbone"]
    splits = data["splits"]
    if "test_unseen" not in splits:
        raise ValueError(f"{path}: no unknown classes (last step) - nothing to compare open-set on")
    modalities = list(splits["train"]["features"])
    joint = JOINT[backbone]
    train = per_class(splits["train"], classes, modalities)
    seen, unseen = splits["test_seen"], splits["test_unseen"]
    results = {}

    def show(name):
        print(f"  {name:22s} " + "  ".join(f"{k} {v:.4f}" for k, v in results[name].items()), flush=True)

    def run(name, mods, theta=None):
        model = MultivariateEVM(tailsize=0.3, max_fit_samples=50, theta=theta)
        model.fit({c: {m: train[c][m] for m in mods} for c in classes})
        ps, pu = model.log_proba_tensor(tensors(seen, mods)), model.log_proba_tensor(tensors(unseen, mods))
        cols = [list(model.weibull_models).index(c) for c in classes]          # columns in class order
        results[name] = evaluate(ps.numpy()[:, cols], pu.numpy()[:, cols], seen["labels"], classes)
        if len(mods) > 1:
            results[name]["theta_mean"] = float(np.mean(list(model.theta.values())))
        show(name)

    print(f"{path}  ({backbone}, {len(classes)} known / {len(data['unknown_classes'])} unknown classes)")
    run("EVM", [joint])
    run("MEVM", modalities)
    run("MEVM theta=1", modalities, theta=1.0)
    for m in modalities:
        if m != joint:
            run(f"EVM {m}", [m])
    z = lambda s: torch.softmax(torch.as_tensor(np.asarray(s["logits"], dtype=np.float32)), dim=1).numpy()
    results["Softmax"] = evaluate(z(seen), z(unseen), seen["labels"], classes)
    show("Softmax")
    d_auroc = results["MEVM"]["auroc"] - results["EVM"]["auroc"]
    d_acc = results["MEVM"]["acc"] - results["EVM"]["acc"]
    passed = bool(d_auroc >= auroc_gain and d_acc >= -acc_loss)
    return {"file": path, "backbone": backbone, "llmv3_model": data["meta"].get("llmv3_model", ""),
            "checkpoint": data["meta"]["checkpoint"], "results": results,
            "delta_auroc": d_auroc, "delta_acc": d_acc, "passes_rule": passed}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", nargs="*", help="feature files of extract_step_features.py")
    ap.add_argument("--out", default="results/mevm_go")
    ap.add_argument("--auroc_gain", type=float, default=AUROC_GAIN, help="required AUROC gain of MEVM over EVM")
    ap.add_argument("--acc_loss", type=float, default=ACC_LOSS, help="allowed closed-set accuracy loss")
    ap.add_argument("--rejudge", help="posthoc_evt.json of a previous run: apply the rule to it, no refitting")
    args = ap.parse_args()
    if args.rejudge:
        with open(args.rejudge) as f:
            reports = json.load(f)["reports"]
        for r in reports:
            r["passes_rule"] = bool(r["delta_auroc"] >= args.auroc_gain and r["delta_acc"] >= -args.acc_loss)
    else:
        reports = [compare(f, args.auroc_gain, args.acc_loss) for f in args.features]
    go = all(r["passes_rule"] for r in reports)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "posthoc_evt.json"), "w") as f:
        json.dump({"rule": {"auroc_gain": args.auroc_gain, "acc_loss": args.acc_loss}, "go": go, "reports": reports},
                  f, indent=1)
    lines = ["# MEVM post-hoc go / no-go", "",
             f"Rule: GO if MEVM AUROC >= EVM {args.auroc_gain * 100:+.1f} points and accuracy >= EVM "
             f"{-args.acc_loss * 100:+.1f} points on every file. **Decision: {'GO' if go else 'NO-GO'}**", ""]
    for r in reports:
        lines += [f"## {r['backbone']}{' (' + r['llmv3_model'] + ')' if r['llmv3_model'] else ''}: "
                  f"{os.path.basename(r['checkpoint'])}", "",
                  f"MEVM vs EVM: AUROC {r['delta_auroc'] * 100:+.2f}, accuracy {r['delta_acc'] * 100:+.2f} -> "
                  f"{'passes' if r['passes_rule'] else 'fails'}", "",
                  "| Model | Closed-set acc | AUROC | FPR95 | OSCR | mean theta |", "|---|---|---|---|---|---|"]
        for name, m in r["results"].items():
            theta = f"{m['theta_mean']:.2f}" if "theta_mean" in m else "–"
            lines.append(f"| {name} | {m['acc'] * 100:.2f} | {m['auroc'] * 100:.2f} | {m['fpr95'] * 100:.2f} | "
                         f"{m['oscr'] * 100:.2f} | {theta} |")
        lines.append("")
    with open(os.path.join(args.out, "posthoc_evt.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nDecision: {'GO' if go else 'NO-GO'}  ->  {args.out}/posthoc_evt.md")


if __name__ == "__main__":
    main()
