"""Frozen-backbone nearest-class-mean (NCM) baseline for class-incremental learning (JOURNAL_EXTENSION_PLAN.md,
item 16), on cached features of the base model (tools/eval/extract_step_features.py with --train_classes all;
scripts/eval/run_extract_features.sh <backbone> base_all).

The base model (step 0) is frozen. At CIL step k the model knows the 11 base classes plus the first k classes of the
class order; every known class is represented by the mean of its training-document features (joint feature: EAML
"fused", LayoutLMv3 "cls"), and a document goes to the nearest mean. Nothing is trained, so nothing is forgotten
except through the growing number of classes.
  distance   cosine (similarity to the L2-normalised means of L2-normalised features) or Euclidean
  means      from all cached training documents of a class (1000 per class) or from 20 (the exemplar budget of the
             IL methods)
  open set   the score of the nearest mean (cosine similarity, or minus the Euclidean distance); the unknowns are the
             test documents of the classes not learned yet

Outputs:
  <out>/ncm_baseline.md / .json   metrics of steps 0..5 with NCM also at step 0 (utils/metrics/il_metrics.py)
  with --run_root: predictions files of steps 1..5 in the IL format (utils/eval/predictions.py),
                   <run_root>/cil/<backbone>/ncm_<cos|euc>_<all|20>_frozen/seed<seed>/predictions/cil_<class>_seed<seed>.pt
                   so tools/eval/build_results.py lists NCM next to the IL methods. In those tables step 0 is the
                   shared base model (its classifier head), as for every method; average incremental accuracy and
                   forgetting with NCM at step 0 are the ones in ncm_baseline.md. The open-set score is stored as
                   ood_score (the "ood" columns), the NCM scores as logits.

Example:
  python tools/eval/ncm_baseline.py $RUN_ROOT/features/eaml/base_all_cil.pt --order $(IFS=,; echo "${CIL_ORDER[*]}") \
      --tag eaml --out $RUN_ROOT/features/ncm/eaml --run_root $RUN_ROOT
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tools.eval.posthoc_evt import JOINT  # noqa: E402
from utils.data_subset import DEFAULT_SEED, select_per_class  # noqa: E402
from utils.eval.predictions import FORMAT_VERSION  # noqa: E402
from utils.metrics.il_metrics import cil_summary  # noqa: E402
from utils.metrics.openset_metrics import openset_summary  # noqa: E402

VARIANTS = [("cos", None), ("euc", None), ("cos", 20), ("euc", 20)]     # (distance, documents per class mean)


def normalise(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def class_means(feats, labels, doc_ids, classes, n=None):
    """{class: mean feature} over all its training documents, or over a reproducible n of them."""
    means = {}
    for c in classes:
        idx = np.flatnonzero(labels == c)
        if n is not None and len(idx) > n:
            keep = set(select_per_class([doc_ids[i] for i in idx], n, DEFAULT_SEED, c))
            idx = np.array([i for i in idx if doc_ids[i] in keep], dtype=int)
        means[c] = feats[idx].mean(axis=0)
    return means


def ncm_scores(x, means, distance):
    """(N, C) scores, higher = closer: cosine similarity, or minus the Euclidean distance."""
    M = np.stack(list(means.values()))
    if distance == "cos":
        return normalise(x) @ normalise(M).T
    return -torch.cdist(torch.as_tensor(x), torch.as_tensor(M)).numpy()


def step_prediction(scores_by_set, sets, classes, meta):
    """A predictions dict in the format of utils/eval/predictions.py for one step."""
    out = {"format_version": FORMAT_VERSION, "meta": meta, "output_classes": list(classes), "evm_classes": None,
           "sets": {}}
    for name, (s, docs, labels) in scores_by_set.items():
        names = sets[name]
        index = {c: i for i, c in enumerate(names)}
        out["sets"][name] = {"doc_id": list(docs), "y_true": np.array([index[l] for l in labels], dtype=np.int16),
                             "class_names": list(names), "logits": s.astype(np.float16),
                             "pred": s.argmax(axis=1).astype(np.int16),
                             "ood_score": s.max(axis=1).astype(np.float32)}
    return out


def run(path, order, tag, seed, run_root):
    data = torch.load(path, map_location="cpu", weights_only=False)
    base, joint = data["known_classes"], JOINT[data["meta"]["backbone"]]
    S = data["splits"]
    tr_labels = np.asarray(S["train"]["labels"], dtype=object)
    missing = [c for c in base + order if c not in set(tr_labels)]
    if missing:
        raise ValueError(f"{path}: no training features of {missing}; extract with --train_classes all (base_all)")
    tr = np.asarray(S["train"]["features"][joint], dtype=np.float32)
    test = {k: (np.asarray(S[k]["features"][joint], dtype=np.float32), list(S[k]["doc_id"]),
                np.asarray(S[k]["labels"], dtype=object)) for k in ("test_seen", "test_unseen")}
    x_all = np.concatenate([test["test_seen"][0], test["test_unseen"][0]])
    docs_all = test["test_seen"][1] + test["test_unseen"][1]
    y_all = np.concatenate([test["test_seen"][2], test["test_unseen"][2]])
    report = {"file": path, "backbone": data["meta"]["backbone"], "tag": tag, "feature": joint, "order": order,
              "variants": {}}
    for distance, n in VARIANTS:
        name = f"ncm_{distance}_{'all' if n is None else n}"
        means = class_means(tr, tr_labels, list(S["train"]["doc_id"]), base + order, n)
        steps = []
        for k in range(len(order) + 1):
            known = base + order[:k]
            unknown = [c for c in base + order if c not in known]
            s = ncm_scores(x_all, {c: means[c] for c in known}, distance)
            seen = np.isin(y_all, known)
            sets = {"seen": known, "unseen": unknown}
            by_set = {"seen": (s[seen], [d for d, m in zip(docs_all, seen) if m], y_all[seen])}
            if unknown:
                by_set["unseen"] = (s[~seen], [d for d, m in zip(docs_all, seen) if not m], y_all[~seen])
            meta = {"run_id": f"{tag}_{name}_seed{seed}", "git_commit": "", "script": "tools/eval/ncm_baseline.py",
                    "backbone": tag, "setting": "CIL", "method": f"NCM ({'cosine' if distance == 'cos' else 'Euclidean'}"
                    f", {'all' if n is None else n} per class)", "strategy": "frozen", "bias_correction": False,
                    "seed": seed, "step": k, "new_class": order[k - 1] if k else None, "checkpoint": data["meta"][
                        "checkpoint"], "created": datetime.datetime.now().isoformat(timespec="seconds")}
            steps.append(step_prediction(by_set, sets, known, meta))
        summary, A, tasks = cil_summary(steps)
        opens = [openset_summary(p, kinds=("ood",)) for p in steps[1:]]
        summary.update({k.replace("ood_", "open_"): float(np.mean([o[k] for o in opens if k in o]))
                        for k in sorted({k for o in opens for k in o})})
        summary["open_auroc_step0"] = openset_summary(steps[0], kinds=("ood",)).get("ood_auroc", float("nan"))
        report["variants"][name] = {"summary": summary, "A": np.round(A, 6).tolist(), "tasks": tasks}
        print(f"  {name:12s} acc step0 {summary['acc_step0']:.4f}  final {summary['acc_final']:.4f}  "
              f"avg inc {summary['avg_inc_acc']:.4f}  forgetting {summary['forgetting']:.4f}  "
              f"open AUROC (steps 1-{len(order) - 1}) {summary.get('open_auroc', float('nan')):.4f}", flush=True)
        if run_root:
            d = os.path.join(run_root, "cil", tag, f"{name}_frozen", f"seed{seed}", "predictions")
            os.makedirs(d, exist_ok=True)
            for p in steps[1:]:
                f = os.path.join(d, f"cil_{p['meta']['new_class']}_seed{seed}.pt")
                torch.save(p, f + ".tmp")
                os.replace(f + ".tmp", f)
            print(f"    predictions of steps 1-{len(order)} -> {d}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", help="base_all feature file of extract_step_features.py (--train_classes all)")
    ap.add_argument("--order", required=True, help="CIL class order, comma-separated (scripts/config.sh CIL_ORDER)")
    ap.add_argument("--tag", required=True, help="backbone folder name: eaml, llmv3 or llmv3hf")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default="results/ncm")
    ap.add_argument("--run_root", help="also write IL-format predictions files under <run_root>/cil/<tag>/")
    args = ap.parse_args()
    report = run(args.features, [c.strip() for c in args.order.split(",")], args.tag, args.seed, args.run_root)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ncm_baseline.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    pct = lambda v: f"{v * 100:.2f}"
    lines = [f"# Frozen-backbone NCM: {args.tag} ({report['feature']})", "",
             "Steps 0..K with NCM at every step. Open set: steps 1..K-1 (the last step has no unknowns).", "",
             "| Variant | Acc step 0 | Acc final | Avg inc. acc | Forgetting | BWT | Acc old | Acc new | "
             "Open AUROC | FPR95 | OSCR |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in report["variants"].items():
        s = v["summary"]
        lines.append(f"| {name} | {pct(s['acc_step0'])} | {pct(s['acc_final'])} | {pct(s['avg_inc_acc'])} | "
                     f"{pct(s['forgetting'])} | {pct(s['bwt'])} | {pct(s['acc_old_final'])} | "
                     f"{pct(s['acc_new_final'])} | {pct(s.get('open_auroc', float('nan')))} | "
                     f"{pct(s.get('open_fpr95', float('nan')))} | {pct(s.get('open_oscr', float('nan')))} |")
    with open(os.path.join(args.out, "ncm_baseline.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"-> {args.out}/ncm_baseline.md")


if __name__ == "__main__":
    main()
