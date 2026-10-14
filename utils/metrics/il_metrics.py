"""Incremental-learning metrics computed from the per-step predictions files (utils/eval/predictions.py).

CIL: the base classes form task 0, the class added in step j forms task j. After step k the accuracy matrix holds
  A[k, j] = accuracy on the test documents of task j (j <= k; NaN for tasks not learned yet).
  - accuracy after step k: over all test documents of the classes learned so far (document-weighted, as logged)
  - average incremental accuracy (Rebuffi et al., 2017): mean of the accuracies after steps 0..K
  - forgetting (Chaudhry et al., 2018): mean over tasks j < K of max_{l in [j, K-1]} A[l, j] - A[K, j]
  - backward transfer (Lopez-Paz & Ranzato, 2017): mean over tasks j < K of A[K, j] - A[j, j]
  - G_IL (workshop paper): (acc - acc_base) / (1 - acc_base), acc_base = accuracy of the base model (step 0)
DIL: accuracy per domain before (base model) and after adaptation, forgetting on the first domain, and macro-F1
  per domain averaged only over the classes that occur in that domain.
"""
import numpy as np


def named(set_, output_classes):
    """(true class names, predicted class names) of a predictions set."""
    names = np.asarray(set_["class_names"], dtype=object)
    out = np.asarray(output_classes, dtype=object)
    return names[np.asarray(set_["y_true"], dtype=int)], out[np.asarray(set_["pred"], dtype=int)]


def accuracy(true, pred, mask=None):
    """Fraction of correct predictions (over `mask` if given); NaN if there is nothing to score."""
    true, pred = np.asarray(true, dtype=object), np.asarray(pred, dtype=object)
    if mask is not None:
        true, pred = true[mask], pred[mask]
    return float(np.mean(true == pred)) if len(true) else float("nan")


def macro_f1(true, pred):
    """Macro-F1 averaged over the classes that occur in `true` (classes absent from a domain do not count)."""
    true, pred = np.asarray(true, dtype=object), np.asarray(pred, dtype=object)
    scores = []
    for c in sorted(set(true)):
        tp = np.sum((pred == c) & (true == c))
        fp = np.sum((pred == c) & (true != c))
        fn = np.sum((pred != c) & (true == c))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else float("nan")


def g_il(acc, acc_base):
    return (acc - acc_base) / (1 - acc_base) if acc_base is not None and acc_base < 1 else float("nan")


def cil_accuracy_matrix(steps):
    """steps: predictions dicts of steps 0..K in order (step 0 = base model). Returns (tasks, A, overall) with
    tasks[j] = list of class names of task j, A the (K+1) x (K+1) accuracy matrix and overall[k] the accuracy over
    all learned classes after step k."""
    tasks = [list(steps[0]["output_classes"])]
    for s in steps[1:]:
        tasks.append([s["meta"]["new_class"]])
    K = len(steps) - 1
    A = np.full((K + 1, K + 1), np.nan)
    overall = []
    for k, s in enumerate(steps):
        true, pred = named(s["sets"]["seen"], s["output_classes"])
        overall.append(accuracy(true, pred))
        for j in range(k + 1):
            A[k, j] = accuracy(true, pred, np.isin(true, tasks[j]))
    return tasks, A, np.asarray(overall)


def forgetting(A):
    """Average forgetting after the last step (Chaudhry et al., 2018); NaN with fewer than two steps."""
    K = A.shape[0] - 1
    if K < 1:
        return float("nan")
    return float(np.mean([np.nanmax(A[j:K, j]) - A[K, j] for j in range(K)]))


def backward_transfer(A):
    """Backward transfer after the last step (Lopez-Paz & Ranzato, 2017); negative values mean forgetting."""
    K = A.shape[0] - 1
    if K < 1:
        return float("nan")
    return float(np.mean([A[K, j] - A[j, j] for j in range(K)]))


def cil_summary(steps):
    """All CIL metrics of one run (steps 0..K, step 0 = base model) as a flat dict."""
    tasks, A, overall = cil_accuracy_matrix(steps)
    K = len(steps) - 1
    base_classes = tasks[0]
    true, pred = named(steps[-1]["sets"]["seen"], steps[-1]["output_classes"])
    old = np.isin(true, [c for t in tasks[:K] for c in t])
    out = {
        "steps": K,
        "acc_base": overall[0],
        "acc_final": overall[-1],
        "avg_inc_acc": float(np.mean(overall)),
        "forgetting": forgetting(A),
        "bwt": backward_transfer(A),
        "g_il_final": g_il(overall[-1], overall[0]),
        "acc_old_final": accuracy(true, pred, old) if K else float("nan"),
        "acc_new_final": accuracy(true, pred, ~old) if K else float("nan"),
        "acc_base_classes_final": accuracy(true, pred, np.isin(true, base_classes)),
        "macro_f1_final": macro_f1(true, pred),
    }
    for k in range(K + 1):
        out[f"acc_step{k}"] = overall[k]
    return out, A, tasks


def dil_summary(base, adapted, first="rvl", second="tobacco"):
    """DIL metrics: the base model (before) and the adapted model (after) on both domains."""
    out = {}
    for tag, r in (("before", base), ("after", adapted)):
        for d in (first, second):
            true, pred = named(r["sets"][d], r["output_classes"])
            out[f"acc_{d}_{tag}"] = accuracy(true, pred)
            out[f"macro_f1_{d}_{tag}"] = macro_f1(true, pred)
    out[f"forgetting_{first}"] = out[f"acc_{first}_before"] - out[f"acc_{first}_after"]
    out[f"gain_{second}"] = out[f"acc_{second}_after"] - out[f"acc_{second}_before"]
    out["acc_mean_after"] = (out[f"acc_{first}_after"] + out[f"acc_{second}_after"]) / 2
    out[f"g_il_{first}"] = g_il(out[f"acc_{first}_after"], out[f"acc_{first}_before"])
    return out
