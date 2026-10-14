"""Open-set metrics from the saved predictions: in CIL the test documents of the classes learned so far are the knowns
and those of the classes not learned yet are the unknowns (the predictions files' "seen" and "unseen" sets).

Scores ("higher = known"):
  msp  - maximum softmax probability of the classifier
  evm  - maximum EVM inclusion probability (EVM methods)
  ood  - the method's OOD detector score (OOD methods)
Metrics:
  AUROC      - known vs unknown, threshold-free
  FPR95      - fraction of unknowns accepted at the threshold that accepts 95 % of the knowns
  OSCR       - area under the correct-classification-rate vs false-positive-rate curve (Dhamija et al., 2018):
               a known document counts only if it is accepted AND classified correctly
"""
import numpy as np
from scipy.stats import rankdata

SCORES = ("msp", "evm", "ood")


def scores(set_, kind):
    """Known-ness score per document, or None if the set does not have it."""
    if kind == "msp":
        z = np.asarray(set_["logits"], dtype=np.float64)
        p = np.exp(z - z.max(axis=1, keepdims=True))
        return (p / p.sum(axis=1, keepdims=True)).max(axis=1)
    if kind == "evm":
        return np.asarray(set_["evm_prob"], dtype=np.float64).max(axis=1) if "evm_prob" in set_ else None
    if kind == "ood":
        return np.asarray(set_["ood_score"], dtype=np.float64) if "ood_score" in set_ else None
    raise ValueError(kind)


def auroc(known, unknown):
    """Probability that a random known scores higher than a random unknown (ties count half)."""
    known, unknown = np.asarray(known, dtype=np.float64), np.asarray(unknown, dtype=np.float64)
    if not len(known) or not len(unknown):
        return float("nan")
    ranks = rankdata(np.concatenate([known, unknown]))      # average ranks over ties
    n1, n0 = len(known), len(unknown)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fpr_at_tpr(known, unknown, tpr=0.95):
    """Fraction of unknowns with a score >= the threshold that accepts `tpr` of the knowns."""
    known, unknown = np.asarray(known, dtype=np.float64), np.asarray(unknown, dtype=np.float64)
    if not len(known) or not len(unknown):
        return float("nan")
    thr = np.quantile(known, 1 - tpr)
    return float(np.mean(unknown >= thr))


def oscr(known, known_correct, unknown):
    """Open-set classification rate (area under CCR(FPR), both in [0, 1])."""
    known, unknown = np.asarray(known, dtype=np.float64), np.asarray(unknown, dtype=np.float64)
    known_correct = np.asarray(known_correct, dtype=bool)
    if not len(known) or not len(unknown):
        return float("nan")
    thresholds = np.unique(np.concatenate([known, unknown]))[::-1]      # descending: from strict to permissive
    # counts of scores >= each threshold, via sorted arrays (linear-logarithmic, not thresholds x documents)
    ge = lambda values: len(values) - np.searchsorted(np.sort(values), thresholds, side="left")
    ccr = ge(known[known_correct]) / len(known)
    fpr = ge(unknown) / len(unknown)
    ccr, fpr = np.concatenate([[0.0], ccr, [ccr[-1]]]), np.concatenate([[0.0], fpr, [1.0]])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(ccr, fpr))


def openset_summary(prediction, kinds=SCORES):
    """AUROC, FPR95 and OSCR of every available score of one CIL step file; {} without unknowns."""
    sets = prediction["sets"]
    if "unseen" not in sets:
        return {}
    known_set, unknown_set = sets["seen"], sets["unseen"]
    out_classes = np.asarray(prediction["output_classes"], dtype=object)
    true = np.asarray(known_set["class_names"], dtype=object)[np.asarray(known_set["y_true"], dtype=int)]
    correct = true == out_classes[np.asarray(known_set["pred"], dtype=int)]
    out = {}
    for kind in kinds:
        k, u = scores(known_set, kind), scores(unknown_set, kind)
        if k is None or u is None:
            continue
        out[f"{kind}_auroc"] = auroc(k, u)
        out[f"{kind}_fpr95"] = fpr_at_tpr(k, u)
        out[f"{kind}_oscr"] = oscr(k, correct, u)
    return out
