"""Statistics for the paper tables: seed aggregation, bootstrap confidence intervals over test documents, and the
exact McNemar test between two methods on the same test documents."""
import numpy as np
from scipy.stats import binomtest


def mean_std(values):
    """(mean, sample std) ignoring NaN; std is NaN for a single value."""
    v = np.asarray([x for x in values if x is not None and not np.isnan(x)], dtype=np.float64)
    if not len(v):
        return float("nan"), float("nan")
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else float("nan"))


def bootstrap_ci(correct, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the accuracy (mean of the 0/1 `correct` array), resampling test documents."""
    correct = np.asarray(correct, dtype=np.float64)
    if not len(correct):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(correct, size=(n_boot, len(correct)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired(doc_ids_a, correct_a, doc_ids_b, correct_b):
    """Correctness of two runs on their common documents, aligned by document ID."""
    b = dict(zip(doc_ids_b, correct_b))
    common = [(c, b[d]) for d, c in zip(doc_ids_a, correct_a) if d in b]
    if not common:
        return np.zeros(0, bool), np.zeros(0, bool)
    a_c, b_c = zip(*common)
    return np.asarray(a_c, bool), np.asarray(b_c, bool)


def mcnemar(correct_a, correct_b):
    """Exact McNemar test on paired correctness: returns (n_a_only, n_b_only, two-sided p-value)."""
    correct_a, correct_b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    if a_only + b_only == 0:
        return a_only, b_only, 1.0
    return a_only, b_only, float(binomtest(min(a_only, b_only), a_only + b_only, 0.5).pvalue)
