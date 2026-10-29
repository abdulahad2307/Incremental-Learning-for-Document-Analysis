"""Adjacency-aware replay: split the exemplar budget of a CIL step over the old classes by how close each is to the
new class, instead of uniformly.

In every step analysed so far, forgetting concentrates in the old class closest to the new one (step 1: adding
scientific_publication costs scientific_report up to 25 points; step 2: specification costs form; step 3:
file_folder costs presentation), and about a third of the new errors on old classes are predictions of the new
class. The total budget stays the same; the at-risk classes get more of it.

  a_c = cos(mean of class c, mean of the new class)        joint feature of the previous model, L2-normalised
  n_c = n_min + (B - n_min * K) * softmax(a / tau)_c, rounded so that sum_c n_c = B   (K old classes)
Defaults (fixed before any result): tau = 0.2, n_min = 10, i.e. every class keeps at least half of the uniform
budget of 20; at step 1 the most adjacent class (scientific_report) gets 100 of the 220 exemplars.

With the class-balanced sampler (--use_balanced_sampler), every class is drawn equally often whatever its number of
exemplars, so the split above changes only how varied a class's replay is, not how much it weighs in training.
--replay_weighting allocation (replay_weights) draws the old classes in proportion to their exemplars instead: the
new classes keep their balanced share 1 / K, and the old classes share the rest by their budget. With a uniform split
this is the balanced sampler again. Pre-registered as the second and last adjacency pilot (step 1, both backbones).
"""
import numpy as np
import torch

REPLAY_WEIGHTING = ("balanced", "allocation")


def adjacency_method_name(args):
    """Method name of an adjacency-replay run in the result tables."""
    weighted = getattr(args, "replay_weighting", "balanced") == "allocation"
    return "Adjacency replay (weighted)" if weighted else "Adjacency replay"


def replay_weights(labels, new_labels, mode="balanced"):
    """Sampling weight of every training sample (labels: one per sample; new_labels: the step's new classes).
    balanced: every class equally often. allocation: each new class 1 / K of the draws (K classes), the old classes
    the rest in proportion to their number of exemplars (every exemplar equally often)."""
    if mode not in REPLAY_WEIGHTING:
        raise ValueError(f"replay weighting must be one of {REPLAY_WEIGHTING}, got {mode!r}")
    labels = np.asarray(labels)
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    w = 1.0 / counts[inverse]
    if mode == "allocation":
        new = np.isin(labels, list(new_labels))
        n_new_classes = len(set(labels[new].tolist()))
        old_share = (len(classes) - n_new_classes) / len(classes)
        if (~new).any():
            w = np.where(new, 1.0 / (len(classes) * counts[inverse]), old_share / (~new).sum())
    return torch.as_tensor(w / w.sum(), dtype=torch.double)


def normalised_mean(feats):
    f = np.asarray(feats, dtype=np.float64)
    f = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    m = f.mean(axis=0)
    return m / max(np.linalg.norm(m), 1e-12)


def class_adjacency(old_feats, new_feats):
    """{old class: cosine between its mean and the new class's mean} (features L2-normalised per sample)."""
    new_mean = normalised_mean(new_feats)
    return {c: float(normalised_mean(f) @ new_mean) for c, f in old_feats.items()}


def allocate(adjacency, total, tau=0.2, n_min=10):
    """Integer exemplars per class summing to `total`: n_min each, the rest shared by softmax(a / tau) (largest
    remainder rounding). Equal adjacencies give the uniform split."""
    classes = list(adjacency)
    k = len(classes)
    if k == 0:
        return {}
    n_min = min(n_min, total // k)
    a = np.array([adjacency[c] for c in classes], dtype=np.float64) / tau
    w = np.exp(a - a.max())
    w /= w.sum()
    spare = total - n_min * k                       # what is left after every class has its minimum
    raw = n_min + spare * w
    n = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - n))[: total - n.sum()]:
        n[i] += 1
    return {c: int(x) for c, x in zip(classes, n)}


@torch.no_grad()
def features_by_class(extract, items_by_class, batch_size=32):
    """{class: (n, d) features} with extract(list of items) -> (b, d) tensor, in batches."""
    out = {}
    for c, items in items_by_class.items():
        chunks = [extract(items[s:s + batch_size]).float().cpu() for s in range(0, len(items), batch_size)]
        out[c] = torch.cat(chunks).numpy() if chunks else np.zeros((0, 0))
    return out


def describe(adjacency, alloc):
    """One line per class, most adjacent first (for the logs)."""
    return "\n".join(f"  {c:24s} adjacency {adjacency[c]:+.3f}  exemplars {alloc[c]}"
                     for c in sorted(adjacency, key=lambda c: -adjacency[c]))
