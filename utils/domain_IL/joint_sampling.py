"""Domain-balanced sampling for the DIL joint-training upper bound (--joint_balance domain).

Shuffled together, RVL-CDIP's training set outnumbers the new domain's ~100:1 (Tobacco-3482: 2,436 documents),
so the joint model hardly learns the new domain. The balanced variant draws each domain with equal probability
(with replacement) and keeps the epoch length of the imbalanced one, so both variants cost the same."""
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

JOINT_BALANCE = ("none", "domain")


def domain_weights(sizes):
    """Per-sample weights for datasets of the given sizes: each dataset gets total weight 1/len(sizes)."""
    return torch.cat([torch.full((n,), 1.0 / (len(sizes) * n), dtype=torch.double) for n in sizes])


def joint_loader(datasets, batch_size, balance="none", seed=42, num_workers=4, collate_fn=None):
    """One loader over all datasets: shuffled together (balance "none") or domain-balanced ("domain")."""
    if balance not in JOINT_BALANCE:
        raise ValueError(f"joint balance must be one of {JOINT_BALANCE}, got {balance!r}")
    combined = ConcatDataset(datasets)
    if balance == "none":
        return DataLoader(combined, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    sizes = [len(d) for d in datasets]
    sampler = WeightedRandomSampler(domain_weights(sizes), num_samples=len(combined), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    print(f"Domain-balanced joint sampling: datasets of {sizes} samples, each drawn with probability "
          f"{1 / len(sizes):.2f}; {len(combined)} samples per epoch")
    return DataLoader(combined, batch_size=batch_size, sampler=sampler, num_workers=num_workers, collate_fn=collate_fn)
