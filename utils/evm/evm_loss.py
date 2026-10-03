import math

import numpy as np
import torch

# Same ceiling as the former -log(Psi + 1e-8)
_MAX_NLL = -math.log(1e-8)


def _extreme_vectors(evm, key, device):
    """(points, scales, shapes) of one class's extreme vectors as tensors (EVMClassifier or IncrementalEVM)."""
    evs = evm.weibull_models[key] if hasattr(evm, "weibull_models") else evm.class_evs[key]
    points = torch.tensor(np.stack([ev[0] for ev in evs]), dtype=torch.float32, device=device)
    scales = torch.tensor([float(ev[1]) + 1e-8 for ev in evs], dtype=torch.float32, device=device)
    shapes = torch.tensor([float(ev[2]) for ev in evs], dtype=torch.float32, device=device)
    return points, scales, shapes


def evm_nll_loss(evm, features, labels, class_names=None):
    """Differentiable EVM loss: mean -log Psi_true(x) over samples whose class the EVM has been fitted on.

    Psi_class(x) = max_i exp(-(d_i / scale_i) ** shape_i), so -log Psi_class(x) = min_i (d_i / scale_i) ** shape_i.
    It is computed in log space (no exp underflow / inf * 0 = NaN gradients) and capped like -log(Psi + 1e-8).

    features:    (N, D) tensor attached to the graph (gradients flow into the model through the features).
    labels:      (N,) label ids.
    class_names: label id -> EVM key (e.g. class names) when the EVM is keyed by names; if None the EVM keys
                 are assumed to be the label ids themselves.
    Returns None if no sample's class is known to the EVM (e.g. the class being added in this step).
    """
    if evm is None or not getattr(evm, "initialized", False):
        return None
    keys = list(evm.weibull_models.keys()) if hasattr(evm, "weibull_models") else list(evm.class_evs.keys())
    key_of = (lambda l: class_names[l]) if class_names is not None else (lambda l: l)
    sample_keys = [key_of(l) for l in labels.cpu().tolist()]
    losses = []
    for key in set(k for k in sample_keys if k in keys):
        idx = torch.tensor([i for i, k in enumerate(sample_keys) if k == key], device=features.device)
        points, scales, shapes = _extreme_vectors(evm, key, features.device)
        d = torch.cdist(features[idx].float(), points)                             # (n, M)
        log_u = shapes * torch.log(d / scales + 1e-12)                             # log (d/scale)^shape
        nll = torch.exp(torch.clamp(log_u, max=math.log(_MAX_NLL))).min(dim=1).values
        losses.append(nll)
    if not losses:
        return None
    return torch.cat(losses).mean()
