import torch
import torch.nn.functional as F

# Which routes of L_OOD's gradient are kept (--ood_grad), from the most constrained to the original formulation
OOD_GRAD = ("direction", "radial", "full")


def vim_ood_loss(vim, features, logits, labels, class_names=None, grad="direction"):
    """Training-time L_OOD from ViM (Eq. 4/7): mean -log(1 - p_virtual) over samples of classes the ViM was fitted on.

    The ViM virtual logit v = alpha * ||residual|| is appended to the logits z; p_virtual = softmax([z, v])[-1], so
        -log(1 - p_virtual) = softplus(v - logsumexp(z)).
    Its purpose is to keep known-class features inside the principal subspace fitted on earlier data. Taken as is
    (grad="full"), the loss is also lowered without bound by two changes of scale that do not do that:
      - raising the energy logsumexp(z): the classifier rows grow at every CIL step;
      - shrinking the features towards the ViM origin u, which shrinks the residual: LayoutLMv3's features halve at
        every CIL step (cls norm 15.7 -> 8.2 -> 4.6) and the classifier rows grow to compensate; at step 5 the old
        rows are 5.6x the new class's row and the new class is never predicted.
    grad="radial" detaches the energy only; grad="direction" (default) also computes the residual on the feature
    rescaled to its own (detached) distance from u, so the gradient can turn the feature but not shrink it.
    The loss value is the same for all three.
    vim.fit_keys holds the fitted classes (class names if `class_names` maps label ids to names, else label ids).
    Returns None if the ViM is not fitted or no sample belongs to a fitted class.
    """
    if grad not in OOD_GRAD:
        raise ValueError(f"grad must be one of {OOD_GRAD}, not {grad!r}")
    if vim is None or not getattr(vim, "initialized", False):
        return None
    keys = getattr(vim, "fit_keys", None)
    if keys is not None:
        key_of = (lambda l: class_names[l]) if class_names is not None else (lambda l: l)
        keep = torch.tensor([key_of(l) in keys for l in labels.cpu().tolist()], device=features.device)
        if not keep.any():
            return None
        features, logits = features[keep], logits[keep]
    features = features.float()
    if grad == "direction":
        u = torch.as_tensor(vim.u, dtype=features.dtype, device=features.device)
        dist = (features - u).norm(dim=1, keepdim=True).clamp_min(1e-12)
        features = u + (features - u) * (dist.detach() / dist)
    v = vim.virtual_logit_tensor(features)
    energy = torch.logsumexp(logits.float(), dim=1)
    return F.softplus(v - (energy if grad == "full" else energy.detach())).mean()
