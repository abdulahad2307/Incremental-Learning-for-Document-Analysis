import torch
import torch.nn.functional as F


def vim_ood_loss(vim, features, logits, labels, class_names=None, energy_grad=False):
    """Training-time L_OOD from ViM (Eq. 4/7): mean -log(1 - p_virtual) over samples of classes the ViM was fitted on.

    The ViM virtual logit v = alpha * ||residual|| is appended to the logits z; p_virtual = softmax([z, v])[-1], so
        -log(1 - p_virtual) = softplus(v - logsumexp(z)).
    Minimising it keeps known-class features inside the principal subspace fitted on earlier data.
    The energy logsumexp(z) is detached (energy_grad=False, default): otherwise the loss is lowered without bound by
    scaling the logits up, and the classifier rows of the classes it is applied to grow at every CIL step (LayoutLMv3:
    mean row norm 0.65 -> 3.3 over four steps, and the step-5 class is never predicted). energy_grad=True is the
    original formulation (--ood_energy_grad), kept for the ablation.
    vim.fit_keys holds the fitted classes (class names if `class_names` maps label ids to names, else label ids).
    Returns None if the ViM is not fitted or no sample belongs to a fitted class.
    """
    if vim is None or not getattr(vim, "initialized", False):
        return None
    keys = getattr(vim, "fit_keys", None)
    if keys is not None:
        key_of = (lambda l: class_names[l]) if class_names is not None else (lambda l: l)
        keep = torch.tensor([key_of(l) in keys for l in labels.cpu().tolist()], device=features.device)
        if not keep.any():
            return None
        features, logits = features[keep], logits[keep]
    v = vim.virtual_logit_tensor(features.float())
    energy = torch.logsumexp(logits.float(), dim=1)
    return F.softplus(v - (energy if energy_grad else energy.detach())).mean()
