import numpy as np
from scipy.special import logsumexp


class VIM_OOD:
    """ViM: Virtual-logit Matching (Wang et al., CVPR 2022), following the official implementation.

    fit():   origin u = -pinv(W) @ b of the last linear layer (feature mean if W/b are not given);
             principal subspace = top `dim` eigenvectors of the centred feature covariance;
             alpha = mean max-logit / mean residual norm on the fitting (ID training) data.
    score(): energy(logits) - alpha * ||residual||. Higher = more in-distribution.
    """

    name = "vim"

    def __init__(self, dim=None, **kwargs):
        self.dim = dim
        self.initialized = False

    def fit(self, feats, logits, W=None, b=None):
        feats = np.asarray(feats, dtype=np.float64)
        logits = np.asarray(logits, dtype=np.float64)
        D = feats.shape[1]
        dim = self.dim or (1000 if D >= 2048 else 512 if D >= 768 else D // 2)
        if dim >= D:
            raise ValueError(f"ViM principal dim {dim} must be smaller than feature dim {D}")
        if feats.shape[0] < 2 * D:
            print(f"Warning: ViM fitted on {feats.shape[0]} samples for {D}-dim features; the residual space is "
                  f"poorly estimated (use >= {2 * D} samples, e.g. a larger --ood_max_per_class).")
        if W is not None and b is not None:
            self.u = -np.linalg.pinv(np.asarray(W, dtype=np.float64)) @ np.asarray(b, dtype=np.float64)
        else:
            self.u = feats.mean(axis=0)
        X = feats - self.u
        cov = X.T @ X / X.shape[0]
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        order = np.argsort(eig_vals)[::-1]
        self.NS = np.ascontiguousarray(eig_vecs[:, order[dim:]])  # residual (null) space
        vlogit = np.linalg.norm(X @ self.NS, axis=1)
        self.alpha = logits.max(axis=1).mean() / vlogit.mean()
        self.dim_used = dim
        self.initialized = True
        return self

    def virtual_logit_tensor(self, feats):
        """alpha * ||residual|| as a differentiable torch tensor (for the training-time L_OOD)."""
        import torch
        u = torch.as_tensor(self.u, dtype=feats.dtype, device=feats.device)
        NS = torch.as_tensor(self.NS, dtype=feats.dtype, device=feats.device)
        return torch.linalg.norm((feats - u) @ NS, dim=1) * float(self.alpha)

    def score(self, feats, logits):
        feats = np.asarray(feats, dtype=np.float64)
        vlogit = np.linalg.norm((feats - self.u) @ self.NS, axis=1) * self.alpha
        energy = logsumexp(np.asarray(logits, dtype=np.float64), axis=1)
        return energy - vlogit
