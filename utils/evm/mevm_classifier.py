"""Multivariate Extreme Value Machine (MEVM): one Weibull per extreme vector and modality, joined by a Gumbel copula
(JOURNAL_EXTENSION_PLAN.md, section B).

  modalities j = 1..m (EAML: image, text, fused; LayoutLMv3: text, visual, cls)
  per extreme vector i and modality j: Weibull (kappa_ij, lambda_ij) on the half-distances to the nearest negatives
      t_ij(x) = (d_j(x_i, x) / lambda_ij) ** kappa_ij                     per-modality EVM energy
  joint inclusion (Gumbel copula, theta >= 1):
      -log Psi_i(x) = ( sum_j t_ij ** theta ) ** (1 / theta)              l_theta-norm of the energies
  class score:  P(c | x) = max_{i in c} Psi_i(x)

theta is estimated per class, not tuned: theta = 1 / (1 - tau), with tau Kendall's tau between the modality margins
(each training sample's half-distance to its nearest negative), averaged over modality pairs, theta >= 1.
With m = 1 this is exactly EVMClassifier; theta = 1 gives the product of the modality inclusions; theta -> infinity
gives their minimum. Everything is computed in log space (-log Psi = exp(logsumexp(theta * log t) / theta)).
"""
import itertools
import math

import numpy as np
import torch
from scipy.stats import kendalltau

from .evm_classifier import EVMClassifier

THETA_MAX = 50.0                 # tau -> 1 means fully dependent modalities; beyond this the l_theta-norm is the max
_MAX_NLL = -math.log(1e-8)       # cap of the loss, as evm_nll_loss


class MultivariateEVM(EVMClassifier):
    """EVMClassifier interface (fit / predict_proba / predict / weibull_models / initialized), with per-modality
    features: every features argument is {class: {modality: (n, d_modality) array}} for fitting and
    {modality: (N, d_modality)} for scoring, with the same samples in the same order in every modality."""

    def __init__(self, tailsize=0.3, cover_threshold=0.7, max_fit_samples=50, theta=None, batch_size=128):
        super().__init__(tailsize=tailsize, cover_threshold=cover_threshold, batch_size=batch_size,
                         max_fit_samples=max_fit_samples)
        self.fixed_theta = theta          # None: estimate per class from Kendall's tau
        self.theta = {}                   # class -> theta
        self.modalities = None

    # ---------------------------------------------------------------- fitting
    def _tail_count(self, n):
        return (max(int(n * self.tailsize), 1) if isinstance(self.tailsize, float) and self.tailsize < 1.0
                else min(int(self.tailsize), n))

    @staticmethod
    def estimate_theta(margins):
        """theta = 1 / (1 - tau), tau = mean pairwise Kendall's tau between the modality margins (n, m); >= 1."""
        m = margins.shape[1]
        if m < 2 or margins.shape[0] < 3:
            return 1.0
        taus = [kendalltau(margins[:, a], margins[:, b])[0] for a, b in itertools.combinations(range(m), 2)]
        tau = float(np.nanmean(taus)) if not all(np.isnan(taus)) else 0.0
        return float(min(max(1.0 / (1.0 - tau), 1.0), THETA_MAX)) if tau < 1 else THETA_MAX

    def fit(self, features):
        classes = list(features)
        self.modalities = list(features[classes[0]])
        pruned = {}
        for c in classes:
            arrs = {j: np.asarray(features[c][j], dtype=np.float32) for j in self.modalities}
            n = len(next(iter(arrs.values())))
            idx = (np.linspace(0, n - 1, self.max_fit_samples).astype(int)
                   if self.max_fit_samples and n > self.max_fit_samples else np.arange(n))
            pruned[c] = {j: a[idx] for j, a in arrs.items()}          # same samples in every modality
        self.weibull_models, self.theta = {}, {}
        for c in classes:
            others = [o for o in classes if o != c] or [c]
            n_c = len(pruned[c][self.modalities[0]])
            scales = np.zeros((n_c, len(self.modalities)))
            shapes = np.zeros((n_c, len(self.modalities)))
            margins = np.zeros((n_c, len(self.modalities)))
            for jj, j in enumerate(self.modalities):
                neg = torch.from_numpy(np.vstack([pruned[o][j] for o in others]))
                d = torch.cdist(torch.from_numpy(pruned[c][j]), neg).numpy()
                for i in range(n_c):
                    tail = np.sort(d[i])[:self._tail_count(d.shape[1])] / 2.0      # half-distances to nearest negatives
                    scales[i, jj], shapes[i, jj] = self._fit_weibull(tail)
                    margins[i, jj] = tail[0]
            self.theta[c] = self.fixed_theta if self.fixed_theta is not None else self.estimate_theta(margins)
            self.weibull_models[c] = [({j: pruned[c][j][i] for j in self.modalities}, scales[i], shapes[i])
                                      for i in range(n_c)]
        self.initialized = True
        return self

    # ---------------------------------------------------------------- scoring
    def _class_tensors(self, c, device):
        evs = self.weibull_models[c]
        points = [torch.tensor(np.stack([ev[0][j] for ev in evs]), dtype=torch.float32, device=device)
                  for j in self.modalities]
        scales = torch.tensor(np.stack([ev[1] for ev in evs]), dtype=torch.float32, device=device)   # (M, m)
        shapes = torch.tensor(np.stack([ev[2] for ev in evs]), dtype=torch.float32, device=device)
        return points, scales, shapes

    def log_energies(self, features, c):
        """log t_ij(x) = kappa_ij * (log d_j(x_i, x) - log lambda_ij) of class c: (N, M extreme vectors, m) tensor."""
        device = next(iter(features.values())).device
        points, scales, shapes = self._class_tensors(c, device)
        return torch.stack([shapes[:, jj] * (torch.log(torch.cdist(features[j].float(), points[jj]) + 1e-12)
                                              - torch.log(scales[:, jj] + 1e-12))
                            for jj, j in enumerate(self.modalities)], dim=-1)

    def class_nll(self, features, c, cap=True):
        """-log Psi_c(x) = min_i || t_i(x) ||_theta for every sample: (N,) tensor, differentiable in `features`.
        cap: clip at -log(1e-8) as evm_nll_loss (training); False keeps far-away samples ranked (evaluation)."""
        theta = float(self.theta[c])
        log_t = self.log_energies(features, c)
        log_norm = torch.logsumexp(theta * log_t, dim=-1) / theta            # log || t_i ||_theta
        if cap:
            log_norm = torch.clamp(log_norm, max=math.log(_MAX_NLL))
        return torch.exp(log_norm).min(dim=1).values

    def predict_proba_tensor(self, features):
        """(N, n_classes) inclusion probabilities, columns in the order of weibull_models (CPU tensor)."""
        with torch.no_grad():
            return torch.stack([torch.exp(-self.class_nll(features, c)) for c in self.weibull_models], dim=1).cpu()

    def log_proba_tensor(self, features):
        """(N, n_classes) log inclusion probabilities, uncapped: unlike the probabilities they do not underflow to a
        common floor far from all extreme vectors, so they rank every sample (open-set scores)."""
        with torch.no_grad():
            return torch.stack([-self.class_nll(features, c, cap=False) for c in self.weibull_models], dim=1).cpu()

    def predict_proba(self, features):
        feats = {j: torch.as_tensor(np.asarray(v), dtype=torch.float32) for j, v in features.items()}
        probs = self.predict_proba_tensor(feats).numpy()
        return {c: probs[:, k] for k, c in enumerate(self.weibull_models)}

    def predict(self, features, threshold=None):
        """Class with the highest inclusion probability, or "unknown" below `threshold` (cover_threshold)."""
        threshold = self.cover_threshold if threshold is None else threshold
        probs = self.predict_proba(features)
        classes = list(probs)
        stacked = np.stack([probs[c] for c in classes], axis=1)
        best = stacked.argmax(axis=1)
        return [classes[b] if stacked[i, b] >= threshold else "unknown" for i, b in enumerate(best)]


def mevm_nll_loss(mevm, features, labels, class_names=None):
    """Differentiable MEVM loss, drop-in for evm_nll_loss: mean -log Psi_true(x) = min_i || t_i(x) ||_theta over the
    samples whose class the MEVM has been fitted on. features: {modality: (N, d) tensor attached to the graph}.
    Returns None if no sample's class is known to the MEVM."""
    if mevm is None or not getattr(mevm, "initialized", False):
        return None
    key_of = (lambda l: class_names[l]) if class_names is not None else (lambda l: l)
    keys = [key_of(l) for l in labels.cpu().tolist()]
    losses = []
    for c in set(k for k in keys if k in mevm.weibull_models):
        idx = torch.tensor([i for i, k in enumerate(keys) if k == c], device=next(iter(features.values())).device)
        losses.append(mevm.class_nll({j: f[idx] for j, f in features.items()}, c))
    return torch.cat(losses).mean() if losses else None
