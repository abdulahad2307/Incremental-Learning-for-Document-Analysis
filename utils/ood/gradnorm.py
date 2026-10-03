import numpy as np
from scipy.special import softmax


class GradNorm_OOD:
    """GradNorm (Huang et al., NeurIPS 2021).

    The score is the L1 norm of the gradient of KL(uniform || softmax(z / T)) w.r.t. the last linear
    layer's weights, per sample. For z = W h + b that gradient is (p - 1/C) h^T / T, so
        score = (1/T) * sum_c |p_c - 1/C| * sum_j |h_j|
    computed exactly from the penultimate features h and the logits z (no backward pass needed).
    Only valid when the logits come from a linear layer applied to `feats` (true for EAML:
    logits = fusion_classifier(fused features)). Higher = more in-distribution.
    """

    name = "gradnorm"

    def __init__(self, temperature=1.0, **kwargs):
        self.temperature = temperature
        self.initialized = False

    def fit(self, feats, logits, W=None, b=None):
        # GradNorm needs no fitting
        self.initialized = True
        return self

    def score(self, feats, logits):
        logits = np.asarray(logits, dtype=np.float64)
        feats = np.asarray(feats, dtype=np.float64)
        C = logits.shape[1]
        p = softmax(logits / self.temperature, axis=1)
        return np.abs(p - 1.0 / C).sum(axis=1) * np.abs(feats).sum(axis=1) / self.temperature
