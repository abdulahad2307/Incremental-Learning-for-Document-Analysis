import numpy as np
from scipy.special import softmax


class MSP_OOD:
    """Maximum Softmax Probability (Hendrycks & Gimpel, 2017).

    Score = max_c softmax(logits)_c. Higher = more in-distribution.
    """

    name = "msp"

    def __init__(self, **kwargs):
        self.initialized = False

    def fit(self, feats, logits, W=None, b=None):
        # MSP needs no fitting
        self.initialized = True
        return self

    def score(self, feats, logits):
        return softmax(np.asarray(logits, dtype=np.float64), axis=1).max(axis=1)
