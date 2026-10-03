import numpy as np
from scipy.optimize import minimize

from utils.evm.evm_classifier import EVMClassifier


class RegularizedEVMClassifier(EVMClassifier):
    """RegEVM: EVM whose per-extreme-vector Weibull fit is regularised on the scale (Eq. 10):

        (kappa_i, lambda_i) = argmin  - sum_j log f_Weibull(d_j; kappa, lambda) + alpha * lambda^2

    over the tail of half-distances d_j of extreme vector i. alpha = lambda_reg. Large scales (overly permissive class
    tails) are penalised; alpha = 0 gives the standard maximum-likelihood EVM fit. The penalty is in the units of the
    feature distances, so alpha has to be chosen relative to the feature scale.
    Used by fit() and fit_cuda() (the fitting hook of EVMClassifier).
    """

    def __init__(self, lambda_reg=0.1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_reg = lambda_reg  # alpha in Eq. 10

    def _fit_weibull(self, tail):
        scale0, shape0 = super()._fit_weibull(tail)
        if self.lambda_reg <= 0:
            return scale0, shape0
        d = np.maximum(np.asarray(tail, dtype=np.float64), 1e-12)
        log_d = np.log(d)
        n = len(d)

        def objective(theta):  # theta = (log shape, log scale) keeps both positive
            k, lam = np.exp(theta)
            z = np.exp(np.clip(k * (log_d - np.log(lam)), -700, 700))  # (d/lam)^k
            nll = -(n * np.log(k) - n * k * np.log(lam) + (k - 1) * log_d.sum() - z.sum())
            return nll + self.lambda_reg * lam ** 2

        theta0 = np.log([max(shape0, 1e-6), max(scale0, 1e-12)])
        res = minimize(objective, theta0, method="L-BFGS-B")
        if not res.success or not np.all(np.isfinite(res.x)):
            return scale0, shape0
        shape, scale = np.exp(res.x)
        return float(scale), float(shape)
