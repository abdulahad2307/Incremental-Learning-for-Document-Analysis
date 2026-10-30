"""Unit tests of the iEVM's Weibull fit (utils/ievm/ievm.py)."""
import os
import unittest

import numpy as np
from scipy.stats import weibull_min

import tests.runner  # noqa: F401  (puts the repo on sys.path)
from utils.ievm.ievm import IncrementalEVM

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


class TestWeibullFit(unittest.TestCase):
    def test_degenerate_default_start(self):
        # Half-distances of an EAML CIL extreme vector (specification, step 2) where scipy's default start ends at
        # shape 0.2, scale 5559: Psi >= 0.7 for any distance below ~37, i.e. every sample
        tails = np.load(os.path.join(ASSETS, "ievm_degenerate_tail.npy")).astype(np.float64)
        shape0, _, _ = weibull_min.fit(tails, floc=0)
        self.assertLess(shape0, 1)
        scale, shape = IncrementalEVM.weibull_mle(tails)
        self.assertGreater(shape, 5)
        self.assertLess(scale, tails.max() * 1.5)
        self.assertGreater(weibull_min.logpdf(tails, shape, scale=scale).sum(),
                           weibull_min.logpdf(tails, shape0, scale=weibull_min.fit(tails, floc=0)[2]).sum())

    def test_regular_fit_unchanged(self):
        rng = np.random.RandomState(0)
        for _ in range(5):
            tails = np.sort(rng.normal(20, 1.5, 300))[:90] / 2
            shape0, _, scale0 = weibull_min.fit(tails, floc=0)
            self.assertEqual(IncrementalEVM.weibull_mle(tails), (scale0, shape0))

    def test_set_cover_keeps_regular_evs(self):
        # Two separated classes: every kept EV has a regular fit (shape >= 1), and far-away samples are not included
        rng = np.random.RandomState(1)
        feats = {"a": rng.normal(0, 1, (40, 8)).astype(np.float32),
                 "b": rng.normal(3, 1, (40, 8)).astype(np.float32)}
        evm = IncrementalEVM(tailsize=0.3, ev_budget=10).fit(feats)
        for evs in evm.class_evs.values():
            self.assertTrue(all(ev[2] >= 1 for ev in evs))
        probs = evm.predict_proba(rng.normal(30, 1, (20, 8)))
        self.assertLess(max(p.max() for p in probs.values()), 0.1)


if __name__ == "__main__":
    unittest.main()
