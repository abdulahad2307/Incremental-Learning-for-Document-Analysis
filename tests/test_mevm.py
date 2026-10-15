"""Unit tests of the Multivariate EVM (utils/evm/mevm_classifier.py): the properties stated in
JOURNAL_EXTENSION_PLAN.md, section B."""
import math
import unittest

import numpy as np
import torch

import tests.runner  # noqa: F401  (puts the repo on sys.path)
from utils.evm.evm_classifier import EVMClassifier
from utils.evm.mevm_classifier import THETA_MAX, MultivariateEVM, mevm_nll_loss


def blobs(seed=0, n=40, d=4, centers=((0, 0), (4, 0), (0, 4))):
    rng = np.random.default_rng(seed)
    return {f"c{k}": rng.normal(0, 1, (n, d)).astype(np.float32) + np.r_[c, np.zeros(d - 2)].astype(np.float32)
            for k, c in enumerate(centers)}


def single_ev_mevm(scales, shapes, theta, point=(0.0,)):
    """An MEVM with one class and one extreme vector, set by hand (two modalities 'a', 'b')."""
    m = MultivariateEVM(theta=theta)
    m.modalities = ["a", "b"]
    p = np.asarray(point, dtype=np.float32)
    m.weibull_models = {"c": [({"a": p, "b": p}, np.asarray(scales, float), np.asarray(shapes, float))]}
    m.theta = {"c": theta}
    m.initialized = True
    return m


class TestMEVM(unittest.TestCase):
    def test_single_modality_equals_evm(self):
        data, test = blobs(0), blobs(1, n=15)
        evm = EVMClassifier(tailsize=0.3, max_fit_samples=50).fit(data)
        mevm = MultivariateEVM(tailsize=0.3, max_fit_samples=50).fit({c: {"fused": x} for c, x in data.items()})
        x = np.vstack(list(test.values()))
        p_evm, p_mevm = evm.predict_proba(x), mevm.predict_proba({"fused": x})
        for c in data:
            np.testing.assert_allclose(p_mevm[c], np.maximum(p_evm[c], 1e-8), rtol=1e-3, atol=1e-6)
        log_p = mevm.log_proba_tensor({"fused": torch.as_tensor(x)}).numpy()     # uncapped: same up to the floor
        for k, c in enumerate(mevm.weibull_models):
            ok = p_evm[c] > 1e-6
            np.testing.assert_allclose(log_p[ok, k], np.log(p_evm[c][ok]), rtol=1e-3, atol=1e-4)

    def test_theta_one_is_product(self):
        m = single_ev_mevm(scales=[1.5, 0.7], shapes=[2.0, 1.3], theta=1.0)
        x = {"a": torch.tensor([[0.8]]), "b": torch.tensor([[0.5]])}
        expected = math.exp(-(0.8 / 1.5) ** 2.0) * math.exp(-(0.5 / 0.7) ** 1.3)
        self.assertAlmostEqual(float(m.predict_proba_tensor(x)[0, 0]), expected, places=5)

    def test_large_theta_is_minimum(self):
        m = single_ev_mevm(scales=[1.5, 0.7], shapes=[2.0, 1.3], theta=THETA_MAX)
        x = {"a": torch.tensor([[0.8]]), "b": torch.tensor([[0.5]])}
        expected = min(math.exp(-(0.8 / 1.5) ** 2.0), math.exp(-(0.5 / 0.7) ** 1.3))
        self.assertAlmostEqual(float(m.predict_proba_tensor(x)[0, 0]), expected, places=2)

    def test_theta_estimation(self):
        rng = np.random.default_rng(0)
        a = rng.random(200)
        self.assertLess(MultivariateEVM.estimate_theta(np.c_[a, rng.random(200)]), 1.3)        # independent
        self.assertEqual(MultivariateEVM.estimate_theta(np.c_[a, a]), THETA_MAX)               # identical
        self.assertEqual(MultivariateEVM.estimate_theta(np.c_[a, -a]), 1.0)                    # negative: >= 1
        self.assertEqual(MultivariateEVM.estimate_theta(a[:, None]), 1.0)                      # one modality

    def test_fit_two_modalities_and_predict(self):
        img, txt = blobs(0), blobs(2)
        mevm = MultivariateEVM().fit({c: {"image": img[c], "text": txt[c]} for c in img})
        self.assertTrue(all(1.0 <= t <= THETA_MAX for t in mevm.theta.values()))
        x = {"image": img["c1"][:10], "text": txt["c1"][:10]}
        self.assertGreater(np.mean(np.array(mevm.predict(x, threshold=0.0)) == "c1"), 0.8)

    def test_loss_gradients_finite(self):
        img, txt = blobs(0), blobs(2)
        mevm = MultivariateEVM().fit({c: {"image": img[c], "text": txt[c]} for c in img})
        ev = mevm.weibull_models["c0"][0][0]               # a sample exactly on an extreme vector (distance 0)
        feats = {"image": torch.tensor(np.vstack([img["c0"][:5], ev["image"][None]]), requires_grad=True),
                 "text": torch.tensor(np.vstack([txt["c0"][:5], ev["text"][None]]), requires_grad=True)}
        labels = torch.zeros(6, dtype=torch.long)
        loss = mevm_nll_loss(mevm, feats, labels, class_names=["c0", "c1", "c2"])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(f.grad).all() for f in feats.values()))
        self.assertIsNone(mevm_nll_loss(mevm, feats, torch.full((6,), 9), class_names=list(range(10))))


if __name__ == "__main__":
    unittest.main()
