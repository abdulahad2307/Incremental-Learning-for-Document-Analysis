"""Unit tests of the training-time L_OOD (utils/ood/ood_loss.py)."""
import unittest

import numpy as np
import torch

import tests.runner  # noqa: F401  (puts the repo on sys.path)
from utils.ood.ood_loss import vim_ood_loss
from utils.ood.vim import VIM_OOD

U = np.array([0.5, -0.5, 0.2, 0.1])


def toy_vim():
    """ViM with origin U and the residual space = the last two of four dimensions."""
    vim = VIM_OOD()
    vim.u, vim.NS, vim.alpha = U, np.eye(4)[:, 2:], 2.0
    vim.fit_keys = None
    vim.initialized = True
    return vim


class TestVimOodLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.features = torch.randn(8, 4, requires_grad=True)
        self.W = torch.randn(3, 4, requires_grad=True)
        self.labels = torch.zeros(8, dtype=torch.long)

    def loss(self, grad, logits=None):
        logits = self.features @ self.W.T if logits is None else logits
        return vim_ood_loss(toy_vim(), self.features, logits, self.labels, grad=grad)

    def grads(self, grad):
        """Gradients of the features and the classifier, and each feature gradient's component along f - u."""
        self.features.grad, self.W.grad = None, None
        self.loss(grad).backward()
        radial = (self.features.grad * (self.features.detach() - torch.tensor(U).float())).sum(1)
        return self.features.grad, self.W.grad, radial

    def test_direction_only_turns_the_features(self):
        f_grad, w_grad, radial = self.grads("direction")
        self.assertIsNone(w_grad)
        self.assertGreater(f_grad.abs().sum().item(), 0)
        self.assertLess(radial.abs().max().item(), 1e-6)

    def test_radial_shrinks_the_features(self):
        _, w_grad, radial = self.grads("radial")
        self.assertIsNone(w_grad)
        self.assertGreater(radial.min().item(), 0)   # gradient descent moves every feature towards u

    def test_full_raises_the_logits(self):
        _, w_grad, _ = self.grads("full")
        self.assertGreater(w_grad.abs().sum().item(), 0)
        z = (self.features @ self.W.T).detach()
        self.assertLess(self.loss("full", z + 2).item(), self.loss("full", z).item())

    def test_same_value_for_all(self):
        values = [self.loss(g).item() for g in ("direction", "radial", "full")]
        self.assertAlmostEqual(min(values), max(values), places=5)

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            self.loss("energy")


if __name__ == "__main__":
    unittest.main()
