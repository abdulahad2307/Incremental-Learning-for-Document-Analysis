"""Unit tests of the training-time L_OOD (utils/ood/ood_loss.py)."""
import unittest

import torch

import tests.runner  # noqa: F401  (puts the repo on sys.path)
from utils.ood.ood_loss import vim_ood_loss


class NormVim:
    """ViM stand-in: virtual logit = norm of the features."""
    initialized = True
    fit_keys = None

    def virtual_logit_tensor(self, features):
        return features.norm(dim=1)


class TestVimOodLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.features = torch.randn(8, 4, requires_grad=True)
        self.W = torch.randn(3, 4, requires_grad=True)
        self.labels = torch.zeros(8, dtype=torch.long)

    def grads(self, energy_grad):
        self.features.grad, self.W.grad = None, None
        vim_ood_loss(NormVim(), self.features, self.features @ self.W.T, self.labels, energy_grad=energy_grad).backward()
        return self.features.grad, self.W.grad

    def test_no_gradient_into_the_classifier(self):
        f_grad, w_grad = self.grads(energy_grad=False)
        self.assertIsNone(w_grad)
        self.assertGreater(f_grad.abs().sum().item(), 0)   # it still acts on the features, through the residual

    def test_original_formulation_rewards_larger_logits(self):
        f, z = self.features.detach(), self.features.detach() @ self.W.detach().T
        self.assertLess(vim_ood_loss(NormVim(), f, z + 2, self.labels, energy_grad=True),
                        vim_ood_loss(NormVim(), f, z, self.labels, energy_grad=True))
        _, w_grad = self.grads(energy_grad=True)
        self.assertGreater(w_grad.abs().sum().item(), 0)

    def test_same_value_both_ways(self):
        z = self.features @ self.W.T
        self.assertAlmostEqual(vim_ood_loss(NormVim(), self.features, z, self.labels).item(),
                               vim_ood_loss(NormVim(), self.features, z, self.labels, energy_grad=True).item(), places=6)


if __name__ == "__main__":
    unittest.main()
