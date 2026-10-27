"""Domain-balanced sampling of the DIL joint upper bound (utils/domain_IL/joint_sampling.py)."""
import unittest

import torch
from torch.utils.data import TensorDataset

from utils.domain_IL.joint_sampling import domain_weights, joint_loader


def domain_ids(n_big, n_small, balance, seed=0):
    big = TensorDataset(torch.zeros(n_big, dtype=torch.long))
    small = TensorDataset(torch.ones(n_small, dtype=torch.long))
    return torch.cat([b[0] for b in joint_loader([big, small], 50, balance, seed, num_workers=0)])


class TestJointSampling(unittest.TestCase):
    def test_weights_sum_to_one_per_domain(self):
        w = domain_weights([1000, 10])
        self.assertAlmostEqual(w[:1000].sum().item(), 0.5)
        self.assertAlmostEqual(w[1000:].sum().item(), 0.5)

    def test_none_keeps_the_natural_mix(self):
        ids = domain_ids(1000, 10, "none")
        self.assertEqual(len(ids), 1010)
        self.assertEqual(ids.sum().item(), 10)                          # every sample exactly once

    def test_domain_balances_and_keeps_epoch_length(self):
        ids = domain_ids(1000, 10, "domain")
        self.assertEqual(len(ids), 1010)                                # same epoch length as "none"
        self.assertAlmostEqual(ids.float().mean().item(), 0.5, delta=0.06)

    def test_seeded(self):
        self.assertTrue(torch.equal(domain_ids(100, 5, "domain", 3), domain_ids(100, 5, "domain", 3)))

    def test_unknown_balance(self):
        with self.assertRaises(ValueError):
            joint_loader([TensorDataset(torch.zeros(2))], 1, "class")


if __name__ == "__main__":
    unittest.main()
