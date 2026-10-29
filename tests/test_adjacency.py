"""Adjacency-aware replay (utils/class_IL/adjacency.py): the allocation, the per-class quotas of both exemplar
managers, the replay weighting, and one CIL step per backbone with --exemplar_allocation adjacency (with balanced and
with allocation-proportional replay weighting)."""
import glob
import os
import unittest

import numpy as np

from tests.fixtures import CIL_BASE, CIL_NEW
from tests.runner import PipelineTestCase, check_predictions, out_dir, run_script
from tests.test_eaml_cil import cil_args as eaml_args
from tests.test_layout_cil import cil_args as layout_args
from utils.class_IL.adjacency import allocate, class_adjacency, replay_weights


class TestAllocation(unittest.TestCase):
    def test_budget_minimum_and_order(self):
        adj = {"near": 0.9, "mid": 0.4, "far": 0.0, "farther": -0.3}
        n = allocate(adj, 80, tau=0.2, n_min=10)
        self.assertEqual(sum(n.values()), 80)                                  # the uniform budget, unchanged
        self.assertTrue(all(v >= 10 for v in n.values()))
        self.assertGreater(n["near"], n["mid"])
        self.assertGreaterEqual(n["mid"], n["far"])
        self.assertGreaterEqual(n["far"], n["farther"])

    def test_equal_adjacency_is_uniform(self):
        self.assertEqual(set(allocate({c: 0.5 for c in "abcd"}, 80).values()), {20})

    def test_small_budget(self):
        n = allocate({"a": 1.0, "b": 0.0}, 6, n_min=10)                        # minimum capped at budget / classes
        self.assertEqual(sum(n.values()), 6)
        self.assertEqual(n["b"], 3)

    def test_adjacency(self):
        rng = np.random.default_rng(0)
        new = rng.normal(0, 0.1, (50, 8)) + np.eye(8)[0]
        adj = class_adjacency({"same": rng.normal(0, 0.1, (50, 8)) + np.eye(8)[0],
                               "other": rng.normal(0, 0.1, (50, 8)) + np.eye(8)[1]}, new)
        self.assertGreater(adj["same"], 0.9)
        self.assertLess(adj["other"], 0.2)


class TestReplayWeights(unittest.TestCase):
    LABELS = ["near"] * 60 + ["far"] * 20 + ["new"] * 500

    def share(self, w, c):
        return float(w[np.array(self.LABELS) == c].sum())

    def test_balanced_draws_classes_equally(self):
        w = replay_weights(self.LABELS, ["new"], "balanced")
        for c in ("near", "far", "new"):
            self.assertAlmostEqual(self.share(w, c), 1 / 3)

    def test_allocation_follows_the_budget(self):
        w = replay_weights(self.LABELS, ["new"], "allocation")
        self.assertAlmostEqual(self.share(w, "new"), 1 / 3)                     # the new class keeps its share
        self.assertAlmostEqual(self.share(w, "near"), 2 / 3 * 60 / 80)          # old classes by their exemplars
        self.assertAlmostEqual(self.share(w, "far"), 2 / 3 * 20 / 80)

    def test_uniform_budget_is_balanced(self):
        labels = ["a"] * 20 + ["b"] * 20 + ["new"] * 300
        self.assertTrue(np.allclose(replay_weights(labels, ["new"], "allocation"),
                                    replay_weights(labels, ["new"], "balanced")))

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            replay_weights(["a"], [], "inverse")


class TestQuotas(unittest.TestCase):
    def test_eaml_manager(self):
        from utils.class_IL.cil_utils import ExemplarManager

        class DS:
            samples = [(f"{c}/{i}", None, c) for c in "ab" for i in range(30)]
        m = ExemplarManager(max_exemplars=40, max_per_class=20, selection_strategy="random")
        m.per_class = {"a": 28, "b": 12}
        m.update(DS(), "a")
        m.update(DS(), "b")
        self.assertEqual((len(m.exemplars["a"]), len(m.exemplars["b"])), (28, 12))   # no rebalancing
        u = ExemplarManager(max_exemplars=40, max_per_class=20, selection_strategy="random")
        u.update(DS(), "a")
        self.assertEqual(len(u.exemplars["a"]), 20)                                   # uniform as before


class TestAdjacencyPipeline(PipelineTestCase):
    def check(self, out, method="Adjacency replay"):
        from utils.eval.predictions import load_predictions
        check_predictions(self, out, f"cil_{CIL_NEW}", ["seen", "unseen"], method=method)
        meta = load_predictions(glob.glob(os.path.join(out, "predictions", f"cil_{CIL_NEW}_seed*.pt"))[0])["meta"]
        alloc = meta["allocation"]
        self.assertEqual(sorted(alloc["exemplars"]), sorted(CIL_BASE))
        self.assertEqual(set(alloc["adjacency"]), set(CIL_BASE))

    def test_eaml(self):
        out = out_dir("adjacency", "eaml")
        base = os.path.join(self.data, "ckpt", "eaml_base_3.pt")
        run_script(self, "src/class_incremental/eaml/class_incremental.py",
                   eaml_args(self.data, base, CIL_BASE, CIL_NEW, out, "standard") + ["--exemplar_allocation", "adjacency"],
                   cwd=out, expect_files=[os.path.join(out, f"best_model_{CIL_NEW}.pth")],
                   expect_logs=["Adjacency-aware replay"])
        self.check(out)

    def test_layout(self):
        out = out_dir("adjacency", "layout")
        base = os.path.join(self.data, "ckpt", "llmv3_base_3.pt")
        run_script(self, "src/class_incremental/llmv3/llmv3_class_incremental.py",
                   layout_args(self.data, base, CIL_BASE, CIL_NEW, out, "standard") + ["--exemplar_allocation", "adjacency"],
                   cwd=out, expect_logs=["Adjacency-aware replay"])
        self.check(out)

    def test_eaml_weighted(self):
        out = out_dir("adjacency", "eaml_weighted")
        base = os.path.join(self.data, "ckpt", "eaml_base_3.pt")
        run_script(self, "src/class_incremental/eaml/class_incremental.py",
                   eaml_args(self.data, base, CIL_BASE, CIL_NEW, out, "standard")
                   + ["--exemplar_allocation", "adjacency", "--replay_weighting", "allocation"],
                   cwd=out, expect_files=[os.path.join(out, f"best_model_{CIL_NEW}.pth")],
                   expect_logs=["Adjacency-aware replay", "Replay weighting 'allocation'"])
        self.check(out, "Adjacency replay (weighted)")

    def test_layout_weighted(self):
        out = out_dir("adjacency", "layout_weighted")
        base = os.path.join(self.data, "ckpt", "llmv3_base_3.pt")
        run_script(self, "src/class_incremental/llmv3/llmv3_class_incremental.py",
                   layout_args(self.data, base, CIL_BASE, CIL_NEW, out, "standard")
                   + ["--exemplar_allocation", "adjacency", "--replay_weighting", "allocation"],
                   cwd=out, expect_logs=["Adjacency-aware replay", "replay weighting 'allocation'"])
        self.check(out, "Adjacency replay (weighted)")


if __name__ == "__main__":
    unittest.main()
