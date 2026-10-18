"""Unit tests of the drift analysis (tools/eval/drift_analysis.py): the measures on constructed data, and a run over
synthetic step files in the format of extract_step_features.py."""
import os
import tempfile
import unittest

import numpy as np
import torch

import tests.runner  # noqa: F401  (puts the repo on sys.path)
from tools.eval import drift_analysis as da

BASE, ORDER = ["a", "b", "c", "d"], ["e", "f"]


def step_file(path, known, drift, rng, d=8, n_train=40, n_test=30):
    """One step: "stable" keeps the class structure, "drifting" loses it with `drift` (0 = step 0).
    The classifier predicts the true class for the base classes with probability 1 - drift."""
    classes = BASE + ORDER
    centers = {c: 5 * np.eye(d)[k] for k, c in enumerate(classes)}
    noise = np.random.default_rng(1)          # the same documents (and noise) at every step

    def split(cls, n, prefix):
        labels = [c for c in cls for _ in range(n)]
        base = np.vstack([noise.normal(0, 1, (n, d)) + centers[c] for c in cls]).astype(np.float32)
        shrink = np.array([1 - drift if c in BASE else 1.0 for c in labels], dtype=np.float32)[:, None]
        drifting = base * shrink + (1 - shrink) * base.mean(axis=0)     # base classes move towards the centre
        logits = np.zeros((len(labels), len(known)), dtype=np.float32)
        for i, c in enumerate(labels):
            j = known.index(c) if c in known and rng.random() >= (drift if c in BASE else 0) else rng.integers(len(known))
            logits[i, j] = 1
        return {"features": {"stable": base, "drifting": drifting}, "logits": logits, "labels": labels,
                "doc_id": [f"{c}/{prefix}{i}.tif" for i, c in enumerate(labels)]}
    unknown = [c for c in classes if c not in known]
    splits = {"train": split(classes, n_train, "tr"), "test_seen": split(known, n_test, "ts")}
    if unknown:
        splits["test_unseen"] = split(unknown, n_test, "tu")
    torch.save({"meta": {"checkpoint": path, "backbone": "eaml", "llmv3_model": ""}, "known_classes": known,
                "unknown_classes": unknown, "splits": splits}, path)
    return path


class TestMeasures(unittest.TestCase):
    def test_cka(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 6))
        q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        self.assertAlmostEqual(da.linear_cka(x, x), 1.0, places=6)
        self.assertAlmostEqual(da.linear_cka(x, 3 * x @ q), 1.0, places=6)          # rotation and scale
        self.assertLess(da.linear_cka(x, rng.normal(size=(200, 6))), 0.2)            # unrelated

    def test_margins(self):
        x = np.array([[0.0], [0.0], [4.0], [5.0]])
        m = da.class_margins(x, ["a", "a", "b", "b"], ["a", "b"])
        self.assertAlmostEqual(m["a"], 2.0)                                           # half of the distance 4
        self.assertAlmostEqual(m["b"], 2.25)                                          # median of 2, 2.5

    def test_ncm(self):
        tr = np.array([[1.0, 0.0], [0.0, 1.0]])
        te = np.array([[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]])
        acc = da.ncm_accuracy(tr, ["a", "b"], te, ["a", "b", "a"], ["a", "b"], ["a"])
        self.assertAlmostEqual(acc, 0.5)

    def test_spearman_sign(self):
        x = np.arange(20.0)
        rho, lo, hi = da.spearman_ci(x, 2 * x + 1)
        self.assertAlmostEqual(rho, 1.0)
        self.assertTrue(np.isnan(da.spearman_ci([1, 2, 3], [1, 2, 3])[0]))           # too few pairs


class TestRun(unittest.TestCase):
    def test_synthetic_run(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            paths = {0: step_file(os.path.join(d, "s0.pt"), BASE, 0.0, rng)}
            for k, drift in ((1, 0.3), (2, 0.6)):
                paths[k] = step_file(os.path.join(d, f"s{k}.pt"), BASE + ORDER[:k], drift, rng)
            steps = {k: da.load_step(p) for k, p in paths.items()}
            rows, crows = da.analyse_run("eaml", "toy", steps)
        m = {(r[2], r[3], r[4]): r[5] for r in rows}
        self.assertAlmostEqual(m[(2, "stable", "cka_base")], 1.0, places=5)
        self.assertAlmostEqual(m[(2, "stable", "margin_rel")], 1.0, places=5)
        self.assertLess(m[(2, "drifting", "margin_rel")], m[(1, "drifting", "margin_rel")])   # margins shrink
        self.assertLess(m[(2, "drifting", "margin_rel")], 0.9)
        self.assertLess(m[(2, "classifier", "acc_old")], m[(0, "classifier", "acc_old")])     # and it forgets
        self.assertIn((2, "all", "evt_theta"), m)
        self.assertTrue(all(c[4] in BASE + ORDER for c in crows))


if __name__ == "__main__":
    unittest.main()
