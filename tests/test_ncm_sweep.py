"""Unit tests of the frozen-backbone NCM baseline (tools/eval/ncm_baseline.py) and the post-hoc EVT sweep
(tools/eval/posthoc_evt_sweep.py) on a small synthetic feature file in the format of extract_step_features.py."""
import os
import tempfile
import unittest

import numpy as np
import torch

import tests.runner  # noqa: F401  (puts the repo on sys.path)
from tools.eval import ncm_baseline, posthoc_evt_sweep

BASE, ORDER = ["a", "b", "c"], ["d", "e"]


def feature_file(path, unit_norm=False, n_train=60, n_test=30, d=8):
    """Five well-separated Gaussian classes; the checkpoint knows the base classes, training features of all."""
    rng = np.random.default_rng(0)
    centers = {c: 6 * np.eye(d)[k] for k, c in enumerate(BASE + ORDER)}

    def split(classes, n, prefix):
        x = np.vstack([rng.normal(0, 1, (n, d)) + centers[c] for c in classes]).astype(np.float32)
        if unit_norm:
            x /= np.linalg.norm(x, axis=1, keepdims=True)
        labels = [c for c in classes for _ in range(n)]
        return {"features": {"image": x, "text": x, "fused": x.astype(np.float16)},
                "logits": np.zeros((len(x), len(BASE)), dtype=np.float16), "labels": labels,
                "doc_id": [f"{c}/{prefix}{i}.tif" for i, c in enumerate(labels)]}
    torch.save({"meta": {"checkpoint": "/x/eaml_best_model.pt", "backbone": "eaml", "llmv3_model": ""},
                "known_classes": BASE, "unknown_classes": ORDER,
                "splits": {"train": split(BASE + ORDER, n_train, "tr"), "test_seen": split(BASE, n_test, "ts"),
                           "test_unseen": split(ORDER, n_test, "tu")}}, path)
    return path


class TestNCM(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = feature_file(os.path.join(self.dir.name, "base_all_cil.pt"))

    def tearDown(self):
        self.dir.cleanup()

    def test_steps_metrics_and_predictions(self):
        report = ncm_baseline.run(self.path, ORDER, "eaml", 42, run_root=self.dir.name)
        for name, v in report["variants"].items():
            s = v["summary"]
            self.assertEqual(s["steps"], len(ORDER))
            self.assertGreater(s["acc_final"], 0.95, name)              # separable classes, nothing is trained
            self.assertLess(abs(s["forgetting"]), 0.05, name)
            self.assertGreater(s["open_auroc"], 0.9, name)
        pred_dir = os.path.join(self.dir.name, "cil", "eaml", "ncm_cos_all_frozen", "seed42", "predictions")
        files = sorted(os.listdir(pred_dir))
        self.assertEqual(files, [f"cil_{c}_seed42.pt" for c in sorted(ORDER)])
        p = torch.load(os.path.join(pred_dir, "cil_d_seed42.pt"), weights_only=False)
        self.assertEqual(p["output_classes"], BASE + ["d"])
        self.assertEqual(p["meta"]["step"], 1)
        self.assertEqual(p["sets"]["unseen"]["class_names"], ["e"])

    def test_means_from_subset(self):
        x = np.arange(10, dtype=np.float32)[:, None]
        labels = np.array(["a"] * 10, dtype=object)
        docs = [f"a/{i}.tif" for i in range(10)]
        self.assertEqual(ncm_baseline.class_means(x, labels, docs, ["a"])["a"][0], 4.5)
        sub = ncm_baseline.class_means(x, labels, docs, ["a"], n=3)["a"][0]
        self.assertEqual(sub, ncm_baseline.class_means(x, labels, docs, ["a"], n=3)["a"][0])   # reproducible
        self.assertNotEqual(sub, 4.5)

    def test_missing_training_features(self):
        data = torch.load(self.path, weights_only=False)
        keep = [l in BASE for l in data["splits"]["train"]["labels"]]
        tr = data["splits"]["train"]
        data["splits"]["train"] = {"features": {k: v[keep] for k, v in tr["features"].items()},
                                   "logits": tr["logits"][keep], "labels": [l for l, k in zip(tr["labels"], keep) if k],
                                   "doc_id": [d for d, k in zip(tr["doc_id"], keep) if k]}
        torch.save(data, self.path)
        with self.assertRaises(ValueError):
            ncm_baseline.run(self.path, ORDER, "eaml", 42, run_root=None)


class TestSweep(unittest.TestCase):
    def test_configurations(self):
        cfgs = posthoc_evt_sweep.configurations()
        self.assertEqual(cfgs[0], posthoc_evt_sweep.DEFAULT)
        keys = [tuple(sorted(c.items(), key=lambda kv: kv[0])) for c in cfgs]
        self.assertEqual(len(keys), len(set(keys)))

    def test_cosine_equals_euclidean_on_unit_features(self):
        with tempfile.TemporaryDirectory() as d:
            report = posthoc_evt_sweep.run_file(feature_file(os.path.join(d, "f.pt"), unit_norm=True, n_train=30))
        rows = {(r["tailsize"], r["max_fit_samples"], r["distance"]): r for r in report["rows"]}
        for t in posthoc_evt_sweep.TAILSIZES:
            e, c = rows[(t, 50, "euclidean")], rows[(t, 50, "cosine")]
            for k in ("acc", "auroc", "fpr95", "oscr"):
                self.assertAlmostEqual(e[k], c[k], places=2, msg=f"tail {t} {k}")
        self.assertIn("cover", rows[(0.3, 50, "euclidean")])


if __name__ == "__main__":
    unittest.main()
