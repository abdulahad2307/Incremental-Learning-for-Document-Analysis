"""Unit tests of the evaluation module (utils/metrics/) on small hand-made inputs with known answers."""
import os
import tempfile
import unittest

import numpy as np
import torch

import tests.runner  # noqa: F401  (puts the repo on sys.path)
from utils.metrics.il_metrics import backward_transfer, cil_summary, dil_summary, forgetting, macro_f1
from utils.metrics.openset_metrics import auroc, fpr_at_tpr, openset_summary, oscr
from utils.metrics.runs import load_cil_run
from utils.metrics.stats import bootstrap_ci, mcnemar, mean_std, paired


def pred_file(output_classes, true_names, pred_names, step, new_class=None, unseen=None, extra=None):
    """A predictions dict like utils/eval/predictions.py writes (logits favour the predicted class)."""
    def set_(true, pred):
        names = sorted(set(true))
        logits = np.full((len(pred), len(output_classes)), -5.0, np.float16)
        for i, p in enumerate(pred):
            logits[i, output_classes.index(p)] = 5.0
        return {"doc_id": [f"{t}/{i}" for i, t in enumerate(true)], "class_names": names,
                "y_true": np.array([names.index(t) for t in true], np.int16), "logits": logits,
                "pred": np.array([output_classes.index(p) for p in pred], np.int16)}
    sets = {"seen": set_(true_names, pred_names)}
    if unseen is not None:
        sets["unseen"] = set_(unseen, [output_classes[0]] * len(unseen))
        if extra:
            for k, (seen_v, unseen_v) in extra.items():
                sets["seen"][k], sets["unseen"][k] = np.asarray(seen_v), np.asarray(unseen_v)
    return {"meta": {"step": step, "new_class": new_class, "method": "No EVM", "strategy": "standard"},
            "output_classes": list(output_classes), "evm_classes": None, "sets": sets}


class TestILMetrics(unittest.TestCase):
    def test_accuracy_matrix_forgetting_bwt(self):
        # base task {a, b}; step 1 adds c, step 2 adds d. Base accuracy 100 %, then the base task drops to 50 %.
        s0 = pred_file(["a", "b"], ["a", "a", "b", "b"], ["a", "a", "b", "b"], 0)
        s1 = pred_file(["a", "b", "c"], ["a", "a", "b", "b", "c", "c"], ["a", "a", "b", "a", "c", "c"], 1, "c")
        s2 = pred_file(["a", "b", "c", "d"], list("aabbccdd"), list("aaaaccdd"), 2, "d")
        out, A, tasks = cil_summary([s0, s1, s2])
        self.assertEqual(tasks, [["a", "b"], ["c"], ["d"]])
        np.testing.assert_allclose(A[0, 0], 1.0)
        np.testing.assert_allclose(A[1, :2], [0.75, 1.0])
        np.testing.assert_allclose(A[2], [0.5, 1.0, 1.0])
        # forgetting: task 0: max(1.0, 0.75) - 0.5 = 0.5; task 1: 1.0 - 1.0 = 0 -> 0.25
        self.assertAlmostEqual(forgetting(A), 0.25)
        # BWT: task 0: 0.5 - 1.0; task 1: 1.0 - 1.0 -> -0.25
        self.assertAlmostEqual(backward_transfer(A), -0.25)
        self.assertAlmostEqual(out["acc_final"], 6 / 8)
        self.assertAlmostEqual(out["avg_inc_acc"], np.mean([1.0, 5 / 6, 6 / 8]))
        self.assertAlmostEqual(out["acc_new_final"], 1.0)                      # class d
        self.assertAlmostEqual(out["acc_old_final"], 4 / 6)                    # a, b, c

    def test_macro_f1_ignores_absent_classes(self):
        # 'c' never occurs in this domain: it must not pull the average down
        self.assertAlmostEqual(macro_f1(["a", "a", "b"], ["a", "a", "b"]), 1.0)
        self.assertAlmostEqual(macro_f1(["a", "b"], ["a", "c"]), (1.0 + 0.0) / 2)

    def test_dil_summary(self):
        base = pred_file(["a", "b"], ["a", "b", "a", "b"], ["a", "b", "a", "b"], 0)
        base["sets"] = {"rvl": base["sets"]["seen"], "tobacco": base["sets"]["seen"]}
        adapted = pred_file(["a", "b"], ["a", "b", "a", "b"], ["a", "b", "a", "a"], None)
        adapted["sets"] = {"rvl": adapted["sets"]["seen"], "tobacco": base["sets"]["rvl"]}
        out = dil_summary(base, adapted)
        self.assertAlmostEqual(out["acc_rvl_before"], 1.0)
        self.assertAlmostEqual(out["acc_rvl_after"], 0.75)
        self.assertAlmostEqual(out["forgetting_rvl"], 0.25)
        self.assertAlmostEqual(out["acc_mean_after"], (0.75 + 1.0) / 2)
        self.assertAlmostEqual(out["acc_pooled_after"], 7 / 8)        # 3/4 on rvl + 4/4 on tobacco, pooled
        self.assertNotIn("acc_tobacco_after_clean", out)
        docs = list(adapted["sets"]["tobacco"]["doc_id"])
        clean = dil_summary(base, adapted, exclude={docs[0]})
        self.assertAlmostEqual(clean["acc_tobacco_after_clean"], 1.0)
        self.assertEqual(clean["n_tobacco_excluded"], 1)


class TestOpenSet(unittest.TestCase):
    def test_auroc(self):
        self.assertEqual(auroc([0.9, 0.8], [0.1, 0.2]), 1.0)
        self.assertEqual(auroc([0.5, 0.5], [0.5, 0.5]), 0.5)
        self.assertEqual(auroc([0.1], [0.9]), 0.0)
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(0)
        k, u = rng.normal(1, 1, 300).round(1), rng.normal(0, 1, 200).round(1)       # with ties
        ref = roc_auc_score(np.r_[np.ones(300), np.zeros(200)], np.r_[k, u])
        self.assertAlmostEqual(auroc(k, u), ref, places=10)

    def test_fpr95(self):
        known = np.arange(100) / 100.0          # threshold accepting 95 % of knowns: 0.05
        self.assertAlmostEqual(fpr_at_tpr(known, [0.0, 0.01, 0.5, 0.9]), 0.5)

    def test_oscr(self):
        # perfectly separated and all knowns correct -> 1; all knowns wrong -> 0
        self.assertAlmostEqual(oscr([0.9, 0.8], [True, True], [0.1, 0.2]), 1.0)
        self.assertAlmostEqual(oscr([0.9, 0.8], [False, False], [0.1, 0.2]), 0.0)
        # perfectly separated, half of the knowns correct -> 0.5
        self.assertAlmostEqual(oscr([0.9, 0.8], [True, False], [0.1, 0.2]), 0.5)

    def test_openset_summary_uses_available_scores(self):
        s = pred_file(["a", "b"], ["a", "b"], ["a", "b"], 1, "b", unseen=["c", "d"],
                      extra={"evm_prob": ([[0.9, 0.1], [0.1, 0.8]], [[0.1, 0.2], [0.3, 0.1]])})
        out = openset_summary(s)
        self.assertIn("msp_auroc", out)
        self.assertEqual(out["evm_auroc"], 1.0)
        self.assertNotIn("ood_auroc", out)
        self.assertEqual(openset_summary(pred_file(["a"], ["a"], ["a"], 1, "a")), {})   # last step: no unknowns


class TestStats(unittest.TestCase):
    def test_mcnemar(self):
        a = np.array([1, 1, 1, 1, 1, 0, 1, 1], bool)
        b = np.array([0, 0, 0, 0, 0, 0, 1, 1], bool)
        a_only, b_only, p = mcnemar(a, b)
        self.assertEqual((a_only, b_only), (5, 0))
        self.assertAlmostEqual(p, 2 * 0.5 ** 5)
        self.assertEqual(mcnemar(a, a)[2], 1.0)

    def test_paired_aligns_by_document(self):
        a, b = paired(["x", "y", "z"], [True, False, True], ["z", "x"], [False, True])
        np.testing.assert_array_equal(a, [True, True])
        np.testing.assert_array_equal(b, [True, False])

    def test_bootstrap_and_mean_std(self):
        lo, hi = bootstrap_ci(np.r_[np.ones(80), np.zeros(20)])
        self.assertTrue(lo < 0.8 < hi)
        self.assertEqual(mean_std([0.5, float("nan")])[0], 0.5)
        m, s = mean_std([0.2, 0.4])
        self.assertAlmostEqual(m, 0.3)
        self.assertAlmostEqual(s, np.std([0.2, 0.4], ddof=1))


class TestRuns(unittest.TestCase):
    def test_load_cil_run_stops_at_gap(self):
        base = pred_file(["a"], ["a"], ["a"], 0)
        with tempfile.TemporaryDirectory() as d:
            files = []
            for k, c in ((1, "b"), (2, "c"), (4, "e")):       # step 3 missing
                f = os.path.join(d, f"cil_{c}_seed42.pt")
                torch.save(pred_file(["a", c], ["a"], ["a"], str(k), c), f)
                files.append(f)
            steps, complete = load_cil_run(files, base)
        self.assertEqual([s["meta"]["new_class"] for s in steps[1:]], ["b", "c"])
        self.assertFalse(complete)


if __name__ == "__main__":
    unittest.main()
