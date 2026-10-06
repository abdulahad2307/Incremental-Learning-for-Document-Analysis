"""Shared helpers for the end-to-end pipeline tests (standard-library unittest; no extra packages).

Environment variables:
    PIPELINE_TEST_DIR         where the synthetic data, checkpoints and run outputs go (default: $TMPDIR or /tmp)
    PIPELINE_TEST_METHODS     comma-separated subset of methods to run (default: all)
    PIPELINE_TEST_STRATEGIES  comma-separated subset of strategies (default: standard,distillation)
    PIPELINE_TEST_TIMEOUT     seconds per script run (default: 3600)
    IL_RESULTS_TABLE          results table the runs log to (default: <test dir>/il_runs.csv, never the repo's table)
"""
import csv
import os
import shutil
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from tests.fixtures import build_fixture  # noqa: E402

STRATEGIES = ("standard", "distillation")
TEST_ROOT = os.path.join(os.environ.get("PIPELINE_TEST_DIR") or os.environ.get("TMPDIR") or "/tmp", "il_pipeline_tests")
DATA = os.path.join(TEST_ROOT, "data")
TIMEOUT = int(os.environ.get("PIPELINE_TEST_TIMEOUT", "3600"))
RESULTS_TABLE = os.environ.get("IL_RESULTS_TABLE") or os.path.join(TEST_ROOT, "il_runs.csv")


def data_dir():
    return build_fixture(DATA)


def selected(names, env_var):
    wanted = os.environ.get(env_var)
    return [n for n in names if not wanted or n in wanted.split(",")]


def out_dir(*parts):
    d = os.path.join(TEST_ROOT, "runs", *parts)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def run_script(test, script, args, cwd, expect_files=(), expect_logs=()):
    """Run a pipeline entry point like the SLURM scripts do (PYTHONPATH = repo) and check it finished cleanly.
    `cwd` is the test's output folder, so files a script writes to relative paths stay there; the full
    command and output are saved there as run.log."""
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""),
               IL_RESULTS_TABLE=RESULTS_TABLE)
    cmd = [sys.executable, os.path.join(REPO, script)] + [str(a) for a in args]
    proc = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=TIMEOUT)
    with open(os.path.join(cwd, "run.log"), "w") as f:
        f.write(" ".join(cmd) + "\n\n" + proc.stdout + proc.stderr)
    log = proc.stdout + proc.stderr
    tail = "\n".join(log.splitlines()[-40:])
    test.assertEqual(proc.returncode, 0, f"{script} exited with {proc.returncode}\n{tail}")
    test.assertNotIn("Traceback (most recent call last)", log, f"{script} logged a traceback\n{tail}")
    for f in expect_files:
        test.assertTrue(os.path.exists(f), f"{script} did not write {f}\n{tail}")
    for text in expect_logs:
        test.assertIn(text, log, f"{script} log is missing '{text}'\n{tail}")
    return log


def _logged_final_accs(run_id):
    """{"Test" | "rvl" | "tobacco": accuracy} from the run's 'final' rows in the results table."""
    accs = {}
    with open(RESULTS_TABLE) as f:
        for row in csv.DictReader(f):
            if row["run_id"] != run_id or row["phase"] != "final":
                continue
            if row["split"] == "Test" and row["acc"]:
                accs["Test"] = float(row["acc"])
            for col, name in (("acc_rvl", "rvl"), ("acc_tob", "tobacco")):
                if row[col]:
                    accs[name] = float(row[col])
    return accs


def check_predictions(test, out, tag, sets, evm=False, ood=False):
    """Check the predictions file a run saved (utils/eval/predictions.py): <out>/predictions/<tag>_seed*.pt with
    the expected sets and fields, and accuracies that reproduce the 'final' accuracies the run logged."""
    import glob
    from utils.eval.predictions import load_predictions
    files = glob.glob(os.path.join(out, "predictions", f"{tag}_seed*.pt"))
    test.assertEqual(len(files), 1, f"expected one predictions file {tag}_seed*.pt in {out}/predictions, got {files}")
    r = load_predictions(files[0])
    test.assertEqual(sorted(r["sets"]), sorted(sets), "predictions sets")
    test.assertTrue(r["meta"].get("run_id") and r["meta"].get("git_commit") is not None, f"meta: {r['meta']}")
    n_out = len(r["output_classes"])
    for name, s in r["sets"].items():
        n = len(s["y_true"])
        test.assertGreater(n, 0, f"set {name} is empty")
        test.assertEqual(s["logits"].shape, (n, n_out), f"{name}: logits shape")
        test.assertEqual(len(s["doc_id"]), n, f"{name}: doc_id")
        test.assertEqual(len(set(s["doc_id"])), n, f"{name}: doc ids are not unique")
        test.assertTrue(all("/" in d for d in s["doc_id"]), f"{name}: doc ids are not '<class>/<file>' paths")
        test.assertTrue(((s["pred"] >= 0) & (s["pred"] < n_out)).all(), f"{name}: pred out of range")
        test.assertTrue(((s["y_true"] >= 0) & (s["y_true"] < len(s["class_names"]))).all(), f"{name}: y_true")
        test.assertEqual("evm_prob" in s, evm, f"{name}: evm_prob present = {'evm_prob' in s}, expected {evm}")
        if evm:
            test.assertEqual(s["evm_prob"].shape, (n, len(r["evm_classes"])), f"{name}: evm_prob shape")
        test.assertEqual("ood_score" in s, ood, f"{name}: ood_score present = {'ood_score' in s}, expected {ood}")
    # the saved predictions reproduce the accuracies the script logged
    logged = _logged_final_accs(r["meta"]["run_id"])
    key = {"seen": "Test", "rvl": "rvl", "tobacco": "tobacco"}
    out_index = {c: i for i, c in enumerate(r["output_classes"])}
    for name, s in r["sets"].items():
        if key.get(name) not in logged:
            continue
        target = [out_index.get(s["class_names"][y], -1) for y in s["y_true"]]
        acc = sum(int(p) == t for p, t in zip(s["pred"], target)) / len(target)
        test.assertAlmostEqual(acc, logged[key[name]], places=4,
                               msg=f"{name}: accuracy of the saved predictions {acc:.4f} != logged {logged[key[name]]:.4f}")
    return r


def add_matrix_tests(cls, methods, run_one):
    """Add one test per (method, strategy): test_<method>__<strategy> calling run_one(self, method, strategy)."""
    for method in selected(list(methods), "PIPELINE_TEST_METHODS"):
        for strategy in selected(STRATEGIES, "PIPELINE_TEST_STRATEGIES"):
            def test(self, method=method, strategy=strategy):
                run_one(self, method, strategy)
            test.__name__ = f"test_{method}__{strategy}"
            setattr(cls, test.__name__, test)
    return cls


class PipelineTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = data_dir()
