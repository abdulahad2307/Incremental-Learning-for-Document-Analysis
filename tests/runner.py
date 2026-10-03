"""Shared helpers for the end-to-end pipeline tests (standard-library unittest; no extra packages).

Environment variables:
    PIPELINE_TEST_DIR         where the synthetic data, checkpoints and run outputs go (default: $TMPDIR or /tmp)
    PIPELINE_TEST_METHODS     comma-separated subset of methods to run (default: all)
    PIPELINE_TEST_STRATEGIES  comma-separated subset of strategies (default: standard,distillation)
    PIPELINE_TEST_TIMEOUT     seconds per script run (default: 3600)
"""
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
    env = dict(os.environ, PYTHONPATH=REPO + os.pathsep + os.environ.get("PYTHONPATH", ""))
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
