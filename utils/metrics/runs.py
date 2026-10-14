"""Find the runs under the pipeline's output folder (scripts/config.sh: RUN_ROOT) and load their predictions files.

  CIL step k of a run:  <run_root>/cil/<backbone>/<method>_<std|kd>/seed<seed>/predictions/cil_<class>_seed<seed>.pt
  CIL step 0:           <run_root>/base/<backbone>_11cls/predictions/base_cil_seed*.pt   (shared by all CIL runs)
  DIL run:              <run_root>/dil/<backbone>/<method>_<std|kd>/seed<seed>/predictions/dil_seed<seed>.pt
  DIL step 0:           <run_root>/base/<backbone>_16cls/predictions/base_dil_seed*.pt   (shared by all DIL runs)
<backbone> is eaml, llmv3 (Custom LayoutLMv3) or llmv3hf (pre-trained LayoutLMv3).
"""
import glob
import os
import re

from utils.eval.predictions import load_predictions

STRATEGY = {"standard": "std", "distillation": "KD"}


def _step(meta):
    s = meta.get("step")
    return int(s) if s not in (None, "") else None


def base_predictions(run_root, backbone, setting):
    """Step-0 predictions of the base model of `backbone` for CIL (11 classes) or DIL (16 classes), or None."""
    n = 11 if setting == "CIL" else 16
    files = sorted(glob.glob(os.path.join(run_root, "base", f"{backbone}_{n}cls", "predictions",
                                          f"base_{setting.lower()}_seed*.pt")))
    return load_predictions(files[0]) if files else None


def find_runs(run_root, setting):
    """{(backbone, method_dir, seed): [predictions files]} of all CIL or DIL runs."""
    pattern = "cil_*_seed*.pt" if setting == "CIL" else "dil_seed*.pt"
    runs = {}
    for f in glob.glob(os.path.join(run_root, setting.lower(), "*", "*", "seed*", "predictions", pattern)):
        parts = f.split(os.sep)
        backbone, method_dir, seed = parts[-5], parts[-4], int(re.sub(r"\D", "", parts[-3]))
        runs.setdefault((backbone, method_dir, seed), []).append(f)
    return runs


def load_cil_run(files, base):
    """The run's steps in order, starting with the base model (step 0) and stopping at the first missing step.
    Returns (steps, complete) with complete = False if a later step file exists after a gap."""
    by_step = {}
    for f in files:
        r = load_predictions(f)
        by_step[_step(r["meta"])] = r
    steps, k = [base], 1
    while k in by_step:
        steps.append(by_step[k])
        k += 1
    return steps, len(steps) - 1 == len(by_step)


def run_label(prediction):
    """(method, strategy) as written in the paper tables, from a step file's metadata."""
    meta = prediction["meta"]
    return meta.get("method", ""), STRATEGY.get(meta.get("strategy"), meta.get("strategy") or "")
