"""One results table for all incremental-learning runs.

Every IL entry script calls init() once after parsing its arguments and log() wherever it reports metrics. Each call
appends one row to a single CSV shared by all runs (default: <repo>/results/il_runs.csv, override with the
IL_RESULTS_TABLE environment variable). Concurrent SLURM jobs are serialised with a file lock. The full argument set of
each run goes to a JSONL file next to the table (il_runs_config.jsonl), keyed by run_id. Every row also records the
run's seed and the git commit of the code (with "-dirty" if tracked files had uncommitted changes).

Row phases:
  epoch      - one per training epoch (train/val loss and accuracy)
  step_test  - test evaluation of a new best model within an incremental step
  open_set   - EVM open-set evaluation (known accuracy / unknown rejection)
  ood        - OOD detector evaluation (logged by utils.ood.ood_eval.evaluate_ood)
  final      - evaluation of the final model (one row per split / domain)
"""
import csv
import datetime
import fcntl
import json
import math
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TABLE = os.path.join(_REPO_ROOT, "results", "il_runs.csv")

COLUMNS = [
    "timestamp", "run_id", "slurm_job_id", "git_commit", "script", "backbone", "setting", "method", "strategy",
    "bias_correction", "seed",
    "phase", "step", "new_class", "epoch", "split",
    "train_loss", "train_acc", "val_loss", "val_acc",
    "loss", "acc", "precision", "recall", "f1",
    "acc_rvl", "acc_tob", "gil_base", "gil_prev",
    "evm_known_acc", "evm_unknown_rej", "ood_auroc", "ood_fpr95",
    "extra",
]

_run = None


def _git_commit():
    """Short commit hash of the repo the code runs from, "-dirty" if tracked files differ from it; "" without git."""
    try:
        run = lambda *a: subprocess.run(["git", "-C", _REPO_ROOT, *a], capture_output=True, text=True, timeout=10)
        commit = run("rev-parse", "--short", "HEAD").stdout.strip()
        if commit and run("status", "--porcelain", "--untracked-files=no").stdout.strip():
            commit += "-dirty"
        return commit
    except (OSError, subprocess.SubprocessError):
        return ""


def _table_path():
    return os.environ.get("IL_RESULTS_TABLE", DEFAULT_TABLE)


def _fmt(v):
    if v is None:
        return ""
    if hasattr(v, "item"):  # torch / numpy scalars
        v = v.item()
    if isinstance(v, float):
        return "" if math.isnan(v) else f"{v:.6g}"
    return v


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)) or (hasattr(v, "tolist") and getattr(v, "ndim", 0) > 0):
        return [_jsonable(x) for x in (v.tolist() if hasattr(v, "tolist") else v)]
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float):
        return None if math.isnan(v) else round(v, 6)
    return v if isinstance(v, (int, str, bool)) or v is None else str(v)


def init(backbone, setting, method, args=None, **fields):
    """Start a run. backbone: eaml | llmv3; setting: CIL | DIL; method: e.g. No EVM, EVM, EVM+OOD, iEVM, RegEVM.
    strategy / bias_correction are read from args (or passed as fields) so the table shows how the run was configured."""
    global _run
    if backbone == "llmv3" and os.environ.get("LLMV3_MODEL") == "hf":  # pre-trained LayoutLMv3 (scripts/config.sh)
        backbone = "llmv3hf"
    cfg = vars(args) if args is not None and hasattr(args, "__dict__") else dict(args or {})
    now = datetime.datetime.now()
    job = os.environ.get("SLURM_JOB_ID", "")
    _run = {
        "run_id": f"{job or 'local'}_{now:%Y%m%d_%H%M%S}_{os.getpid()}",
        "slurm_job_id": job,
        "git_commit": _git_commit(),
        "script": os.path.basename(sys.argv[0]),
        "backbone": backbone,
        "setting": setting,
        "method": method,
        "strategy": fields.pop("strategy", cfg.get("strategy", "")),
        "bias_correction": fields.pop("bias_correction", cfg.get("use_bias_correction", "")),
        "seed": fields.pop("seed", cfg.get("seed", "")),
    }
    _run.update(fields)

    path = _table_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    config_path = os.path.join(os.path.dirname(path) or ".", "il_runs_config.jsonl")
    with open(config_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps({"run_id": _run["run_id"], "timestamp": now.isoformat(timespec="seconds"),
                            **{k: _run[k] for k in ("git_commit", "script", "backbone", "setting", "method", "seed")},
                            "argv": sys.argv[1:], "args": cfg}, default=str) + "\n")
    print(f"[run_log] run_id={_run['run_id']} -> {path}")
    return _run["run_id"]


def log(phase, **metrics):
    """Append one row. Keys that are not table columns are stored as JSON in the 'extra' column."""
    if _run is None:
        init(backbone="", setting="", method="")
    row = dict(_run)
    row["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    row["phase"] = phase
    extra = {}
    for k, v in metrics.items():
        if k in COLUMNS:
            row[k] = v
        elif v is not None:
            extra[k] = _jsonable(v)
    row["extra"] = json.dumps(extra) if extra else ""

    path = _table_path()
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if f.tell() == 0:
            writer.writeheader()
        else:  # never append rows whose columns do not match the existing header
            f.seek(0)
            header = next(csv.reader(f), [])
            if header != COLUMNS:
                raise RuntimeError(f"{path} was written with other columns than utils/run_log.py now uses; "
                                   f"move it aside or set IL_RESULTS_TABLE to a new file")
            f.seek(0, os.SEEK_END)
        writer.writerow({k: _fmt(row.get(k)) for k in COLUMNS})
