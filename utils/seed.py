"""Seeding for reproducible runs (shared by the base-model, class- and domain-incremental scripts)."""
import os
import random

import numpy as np
import torch

DEFAULT_SEED = 42


def add_seed_arg(parser, default=DEFAULT_SEED):
    """Add --seed to an argparse parser unless the script already defines it."""
    if "--seed" not in parser._option_string_actions:
        parser.add_argument("--seed", type=int, default=default, help="Random seed (Python, NumPy, PyTorch)")


def set_seed(seed=DEFAULT_SEED, deterministic=True):
    """Seed Python, NumPy and PyTorch (CPU and all GPUs). This also makes the unseeded random.sample / np.random /
    DataLoader-shuffle calls in the pipeline reproducible, because they draw from these global generators.
    deterministic=True selects deterministic cuDNN kernels (slightly slower, same results on the same GPU type)."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # affects subprocesses such as DataLoader workers started afterwards
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Seed: {seed}" + (" (deterministic cuDNN)" if deterministic else ""))
    return seed
