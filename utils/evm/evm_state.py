"""Saving / loading open-set state (EVM, and the ViM used by L_OOD) between incremental steps and jobs.

The state of a step is stored next to that step's best checkpoint:
    best_model_<cls>.pth (EAML)  ->  evm_state_<cls>.pt
    <name>.pt (LayoutLMv3)       ->  <name>_evm_state.pt
so a job whose BASE_MODEL is that checkpoint finds the state of the step that produced it.
"""
import os

import torch


def evm_state_path(model_path):
    """evm_state_<cls>.pt next to best_model_<cls>.pth (EAML); <checkpoint stem>_evm_state.pt for other names (LayoutLMv3).
    An initial base model has no such file, so the EVM starts empty."""
    if not model_path:
        return None
    folder, name = os.path.split(model_path)
    stem = os.path.splitext(name)[0]
    if name.startswith("best_model_"):
        return os.path.join(folder, "evm_state_" + stem[len("best_model_"):] + ".pt")
    return os.path.join(folder, stem + "_evm_state.pt")


def save_open_set_state(model_path, evm=None, vim=None):
    path = evm_state_path(model_path)
    state = {
        "evm_class": type(evm).__name__ if evm is not None else None,
        "evm": dict(vars(evm)) if evm is not None and getattr(evm, "initialized", False) else None,
        "vim": dict(vars(vim)) if vim is not None and getattr(vim, "initialized", False) else None,
    }
    torch.save(state, path)
    print(f"Saved open-set state (EVM{'+ViM' if state['vim'] else ''}) to {path}")
    return path


def load_open_set_state(model_path, evm=None, vim=None):
    """Restore EVM / ViM saved for the step that produced `model_path`. Returns True if an EVM state was loaded."""
    path = evm_state_path(model_path)
    if path is None or not os.path.exists(path):
        print(f"No saved open-set state for {model_path}; EVM starts empty.")
        return False
    state = torch.load(path, map_location="cpu", weights_only=False)
    loaded = False
    if evm is not None and state.get("evm"):
        if state["evm_class"] != type(evm).__name__:
            raise ValueError(f"{path} holds a {state['evm_class']}, but this run uses {type(evm).__name__}")
        vars(evm).update(state["evm"])
        loaded = True
    if vim is not None and state.get("vim"):
        vars(vim).update(state["vim"])
    print(f"Loaded open-set state from {path} (EVM: {loaded}, ViM: {bool(state.get('vim'))})")
    return loaded
