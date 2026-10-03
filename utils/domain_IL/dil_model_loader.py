import re
import torch
from utils.training_scope import set_training_scope
from utils.eaml.eaml_model import EAMLModel


def load_eaml_model(checkpoint_path, num_classes, device, text_branch=True):
    model = EAMLModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    return model

def load_eaml_model_partial(
    checkpoint_path: str,
    old_num_classes: int,
    new_num_classes: int,
    device,
    text_branch: bool = True,
):
    """
    Load an EAML model with pretrained weights on old classes,
    expanding classifier heads to accommodate new classes.
    """
    model = EAMLModel(num_classes=new_num_classes).to(device)
        
    load_partial_checkpoint(
        model,
        checkpoint_path,
        device,
        old_num_classes,
        new_num_classes,
        classifier_names=["image_classifier", "text_classifier", "fusion_classifier"]
    )
    
    model.to(device)
    return model

def set_finetune_mode(model, mode="head_only", encoder_unfreeze_depth=0):
    """head_only: classifier heads; last_layer: fusion_module + heads; partial / partial_finetune: encoder layers with
    index >= encoder_unfreeze_depth + fusion_module + heads; full / full_finetune: everything."""
    mode = {"partial_finetune": "partial", "full_finetune": "full"}.get(mode, mode)
    if mode in ("head_only", "last_layer", "full"):
        return set_training_scope(model, mode)
    if mode == "partial":
        set_training_scope(model, "last_layer")
        for name, param in model.named_parameters():
            m = re.search(r"layer\.?(\d+)", name)  # e.g. text_encoder...encoder.layer.11..., image_encoder...layer4...
            if m and int(m.group(1)) >= encoder_unfreeze_depth:
                param.requires_grad = True
        return model
    raise ValueError(f"Unknown finetune mode: {mode}")


def load_partial_checkpoint(
    model,
    checkpoint_path: str,
    device,
    old_num_classes: int,
    new_num_classes: int,
    classifier_names: list = ["image_classifier", "text_classifier", "fusion_classifier"]
):
    """
    Load checkpoint weights into model, allowing classifier layers to expand for new classes.

    Args:
        model: nn.Module, your model instance.
        checkpoint_path: str, path to checkpoint file.
        device: torch.device, device to load checkpoint.
        old_num_classes: int, number of classes in the checkpoint (old model).
        new_num_classes: int, desired number of classes in current model.
        classifier_names: list[str], names of classifier heads in the model.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model_state = model.state_dict()
    for key, param in state_dict.items():
        if any(clf in key for clf in classifier_names):
            # We'll handle classifier layers later
            continue
        if key in model_state and param.size() == model_state[key].size():
            model_state[key].copy_(param)
        else:
            print(f"Skipping key {key} due to size mismatch.")

    # Function to expand classifier parameters
    def expand_classifier_weight_bias(w_old, b_old, new_classes):
        # w_old: [old_classes, features]
        # b_old: [old_classes]
        num_old = w_old.size(0)
        assert new_classes >= num_old
        if new_classes == num_old:
            return w_old.clone(), b_old.clone()
        w_new = torch.zeros(new_classes, w_old.size(1), device=w_old.device)
        b_new = torch.zeros(new_classes, device=b_old.device)
        w_new[:num_old, :] = w_old
        b_new[:num_old] = b_old
        return w_new, b_new

    for clf_name in classifier_names:
        w_key = f"{clf_name}.weight"
        b_key = f"{clf_name}.bias"
        if w_key in state_dict and b_key in state_dict:
            w_old = state_dict[w_key]
            b_old = state_dict[b_key]
            w_exp, b_exp = expand_classifier_weight_bias(w_old, b_old, new_num_classes)
            getattr(model, clf_name).weight.data.copy_(w_exp)
            getattr(model, clf_name).bias.data.copy_(b_exp)
            print(f"Expanded {clf_name} from {old_num_classes} to {new_num_classes} classes.")
        else:
            print(f"Warning: {clf_name} weights not found in checkpoint.")

    model.load_state_dict(model_state)

    print("Partial checkpoint loading completed.")

