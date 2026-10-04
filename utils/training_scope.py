"""Which parameters are trained during an incremental step (shared by EAML and LayoutLMv3, CIL and DIL).

  classifier_only : classifier heads only (the features are frozen: EWC and L_EVM cannot act)
  last_layer      : the model's last feature layer + the classifier heads
                    EAML -> fusion_module, Custom LayoutLMv3 -> last fusion-transformer layer,
                    HF LayoutLMv3 -> last encoder layer
  full            : all parameters
"""
HEAD_NAMES = ("classifier", "image_classifier", "text_classifier", "fusion_classifier")
SCOPES = ("classifier_only", "last_layer", "full")
_ALIASES = {"full_model": "full", "full_finetune": "full", "head_only": "classifier_only"}


def last_feature_layer_prefixes(model):
    if hasattr(model, "fusion_module"):  # EAML: fused image+text features
        return ["fusion_module."]
    if hasattr(model, "fusion_transformer"):  # Custom LayoutLMv3: [CLS] of the last fusion layer
        return [f"fusion_transformer.layers.{len(model.fusion_transformer.layers) - 1}."]
    if hasattr(model, "backbone") and hasattr(model.backbone, "encoder"):  # HF LayoutLMv3: last encoder layer
        return [f"backbone.encoder.layer.{len(model.backbone.encoder.layer) - 1}."]
    raise ValueError(f"No last-layer definition for {type(model).__name__}")


def _is_head(name):
    return name.split(".")[0] in HEAD_NAMES


def set_training_scope(model, scope):
    scope = _ALIASES.get(scope, scope)
    if scope not in SCOPES:
        raise ValueError(f"Unknown training scope '{scope}'; choose from {SCOPES}")
    prefixes = last_feature_layer_prefixes(model) if scope == "last_layer" else []
    for name, p in model.named_parameters():
        p.requires_grad = (scope == "full" or _is_head(name) or any(name.startswith(x) for x in prefixes))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"Training scope '{scope}': {n_train / 1e6:.2f}M of {n_all / 1e6:.2f}M parameters trainable"
          + (f" ({', '.join(prefixes)} + classifier heads)" if prefixes else ""))
    return model
