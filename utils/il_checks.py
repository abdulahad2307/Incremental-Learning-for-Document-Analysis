def warn_inactive_terms(model, use_ewc=False, lambda_evm=0.0, lambda_ood=0.0):
    """Warn when loss terms cannot change any trainable parameter because only the classifier layers are trainable
    (e.g. --training_mode last_layer): EWC skips the classifier layers and L_EVM depends on the features only."""
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    if trainable and all("classifier" in n for n in trainable):
        if use_ewc:
            print("Warning: only classifier layers are trainable, so the EWC penalty (which excludes classifiers) has no effect.")
        if lambda_evm:
            print("Warning: only classifier layers are trainable, so L_EVM (a loss on the features) has no effect.")
        if lambda_ood:
            print("Warning: only classifier layers are trainable, so L_OOD acts only through the logits' energy, not the features.")
