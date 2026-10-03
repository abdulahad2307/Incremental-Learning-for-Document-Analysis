import numpy as np

def evm_openset_metrics(evm, features_dict, known_classes=None, threshold=None, unknown_label="unknown"):
    """
    Compute open set metrics using an EVM on extracted features.
    Args:
        evm: fitted EVMClassifier instance
        features_dict: dict[class_name, np.ndarray], e.g. from extract_features()
        known_classes: list of known/seen classes (optional)
        threshold: float, EVM threshold (optional, default to evm.cover_threshold)
        unknown_label: string
    Returns:
        metrics dict including: open_set_accuracy, unknown_rejection, y_true, y_pred
    """
    X = []
    y_true = []
    for cls_name, feats in features_dict.items():
        X.append(feats)
        y_true.extend([cls_name]*feats.shape[0])
    if not X:
        return {}
    X = np.vstack(X)
    y_true = np.array(y_true)

    y_pred, scores = evm.predict(X, threshold=threshold if threshold is not None else evm.cover_threshold)
    n_known = sum([yt in evm.class_features for yt in y_true])
    n_correct_known = sum([(yt == yp) for yt, yp in zip(y_true, y_pred) if yt in evm.class_features])
    n_unknown = sum([yt not in evm.class_features for yt in y_true])
    n_correct_unknown = sum([(yp == unknown_label) for yt, yp in zip(y_true, y_pred) if yt not in evm.class_features])
    known_acc = n_correct_known / n_known if n_known > 0 else 0.0
    unknown_rej = n_correct_unknown / n_unknown if n_unknown > 0 else 0.0

    return {
        "n_known": n_known,
        "n_unknown": n_unknown,
        "open_set_accuracy": known_acc,
        "unknown_rejection": unknown_rej,
        "y_true": y_true,
        "y_pred": y_pred,
        "is_known": np.array([yt in evm.class_features for yt in y_true]),
        "scores": scores
    }
