"""Open-set / OOD evaluation shared by the *_ood scripts.

Protocol
--------
1. fit      the detector on in-distribution (ID) training features/logits,
2. calibrate the accept/reject threshold on ID validation scores so that `tpr` of ID samples are accepted,
3. evaluate on ID test vs OOD test samples.

All detectors return scores where higher = more in-distribution. Reported:
AUROC, AUPR-In, AUPR-Out, FPR@95%TPR (threshold-free), and at the calibrated threshold:
known acceptance, unknown rejection and open-set accuracy (ID sample accepted AND correctly classified).
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.ood.msp import MSP_OOD
from utils.ood.vim import VIM_OOD
from utils.ood.gradnorm import GradNorm_OOD
from utils import run_log

DETECTORS = {"msp": MSP_OOD, "vim": VIM_OOD, "gradnorm": GradNorm_OOD}


def make_detector(method, **kwargs):
    method = method.lower()
    if method not in DETECTORS:
        raise ValueError(f"OOD method '{method}' not implemented; choose from {list(DETECTORS)}")
    return DETECTORS[method](**kwargs)


class _Subset(Subset):
    """Subset that still exposes the wrapped dataset's attributes (e.g. current_classes)."""

    def __getattr__(self, name):
        if name in ("dataset", "indices"):
            raise AttributeError(name)
        return getattr(self.dataset, name)


def subsample_loader(loader, max_per_class=None, seed=0):
    """Loader over at most `max_per_class` samples per class (same collate_fn / batch size)."""
    if loader is None or not max_per_class:
        return loader
    ds = loader.dataset
    base, idx_map = (ds.dataset, list(ds.indices)) if isinstance(ds, Subset) else (ds, list(range(len(ds))))
    samples = getattr(base, "samples", None)
    if samples is None:
        return loader
    rng = np.random.default_rng(seed)
    by_class = {}
    for i in idx_map:
        by_class.setdefault(samples[i][-1], []).append(i)
    keep = []
    for cls_idx in by_class.values():
        keep.extend(cls_idx if len(cls_idx) <= max_per_class else rng.choice(cls_idx, max_per_class, replace=False).tolist())
    return DataLoader(_Subset(base, sorted(keep)), batch_size=loader.batch_size, shuffle=False,
                      num_workers=loader.num_workers, collate_fn=loader.collate_fn)


@torch.no_grad()
def collect_features_logits(model, loader, device):
    """Penultimate features, logits and labels for every sample of an EAML or LayoutLMv3 loader."""
    model.eval()
    feats, logits, labels = [], [], []
    for batch in loader:
        if "images" in batch:  # EAML
            images = batch["images"].to(device)
            texts = {k: v.to(device) for k, v in batch["texts"].items()}
            f = model.extract_features(images=images, texts=texts)
            z = model(images=images, texts=texts)
        else:  # LayoutLMv3
            ids, bbox, mask, pix = (batch[k].to(device) for k in ("input_ids", "bbox", "attention_mask", "pixel_values"))
            f = model.extract_features(ids, bbox, mask, pix)
            z = model.classifier(f)
        feats.append(f.float().cpu())
        logits.append(z.float().cpu())
        labels.append(torch.as_tensor(batch["labels"]).cpu())
    return torch.cat(feats).numpy(), torch.cat(logits).numpy(), torch.cat(labels).numpy()


def last_linear_layer(model):
    """(W, b) of the layer that maps features to logits, if the model exposes one."""
    layer = getattr(model, "fusion_classifier", None) or getattr(model, "classifier", None)
    if isinstance(layer, torch.nn.Linear):
        return layer.weight.detach().cpu().numpy(), layer.bias.detach().cpu().numpy()
    return None, None


def ood_metrics(id_scores, ood_scores, threshold, id_correct=None):
    y_true = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))]).astype(int)
    scores = np.concatenate([id_scores, ood_scores])
    y_pred = (scores >= threshold).astype(int)  # 1 = accepted as known
    thr95 = np.percentile(id_scores, 5)
    res = {
        "auroc": roc_auc_score(y_true, scores),
        "aupr_in": average_precision_score(y_true, scores),
        "aupr_out": average_precision_score(1 - y_true, -scores),
        "fpr_at_95tpr": float(np.mean(ood_scores >= thr95)),
        "threshold": float(threshold),
        "known_acceptance": float(np.mean(id_scores >= threshold)),
        "unknown_rejection": float(np.mean(ood_scores < threshold)),
        "n_known": len(id_scores),
        "n_unknown": len(ood_scores),
        "y_true": y_true,
        "y_pred": y_pred,
        "scores": scores,
    }
    if id_correct is not None:
        res["open_set_accuracy"] = float(np.mean((id_scores >= threshold) & id_correct))
    return res


def evaluate_ood(detector, model, device, fit_loader, calib_loader, id_loader, ood_loader,
                 tpr=0.95, max_per_class=None, tag="OOD", savepath=None):
    """Fit, calibrate and evaluate `detector`. Returns the metrics dict, or None if there are no OOD samples."""
    if ood_loader is None or len(ood_loader.dataset) == 0:
        print(f"[{tag}] No unknown samples to evaluate against; OOD metrics skipped.")
        return None
    fit_loader, calib_loader, id_loader, ood_loader = (
        subsample_loader(l, max_per_class) for l in (fit_loader, calib_loader, id_loader, ood_loader))
    W, b = last_linear_layer(model)

    f, z, _ = collect_features_logits(model, fit_loader, device)
    detector.fit(f, z, W=W, b=b)
    f, z, _ = collect_features_logits(model, calib_loader, device)
    threshold = np.quantile(detector.score(f, z), 1.0 - tpr)
    f, z, y = collect_features_logits(model, id_loader, device)
    id_scores, id_correct = detector.score(f, z), z.argmax(axis=1) == y
    f, z, _ = collect_features_logits(model, ood_loader, device)
    ood_scores = detector.score(f, z)

    res = ood_metrics(id_scores, ood_scores, threshold, id_correct)
    print(f"[{tag} {detector.name.upper()}] AUROC {res['auroc']:.4f} | FPR@95TPR {res['fpr_at_95tpr']:.4f} | "
          f"AUPR-In {res['aupr_in']:.4f} | AUPR-Out {res['aupr_out']:.4f}")
    print(f"[{tag} {detector.name.upper()}] @TPR{int(tpr * 100)} threshold {res['threshold']:.4f}: "
          f"known accepted {res['known_acceptance']:.4f}, unknown rejected {res['unknown_rejection']:.4f}, "
          f"open-set acc {res['open_set_accuracy']:.4f} (n_known={res['n_known']}, n_unknown={res['n_unknown']})")
    run_log.log("ood", split=tag, ood_auroc=res["auroc"], ood_fpr95=res["fpr_at_95tpr"], ood_method=detector.name,
                aupr_in=res["aupr_in"], aupr_out=res["aupr_out"], threshold=res["threshold"],
                known_acceptance=res["known_acceptance"], unknown_rejection=res["unknown_rejection"],
                open_set_accuracy=res["open_set_accuracy"], n_known=res["n_known"], n_unknown=res["n_unknown"])
    if savepath:
        from utils.ood.ood_viz import plot_ood_histograms
        plot_ood_histograms(res["y_true"], res["y_pred"], res["scores"], savepath=savepath,
                            method_name=detector.name.upper())
    return res
