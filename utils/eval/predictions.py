"""Per-document predictions of every incremental-learning run, saved once per step.

At the end of each CIL step, of the DIL step, and once per base model, the scripts evaluate the final model on the
test sets and call save_predictions(). All metrics (accuracy matrix, forgetting, BWT, open-set AUROC / FPR95 / OSCR,
confidence intervals, paired tests between methods) can then be computed offline, without retraining.

File: <out_dir>/predictions/<tag>_seed<seed>.pt, a torch.save'd dict:
  format_version   1
  meta             run_id, git_commit, script, backbone, setting, method, strategy, bias_correction, seed (from
                   utils.run_log), plus step, new_class, checkpoint, created and whatever the caller passes
  output_classes   class names of the model's output columns (logit order)
  evm_classes      class names of the evm_prob columns (EVM methods), else None
  sets             {set name: arrays over the documents of that test set}, e.g. CIL "seen" / "unseen",
                   DIL "rvl" / "tobacco"
    doc_id         image path relative to the class folder's parent ("<class>/<file>"); joins documents across runs
    y_true         int16, index into the set's class_names
    class_names    names for y_true
    logits         float16 (N, len(output_classes)), the prediction head (EAML: fusion head)
    pred           int16, index into output_classes (argmax of logits)
    logits_image, logits_text   float16, EAML only: its image and text heads
    evm_prob       float16 (N, len(evm_classes)), EVM inclusion probabilities (EVM methods only)
    ood_score      float32 (N,), OOD detector score, higher = more in-distribution (OOD methods only)

Features are not stored (about 40 MB per step and run); the drift analysis saves them for a fixed subsample.
"""
import datetime
import os

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset

from utils import run_log

FORMAT_VERSION = 1


class _Indexed(Dataset):
    """Returns (index, sample) so that the dataset index of every document that survives the collate is known."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return i, self.dataset[i]


def _indexed_collate(collate):
    """Wrap a collate function that may drop samples (e.g. DIL's safe_collate drops documents without OCR text):
    keep only the samples it accepts on their own, and add their dataset indices as batch["_idx"]."""
    collate = collate or torch.utils.data.default_collate

    def wrapped(batch):
        kept = []
        for i, sample in batch:
            try:
                collate([sample])
                kept.append((i, sample))
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
        if not kept:
            return None
        out = collate([s for _, s in kept])
        out["_idx"] = torch.tensor([i for i, _ in kept])
        return out

    return wrapped


def _doc_ids(dataset):
    """'<class>/<file>' for every sample, if the dataset keeps (path, ...) tuples or dicts in .samples
    (also through Subset and ConcatDataset, e.g. the Tobacco-3482 splits)."""
    if isinstance(dataset, Subset):
        ids = _doc_ids(dataset.dataset)
        return [ids[i] for i in dataset.indices] if ids is not None else None
    if isinstance(dataset, ConcatDataset):
        parts = [_doc_ids(d) for d in dataset.datasets]
        return None if any(p is None for p in parts) else [i for p in parts for i in p]
    samples = getattr(dataset, "samples", None)
    if not samples:
        return None
    ids = []
    for s in samples:
        path = s[0] if isinstance(s, (tuple, list)) else (s.get("img_path") or s.get("image_path") or s.get("path")
                                                          if isinstance(s, dict) else None)
        if path is None:
            return None
        path = str(path)
        ids.append(os.path.join(os.path.basename(os.path.dirname(path)), os.path.basename(path)))
    return ids


def _forward(model, batch, device):
    """(logits, features, extra heads) for an EAML or LayoutLMv3 batch, in one pass (eval mode: no dropout)."""
    if "images" in batch:  # EAML
        images = batch["images"].to(device)
        texts = {k: v.to(device) for k, v in batch["texts"].items()}
        out = model(images=images, texts=texts, return_features=True)
        return out["fusion_logits"], out["fused_feat"], {"logits_image": out["image_logits"],
                                                         "logits_text": out["text_logits"]}
    ids, bbox, mask, pix = (batch[k].to(device) for k in ("input_ids", "bbox", "attention_mask", "pixel_values"))
    feats = model.extract_features(ids, bbox, mask, pix)
    return model.classifier(feats), feats, {}


def evm_class_names(evm):
    """Class names of the columns of evm.predict_proba_tensor (EVM / RegEVM: weibull_models, iEVM: class_evs)."""
    models = getattr(evm, "weibull_models", None) or getattr(evm, "class_evs", None) or {}
    return [str(c) for c in models]


@torch.no_grad()
def collect_predictions(model, loader, device, evm=None, ood_detector=None):
    """Run `model` over the dataset of `loader` (in dataset order, no shuffling) and return per-document arrays."""
    model.eval()
    dataset = loader.dataset
    eval_loader = DataLoader(_Indexed(dataset), batch_size=loader.batch_size or 32, shuffle=False,
                             num_workers=loader.num_workers, collate_fn=_indexed_collate(loader.collate_fn))
    use_evm = evm is not None and getattr(evm, "initialized", False)
    use_ood = ood_detector is not None and getattr(ood_detector, "initialized", False)
    cols = {"_idx": [], "y_true": [], "logits": [], "logits_image": [], "logits_text": [], "evm_prob": [],
            "ood_score": []}
    for batch in eval_loader:
        if batch is None:
            continue
        logits, feats, extra = _forward(model, batch, device)
        cols["_idx"].append(batch["_idx"])
        cols["y_true"].append(torch.as_tensor(batch["labels"]).cpu())
        cols["logits"].append(logits.float().cpu())
        for k, v in extra.items():
            cols[k].append(v.float().cpu())
        if use_evm:
            cols["evm_prob"].append(evm.predict_proba_tensor(feats.float()).float().cpu())
        if use_ood:
            cols["ood_score"].append(torch.as_tensor(np.asarray(
                ood_detector.score(feats.float().cpu().numpy(), logits.float().cpu().numpy()), dtype=np.float32)))

    if not cols["_idx"]:
        raise RuntimeError(f"No documents could be evaluated from {type(dataset).__name__} ({len(dataset)} samples)")
    idx = torch.cat(cols["_idx"]).numpy()
    logits = torch.cat(cols["logits"])
    out = {
        "y_true": torch.cat(cols["y_true"]).numpy().astype(np.int16),
        "logits": logits.numpy().astype(np.float16),
        "pred": logits.argmax(dim=1).numpy().astype(np.int16),
    }
    for k in ("logits_image", "logits_text", "evm_prob"):
        if cols[k]:
            out[k] = torch.cat(cols[k]).numpy().astype(np.float16)
    if cols["ood_score"]:
        out["ood_score"] = torch.cat(cols["ood_score"]).numpy().astype(np.float32)
    ids = _doc_ids(dataset)
    out["doc_id"] = [ids[i] for i in idx] if ids is not None else [str(i) for i in idx]
    if len(idx) < len(dataset):
        print(f"[predictions] {len(dataset) - len(idx)} of {len(dataset)} documents dropped by the collate function")
    return out


def save_predictions(model, device, out_dir, tag, output_classes, sets, evm=None, ood_detector=None, **meta):
    """Evaluate `model` on every test set and save one predictions file.

    out_dir:         run directory (usually the checkpoint dir); the file goes to <out_dir>/predictions/
    tag:             e.g. "cil_<new class>", "dil", "base_cil"; the file is <tag>_seed<seed>.pt
    output_classes:  class names of the model's output columns, in logit order
    sets:            {name: (loader, class_names)}; class_names are the names of the loader's label indices.
                     Sets whose loader is None or empty are skipped.
    meta:            stored in meta (e.g. step, new_class, checkpoint); overrides the run_log fields, e.g. seed for
                     the base models, which do not log to the results table
    Returns (path, {set name: accuracy over its documents whose class is an output class}).
    """
    run = dict(run_log._run or {})
    meta = {**{k: run.get(k) for k in ("run_id", "git_commit", "script", "backbone", "setting", "method",
                                        "strategy", "bias_correction", "seed")},
            "step": os.environ.get("IL_STEP"), "created": datetime.datetime.now().isoformat(timespec="seconds"),
            **meta}
    seed = meta.get("seed") if meta.get("seed") is not None else ""
    result = {"format_version": FORMAT_VERSION, "meta": meta, "output_classes": [str(c) for c in output_classes],
              "evm_classes": evm_class_names(evm) if evm is not None and getattr(evm, "initialized", False) else None,
              "sets": {}}
    accs = {}
    out_index = {c: i for i, c in enumerate(result["output_classes"])}
    for name, (loader, class_names) in sets.items():
        if loader is None or len(loader.dataset) == 0:
            continue
        s = collect_predictions(model, loader, device, evm=evm, ood_detector=ood_detector)
        s["class_names"] = [str(c) for c in class_names]
        result["sets"][name] = s
        known = np.array([s["class_names"][y] in out_index for y in s["y_true"]])
        if known.any():
            target = np.array([out_index.get(s["class_names"][y], -1) for y in s["y_true"]])
            accs[name] = float((s["pred"][known] == target[known]).mean())

    pred_dir = os.path.join(out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    path = os.path.join(pred_dir, f"{tag}_seed{seed}.pt" if seed != "" else f"{tag}.pt")
    torch.save(result, path + ".tmp")
    os.replace(path + ".tmp", path)  # never leave a half-written file behind
    summary = ", ".join(f"{n}: {len(s['y_true'])} docs" + (f", acc {accs[n]:.4f}" if n in accs else "")
                        for n, s in result["sets"].items())
    print(f"[predictions] {path} ({summary})")
    return path, accs


def load_predictions(path):
    """Load a predictions file (dict described in the module docstring)."""
    return torch.load(path, map_location="cpu", weights_only=False)
