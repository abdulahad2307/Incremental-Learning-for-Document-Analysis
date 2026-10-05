"""Pieces shared by the LayoutLMv3 class- and domain-incremental scripts, following the same workflow as EAML
(ER, EWC, BC, KD via --strategy, and No-EVM / EVM / EVM+OOD / iEVM / RegEVM)."""
import copy
import math

import numpy as np
import torch

from utils.llmv3.llmv3_incremental_utils import distillation_loss
from utils.ievm.ievm import IncrementalEVM
from utils.evm.evm_eval import evm_openset_metrics
from utils.ood.ood_eval import collect_features_logits, subsample_loader
from utils import run_log
from torch.utils.data import ConcatDataset, DataLoader, Subset, WeightedRandomSampler


def _add(parser, name, **kwargs):
    if name not in parser._option_string_actions:
        parser.add_argument(name, **kwargs)


def add_il_args(parser, strategy_default="standard", evm=False, ood=False, persist=False, cil=False):
    _add(parser, "--strategy", choices=["standard", "distillation"], default=strategy_default,
         help="standard: CE (+EWC); distillation: CE + KD from the frozen base model (+EWC)")
    _add(parser, "--temperature", type=float, default=2.0)
    _add(parser, "--lambda_distill", type=float, default=1.0)
    _add(parser, "--use_bias_correction", action="store_true", help="Centre the classifier bias after training (as for EAML)")
    _add(parser, "--exemplar_pool", type=int, default=1000,
         help="Images per previous class that exemplars are selected from (herding) and EWC is estimated on "
              "(EAML: 1000); --max_exemplars is the number kept per class")
    _add(parser, "--base_model_acc", type=float, default=None, help="Base model accuracy (fraction) for GIL w.r.t. the base model")
    if cil:
        _add(parser, "--joint_training", action="store_true",
             help="Train on all training data of all seen classes (upper-bound baseline) instead of new class + exemplars")
        _add(parser, "--use_balanced_sampler", action="store_true", help="Class-balanced sampling of the training data")
    if evm:
        _add(parser, "--lambda_evm", type=float, default=0.1)
        _add(parser, "--evm_tailsize", type=float, default=0.5)
        _add(parser, "--evm_threshold", type=float, default=0.7)
    if persist:
        _add(parser, "--evm_persist", action="store_true",
             help="Save the EVM (and ViM) after the step and load it in the next job (default: in-memory only)")
    if ood:
        _add(parser, "--lambda_ood", type=float, default=0.1)
        _add(parser, "--ood_method", default="vim", choices=["msp", "vim", "gradnorm"], help="Detector for open-set evaluation")
        _add(parser, "--ood_tpr", type=float, default=0.95)
        _add(parser, "--ood_max_per_class", type=int, default=500)


def make_teacher(base_model, strategy):
    """Frozen copy of the base model (before its classifier is expanded), only for the distillation strategy."""
    if strategy != "distillation":
        return None
    teacher = copy.deepcopy(base_model).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def distill_term(new_logits, old_logits, temperature):
    """KL(teacher || student) on the teacher's classes; same gradients as the EAML soft-target cross-entropy."""
    k = old_logits.size(1)
    return distillation_loss(new_logits[:, :k], old_logits, temperature=temperature, alpha=1.0) / temperature ** 2


def center_classifier_bias(model):
    """Bias correction as for EAML: subtract the mean of the classifier bias."""
    with torch.no_grad():
        model.classifier.bias -= model.classifier.bias.mean()
    print("Bias correction applied (classifier bias centred).")


def gil(acc, ref):
    return float("nan") if ref is None else (acc - ref) / (1 - ref)


def features_by_class(model, loader, device, label_names, max_per_class=None):
    """{class name: (N, D) features} for a LayoutLMv3 loader; label ids are mapped through `label_names`."""
    feats, _, labels = collect_features_logits(model, subsample_loader(loader, max_per_class), device)
    out = {}
    for f, l in zip(feats, labels):
        out.setdefault(label_names[int(l)], []).append(f)
    return {k: np.stack(v) for k, v in out.items()}


def fit_evm(evm, model, loader, device, label_names, new_class=None):
    """Fit the EVM on the step's training data; an already fitted iEVM only adds the new class (partial update)."""
    feats = features_by_class(model, loader, device, label_names)
    if isinstance(evm, IncrementalEVM) and evm.initialized and new_class in feats:
        evm.incremental_update({new_class: feats[new_class]})
    else:
        evm.fit(feats)
    print(f"EVM fitted on classes: {list(feats)}")


def evm_open_set_eval(evm, model, device, id_loader, id_names, ood_loader=None, ood_names=None, max_per_class=None, tag="EVM"):
    feats = features_by_class(model, id_loader, device, id_names, max_per_class)
    if ood_loader is not None:
        for k, v in features_by_class(model, ood_loader, device, ood_names, max_per_class).items():
            feats[k] = np.vstack([feats[k], v]) if k in feats else v
    res = evm_openset_metrics(evm, feats)
    print(f"[{tag}] known acc {res['open_set_accuracy']:.4f}, unknown rejection {res['unknown_rejection']:.4f} "
          f"(n_known={res['n_known']}, n_unknown={res['n_unknown']})")
    run_log.log("open_set", split=tag, evm_known_acc=res["open_set_accuracy"], evm_unknown_rej=res["unknown_rejection"],
                n_known=res["n_known"], n_unknown=res["n_unknown"])
    return res


def _dataset_labels(ds):
    """Label ids of every sample without loading images (IncrementalOCRTensorsDataset, Subset, exemplar lists)."""
    if isinstance(ds, ConcatDataset):
        return [l for d in ds.datasets for l in _dataset_labels(d)]
    if isinstance(ds, Subset):
        labels = _dataset_labels(ds.dataset)
        return [labels[i] for i in ds.indices]
    if isinstance(ds, list):  # exemplar items
        return [int(item["labels"]) for item in ds]
    return [ds.class2idx[cls] for _, _, cls in ds.samples]


def build_cil_train_loader(args, unseen_train_loader, exemplar_samples, label_space):
    """Training data of a LayoutLMv3 CIL step: new class + exemplars (default) or, with --joint_training, all
    training data of the classes seen so far; --use_balanced_sampler draws classes equally often."""
    if args.joint_training:
        from utils.llmv3.llmv3_incremental_dataloader import get_incremental_dataloader
        dataset = get_incremental_dataloader(
            dataset_name=args.dataset_name, ocr_tensor_file=args.ocr_tensor_path, classes=label_space,
            label_classes=label_space, image_dir=args.data_dir, split="train", batch_size=args.batch_size,
            images_per_class=args.images_per_class, seed=args.seed,
        ).dataset
        print(f"Joint training on all data of {len(label_space)} classes (upper-bound baseline): {len(dataset)} samples")
    else:
        dataset = ConcatDataset([unseen_train_loader.dataset, exemplar_samples])
    if args.use_balanced_sampler:
        labels = np.array(_dataset_labels(dataset))
        counts = np.bincount(labels)
        weights = 1.0 / counts[labels]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(labels), replacement=True)
        print(f"Balanced sampler over {len(counts[counts > 0])} classes (samples per class: {counts[counts > 0].tolist()})")
        return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
