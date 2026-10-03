import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Dict

from sklearn.metrics import precision_score, recall_score, f1_score

from utils.eaml.eaml_model import EAMLModel
from utils.docformer.model import DocFormer
from utils.docformer.config import DocFormerConfig
from utils.class_IL.dataloader_utils import (
    get_class_il_loader, EAMLClassILDataset, common_transform, eaml_collate_fn
)
from utils.class_IL.train_utils import (
    save_checkpoint, load_checkpoint, CILMetrics, evaluate
)
from utils.class_IL.cil_utils import (
    StandardIncremental, DistillationIncremental, EWC, ExemplarManager,
    AdaptiveLR, extract_features, extract_features_and_logits, extract_feature_vectors, fill_exemplar_memory
)
from utils.class_IL.training_modes import get_training_mode
from utils import run_log
from utils.evm.evm_classifier import EVMClassifier
from utils.evm.evm_eval import evm_openset_metrics
from utils.evm.evm_viz import plot_openset_histograms
from utils.ood.ood_eval import make_detector, evaluate_ood, subsample_loader, collect_features_logits, last_linear_layer
from utils.ood.vim import VIM_OOD
from utils.ood.ood_loss import vim_ood_loss
from utils.evm.evm_state import load_open_set_state, save_open_set_state
from utils.evm.evm_loss import evm_nll_loss
from utils.il_checks import warn_inactive_terms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    else:
        return data


def train_one_epoch_evm_ood(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    metrics,
    inc_strategy,
    evm=None,
    ood_vim=None,
    lambda_evm=0.1,
    lambda_ood=0.1,
    old_model=None,
    ewc=None
):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        # Supporting original batch structure with dictionary keys or tuple/list
        if isinstance(batch, dict) and ("images" in batch or "pixel_values" in batch):
            if "images" in batch:
                images = batch["images"].to(device)
                texts = batch["texts"]
                input_ids = batch["texts"]["input_ids"].to(device)
                attention_mask = batch["texts"]["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                features = model.extract_features(images=images, input_ids=input_ids, attention_mask=attention_mask, texts=texts)
                inputs_for_ood = {
                    "images": images,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask
                }
            else:
                inputs = {
                    "pixel_values": batch["pixel_values"].to(device),
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "bboxes": batch["bboxes"].to(device)
                }
                labels = batch["labels"].to(device)
                texts = batch["texts"]
                outputs = model(**inputs, task="classification")
                logits = outputs["logits"]
                features = model.extract_features(**inputs)
                inputs_for_ood = inputs
        else:
            # fallback for non-dict batches or unknown keys
            inputs_for_ood = None
            labels = batch[1].to(device)
            if hasattr(model, 'extract_features'):
                features = model.extract_features(batch[0].to(device))
            else:
                features = None
            outputs = model(batch[0].to(device))
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        # EVM loss over samples whose class the EVM already knows (the new class is added after the step)
        extra_loss = 0.0
        class_names = getattr(dataloader.dataset, "current_classes", None)
        evm_loss = evm_nll_loss(evm, features, labels, class_names=class_names)
        if evm_loss is not None:
            extra_loss = lambda_evm * evm_loss
        # L_OOD (ViM virtual-logit penalty) over samples of classes the ViM was fitted on
        ood_loss = vim_ood_loss(ood_vim, features, logits, labels, class_names=class_names)
        if ood_loss is not None:
            extra_loss = extra_loss + lambda_ood * ood_loss

        # Incremental strategy loss (already includes the cross-entropy) and predictions
        if inc_strategy:
            inc_loss, preds, target_labels = inc_strategy.compute_loss(
                model, batch, criterion, old_model=old_model, ewc=ewc
            )
            loss = inc_loss + extra_loss
        else:
            loss = criterion(logits, labels) + extra_loss
            preds = torch.argmax(logits, dim=1)
            target_labels = labels

        loss.backward()
        optimizer.step()

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(target_labels.detach().cpu().tolist())
        metrics.update(np.array(preds.detach().cpu().tolist()), np.array(target_labels.detach().cpu().tolist()))  # this batch only
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    acc = metrics.get_accuracy() if hasattr(metrics, "get_accuracy") else metrics.get_metrics().get("top1_acc", 0)

    print(f"Train Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

    return {"loss": avg_loss, "accuracy": acc}


def run_incremental_learning_evm_ood(
    data_root: str,
    ocr_tensor_path: str,
    all_classes: List[str],
    base_classes: List[str],
    unseen_classes: List[str],
    base_model_path: str,
    model_name: str,
    checkpoint_dir: str,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    num_epochs: int = 10,
    strategy: str = "distillation",
    temperature: float = 2.0,
    lambda_distill: float = 1.0,
    lambda_ewc: float = 5000.0,
    use_ewc: bool = True,
    use_exemplars: bool = True,
    joint_training: bool = False,
    max_exemplars: int = 320,
    exemplar_selection: str = "herding",
    training_mode: str = "last_layer",
    trainable_layers: Optional[List[str]] = None,
    resume: bool = False,
    resume_checkpoint: Optional[str] = None,
    global_best_acc: float = 0.0,
    full_model_acc: Optional[float] = None,
    patience: int = 10,
    use_balanced_sampler: bool = True,
    use_bias_correction: bool = True,
    lambda_evm: float = 0.1,
    lambda_ood: float = 0.1,
    ood_method: str = "msp",
    ood_tpr: float = 0.95,
    ood_max_per_class: int = 500,
    evm_tailsize: float = 0.3,
    evm_persist: bool = False,
):
    import shutil
    from torch.utils.data import WeightedRandomSampler, DataLoader


    os.makedirs(checkpoint_dir, exist_ok=True)


    if strategy == "distillation":
        inc_strategy = DistillationIncremental(DEVICE, temperature, lambda_distill)
    else:
        inc_strategy = StandardIncremental(DEVICE)


    exemplar_mgr = (
        ExemplarManager(max_exemplars=max_exemplars, selection_strategy=exemplar_selection)
        if use_exemplars
        else None
    )
    if unseen_classes is None or len(unseen_classes) == 0:
        raise ValueError("Must provide a non-empty list of unseen_classes")
    for cls in unseen_classes:
        if cls not in all_classes:
            raise ValueError(f"Unseen class {cls} not in all_classes")
        if cls in base_classes:
            raise ValueError(f"Unseen class {cls} already in base_classes")


    current_classes = base_classes.copy()
    ewc = None
    evm = EVMClassifier(tailsize=evm_tailsize, cover_threshold=0.7)
    ood_detector = make_detector(ood_method)  # open-set evaluation
    ood_vim = VIM_OOD()                        # ViM behind the training-time L_OOD
    if evm_persist:
        # Continue with the EVM / ViM of the step that produced the base model (saved by a previous job)
        load_open_set_state(base_model_path, evm=evm, vim=ood_vim)


    start_idx = 0
    start_epoch = 0
    epochs_no_improve = 0
    best_acc = 0.0
    best_path = None
    final_best_path = None


    if resume and resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location=DEVICE,weights_only=False)
        current_classes = checkpoint.get("current_classes", base_classes.copy())
        start_idx = checkpoint.get("unseen_index", 0)
        start_epoch = checkpoint.get("epoch", 0)
        epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
        best_acc = checkpoint.get("step_best_acc", 0.0)
        best_path = checkpoint.get("step_best_path", None)
        print(f"Resuming from index {start_idx}, epoch {start_epoch}")
        del checkpoint
        torch.cuda.empty_cache()


    for unseen_idx, new_class in enumerate(unseen_classes[start_idx:], start_idx):
        if unseen_idx > start_idx:
            start_epoch = 0
            epochs_no_improve = 0
            best_acc = 0.0
            best_path = None


        # On resume, current_classes from the checkpoint may already contain new_class
        previous_classes = [c for c in current_classes if c != new_class]
        if new_class not in current_classes:
            current_classes.append(new_class)
        print(f"\n= Incremental step {unseen_idx+1}/{len(unseen_classes)} - Adding {new_class} =")


        new_data_samples = []
        train_dataset = None
        train_loader = None

        if model_name == "eaml":
            train_dataset = EAMLClassILDataset(
                data_dir=os.path.join(data_root, "train"),
                current_classes=current_classes,
                ocr_data_path=ocr_tensor_path,
                transform=common_transform
            )
            if not joint_training:
                # Class-incremental data: the new class's samples plus the exemplar memory of all previous classes
                new_data_samples = [s for s in train_dataset.samples if s[2] == new_class]
                replay_samples = []
                if use_exemplars and exemplar_mgr is not None:
                    fill_exemplar_memory(exemplar_mgr, train_dataset, previous_classes, base_model_path,
                                         len(previous_classes), DEVICE)
                    replay_samples = [s for c in previous_classes for s in exemplar_mgr.exemplars.get(c, [])]
                train_dataset.samples = replay_samples + new_data_samples
                print(f"Training on {len(new_data_samples)} new-class samples and {len(replay_samples)} exemplars "
                      f"of {len(previous_classes)} previous classes.")
            else:
                print(f"Joint training on all data of {len(current_classes)} classes (upper-bound baseline).")
            if use_balanced_sampler:
                from collections import Counter
                class_counts = Counter([s[2] for s in train_dataset.samples])
                total = sum(class_counts.values())
                class_weights = {cls: total/count for cls, count in class_counts.items()}
                sample_weights = [class_weights[s[2]] for s in train_dataset.samples]
                sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    sampler=sampler,
                    num_workers=4,
                    collate_fn=eaml_collate_fn
                )
            else:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4,
                    collate_fn=eaml_collate_fn
                )
        else:
            train_loader = get_class_il_loader(model_name, os.path.join(data_root, "train"), current_classes,
                                               batch_size, ocr_data=ocr_tensor_path)
        val_loader = get_class_il_loader(model_name, os.path.join(data_root, "val"), current_classes,
                                         batch_size, ocr_data=ocr_tensor_path)
        test_loader = get_class_il_loader(model_name, os.path.join(data_root, "test"), current_classes,
                                          batch_size, ocr_data=ocr_tensor_path)
        torch.cuda.empty_cache()


        cfg = DocFormerConfig() if model_name == "docformer" else None

        def build_model(num_classes):
            if model_name == "docformer":
                return DocFormer(cfg, num_classes=num_classes).to(DEVICE)
            return EAMLModel(num_classes=num_classes).to(DEVICE)

        def load_ckpt_into(target, path):
            ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
            target.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
            del ckpt
            torch.cuda.empty_cache()

        start_ep = start_epoch if unseen_idx == start_idx else 0
        ep_no_improve = epochs_no_improve if unseen_idx == start_idx else 0
        best_acc_local = best_acc if unseen_idx == start_idx else 0.0
        best_path_local = best_path if unseen_idx == start_idx else None

        resuming_this_step = (resume and resume_checkpoint and os.path.exists(resume_checkpoint)
                              and unseen_idx == start_idx)
        if resuming_this_step:
            # Resume checkpoint already has the expanded heads for current_classes
            model = build_model(len(current_classes))
            load_ckpt_into(model, resume_checkpoint)
        elif len(previous_classes) > 0 and os.path.exists(base_model_path):
            # Student starts from the previous step's model (incl. old-class heads); heads are expanded below
            model = build_model(len(previous_classes))
            load_ckpt_into(model, base_model_path)
        else:
            print(f"Warning: no previous checkpoint found at {base_model_path}; student starts from scratch.")
            model = build_model(len(current_classes))

        old_model = None
        if len(previous_classes) > 0:
            # Teacher = previous step's best model (base_model_path is advanced after every step)
            old_model = build_model(len(previous_classes))
            if os.path.exists(base_model_path):
                load_ckpt_into(old_model, base_model_path)
            old_model.eval()
            print(f"prev: {len(previous_classes)}, new: {len(current_classes)}")
            if not resuming_this_step:
                model = inc_strategy.adapt_model(model, len(previous_classes), len(current_classes), model_name).to(DEVICE)




        tm = get_training_mode(model, training_mode, trainable_layers)
        model = tm.prepare_for_training()
        warn_inactive_terms(model, use_ewc=use_ewc, lambda_evm=lambda_evm, lambda_ood=lambda_ood)


        optimizer = torch.optim.AdamW(
            tm.get_trainable_params(), lr=lr, weight_decay=weight_decay
        )


        if resuming_this_step:
            ckpt = torch.load(resume_checkpoint, map_location=DEVICE,weights_only=False)
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            del ckpt
            torch.cuda.empty_cache()


        lr_scheduler = AdaptiveLR(optimizer, base_lr=lr)
        criterion = nn.CrossEntropyLoss()


        patience_counter = ep_no_improve


        for epoch in range(start_ep, num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs} for class {new_class}")


            metrics = CILMetrics(current_classes)
            train_results = train_one_epoch_evm_ood(
                model, train_loader, optimizer, criterion, DEVICE, metrics, inc_strategy, evm, ood_vim, lambda_evm, lambda_ood, old_model, ewc if use_ewc else None
            )


            val_metrics = CILMetrics(current_classes)
            val_results = evaluate(model, val_loader, DEVICE, val_metrics, None)


            print(f"Train loss: {train_results['loss']:.4f} | Train acc: {train_results['accuracy']:.4f}")
            print(f"Val loss: {val_results['loss']:.4f} | Val acc: {val_results['top1_acc']:.4f}")
            run_log.log("epoch", step=unseen_idx + 1, new_class=new_class, epoch=epoch + 1,
                        train_loss=train_results['loss'], train_acc=train_results.get('accuracy', train_results.get('top1_acc')),
                        val_loss=val_results['loss'], val_acc=val_results['top1_acc'])


            path_epoch = os.path.join(checkpoint_dir, f"epoch{epoch + 1}_{new_class}.pth")
            path_best = os.path.join(checkpoint_dir, f"best_model_{new_class}.pth")


            if val_results["top1_acc"] > best_acc_local:
                best_acc_local = val_results["top1_acc"]
                best_path_local = path_best
                patience_counter = 0
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    path_best,
                    extra_data={
                        "current_classes": current_classes,
                        "unseen_index": unseen_idx,
                        "epochs_no_improve": patience_counter,
                        "step_best_acc": best_acc_local,
                        "step_best_path": best_path_local,
                        "epoch": epoch + 1,
                        "unseen_class": new_class,
                    },
                )

                # Remove only this class's older epoch checkpoints; keep best models of all steps
                for fname in os.listdir(checkpoint_dir):
                    fpath = os.path.join(checkpoint_dir, fname)
                    if not (fname.startswith("epoch") and fname.endswith(f"_{new_class}.pth")):
                        continue
                    try:
                        os.remove(fpath)
                    except Exception as e:
                        print(f"Failed to remove {fpath}: {e}")
                
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    path_epoch,
                    extra_data={
                        "current_classes": current_classes,
                        "unseen_index": unseen_idx,
                        "epochs_no_improve": patience_counter,
                        "step_best_acc": best_acc_local,
                        "step_best_path": best_path_local,
                        "epoch": epoch + 1,
                        "unseen_class": new_class,
                    },
                )

                
                test_metrics = CILMetrics(current_classes)
                test_results = evaluate(model, test_loader, DEVICE, test_metrics, full_model_acc)
                run_log.log("step_test", step=unseen_idx + 1, new_class=new_class, epoch=epoch + 1, split="test",
                            loss=test_results['loss'], acc=test_results['top1_acc'], gil_base=((test_results['top1_acc'] - full_model_acc) / (1 - full_model_acc)) if full_model_acc else None,
                            class_acc=dict(zip(current_classes, test_results['class_acc'])) if test_results.get('class_acc') is not None else None)
                print("Class-wise Test accuracy:")
                for c, a in zip(current_classes, test_results["class_acc"]):
                    print(f"  {c}: {a:.4f}")
                
                # ===== GIL =====
                gil_base = 0.953
                if full_model_acc is not None:
                    gil = (test_results['top1_acc'] - full_model_acc) / (1 - full_model_acc)
                    print(f"GIL: {gil:.4f}")
                if full_model_acc is not None:
                    gil = (test_results['top1_acc'] - gil_base) / (1 - gil_base)
                    print(f"GIL (wrt base): {gil:.4f}")

            else:
                patience_counter += 1
                print(f"Patience counter: {patience_counter}/{patience}")
                save_checkpoint(
                    model,
                    optimizer,
                    epoch + 1,
                    path_epoch,
                    extra_data={
                        "current_classes": current_classes,
                        "unseen_index": unseen_idx,
                        "epochs_no_improve": patience_counter,
                        "step_best_acc": best_acc_local,
                        "step_best_path": best_path_local,
                        "epoch": epoch + 1,
                        "unseen_class": new_class,
                    },
                )
                if patience_counter >= patience:
                    print("Early stopping triggered")
                    break


            lr_scheduler.step(val_results["top1_acc"])
            torch.cuda.empty_cache()

        # End-of-step updates (EWC, exemplars, EVM) use the step's best model, which is what the next step continues from
        if best_path_local is not None and os.path.exists(best_path_local):
            load_ckpt_into(model, best_path_local)

        if use_ewc:
            ewc = EWC(model, train_loader, DEVICE, lambda_ewc)


        if use_exemplars:
            exemplar_mgr.update(train_dataset, new_class, model)


        if use_bias_correction:
            if hasattr(model, "fusion_classifier"):
                with torch.no_grad():
                    bmean = model.fusion_classifier.bias.mean()
                    model.fusion_classifier.bias -= bmean


        train_feats = extract_features(model, train_loader, DEVICE)
        evm.fit(train_feats)

        # ViM for L_OOD in the next step, fitted on this step's training data (new class + exemplars)
        if lambda_ood > 0:
            f_tr, z_tr, _ = collect_features_logits(model, subsample_loader(train_loader, ood_max_per_class), DEVICE)
            W, b = last_linear_layer(model)
            ood_vim.fit(f_tr, z_tr, W=W, b=b)
            ood_vim.fit_keys = set(current_classes)

        # ---------- OPEN-SET EVALUATION: classes not learned yet are the unknowns ----------
        future_classes = [c for c in all_classes if c not in current_classes]
        if future_classes:
            split_loader = lambda split, classes: get_class_il_loader(
                model_name, os.path.join(data_root, split), classes, batch_size, ocr_data=ocr_tensor_path)
            tag = f"Open-set step {unseen_idx + 1} (+{new_class})"
            evaluate_ood(
                ood_detector, model, DEVICE,
                fit_loader=split_loader("train", current_classes),
                calib_loader=split_loader("val", current_classes),
                id_loader=split_loader("test", current_classes),
                ood_loader=split_loader("test", future_classes),
                tpr=ood_tpr, max_per_class=ood_max_per_class, tag=tag,
                savepath=os.path.join(checkpoint_dir, f"ood_{ood_method}_step{unseen_idx + 1}_{new_class}.png"),
            )
            test_feats = extract_features(
                model, subsample_loader(split_loader("test", current_classes + future_classes), ood_max_per_class), DEVICE)
            evm_res = evm_openset_metrics(evm, test_feats)
            run_log.log("open_set", step=unseen_idx + 1, new_class=new_class, split="test (unseen = future classes)",
                        evm_known_acc=evm_res['open_set_accuracy'], evm_unknown_rej=evm_res['unknown_rejection'],
                        n_known=evm_res['n_known'], n_unknown=evm_res['n_unknown'])
            print(f"[{tag} EVM] known acc {evm_res['open_set_accuracy']:.4f}, unknown rejection {evm_res['unknown_rejection']:.4f} "
                  f"(n_known={evm_res['n_known']}, n_unknown={evm_res['n_unknown']})")
            plot_openset_histograms(evm_res["y_true"], evm_res["y_pred"], evm_res["scores"],
                                    savepath=os.path.join(checkpoint_dir, f"evm_hist_step{unseen_idx + 1}_{new_class}.png"),
                                    known_mask=evm_res["is_known"])


        if best_path_local is not None and os.path.exists(best_path_local):
            base_model_path = best_path_local
            if evm_persist:
                save_open_set_state(best_path_local, evm=evm, vim=ood_vim)
        else:
            print("Warning: No best checkpoint found, skipping base_model_path update.")


        if unseen_idx == len(unseen_classes) - 1:
            if best_path_local is not None and os.path.exists(best_path_local):
                final_best_path = os.path.join(checkpoint_dir, "final_best_model.pth")
                shutil.copy(best_path_local, final_best_path)
                print(f"Saved final best model at: {final_best_path}")


    if final_best_path is not None:
        learned_classes = list(current_classes)  # the final model has one output per class learned so far
        if model_name == "docformer":
            cfg = DocFormerConfig()
            final_model = DocFormer(cfg).to(DEVICE)
        else:
            final_model = EAMLModel(num_classes=len(learned_classes)).to(DEVICE)


        checkpoint = torch.load(final_best_path, map_location=DEVICE,weights_only=False)
        final_model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        final_model.eval()
        torch.cuda.empty_cache()


        train_loader_full = get_class_il_loader(model_name, os.path.join(data_root, "train"), learned_classes, batch_size, ocr_data=ocr_tensor_path)
        val_loader_full = get_class_il_loader(model_name, os.path.join(data_root, "val"), learned_classes, batch_size, ocr_data=ocr_tensor_path)
        test_loader_full = get_class_il_loader(model_name, os.path.join(data_root, "test"), learned_classes, batch_size, ocr_data=ocr_tensor_path)


        def evaluate_print(split_name, loader):
            metric = CILMetrics(learned_classes)
            res = evaluate(final_model, loader, DEVICE, metric, full_model_acc)
            p = precision_score(res['labels'], res['preds'], average='macro', zero_division=0)
            r = recall_score(res['labels'], res['preds'], average='macro', zero_division=0)
            f = f1_score(res['labels'], res['preds'], average='macro', zero_division=0)
            print(f"{split_name} - Loss: {res['loss']:.4f} | Acc: {res['top1_acc']:.4f} | Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f}")
            run_log.log("final", split=split_name, loss=res['loss'], acc=res['top1_acc'], precision=p, recall=r, f1=f,
                        gil_base=((res['top1_acc'] - full_model_acc) / (1 - full_model_acc)) if full_model_acc else None,
                        class_acc=dict(zip(learned_classes, res['class_acc'])) if split_name == "Test" and res.get('class_acc') is not None else None)
            if full_model_acc:
                gil = (res['top1_acc'] - full_model_acc) / (1 - full_model_acc)
                print(f"{split_name} - G_IL: {gil:.4f}")
            if split_name == "Test" and "class_acc" in res:
                print("Class-wise test accuracy:")
                for cls, acc in zip(learned_classes, res["class_acc"]):
                    print(f"  {cls}: {acc:.4f}")
                
            torch.cuda.empty_cache()


        evaluate_print("Train", train_loader_full)
        evaluate_print("Validation", val_loader_full)
        evaluate_print("Test", test_loader_full)


        # All classes are known after the last step; open-set results are reported per step above.


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True)
    p.add_argument('--ocr_tensor_path', required=True)
    p.add_argument('--all_classes', required=True)
    p.add_argument('--base_classes', required=True)
    p.add_argument('--unseen_classes', required=True, help="Comma-separated list of unseen classes")
    p.add_argument('--base_model_path', required=True)
    p.add_argument('--model_name', choices=['eaml', 'docformer'], required=True)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--ood_method', type=str, default='msp', choices=['msp', 'vim', 'gradnorm'])
    p.add_argument('--ood_tpr', type=float, default=0.95, help='Fraction of known val samples accepted; sets the OOD threshold')
    p.add_argument('--ood_max_per_class', type=int, default=500, help='Samples per class used for OOD fitting/calibration/testing')
    p.add_argument('--ood_threshold', type=float, default=None, help='Deprecated, ignored: the threshold is calibrated with --ood_tpr')
    p.add_argument('--evm_tailsize', type=float, default=0.3)
    p.add_argument('--evm_persist', action='store_true', help='Save EVM + ViM after each step and load them in the next job (default: in-memory only)')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_epochs', type=int, default=10)
    p.add_argument('--strategy', choices=['standard','distillation'], default='distillation')
    p.add_argument('--temperature', type=float, default=2.0)
    p.add_argument('--lambda_distill', type=float, default=1.0)
    p.add_argument('--lambda_ewc', type=float, default=5000.0)
    p.add_argument('--use_ewc', action='store_true')
    p.add_argument('--use_exemplars', action='store_true')
    p.add_argument('--joint_training', action='store_true', help='Train each step on all data of all seen classes (upper-bound baseline)')
    p.add_argument('--max_exemplars', type=int, default=320)
    p.add_argument('--exemplar_selection', choices=['random','herding'], default='herding')
    p.add_argument('--training_mode', choices=['classifier_only','last_layer','full','full_model','selective'], default='last_layer')
    p.add_argument('--trainable_layers', nargs='+', default=None)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--resume_checkpoint', type=str, default=None)
    p.add_argument('--global_best_acc', type=float, default=0.0)
    p.add_argument('--full_model_acc', type=float, default=0.953)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--use_balanced_sampler', action='store_true')
    p.add_argument('--use_bias_correction', action='store_true')
    p.add_argument('--lambda_evm', type=float, default=0.1)
    p.add_argument('--lambda_ood', type=float, default=0.1)

    args = p.parse_args()
    run_log.init("eaml", "CIL", "EVM+OOD", args)

    run_incremental_learning_evm_ood(
        data_root=args.data_dir,
        ocr_tensor_path=args.ocr_tensor_path,
        all_classes=args.all_classes.split(','),
        base_classes=args.base_classes.split(','),
        unseen_classes=args.unseen_classes.split(','),
        base_model_path=args.base_model_path,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        ood_method=args.ood_method,
        ood_tpr=args.ood_tpr,
        ood_max_per_class=args.ood_max_per_class,
        evm_tailsize=args.evm_tailsize,
        evm_persist=args.evm_persist,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.num_epochs,
        strategy=args.strategy,
        temperature=args.temperature,
        lambda_distill=args.lambda_distill,
        lambda_ewc=args.lambda_ewc,
        use_ewc=args.use_ewc,
        use_exemplars=args.use_exemplars,
        joint_training=args.joint_training,
        max_exemplars=args.max_exemplars,
        exemplar_selection=args.exemplar_selection,
        training_mode=args.training_mode,
        trainable_layers=args.trainable_layers,
        resume=args.resume,
        resume_checkpoint=args.resume_checkpoint,
        global_best_acc=args.global_best_acc,
        full_model_acc=args.full_model_acc,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_balanced_sampler=args.use_balanced_sampler,
        use_bias_correction=args.use_bias_correction,
        lambda_evm=args.lambda_evm,
        lambda_ood=args.lambda_ood
    )
