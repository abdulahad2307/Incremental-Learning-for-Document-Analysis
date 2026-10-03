import os
import glob
import torch
from utils.domain_IL.dil_utils import mix_replay
from torch import nn
import numpy as np
from tqdm import tqdm

from collections import defaultdict

def save_checkpoint_dil(model, optimizer, epoch, ckpt_path, extra_data=None):
    """
    Save model and optimizer state as a checkpoint for domain incremental learning.
    extra_data: optional dict to be stored in the checkpoint (e.g., metrics, classes, etc.)
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra_data is not None:
        checkpoint.update(extra_data)
    torch.save(checkpoint, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

def save_epoch_checkpoint_dil(model, optimizer, epoch, ckpt_dir, is_best=False, extra_data=None, max_keep_last=2):
    """
    Save a checkpoint for the current epoch in ckpt_dir. 
    Keeps only the best and the last N epoch checkpoints (default N=2).
    """
    # Save current epoch checkpoint
    epoch_ckpt_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    if extra_data is not None:
        checkpoint.update(extra_data)
    torch.save(checkpoint, epoch_ckpt_path)
    print(f"Checkpoint saved: {epoch_ckpt_path}")

    # If this is best checkpoint, save separately
    if is_best:
        best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
        torch.save(checkpoint, best_ckpt_path)
        print(f"Best checkpoint updated: {best_ckpt_path}")

    # Prune: Keep best_model.pth and only last `max_keep_last` epoch_*.pth files
    all_epoch_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pth")))
    # Remove all but the last two
    to_remove = all_epoch_ckpts[:-max_keep_last] if len(all_epoch_ckpts) > max_keep_last else []
    for ckpt_path in to_remove:
        if os.path.basename(ckpt_path) != "best_model.pth":
            os.remove(ckpt_path)
            print(f"Removed old checkpoint: {ckpt_path}")


def load_checkpoint_dil(model, optimizer, ckpt_path, device):
    """
    Load model and optimizer state from checkpoint, for DIL.
    Returns the start epoch.
    """
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}.")
        return 0
    checkpoint = torch.load(ckpt_path, map_location=device,weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0)


def train_one_epoch_dil(
    model,
    train_loaders,
    optimizer,
    criterion,
    device,
    strategy,
    ewc=None,
    exemplar_manager=None,
    old_model=None,
    use_bias_correction=False,
    ):
    model.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    batch_count = 0
    total_batches = sum(len(loader) for loader in train_loaders.values())
    pbar = tqdm(total=total_batches, desc="Training", unit="batch")
    for domain, loader in train_loaders.items():
        for batch in loader:
            # Current-domain batch plus replayed exemplars of the pretrained domain (if a replay memory is used)
            inputs, labels, text_inputs = mix_replay(exemplar_manager, batch["images"], batch["labels"],
                                                     batch.get("texts", None), device)

            batch_data = {"images": inputs, "labels": labels}
            if text_inputs is not None:
                batch_data["texts"] = text_inputs

            loss, preds, _ = strategy.compute_loss(
                model,
                batch_data,
                criterion,
                old_model=old_model,
                ewc=ewc,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += (preds == labels).float().mean().item()
            batch_count += 1

            #loss_accum += loss.item()
            #acc_accum += (preds == labels).float().mean().item()
            #count += 1
            pbar.set_postfix(loss=epoch_loss / batch_count, acc=epoch_acc / batch_count)
            pbar.update(1)
    pbar.close()

    if use_bias_correction and hasattr(model, "fusion_classifier"):
        with torch.no_grad():
            bias_correction = model.fusion_classifier.bias.mean().item()
            model.fusion_classifier.bias -= bias_correction
        print("Bias correction applied after epoch.")

    return epoch_loss / batch_count, epoch_acc / batch_count

def train_one_epoch_dil_evm_ood(
    model,
    train_loaders,
    optimizer,
    criterion,
    device,
    strategy,
    exemplar_manager=None,
    ewc=None,
    old_model=None,
    use_bias_correction=True,
    evm=None,
    ood_detector=None,
    lambda_evm=0.1,
    lambda_ood=0.1,
):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Assume one domain loader (key = current_domain)
    for domain_name, train_loader in train_loaders.items():
        for batch in train_loader:
            optimizer.zero_grad()

            # Unpack batch according to your loader structure
            images = batch["images"].to(device)
            texts = batch.get("texts") # Optional branch
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            features = model.extract_features(images=images, input_ids=input_ids, attention_mask=attention_mask, texts=texts)
            inputs_for_ood = {
                "images": images,
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }

            # Standard CE loss
            loss = criterion(logits, labels)

            # EVM loss
            if evm is not None and hasattr(evm, 'initialized') and evm.initialized and features is not None:
                evm_probs = evm.predict_proba_tensor(features)
                batch_indices = torch.arange(labels.size(0))
                true_class_probs = evm_probs[batch_indices, labels.cpu()]
                evm_loss = -torch.log(true_class_probs + 1e-8).mean()
                loss = loss + lambda_evm * evm_loss

            # OOD loss
            if ood_detector is not None and hasattr(ood_detector, 'initialized') and ood_detector.initialized and inputs_for_ood is not None:
                with torch.no_grad():
                    if hasattr(ood_detector, "score_batch"):
                        ood_scores = ood_detector.score_batch(model, inputs_for_ood, device, texts=texts)
                        if ood_scores is not None:
                            ood_scores_tensor = torch.tensor(ood_scores, device=device)
                            ood_loss = ood_scores_tensor.mean()
                            loss = loss + lambda_ood * ood_loss

            # Strategy-specific loss
            if strategy:
                inc_loss, preds, target_labels = strategy.compute_loss(
                    model, batch, criterion, old_model=old_model, ewc=ewc
                )
                loss = loss + inc_loss
            else:
                preds = torch.argmax(logits, dim=1)
                target_labels = labels

            loss.backward()
            optimizer.step()

            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(target_labels.detach().cpu().tolist())
            total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return avg_loss, acc


def evaluate_dil(model, val_loaders, device):
    model.eval()
    total_batches = sum(len(loader) for loader in val_loaders.values())
    pbar = tqdm(total=total_batches, desc="Validating", unit="batch")

    loss_accum, correct_accum, count = 0., 0., 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for domain, loader in val_loaders.items():
            for batch in loader:
                inputs = batch["images"].to(device)
                labels = batch["labels"].to(device)
                texts = batch.get("texts", None)
                if texts is not None:
                    texts = {k: v.to(device) for k, v in texts.items()}
                    logits = model(inputs, texts)
                else:
                    logits = model(inputs)
                loss = criterion(logits, labels)
                loss_accum += loss.item() * labels.size(0)
                correct_accum += (logits.argmax(1) == labels).sum().item()
                count += labels.size(0)
                pbar.set_postfix(loss=loss_accum/count, acc=correct_accum/count)
                pbar.update(1)
    pbar.close()
    return loss_accum/count, correct_accum/count


def classwise_accuracy(model, loader, device, num_classes):
    model.eval()
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    with torch.no_grad():
        for batch in loader:
            inputs = batch["images"].to(device)
            labels = batch["labels"].to(device)
            texts = batch.get("texts", None)
            if texts is not None:
                texts = {k: v.to(device) for k, v in texts.items()}
                logits = model(inputs, texts)
            else:
                logits = model(inputs)
            preds = logits.argmax(dim=1)
            for i in range(len(labels)):
                total[labels[i]] += 1
                if preds[i] == labels[i]:
                    correct[labels[i]] += 1
    acc = torch.zeros(num_classes)
    nonzero = total != 0
    acc[nonzero] = correct[nonzero] / total[nonzero]
    acc[~nonzero] = float('nan')
    return acc.cpu().numpy()


def evaluate_domain(model, loader, device, num_classes, class_names=None, print_classes=True):
    """
    Evaluate model on a single domain/loader and return overall accuracy and per-class accuracy.
    Args:
      model: Trained model.
      loader: DataLoader for the domain.
      device: torch device.
      num_classes: number of classes.
      class_names: optional list of class names for printing.
    Returns:
      overall_acc: float accuracy.
      class_acc: np.array per-class accuracy.
    """
    overall_loss, overall_acc = evaluate_dil(model, { 'domain': loader }, device)
    class_acc = classwise_accuracy(model, loader, device, num_classes)
    print(f"Domain Loss: {overall_loss:.4f},  Accuracy: {overall_acc:.4f}")

    if print_classes:
        missing = []
        for idx in range(len(class_acc)):
            if np.isnan(class_acc[idx]):
                missing.append(class_names[idx] if class_names else str(idx))
            else:
                print(f"  {class_names[idx]:>16}: {class_acc[idx]:.4f}" if class_names else f"  Class {idx}: {class_acc[idx]:.4f}")
        if missing:
            print(f"  [WARN] No samples for classes: {', '.join(missing)}")

    return overall_acc, class_acc

