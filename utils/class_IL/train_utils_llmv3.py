import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import gc

class CILMetrics:
    def __init__(self, class_names):
        self.num_classes = len(class_names)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, preds, labels):
        for p, l in zip(preds, labels):
            if 0 <= l < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[l, p] += 1

    def get_metrics(self):
        total = np.sum(self.confusion_matrix)
        top1_acc = np.trace(self.confusion_matrix) / total if total > 0 else 0
        correct_per_class = np.diag(self.confusion_matrix)
        total_per_class = self.confusion_matrix.sum(axis=1)
        class_acc = [(c/t if t > 0 else 0.0) for c, t in zip(correct_per_class, total_per_class)]
        return {
            'top1_acc': top1_acc,
            'class_acc': class_acc,
            'confusion_matrix': self.confusion_matrix.tolist(),
        }

def save_checkpoint(model, optimizer, epoch, path, extra_data=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
    }
    if extra_data is not None:
        state.update(extra_data)
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")

def train_one_epoch_cil_v2(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    metrics,
    inc_strategy=None,
    old_model=None,
    ewc=None,
    incremental=True,
    accum_steps: int =  8 # Use 1 if no gradient accumulation needed
):
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(dataloader)):
        with torch.cuda.amp.autocast():
            inputs = {
                "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
                "input_ids": batch["input_ids"].to(device, non_blocking=True),
                "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
                "bbox": batch["bbox"].to(device, non_blocking=True)
            }
            labels = batch["labels"].to(device, non_blocking=True)

            outputs = model(**inputs)
            logits = outputs if not isinstance(outputs, dict) else outputs["logits"]
            loss = criterion(logits, labels)
            if ewc is not None:
                loss = loss + ewc.penalty(model)
            loss = loss / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        metrics.update(preds.detach().cpu().numpy(), labels.detach().cpu().numpy())  # this batch only
        total_loss += loss.item() * accum_steps

        del outputs, logits, loss, inputs, labels
        torch.cuda.empty_cache()
        gc.collect()

    # Handle remaining gradients if not divisible
    if len(dataloader) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    metric_vals = metrics.get_metrics()
    acc = metric_vals.get("top1_acc", 0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Train Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    return {
        "loss": avg_loss,
        "top1_acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def evaluate(model, dataloader, device, metrics, full_acc=None):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = {
                'pixel_values': batch['pixel_values'].to(device, non_blocking=True),
                'input_ids': batch['input_ids'].to(device, non_blocking=True),
                'attention_mask': batch['attention_mask'].to(device, non_blocking=True),
                'bbox': batch['bbox'].to(device, non_blocking=True)
            }
            labels = batch['labels'].to(device, non_blocking=True)
            outputs = model(**inputs)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            metrics.update(preds.cpu().numpy(), labels.cpu().numpy())

            del outputs, logits, loss, inputs, labels
            torch.cuda.empty_cache()
            gc.collect()

    eval_loss = total_loss / num_batches if num_batches > 0 else 0
    acc_score = accuracy_score(all_labels, all_preds)
    eval_metrics = metrics.get_metrics()
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    eval_metrics.update({
        'loss': eval_loss,
        'label_acc': acc_score,
        'preds': all_preds,
        'labels': all_labels,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })

    if full_acc is not None and 'top1_acc' in eval_metrics:
        gil = (eval_metrics['top1_acc'] - full_acc) / (1 - full_acc)
        print(f"Incremental Learning Gap (GIL): {gil:.4f}")
    print(f"\nEvaluation Results:\nLoss: {eval_loss:.4f} | Accuracy: {acc_score:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    return eval_metrics
