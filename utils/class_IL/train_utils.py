import os
import torch
import time
import torch
import torch.nn as nn
import numpy as np
from utils.evm.evm_loss import evm_nll_loss
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

class CILMetrics_old:
    def __init__(self, class_names):
        self.num_classes = len(class_names)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.current_state = 0
        self.class_mapping = {0: class_names}

    def flatten_classes(self):
        """Return full list of classes across all states in order"""
        all_classes = []
        for s in sorted(self.class_mapping.keys()):
            all_classes.extend(self.class_mapping[s])
        return all_classes

    def update(self, preds, labels):
        for p, l in zip(preds, labels):
            if 0 <= l < self.num_classes and 0 <= p < self.num_classes:
                self.confusion_matrix[l, p] += 1
            else:
                print(f"Warning: Label {l} or prediction {p} out of bounds (num_classes={self.num_classes})")

    def get_metrics(self):
        metrics = {}
        total = np.sum(self.confusion_matrix)
        metrics['top1_acc'] = np.trace(self.confusion_matrix) / total if total > 0 else 0
        if self.current_state > 0:
            prev_classes = []
            for s in range(self.current_state):
                prev_classes.extend(self.class_mapping[s])
            n_prev = len(prev_classes)
            if n_prev > 0:
                past_correct = np.trace(self.confusion_matrix[:n_prev, :n_prev])
                past_total = np.sum(self.confusion_matrix[:n_prev, :])
                metrics['past_acc'] = past_correct / past_total if past_total > 0 else 0
                new_correct = np.trace(self.confusion_matrix[n_prev:, n_prev:])
                new_total = np.sum(self.confusion_matrix[n_prev:, :])
                metrics['new_acc'] = new_correct / new_total if new_total > 0 else 0
                metrics['e(p,p)'] = np.sum(self.confusion_matrix[:n_prev, :n_prev]) - past_correct
                metrics['e(p,n)'] = np.sum(self.confusion_matrix[:n_prev, n_prev:])
                metrics['e(n,p)'] = np.sum(self.confusion_matrix[n_prev:, :n_prev])
                metrics['e(n,n)'] = np.sum(self.confusion_matrix[n_prev:, n_prev:]) - new_correct
        return metrics

    def incremental_state_update(self, new_classes):
        self.current_state += 1
        self.class_mapping[self.current_state] = new_classes
        old_size = self.num_classes
        new_size = old_size + len(new_classes)
        new_matrix = np.zeros((new_size, new_size))
        new_matrix[:old_size, :old_size] = self.confusion_matrix
        self.confusion_matrix = new_matrix
        self.num_classes = new_size

    def state_dict(self):
        return {
            'confusion_matrix': self.confusion_matrix,
            'class_mapping': self.class_mapping,
            'current_state': self.current_state,
            'num_classes': self.num_classes
        }

    def load_state_dict(self, state):
        self.confusion_matrix = state['confusion_matrix']
        self.class_mapping = state['class_mapping']
        self.current_state = state['current_state']
        self.num_classes = state['num_classes']

"""
def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Checkpoint saved to {path}")

def save_checkpoint(model, optimizer, epoch, path, extra_data=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if extra_data is not None:
        state.update(extra_data)
    torch.save(state, path)
    print(f"Checkpoint saved to {path}")
"""
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

"""
def load_checkpoint(model, optimizer, path, device):
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    return checkpoint['epoch']
"""

def load_checkpoint(model, optimizer, path, device, metrics=None):
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}")
        return 0
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if metrics is not None and 'metrics_state' in checkpoint:
        metrics.load_state_dict(checkpoint['metrics_state'])
        print(f"Restored metrics state from checkpoint with {metrics.num_classes} classes")
    print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
    #return checkpoint['epoch']
    return checkpoint

"""
def train_one_epoch_cil(model, dataloader, optimizer, criterion, device, metrics):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        # --- EAML ---
        if "images" in batch:  
            images = batch['images'].to(device)
            input_ids = batch['texts']['input_ids'].to(device)
            attention_mask = batch['texts']['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

        # --- DocFormer ---
        else:
            inputs = {
                'pixel_values': batch['pixel_values'].to(device),
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'bboxes': batch['bboxes'].to(device)
            }
            labels = batch['labels'].to(device)
            outputs = model(**inputs, task="classification")
            logits = outputs['logits']

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        metrics.update(preds.cpu().numpy(), labels.cpu().numpy())
        total_loss += loss.item()

    epoch_loss = total_loss / len(dataloader)
    label_acc = accuracy_score(all_labels, all_preds)
    metric_dict = metrics.get_metrics()
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    metric_dict.update({
        'loss': epoch_loss,
        'label_acc': label_acc,
        'preds': all_preds,
        'labels': all_labels,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })

    print(f"Train Loss: {epoch_loss:.4f} | Label Acc: {label_acc:.4f} | Epoch Acc: {metric_dict['top1_acc']:.4f}")
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    return metric_dict
"""

def train_one_epoch_cil_v2(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    metrics,
    inc_strategy,
    old_model=None,
    ewc=None
):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        # The strategy loss includes the cross-entropy (plus distillation / EWC when enabled)
        loss, preds, target_labels = inc_strategy.compute_loss(
            model, batch, criterion, old_model=old_model, ewc=ewc
        )
        
        loss.backward()
        optimizer.step()
        
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(target_labels.detach().cpu().tolist())
        metrics.update(np.array(preds.detach().cpu().tolist()), np.array(target_labels.detach().cpu().tolist()))  # this batch only
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
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


def train_one_epoch_cil(
        model,
        dataloader,
        optimizer,
        criterion,
        device,
        metrics,
        inc_strategy,
        old_model=None,
        ewc=None
    ):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader):
        optimizer.zero_grad()
        # Have to use strategy-specific loss here (handles distillation, EWC, etc)
        loss, preds, labels = inc_strategy.compute_loss(
            model, batch, criterion, old_model=old_model, ewc=ewc
        )
        #print(f"Batch total loss: {loss.item()}")
        loss.backward()
        optimizer.step()

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        metrics.update(preds.cpu().numpy(), labels.cpu().numpy())
        total_loss += loss.item()

    epoch_loss = total_loss / len(dataloader)
    label_acc = accuracy_score(all_labels, all_preds)
    metric_dict = metrics.get_metrics()
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    metric_dict.update({
        'loss': epoch_loss,
        'label_acc': label_acc,
        'preds': all_preds,
        'labels': all_labels,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })

    print(f"Train Loss: {epoch_loss:.4f} | Label Acc: {label_acc:.4f} | Epoch Acc: {metric_dict['top1_acc']:.4f}")
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    return metric_dict
# ---- For EVM integrated training ----- #
def train_one_epoch_cil_with_evm(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    metrics,
    inc_strategy,
    evm=None,
    lambda_evm=0.1,
    old_model=None,
    ewc=None
):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader):
        optimizer.zero_grad()

        if "images" in batch:
            images = batch["images"].to(device)
            texts = batch["texts"]
            input_ids = batch["texts"]["input_ids"].to(device)
            attention_mask = batch["texts"]["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            features = model.extract_features(images=images, input_ids=input_ids, attention_mask=attention_mask,texts=texts)
        else:
            inputs = {
                "pixel_values": batch["pixel_values"].to(device),
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "bbox": batch["bboxes"].to(device)
            }
            labels = batch["labels"].to(device)
            texts = batch["texts"]
            #outputs = model(**inputs, task="classification")
            #logits = outputs["logits"]
            outputs = model(**inputs)
            logits = outputs if not isinstance(outputs, dict) else outputs["logits"]

            features = model.extract_features(**inputs)

        # EVM loss over samples whose class the EVM already knows (EVM is keyed by class name)
        evm_loss = evm_nll_loss(evm, features, labels, class_names=getattr(dataloader.dataset, "current_classes", None))

        if inc_strategy:
            # the strategy loss already includes the cross-entropy
            loss, preds, target_labels = inc_strategy.compute_loss(
                model, batch, criterion, old_model=old_model, ewc=ewc
            )
        else:
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            target_labels = labels
        if evm_loss is not None:
            loss = loss + lambda_evm * evm_loss

        loss.backward()
        optimizer.step()

        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(target_labels.detach().cpu().tolist())
        metrics.update(np.array(preds.detach().cpu().tolist()), np.array(target_labels.detach().cpu().tolist()))  # this batch only
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    acc = metrics.get_metrics().get("top1_acc", 0)

    print(f"Train Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

    return {"loss": avg_loss, "accuracy": acc}



def evaluate(model, dataloader, device, metrics, full_acc=None, evm=None, use_evm=False):
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                # --- EAML ---
                if "images" in batch:
                    images = batch['images'].to(device)
                    input_ids = batch['texts']['input_ids'].to(device)
                    attention_mask = batch['texts']['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    outputs = model(images=images, input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs

                # --- DocFormer ---
                else:
                    inputs = {
                        'pixel_values': batch['pixel_values'].to(device),
                        'input_ids': batch['input_ids'].to(device),
                        'attention_mask': batch['attention_mask'].to(device),
                        'bboxes': batch['bboxes'].to(device)
                    }
                    labels = batch['labels'].to(device)
                    outputs = model(**inputs, task="classification")
                    logits = outputs['logits']

                loss = criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                metrics.update(preds.cpu().numpy(), labels.cpu().numpy())

            except Exception as e:
                print(f"Error processing batch: {e}")
                continue

    eval_loss = total_loss / num_batches if num_batches > 0 else 0
    label_acc = accuracy_score(all_labels, all_preds)
    eval_metrics = metrics.get_metrics()
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    eval_metrics.update({
        'loss': eval_loss,
        'label_acc': label_acc,
        'preds': all_preds,
        'labels': all_labels,
        'precision': prec,
        'recall': rec,
        'f1': f1
    })

    gil = None
    if full_acc is not None and 'top1_acc' in eval_metrics:
        gil = (eval_metrics['top1_acc'] - full_acc) / (1 - full_acc)
        print(f"Incremental Learning Gap (GIL): {gil:.4f}")

    print("\nEvaluation Results:")
    print(f"Loss: {eval_loss:.4f} | Label Acc: {label_acc:.4f} | Epoch Acc: {eval_metrics.get('top1_acc', label_acc):.4f}")
    print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    return eval_metrics

