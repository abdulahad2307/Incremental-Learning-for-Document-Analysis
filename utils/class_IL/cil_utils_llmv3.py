import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

class EWC:
    def __init__(self, model, dataloader, device, lambda_ewc=5000.0):
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {}
        self._fisher = {}
        self._compute_fisher(dataloader)
        for n, p in self.params.items():
            self._means[n] = p.data.clone()

    def _compute_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.model.train()
        samples = 0
        for batch in dataloader:
            inputs = {
                'pixel_values': batch['pixel_values'].to(self.device),
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'bbox': batch['bbox'].to(self.device)
            }
            labels = batch['labels'].to(self.device)
            outputs = self.model(**inputs)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs
            log_probs = F.log_softmax(logits, dim=1)
            for i in range(len(labels)):
                self.model.zero_grad()
                log_probs[i, labels[i]].backward(retain_graph=True)
                for n, p in self.params.items():
                    if p.grad is not None:
                        fisher[n] += p.grad.data ** 2
            samples += len(labels)
        for n in fisher:
            fisher[n] /= samples
        self._fisher = fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._means and p.requires_grad:
                loss += (self._fisher[n] * (p - self._means[n]) ** 2).sum()
        return self.lambda_ewc * loss

class ExemplarManager:
    def __init__(self, max_exemplars=200, selection_strategy="herding"):
        self.exemplars = {}
        self.max_exemplars = max_exemplars
        self.selection_strategy = selection_strategy

    def update(self, dataset, class_name, model=None, batch_size=8, use_cpu=False):
        if self.selection_strategy == "herding" and model is not None:
            self.exemplars[class_name] = self._select_herding(dataset, class_name, model, batch_size, use_cpu)
        else:
            self.exemplars[class_name] = self._select_random(dataset, class_name)
        self._balance_exemplar_set()

    def _select_random(self, dataset, class_name):
        import random
        class_samples = [s for s in dataset.samples if s[2] == class_name]
        if len(class_samples) <= self.max_exemplars:
            return class_samples
        return random.sample(class_samples, self.max_exemplars)

    def _select_herding(self, dataset, class_name, model, batch_size=8, use_cpu=False):
        import warnings
        class_samples = [s for s in dataset.samples if s[2] == class_name]
        if len(class_samples) == 0:
            warnings.warn(f"No samples for {class_name}")
            return []
        if len(class_samples) <= self.max_per_class:
            return class_samples

        features = []
        device = next(model.parameters()).device
        model.eval()

        # Move model to CPU if requested, else keep on GPU
        feature_device = torch.device('cpu') if use_cpu else device
        model = model.to(feature_device)

        with torch.no_grad():
            for i in range(0, len(class_samples), batch_size):
                batch_samples = class_samples[i:i+batch_size]
                images = []
                input_ids = []
                attention_mask = []
                for s in batch_samples:
                    try:
                        img = dataset.transform(Image.open(s[0]).convert("RGB"))
                        images.append(img)
                        input_ids.append(s[1]['input_ids'])
                        attention_mask.append(s[1]['attention_mask'])
                    except:
                        continue
                if len(images) == 0:
                    continue
                images_tensor = torch.stack(images).to(feature_device)
                input_ids_tensor = torch.stack(input_ids).to(feature_device)
                attention_mask_tensor = torch.stack(attention_mask).to(feature_device)
                
                batch_features = model.extract_features(
                    input_ids=input_ids_tensor,
                    bbox=None,  # Modify if bbox needed, or batch appropriately
                    attention_mask=attention_mask_tensor,
                    pixel_values=images_tensor
                )
                features.append(batch_features.cpu().numpy())

        if len(features) == 0:
            return []
        features = np.vstack(features)

        # Herding selection logic unchanged
        class_mean = np.mean(features, axis=0)
        selected_indices = []
        selected_feats = []

        for _ in range(min(self.max_per_class, len(features))):
            cands = [i for i in range(len(features)) if i not in selected_indices]
            if len(cands) == 0:
                break
            if len(selected_feats) == 0:
                dists = np.linalg.norm(features[cands] - class_mean, axis=1)
                idx = cands[np.argmin(dists)]
            else:
                current_mean = np.mean(selected_feats, axis=0)
                cmeans = (current_mean * len(selected_feats) + features[cands]) / (len(selected_feats)+1)
                dists = np.linalg.norm(cmeans - class_mean, axis=1)
                idx = cands[np.argmin(dists)]
            selected_indices.append(idx)
            selected_feats.append(features[idx])
        return [class_samples[i] for i in selected_indices]

    def _balance_exemplar_set(self):
        total = sum(len(v) for v in self.exemplars.values())
        if total <= self.max_exemplars:
            return
        per_class = self.max_exemplars // len(self.exemplars)
        rem = self.max_exemplars % len(self.exemplars)
        for i, c in enumerate(self.exemplars):
            tgt = per_class + (1 if i < rem else 0)
            self.exemplars[c] = self.exemplars[c][:tgt]

    def get_exemplar_dataset(self):
        all_exemples = []
        for exs in self.exemplars.values():
            all_exemples.extend(exs)
        return all_exemples

class AdaptiveLR:
    def __init__(self, optimizer, base_lr=0.001, min_lr=1e-6, decay_factor=0.75, patience=3):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.best_acc = 0
        self.counter = 0
        for g in self.optimizer.param_groups:
            g['lr'] = base_lr

    def step(self, val_acc):
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                for g in self.optimizer.param_groups:
                    g['lr'] = max(g['lr'] * self.decay_factor, self.min_lr)
                self.counter = 0

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def reset(self, task_complexity=1.0):
        self.best_acc = 0
        self.counter = 0
        for g in self.optimizer.param_groups:
            g['lr'] = self.base_lr / (1 + task_complexity)
