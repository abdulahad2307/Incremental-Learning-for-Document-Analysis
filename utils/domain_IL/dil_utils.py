import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
#from typing import Dict, List, Optional, Tuple, Union

# ----------- Domain Incremental Strategies -----------

class DomainIncrementalStrategy:
    """Base class for domain incremental learning strategies."""
    def __init__(self, device):
        self.device = device

    def adapt(self, model, old_domains, new_domain):
        return model

    def compute_loss(self, model, batch, criterion, old_model=None, ewc=None, bias_reg=None):
        images = batch["images"].to(self.device)
        labels = batch["labels"].to(self.device)

        if "texts" in batch and batch["texts"] is not None:
            texts = batch["texts"]
            if isinstance(texts, dict):
                texts = {k: v.to(self.device) for k, v in texts.items()}
                logits = model(images, texts)
            else:
                logits = model(images, {'input_ids': texts.to(self.device)})
        else:
            logits = model(images)

        loss = criterion(logits, labels)
        if ewc:
            loss += ewc.penalty(model)
        if bias_reg:
            loss += bias_reg(model, batch)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels


from utils.eaml.mutual_learning import MutualLearningLoss

# EAML base-model loss (EAML paper: image/text/fusion CE + truncated-KL mutual learning, beta = 0.5)
EAML_BASE_LOSS = MutualLearningLoss(kld_weight=0.5)


def base_loss_and_logits(model, images, texts, labels, criterion):
    """The backbone's own training loss and the prediction logits: for EAML (image and text heads) the EAML base loss
    on all three heads and the fusion logits; otherwise criterion (cross-entropy) on the model's logits."""
    if texts is not None and hasattr(model, "image_classifier") and hasattr(model, "text_classifier"):
        outputs = model(images, texts, return_features=True)
        return EAML_BASE_LOSS(outputs, labels)['total_loss'], outputs['fusion_logits']
    logits = model(images, texts) if texts is not None else model(images)
    return criterion(logits, labels), logits


class StandardDomainIL(DomainIncrementalStrategy):
    """
    Standard domain-incremental learning:
    - Pure cross-entropy classification loss
    - Optional EWC penalty to mitigate forgetting
    - Optional bias/correlation regularizer
    - Supports both dict-style and tuple-style batches
    """
    def adapt_model(self, model, old_domains, new_domain):
        if hasattr(model, "add_domain_head") and callable(getattr(model, "add_domain_head")):
            model.add_domain_head(new_domain)
        return model

    def _move_device(self, x):
        if torch.is_tensor(x):
            return x.to(self.device)
        if isinstance(x, dict):
            return {k: self._move_device(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return type(x)(self._move_device(v) for v in x)
        return x

    def _forward(self, model, inputs):
        if isinstance(inputs, dict):
            if "images" in inputs and "texts" in inputs and inputs["texts"] is not None:
                return model(inputs["images"], inputs["texts"])
            elif "images" in inputs:
                return model(inputs["images"])
            else:
                return model(**inputs)
        return model(inputs)

    def compute_loss(self, model, batch, criterion, old_model=None, ewc=None, bias_reg=None):
        if isinstance(batch, tuple) and len(batch) == 3:
            inputs, domain, labels = batch
            labels = self._move_device(labels)
            inputs = self._move_device(inputs)
        elif isinstance(batch, dict):
            labels = self._move_device(batch["labels"])
            if "texts" in batch:
                inputs = {"images": batch["images"], "texts": batch["texts"]}
            else:
                inputs = {"images": batch["images"]}
            inputs = self._move_device(inputs)
            domain = batch.get("domain", None)
        else:
            raise ValueError("Batch format unsupported")

        if isinstance(inputs, dict) and "images" in inputs:
            loss, logits = base_loss_and_logits(model, inputs["images"], inputs.get("texts"), labels, criterion)
        else:
            logits = self._forward(model, inputs)
            loss = criterion(logits, labels)
        if ewc:
            loss += ewc.penalty(model)
        if bias_reg:
            loss += bias_reg(model, {"inputs": inputs, "labels": labels, "domain": domain})

        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels


class DistillationDomainIL(DomainIncrementalStrategy):
    """Domain-IL with knowledge distillation regularizer."""
    def __init__(self, device, temperature=2.0, lambda_distill=1.0):
        super().__init__(device)
        self.temperature = temperature
        self.lambda_distill = lambda_distill

    def compute_loss(self, model, batch, criterion, old_model=None, ewc=None, bias_reg=None):
        images = batch["images"].to(self.device)
        labels = batch["labels"].to(self.device)

        texts = None
        if "texts" in batch and batch["texts"] is not None:
            texts = {k: v.to(self.device) for k, v in batch["texts"].items()}
        cls_loss, logits = base_loss_and_logits(model, images, texts, labels, criterion)
        old_logits = None
        if old_model:
            with torch.no_grad():  # frozen teacher
                old_logits = old_model(images, texts) if texts is not None else old_model(images)

        dist_loss = 0.0
        if old_logits is not None:
            old_num_classes = old_logits.size(1)
            with torch.no_grad():
                soft_targets = torch.softmax(old_logits / self.temperature, dim=1)
            soft_preds = F.log_softmax(logits[:, :old_num_classes] / self.temperature, dim=1)
            dist_loss = -torch.sum(soft_targets * soft_preds) / soft_preds.size(0)

        total_loss = cls_loss + self.lambda_distill * dist_loss

        if ewc:
            total_loss += ewc.penalty(model)
        if bias_reg:
            total_loss += bias_reg(model, batch)

        preds = torch.argmax(logits, dim=1)
        return total_loss, preds, labels


# ----------- EWC for Domain-IL -----------

class EWC:
    """Elastic Weight Consolidation for domain incremental learning."""
    def __init__(self, model: nn.Module, dataloader, device: torch.device, lambda_ewc: float = 5000.0):
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
        samples_count = 0
        for batch in dataloader:
            images = batch["images"].to(self.device)
            labels = batch["labels"].to(self.device)
            if 'texts' in batch and batch['texts'] is not None:
                text_inputs = {k: v.to(self.device) for k, v in batch['texts'].items()}
                logits = self.model(images, text_inputs)
            else:
                logits = self.model(images)
            log_probs = F.log_softmax(logits, dim=1)
            for i in range(len(labels)):
                self.model.zero_grad()
                log_prob = log_probs[i, labels[i]]
                log_prob.backward(retain_graph=(i < len(labels)-1))
                for n, p in self.params.items():
                    if p.grad is not None:
                        fisher[n] += p.grad.data ** 2
                samples_count += 1
        for n in fisher.keys():
            fisher[n] /= max(samples_count, 1)
        self._fisher = fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if any(classifier_name in n for classifier_name in ['classifier', 'image_classifier', 'text_classifier', 'fusion_classifier']):
                continue
            if n in self._means and p.shape == self._means[n].shape:
                loss += (self._fisher[n] * (p - self._means[n]) ** 2).sum()
        # EWC (Kirkpatrick et al., 2017): (lambda / 2) * sum_i F_i (theta_i - theta*_i)^2
        return 0.5 * self.lambda_ewc * loss
    

# ----------- Exemplar Management -----------

class ExemplarManager:
    def __init__(self, max_exemplars=200, max_per_class=20, selection_strategy="random"):
        self.exemplars = {}  # class_idx -> list of exemplar samples (dict with tensors)
        self.max_exemplars = max_exemplars
        self.max_per_class = max_per_class
        self.selection_strategy = selection_strategy.lower()
        assert self.selection_strategy in ["random", "herding"], "Invalid selection_strategy"

    def add_exemplars(self, class_idx, samples):
        # Replace existing exemplars for that class with new selection
        self.exemplars[class_idx] = samples
        self._balance()

    def _balance(self):
        classes = list(self.exemplars.keys())
        if not classes:
            return
        allowed_per = min(self.max_per_class, self.max_exemplars // len(classes))
        for cl in classes:
            if len(self.exemplars[cl]) > allowed_per:
                self.exemplars[cl] = self.exemplars[cl][:allowed_per]

    def get_replay_batch(self, device, batch_size=32):
        all_samples = sum(self.exemplars.values(), [])
        if len(all_samples) == 0:
            return None
        samples = random.sample(all_samples, min(batch_size, len(all_samples)))
        # Exemplars are dataset items: {"image", "text", "label"}
        images = torch.stack([s["image"] for s in samples]).to(device)
        labels = torch.tensor([int(s["label"]) for s in samples], device=device)
        batch = {"images": images, "labels": labels}
        if samples[0].get("text") is not None:
            batch["texts"] = {k: torch.stack([s["text"][k] for s in samples]).to(device) for k in samples[0]["text"]}
        return batch

    def update_exemplars(self, model, dataset, device, max_candidates=1000, seed=0):
        """Select up to max_per_class exemplars per class of `dataset` (e.g. the pretrained domain's train set).
        Herding runs on a random pool of at most `max_candidates` samples per class."""
        base, idx_map = (dataset.dataset, list(dataset.indices)) if isinstance(dataset, torch.utils.data.Subset) \
            else (dataset, list(range(len(dataset))))
        rng = random.Random(seed)
        # Group by the global label the dataset returns (folder index -> class name -> class_to_idx)
        def global_label(folder_idx):
            name = base.classes[folder_idx].replace(' ', '_')
            return base.class_to_idx.get(name, folder_idx) if hasattr(base, "class_to_idx") else folder_idx
        by_class = {}
        for i, j in enumerate(idx_map):
            by_class.setdefault(global_label(int(base.samples[j][1])), []).append(i)

        model.eval()
        for cls, indices in by_class.items():
            K = min(self.max_per_class, len(indices))
            if self.selection_strategy == "random":
                selected = rng.sample(indices, K)
            else:  # herding: greedily match the class mean in feature space
                pool = indices if len(indices) <= max_candidates else rng.sample(indices, max_candidates)
                feats = []
                with torch.no_grad():
                    for i in pool:
                        sample = dataset[i]
                        img = sample["image"].unsqueeze(0).to(device)
                        txt = sample.get("text")
                        txt = {k: v.unsqueeze(0).to(device) for k, v in txt.items()} if txt is not None else None
                        feats.append((model.extract_features(img, txt) if txt is not None else model.extract_features(img)).cpu().squeeze(0))
                feats = torch.stack(feats)
                class_mean = feats.mean(dim=0)
                chosen, running = [], torch.zeros_like(class_mean)
                remaining = list(range(len(pool)))
                for _ in range(K):
                    dists = torch.norm(class_mean - (running + feats[remaining]) / (len(chosen) + 1), dim=1)
                    best = remaining.pop(int(torch.argmin(dists)))
                    chosen.append(best)
                    running = running + feats[best]
                selected = [pool[c] for c in chosen]
            self.exemplars[cls] = [dataset[i] for i in selected]
        self._balance()
        model.train()
        print(f"Replay memory: {sum(len(v) for v in self.exemplars.values())} exemplars of {len(self.exemplars)} classes "
              f"({self.selection_strategy})")


def mix_replay(exemplar_manager, images, labels, texts, device, batch_size=32):
    """Append a replay batch of exemplars (old domain) to the current batch. Returns images, labels, texts on device."""
    images, labels = images.to(device), labels.to(device)
    texts = {k: v.to(device) for k, v in texts.items()} if texts is not None else None
    replay = exemplar_manager.get_replay_batch(device, batch_size) if exemplar_manager is not None else None
    if replay is None:
        return images, labels, texts
    images = torch.cat([images, replay["images"]], dim=0)
    labels = torch.cat([labels, replay["labels"]], dim=0)
    if texts is not None and "texts" in replay:
        texts = {k: torch.cat([texts[k], replay["texts"][k]], dim=0) for k in texts}
    return images, labels, texts


# ----------- Adaptive Learning Rate -----------

class AdaptiveLR:
    """Adaptive LR scheduler for domain incremental learning."""
    def __init__(self, optimizer, base_lr=1e-3, decay_factor=0.7, patience=3, min_lr=1e-6):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.patience = patience
        self.factor = decay_factor
        self.min_lr = min_lr
        self.counter = 0
        self.best = None

    def step(self, metrics):
        acc = metrics['accuracy']
        if self.best is None or acc > self.best:
            self.best = acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                # Decay the optimizer's current lr (base_lr is only a default and may not be the lr in use,
                # e.g. 1e-3 here vs. --lr 1e-4: deriving the new lr from it raised the lr instead of lowering it)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * self.factor, self.min_lr)
                self.base_lr = self.optimizer.param_groups[0]['lr']
                self.counter = 0
                return True
        return False

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


# ----------- Feature Extraction -----------

def extract_features_old(model, dataloader, device):
    model.eval()
    features = {}
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["images"].to(device)
            if "texts" in batch and batch["texts"] is not None:
                text_inputs = {k: v.to(device) for k, v in batch["texts"].items()}
                feats = model.extract_features(imgs, text_inputs)
            else:
                feats = model.extract_features(imgs)
            labels = batch["labels"].cpu().numpy()
            for f, l in zip(feats.cpu().numpy(), labels):
                if l not in features:
                    features[l] = []
                features[l].append(f)
    for l in features:
        features[l] = np.stack(features[l])
    return features

def extract_features(model, dataloader, device):
    model.eval()
    features_list = []
    with torch.no_grad():
        for batch in dataloader:
            if "images" in batch and "texts" in batch:
                images = batch["images"].to(device)
                texts = {k: v.to(device) for k, v in batch["texts"].items()}
                feats = model.extract_features(images, texts)
            elif "images" in batch:
                images = batch["images"].to(device)
                feats = model.extract_features(images)
            else:
                raise KeyError(f"Batch missing 'images' key: keys are {list(batch.keys())}")
            features_list.append(feats.cpu())

    features = torch.cat(features_list, dim=0)
    return features.numpy()

def flatten_feature(feat):
    if isinstance(feat, dict):
        arrays = []
        for v in feat.values():
            if isinstance(v, dict):
                arrays.append(flatten_feature(v))  # recursive flatten
            elif isinstance(v, torch.Tensor):
                arrays.append(v.cpu().numpy().ravel())
            elif isinstance(v, np.ndarray):
                arrays.append(v.ravel())
            else:
                arrays.append(np.array(v).ravel())
        return np.concatenate(arrays)
    elif isinstance(feat, torch.Tensor):
        return feat.cpu().numpy()
    elif isinstance(feat, np.ndarray):
        return feat
    else:
        return np.array(feat)


def move_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    else:
        return data


def extract_features_and_logits(model, dataloader, device):
    model.eval()
    features_list = []
    logits_list = []
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["images"].to(device)
            if "texts" in batch and batch["texts"] is not None:
                text_inputs = {k: v.to(device) for k, v in batch["texts"].items()}
                feats = model.extract_features(imgs, text_inputs)
                logits = model(imgs, text_inputs)
            else:
                feats = model.extract_features(imgs)
                logits = model(imgs)

            features_list.append(feats.cpu())
            logits_list.append(logits.cpu())

    if len(features_list) == 0:
        # No samples => return None for this class
        return None, None

    features_all = torch.cat(features_list, dim=0)
    logits_all = torch.cat(logits_list, dim=0)

    return features_all.numpy(), logits_all.numpy()

def extract_features_by_class(model, dataloader, device):
    model.eval()
    feats_by_class = {}
    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            images = batch["images"].to(device)
            labels = batch["labels"].cpu().numpy()
            if "texts" in batch:
                texts = {k: v.to(device) for k, v in batch["texts"].items()}
                feats = model.extract_features(images, texts)
            else:
                feats = model.extract_features(images)
            feats = feats.cpu().numpy()
            for i, lbl in enumerate(labels):
                if lbl not in feats_by_class:
                    feats_by_class[lbl] = []
                feats_by_class[lbl].append(feats[i])
    # Stack arrays for each class
    for lbl in feats_by_class:
        feats_by_class[lbl] = np.stack(feats_by_class[lbl])
    return feats_by_class
