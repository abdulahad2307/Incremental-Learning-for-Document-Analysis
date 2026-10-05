import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

from utils.eaml.mutual_learning import MutualLearningLoss

# EAML base-model loss (EAML paper: image/text/fusion CE + truncated-KL mutual learning, beta = 0.5); the IL
# strategies add their own terms (EWC, KD) on top of it, as for LayoutLMv3 (whose base loss is the CE)
EAML_BASE_LOSS = MutualLearningLoss(kld_weight=0.5)

class IncrementalStrategy:
    """Base class for incremental learning strategies"""
    def __init__(self, device):
        self.device = device

    def adapt_model(self, model, old_num_classes, new_num_classes, model_name):
        """Adapt model for new classes"""
        raise NotImplementedError

    def compute_loss(self, model, batch, criterion, old_model=None, ewc=None):
        """Compute loss with strategy-specific components"""
        raise NotImplementedError

class StandardIncremental(IncrementalStrategy):
    """Standard incremental learning without forgetting mitigation"""
    def adapt_model(self, model, old_num_classes, new_num_classes, model_name):
        """Adapt model architecture for new classes"""
        if model_name in ["docformer", "layoutlmv3"] and hasattr(model, 'classifier'):
            # Save old classifier weights
            old_classifier = model.classifier.weight.data.clone()
            old_bias = model.classifier.bias.data.clone() if model.classifier.bias is not None else None
            
            # Initialize new classifier
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, new_num_classes)
            
            # Copying old weights
            with torch.no_grad():
                model.classifier.weight.data[:old_num_classes] = old_classifier
                if old_bias is not None and model.classifier.bias is not None:
                    model.classifier.bias.data[:old_num_classes] = old_bias
                    
        elif model_name == "eaml":
            # Saving old classifier weights
            old_image_classifier = model.image_classifier.weight.data.clone()
            old_image_bias = model.image_classifier.bias.data.clone()
            old_text_classifier = model.text_classifier.weight.data.clone()
            old_text_bias = model.text_classifier.bias.data.clone()
            old_fusion_classifier = model.fusion_classifier.weight.data.clone()
            old_fusion_bias = model.fusion_classifier.bias.data.clone()
            
            # Initializing new classifiers
            model.image_classifier = nn.Linear(model.image_classifier.in_features, new_num_classes)
            model.text_classifier = nn.Linear(model.text_classifier.in_features, new_num_classes)
            model.fusion_classifier = nn.Linear(model.fusion_classifier.in_features, new_num_classes)
            
            # Copy overlapping weights safely
            with torch.no_grad():
                num_copy_img = min(old_num_classes, old_image_classifier.size(0), model.image_classifier.weight.data.size(0))
                model.image_classifier.weight.data[:num_copy_img] = old_image_classifier[:num_copy_img]
                model.image_classifier.bias.data[:num_copy_img] = old_image_bias[:num_copy_img]

                num_copy_txt = min(old_num_classes, old_text_classifier.size(0), model.text_classifier.weight.data.size(0))
                model.text_classifier.weight.data[:num_copy_txt] = old_text_classifier[:num_copy_txt]
                model.text_classifier.bias.data[:num_copy_txt] = old_text_bias[:num_copy_txt]

                num_copy_fus = min(old_num_classes, old_fusion_classifier.size(0), model.fusion_classifier.weight.data.size(0))
                model.fusion_classifier.weight.data[:num_copy_fus] = old_fusion_classifier[:num_copy_fus]
                model.fusion_classifier.bias.data[:num_copy_fus] = old_fusion_bias[:num_copy_fus]

                    
        return model

    def compute_loss(self, model, batch, criterion, old_model=None, ewc=None):
        if "images" in batch:  # EAML
            images = batch['images'].to(self.device)
            texts = batch['texts']
            # Moving text tensors to device
            texts = {k: v.to(self.device) for k, v in texts.items()}
            labels = batch['labels'].to(self.device)
            
            # Forward pass (all three EAML heads, for the EAML base loss)
            outputs = model(images=images, texts=texts, return_features=True)
            logits = outputs['fusion_logits']
            loss = EAML_BASE_LOSS(outputs, labels)['total_loss']
        else:  # DocFormer
            inputs = {
                'pixel_values': batch['pixel_values'].to(self.device),
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'bbox': batch['bbox'].to(self.device)
            }
            labels = batch['labels'].to(self.device)
            outputs = model(**inputs, task="classification")
            logits = outputs['logits']
            # Classification loss
            loss = criterion(logits, labels)
        
        # Adding EWC regularization if available
        if ewc is not None:
            ewc_loss = ewc.penalty(model)
            loss += ewc_loss
            
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

class DistillationIncremental(IncrementalStrategy):
    """Incremental learning with knowledge distillation"""
    def __init__(self, device, temperature=2.0, lambda_distill=1.0):
        super().__init__(device)
        self.temperature = temperature
        self.lambda_distill = lambda_distill

    def adapt_model(self, model, old_num_classes, new_num_classes, model_name):
        """Adapt model architecture for new classes"""
        return StandardIncremental(self.device).adapt_model(model, old_num_classes, new_num_classes, model_name)

    def compute_loss(self, model, batch, criterion, old_model=None, ewc=None):
        """Compute loss with distillation component"""
        if "images" in batch:  # EAML
            images = batch['images'].to(self.device)
            texts = batch['texts']
            # Moving text tensors to device
            texts = {k: v.to(self.device) for k, v in texts.items()}
            labels = batch['labels'].to(self.device)
            
            # Forward pass (all three EAML heads, for the EAML base loss); KD uses the fusion logits
            outputs = model(images=images, texts=texts, return_features=True)
            logits = outputs['fusion_logits']
            cls_loss = EAML_BASE_LOSS(outputs, labels)['total_loss']
            
            # Distillation loss if we have an old model
            if old_model is not None:
                with torch.no_grad():
                    old_outputs = old_model(images=images, texts=texts)
                    old_logits = old_outputs
                old_class_count = old_logits.size(1)
                #print(f"Student logits shape: {logits.shape}")
                #print(f"Teacher logits shape: {old_logits.shape}")
                #print(f"Old class count: {old_class_count}")
                    
                # Only applying distillation to old classes
                old_class_count = old_logits.size(1)
                
                # Getting soft targets from old model
                soft_targets = nn.functional.softmax(old_logits / self.temperature, dim=1)
                
                # Getting soft probabilities from current model (only for old classes)
                soft_probs = nn.functional.log_softmax(logits[:, :old_class_count] / self.temperature, dim=1)
                
                # Calculating distillation loss
                dist_loss = -torch.sum(soft_targets * soft_probs) / soft_probs.size(0)
                
                # Combined loss
                loss = cls_loss + self.lambda_distill * dist_loss
            else:
                loss = cls_loss

        else:  # DocFormer
            inputs = {
                'pixel_values': batch['pixel_values'].to(self.device),
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'bbox': batch['bbox'].to(self.device)
            }
            labels = batch['labels'].to(self.device)
            #outputs = model(**inputs, task="classification")
            outputs = model(**inputs)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            # Classification loss
            cls_loss = criterion(logits, labels)
            
            # Distillation loss if we have an old model
            if old_model is not None:
                with torch.no_grad():
                    old_outputs = old_model(**inputs, task="classification")
                    old_logits = old_outputs['logits']
                    
                # Only applying distillation to old classes
                old_class_count = old_logits.size(1)
                
                # Getting soft targets from old model
                soft_targets = nn.functional.softmax(old_logits / self.temperature, dim=1)
                
                # Getting soft probabilities from current model (only for old classes)
                soft_probs = nn.functional.log_softmax(logits[:, :old_class_count] / self.temperature, dim=1)
                
                # Calculating distillation loss
                dist_loss = -torch.sum(soft_targets * soft_probs) / soft_probs.size(0)
                
                # Combined loss
                loss = cls_loss + self.lambda_distill * dist_loss
            else:
                loss = cls_loss
        
        # Adding EWC regularization if available
        if ewc is not None:
            ewc_loss = ewc.penalty(model)
            loss += ewc_loss
            
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

class EWC:
    """Elastic Weight Consolidation for preventing catastrophic forgetting"""
    def __init__(self, model: nn.Module, dataloader, device: torch.device, lambda_ewc: float = 5000.0):
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {}  # Stores parameter values
        self._fisher = {}  # Stores Fisher information matrix diagonals
        
        # Computing Fisher information matrix
        self._compute_fisher(dataloader)
        
        # Storing current parameter values
        for n, p in self.params.items():
            self._means[n] = p.data.clone()
    
    def _compute_fisher(self, dataloader):
        """Compute Fisher Information Matrix for parameters"""
        # Initializing Fisher information for each parameter
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        
        # Setting model to evaluation mode
        self.model.train()
        
        # Accumulating Fisher information
        samples_count = 0
        for batch in dataloader:
            samples_count += len(batch['labels'])
            
            # Forward pass
            if "images" in batch:  # EAML
                images = batch['images'].to(self.device)
                texts = {k: v.to(self.device) for k, v in batch['texts'].items()}
                labels = batch['labels'].to(self.device)
                outputs = self.model(images=images, texts=texts)
                logits = outputs
            else:  # DocFormer #LayouLMv3
                inputs = {
                    'pixel_values': batch['pixel_values'].to(self.device),
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'bbox': batch['bbox'].to(self.device)
                }
                labels = batch['labels'].to(self.device)
                outputs = self.model(**inputs, task="classification")
                logits = outputs['logits']
                
            #  log probabilities
            log_probs = F.log_softmax(logits, dim=1)
            
            #  gradients calculations
            for i in range(len(labels)):
                self.model.zero_grad()
                # Selecting the log probability of the target class
                log_prob = log_probs[i, labels[i]]
                log_prob.backward(retain_graph=(i < len(labels) - 1))
                
                # Accumulating Fisher information
                for n, p in self.params.items():
                    if p.grad is not None:
                        fisher[n] += p.grad.data ** 2
        
        # Normalizing by number of samples
        for n in fisher.keys():
            fisher[n] /= samples_count
            
        self._fisher = fisher
    """   
    """
        
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute EWC penalty for current model parameters, excluding classifier layers"""
        loss = 0
        for n, p in model.named_parameters():
            # Skipping classifier layers that change size with new classes
            if any(classifier_name in n for classifier_name in 
                ['classifier', 'image_classifier', 'text_classifier', 'fusion_classifier']):
                continue
                
            if n in self._means and p.size() == self._means[n].size():
                # APplying penalty only if parameter exists and sizes match
                loss += (self._fisher[n] * (p - self._means[n]) ** 2).sum()
        # EWC (Kirkpatrick et al., 2017): (lambda / 2) * sum_i F_i (theta_i - theta*_i)^2
        return 0.5 * self.lambda_ewc * loss


    
    def update(self, new_model: nn.Module, dataloader):
        """Update EWC with a new model and dataset"""
        old_means = self._means.copy()
        old_fisher = self._fisher.copy()
        
        # Computing new Fisher information
        self.model = new_model
        self.params = {n: p for n, p in new_model.named_parameters() if p.requires_grad}
        self._compute_fisher(dataloader)
        
        # Storing new parameter values
        for n, p in self.params.items():
            self._means[n] = p.data.clone()
            
        # Merging old and new Fisher information (with equal weighting)
        for n in self._fisher.keys():
            if n in old_fisher:
                self._fisher[n] = (self._fisher[n] + old_fisher[n]) / 2

class ExemplarManager:
    """Manages exemplars for replay-based class incremental learning"""
    def __init__(self, 
                 max_exemplars=200, 
                 max_per_class=20, 
                 selection_strategy="herding",
                 max_candidates=1000):
        self.exemplars = {}
        self.max_candidates = max_candidates  # herding runs on a random pool of this many samples per class
        self.max_exemplars = max_exemplars
        self.max_per_class = max_per_class
        self.selection_strategy = selection_strategy
        self.feature_extractor = None
        
    def set_feature_extractor(self, model):
        """Set the feature extractor model for herding selection"""
        self.feature_extractor = model
        
    def update(self, dataset, class_name, model=None):
        """Update exemplar set with examples from a new class"""
        if self.selection_strategy == "herding" and model is not None:
            self.exemplars[class_name] = self._select_herding(dataset, class_name, model)
        else:
            self.exemplars[class_name] = self._select_random(dataset, class_name)
            
        # Reducing exemplar set if needed
        self._balance_exemplar_set()
        
    def _select_random(self, dataset, class_name):
        """Select random exemplars for a class"""
        import random
        class_samples = [s for s in dataset.samples if s[2] == class_name]
        if len(class_samples) <= self.max_per_class:
            return class_samples
        return random.sample(class_samples, self.max_per_class)
    
    def _select_herding(self, dataset, class_name, model):
        import warnings
        from PIL import Image

        class_samples = [s for s in dataset.samples if s[2] == class_name]
        if len(class_samples) == 0:
            warnings.warn(f"No samples found for class {class_name}. Returning empty exemplar set.")
            return []
        if len(class_samples) <= self.max_per_class:
            return class_samples
        if self.max_candidates and len(class_samples) > self.max_candidates:
            class_samples = random.Random(0).sample(class_samples, self.max_candidates)

        features = []
        valid_labels = []
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            for idx, sample in enumerate(class_samples):
                try:
                    img_path, tokens, _ = sample
                    if tokens is None or not isinstance(tokens, dict):
                        warnings.warn(f"Skipping sample with missing tokens at idx {idx}: {img_path}")
                        continue
                    image = dataset.transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                    if hasattr(model, 'extract_features'):
                        feature = model.extract_features(
                            images=image,
                            texts={
                                'input_ids': tokens['input_ids'].unsqueeze(0).to(device),
                                'attention_mask': tokens['attention_mask'].unsqueeze(0).to(device)
                            }
                        )
                    else:
                        feature = model(image)
                    features.append(feature.cpu().numpy().flatten())
                    valid_labels.append(sample)
                except Exception as e:
                    warnings.warn(f"Skipping sample at idx {idx} ({img_path}): {e}")
                    continue

        features = np.array(features)
        if len(features) == 0:
            warnings.warn(f"No features extracted for class {class_name}. Returning empty exemplars.")
            return []

        class_mean = np.mean(features, axis=0)
        selected_indices = []
        selected_features = []

        pick_n = min(self.max_per_class, len(features))

        for _ in range(pick_n):
            candidate_indices = [i for i in range(len(features)) if i not in selected_indices]
            if not candidate_indices:
                break
            if len(selected_features) == 0:
                distances = np.linalg.norm(features[candidate_indices] - class_mean, axis=1)
                chosen_idx_in_candidates = np.argmin(distances)
                idx = candidate_indices[chosen_idx_in_candidates]
            else:
                current_mean = np.mean(np.array(selected_features), axis=0)
                candidate_means = np.array([
                    (current_mean * len(selected_features) + features[i]) / (len(selected_features) + 1)
                    for i in candidate_indices
                ])
                distances = np.linalg.norm(candidate_means - class_mean, axis=1)
                chosen_idx_in_candidates = np.argmin(distances)
                idx = candidate_indices[chosen_idx_in_candidates]

            selected_indices.append(idx)
            selected_features.append(features[idx])

        return [valid_labels[i] for i in selected_indices]

    
    def _balance_exemplar_set(self):
        """Balance exemplar set to ensure fair representation of all classes"""
        total_exemplars = sum(len(exems) for exems in self.exemplars.values())
        if total_exemplars <= self.max_exemplars:
            return
            
        # Reduce exemplars per class evenly
        target_per_class = self.max_exemplars // len(self.exemplars)
        remainder = self.max_exemplars % len(self.exemplars)
        for i, class_name in enumerate(self.exemplars.keys()):
            target = target_per_class + (1 if i < remainder else 0)
            if len(self.exemplars[class_name]) > target:
                self.exemplars[class_name] = self.exemplars[class_name][:target]
                
    def get_exemplar_dataset(self, transform=None):
        """Convert exemplars to a dataset-like format"""
        all_exemplars = []
        for class_name, examples in self.exemplars.items():
            all_exemplars.extend(examples)
        return all_exemplars

def fill_exemplar_memory(exemplar_mgr, dataset, classes, model_path, num_classes, device):
    """Select exemplars for classes that have none yet (the base classes, or all previous classes when a new job
    starts), using the previous step's model at `model_path` for herding."""
    missing = [c for c in classes if c not in exemplar_mgr.exemplars]
    if not missing:
        return
    model = None
    if exemplar_mgr.selection_strategy == "herding" and model_path and os.path.exists(model_path):
        from utils.eaml.eaml_model import EAMLModel
        model = EAMLModel(num_classes=num_classes).to(device)
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        model.eval()
        del ckpt
    print(f"Selecting exemplars ({exemplar_mgr.selection_strategy}) for {len(missing)} previous classes: {missing}")
    for c in missing:
        exemplar_mgr.update(dataset, c, model)
    del model
    torch.cuda.empty_cache()


class AdaptiveLR:
    """Adaptive learning rate scheduler for class incremental learning"""
    def __init__(self, optimizer, base_lr: float = 0.001, min_lr: float = 1e-6,
                 decay_factor: float = 0.75, patience: int = 3):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.best_acc = 0
        self.no_improvement_count = 0
        
        # Setting initial learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr
            
    def step(self, val_acc: float) -> bool:
        """Update learning rate based on validation accuracy"""
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.no_improvement_count = 0
            return False
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                # Reducing learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * self.decay_factor, self.min_lr)
                self.no_improvement_count = 0
                return True
        return False
    
    def get_lr(self) -> float:
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']
    
    def reset(self, task_complexity: float = 1.0):
        """Reset scheduler for a new task with optional complexity adjustment"""
        self.best_acc = 0
        self.no_improvement_count = 0
        
        # Adjusting learning rate based on task complexity
        adjusted_lr = self.base_lr / (1 + task_complexity)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = adjusted_lr

def extract_features(model, dataloader, device, max_samples_per_class=None):
    """
    Extract features for EVM classifier in a model-agnostic way, with optional per-class sample limit.
    Returns: Dict[class_name, np.ndarray of features]
    """
    from collections import defaultdict
    import random

    model.eval()
    features = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            if "images" in batch:  # EAML
                images = batch['images'].to(device)
                texts = {k: v.to(device) for k, v in batch['texts'].items()}
                labels = batch['labels'].to(device)

                # model.extract_features if available
                if hasattr(model, 'extract_features'):
                    batch_features = model.extract_features(images=images, texts=texts)
                else:
                    outputs = model(images=images, texts=texts, return_features=True)
                    batch_features = outputs['fused_feat']

            else:  # DocFormer
                inputs = {
                    'pixel_values': batch['pixel_values'].to(device),
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'bbox': batch['bbox'].to(device)
                }
                labels = batch['labels'].to(device)
                outputs = model(**inputs, task="classification")
                batch_features = outputs['features'] if 'features' in outputs else outputs['logits']

            # Group features by class
            for i, label in enumerate(labels.cpu().numpy()):
                class_name = dataloader.dataset.current_classes[label]
                features[class_name].append(batch_features[i].cpu().numpy())

    # If max_samples_per_class is set, randomly sample up to that many per samples class
    for class_name in features:
        if max_samples_per_class is not None and len(features[class_name]) > max_samples_per_class:
            features[class_name] = random.sample(features[class_name], max_samples_per_class)
        features[class_name] = np.vstack(features[class_name])

    return features

#---- OOD Feature Extraction ----
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
            if isinstance(batch, dict):
                inputs = {k: move_to_device(v, device) for k, v in batch.items() if k != 'labels'}
                # Extract features
                if hasattr(model, 'extract_features'):
                    feats = model.extract_features(**inputs)
                else:
                    outs = model(**inputs)
                    feats = outs.hidden_states[-1].mean(dim=1) if hasattr(outs, 'hidden_states') else outs
                # Get logits
                logits = model(**inputs)
                if hasattr(logits, 'logits'):
                    logits = logits.logits
            else:
                inputs = batch[0].to(device)
                if hasattr(model, 'extract_features'):
                    feats = model.extract_features(inputs)
                else:
                    outs = model(inputs)
                    feats = outs.hidden_states[-1].mean(dim=1) if hasattr(outs, 'hidden_states') else outs
                logits = model(inputs)
                if hasattr(logits, 'logits'):
                    logits = logits.logits

            features_list.append(feats.cpu())
            logits_list.append(logits.cpu())
    features_all = torch.cat(features_list, dim=0)
    logits_all = torch.cat(logits_list, dim=0)
    return features_all, logits_all


def extract_feature_vectors(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for batch in dataloader:
            if "images" in batch:
                images = batch["images"].to(device)
                input_ids = batch["texts"]["input_ids"].to(device)
                attention_mask = batch["texts"]["attention_mask"].to(device)
                feat = model.extract_features(images=images, input_ids=input_ids, attention_mask=attention_mask)
            elif "pixel_values" in batch:
                inputs = {
                    "pixel_values": batch["pixel_values"].to(device),
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "bbox": batch["bbox"].to(device)
                }
                feat = model.extract_features(**inputs)
            else:
                raise KeyError("Batch missing both 'images' and 'pixel_values' keys: batch keys are {batch.keys()}")
            features.append(feat.cpu())
    features = torch.cat(features, dim=0)
    return features.numpy()


def extract_feature_vectors2(model, dataloader, device,class_names):
    model.eval()
    feature_dict = {}

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            #print(f"Processing batch {i+1}/{len(dataloader)}")
            if "images" in batch:
                images = batch["images"].to(device)
                input_ids = batch["texts"]["input_ids"].to(device)
                attention_mask = batch["texts"]["attention_mask"].to(device)
                feat = model.extract_features(images=images, input_ids=input_ids, attention_mask=attention_mask)
            elif "pixel_values" in batch:
                inputs = {
                    "pixel_values": batch["pixel_values"].to(device),
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                    "bbox": batch["bbox"].to(device)
                }
                feat = model.extract_features(**inputs)
            else:
                raise KeyError(f"Batch missing both 'images' and 'pixel_values' keys: batch keys are {batch.keys()}")
            
            labels = batch['labels']
            # Convert label indices or strings to string keys
            labels_str = []
            for l in labels:
                if isinstance(l, torch.Tensor):
                    l = l.item()
                labels_str.append(class_names[l] if isinstance(class_names[0], str) else l)
            
            feat = feat.cpu().numpy()

            for f_vec, lbl in zip(feat, labels_str):
                if lbl not in feature_dict:
                    feature_dict[lbl] = []
                feature_dict[lbl].append(f_vec)

    # Stack each list of features into numpy arrays
    for key in feature_dict:
        feature_dict[key] = np.stack(feature_dict[key], axis=0)

    return feature_dict

