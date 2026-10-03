import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

class IncrementalStrategy:
    """Base class for incremental learning strategies"""
    def __init__(self, device):
        self.device = device
        
    def adapt_model(self, model, old_classes, new_classes):
        """Adapt model for new classes"""
        raise NotImplementedError
        
    def compute_loss(self, model, batch, criterion, old_model=None):
        """Compute loss with strategy-specific components"""
        raise NotImplementedError

class StandardIncremental(IncrementalStrategy):
    """Standard incremental learning without forgetting mitigation"""
    def adapt_model(self, model, old_classes, new_classes):
        # No special adaptation needed
        return model
        
    def compute_loss(self, model, batch, criterion, old_model=None):
        # Handle both model types
        if "images" in batch:  # EAML
            images = batch['images'].to(self.device)
            text_input_ids = batch['texts']['input_ids'].to(self.device)
            text_attention_mask = batch['texts']['attention_mask'].to(self.device)
            
            outputs = model(
                images=images,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask
            )
            
            labels = batch['labels'].to(self.device)
            logits = outputs
        else:  # DocFormer
            inputs = {
                'pixel_values': batch['pixel_values'].to(self.device),
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'bboxes': batch['bboxes'].to(self.device)
            }
            labels = batch['labels'].to(self.device)
            outputs = model(**inputs, task="classification")
            logits = outputs['logits']
            
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, labels

class RotationAugmentedDistillation(IncrementalStrategy):
    """Rotation Augmented Distillation (RAD) for exemplar-free class incremental learning"""
    def __init__(self, device, temperature=2.0, lambda_distill=1.0):
        super().__init__(device)
        self.temperature = temperature
        self.lambda_distill = lambda_distill
        
    def adapt_model(self, model, old_classes, new_classes):
        """Adapt model architecture for new classes"""
        if hasattr(model, 'classifier'):
            # Save old classifier weights
            old_classifier = model.classifier.weight.data.clone()
            old_bias = model.classifier.bias.data.clone() if model.classifier.bias is not None else None
            
            # Create new classifier with more classes
            old_num_classes = len(old_classes)
            new_num_classes = len(old_classes) + len(new_classes)
            
            # Initialize new classifier
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, new_num_classes)
            
            # Copy old weights
            with torch.no_grad():
                model.classifier.weight.data[:old_num_classes] = old_classifier
                if old_bias is not None and model.classifier.bias is not None:
                    model.classifier.bias.data[:old_num_classes] = old_bias
        
        return model
    
    def _apply_rotation(self, images, angle):
        """Apply rotation to images"""
        if angle == 90:
            return torch.rot90(images, 1, [2, 3])
        elif angle == 180:
            return torch.rot90(images, 2, [2, 3])
        elif angle == 270:
            return torch.rot90(images, 3, [2, 3])
        return images
        
    def compute_loss(self, model, batch, criterion, old_model=None):
        """Compute loss with distillation but without rotation augmentation"""
        # Handle both model types
        if "images" in batch:  # EAML
            images = batch['images'].to(self.device)
            text_input_ids = batch['texts']['input_ids'].to(self.device)
            text_attention_mask = batch['texts']['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Original data forward pass
            """
            outputs = model(
                images=images,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask
            )
            """

            outputs = model(
                images=images,
                texts={
                    'input_ids': text_input_ids,
                    'attention_mask': text_attention_mask
                }
            )


            logits = outputs
            
            # Classification loss
            cls_loss = criterion(logits, labels)
            
            # Distillation loss if we have an old model
            if old_model is not None:
                with torch.no_grad():
                    old_outputs = old_model(
                        images=images,
                        text_input_ids=text_input_ids,
                        text_attention_mask=text_attention_mask
                    )
                
                # Feature distillation
                if isinstance(outputs, dict) and 'features' in outputs:
                    current_features = outputs['features']
                    old_features = old_outputs['features']
                    dist_loss = F.mse_loss(current_features, old_features)
                else:
                    # If features not available, use logit distillation
                    old_logits = old_outputs
                    soft_targets = F.softmax(old_logits / self.temperature, dim=1)
                    soft_probs = F.log_softmax(logits[:, :old_logits.size(1)] / self.temperature, dim=1)
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
                'bboxes': batch['bboxes'].to(self.device)
            }
            labels = batch['labels'].to(self.device)
            
            # Original data forward pass
            outputs = model(**inputs, task="classification")
            logits = outputs['logits']
            
            # Classification loss
            cls_loss = criterion(logits, labels)
            
            # Distillation loss if we have an old model
            if old_model is not None:
                with torch.no_grad():
                    old_outputs = old_model(**inputs, task="classification")
                
                # Feature distillation
                if 'features' in outputs and 'features' in old_outputs:
                    current_features = outputs['features']
                    old_features = old_outputs['features']
                    dist_loss = F.mse_loss(current_features, old_features)
                else:
                    # If features not available, use logit distillation
                    old_logits = old_outputs['logits']
                    soft_targets = F.softmax(old_logits / self.temperature, dim=1)
                    soft_probs = F.log_softmax(logits[:, :old_logits.size(1)] / self.temperature, dim=1)
                    dist_loss = -torch.sum(soft_targets * soft_probs) / soft_probs.size(0)
                
                # Combined loss
                loss = cls_loss + self.lambda_distill * dist_loss
            else:
                loss = cls_loss
        
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, labels
