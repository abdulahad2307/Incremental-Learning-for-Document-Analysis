import torch
import torch.nn as nn
from utils.training_scope import set_training_scope
from typing import List, Dict, Optional

class TrainingMode:
    """Base class for different training modes"""
    def __init__(self, model: nn.Module):
        self.model = model
    
    def prepare_for_training(self):
        """Prepare model for training"""
        raise NotImplementedError
    
    def get_trainable_params(self):
        """Get parameters that should be trained"""
        raise NotImplementedError

class FullModelTraining(TrainingMode):
    """Train the entire model"""
    def prepare_for_training(self):
        for param in self.model.parameters():
            param.requires_grad = True
        return self.model
    
    def get_trainable_params(self):
        return self.model.parameters()

class LastLayerTraining(TrainingMode):
    """Train only the classifier layers (training_mode classifier_only)"""
    def prepare_for_training(self):
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze classifier layers
        if hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'image_classifier'):
            # For EAML model
            for param in self.model.image_classifier.parameters():
                param.requires_grad = True
            for param in self.model.text_classifier.parameters():
                param.requires_grad = True
            for param in self.model.fusion_classifier.parameters():
                param.requires_grad = True
        
        return self.model
    
    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

class SelectiveLayerTraining(TrainingMode):
    """Train selected layers of the model"""
    def __init__(self, model: nn.Module, trainable_layers: List[str]):
        super().__init__(model)
        self.trainable_layers = trainable_layers
    
    def prepare_for_training(self):
        # Freeze all parameters
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            
            # Unfreeze parameters in trainable layers
            for layer_name in self.trainable_layers:
                if layer_name in name:
                    param.requires_grad = True
        
        return self.model
    
    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

class ScopeTraining(TrainingMode):
    """Train the parameters selected by utils.training_scope (e.g. last feature layer + classifier heads)"""
    def __init__(self, model: nn.Module, scope: str):
        super().__init__(model)
        self.scope = scope

    def prepare_for_training(self):
        return set_training_scope(self.model, self.scope)

    def get_trainable_params(self):
        return filter(lambda p: p.requires_grad, self.model.parameters())

def get_training_mode(model: nn.Module, mode: str, trainable_layers: Optional[List[str]] = None):
    """Factory function to get the appropriate training mode.
    classifier_only: heads only; last_layer: last feature layer (fusion_module) + heads; full / full_model: everything"""
    if mode in ("full", "full_model"):
        return FullModelTraining(model)
    elif mode == "classifier_only":
        return LastLayerTraining(model)
    elif mode == "last_layer":
        return ScopeTraining(model, "last_layer")
    elif mode == "selective":
        if trainable_layers is None:
            raise ValueError("trainable_layers must be provided for selective training mode")
        return SelectiveLayerTraining(model, trainable_layers)
    else:
        raise ValueError(f"Unknown training mode: {mode}")
