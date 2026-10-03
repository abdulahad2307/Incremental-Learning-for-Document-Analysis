import json
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DocFormerConfig:
    # Model architecture
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    
    # Visual backbone
    visual_backbone: str = "resnet50"
    visual_feature_dim: int = 2048
    visual_output_dim: int = 768
    
    # Training
    learning_rate: float = 2.5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    batch_size: int = 8
    num_train_epochs: int = 5
    
    # Pre-training tasks
    mm_mlm_probability: float = 0.15
    ltr_weight: float = 1.0
    tdi_weight: float = 5.0
    mm_mlm_weight: float = 5.0
    
    @classmethod
    def from_json(cls, json_path):
        with open(json_path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
