import torch
import torch.nn as nn
from utils.eaml.image_encoder import ImageEncoder
from utils.eaml.text_encoder import TextEncoder
from utils.eaml.fusion_module import FusionModule
from utils.eaml.mutual_learning import TruncatedKLDLoss

class EAMLModel(nn.Module):
    def __init__(self, num_classes=16, embed_dim=512):
        """
        Implementing the Ensemble Self-Attention-based Mutual Learning Network.
        
        Parameters:
            num_classes (int): Number of output classes.
            embed_dim (int): Feature dimension.
        """
        super(EAMLModel, self).__init__()
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.fusion_module = FusionModule(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.tr_kld_loss = TruncatedKLDLoss()

    def forward(self, image, text):
        image_feat = self.image_encoder(image)
        #text_feat = self.text_encoder(text)
        text = {key: val.to(next(self.text_encoder.parameters()).device) for key, val in text.items()}  # Ensure device consistency
        text_feat = self.text_encoder(text)


        fused_feat = self.fusion_module(image_feat, text_feat)
        return self.classifier(fused_feat)
