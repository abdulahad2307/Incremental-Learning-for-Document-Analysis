import torch
import torch.nn as nn
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder
from .fusion_module import ElementwiseSumFusion
from .attention_block import EAMLAttentionBlock

class EAMLModel(nn.Module):
    def __init__(self, num_classes=16, embed_dim=512, dropout_rate=0.2, freeze_image_encoder=False):
        super().__init__()
        # Encoders
        self.image_encoder = ImageEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        
        # Joint image-text attention inside the branches (after Inception "Residual block 0" / BERT block 0)
        self.attention_block = EAMLAttentionBlock(img_channels=320, txt_dim=768, d=256)

        # Fusion as in the EAML paper (Eqs. 8-9): element-wise sum of the image and text features
        self.fusion_module = ElementwiseSumFusion()
        
        # Separate classifiers for mutual learning
        self.image_classifier = nn.Linear(embed_dim, num_classes)
        self.text_classifier = nn.Linear(embed_dim, num_classes)
        self.fusion_classifier = nn.Linear(embed_dim, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize OCR model
        #self.ocr_processor = None
        #self.ocr_model = None

    """
    def forward(self, images, texts, return_features=False):
        # Extract features
        image_feat = self.image_encoder(images)
        text_feat = self.text_encoder(texts)
        
        # Apply dropout for regularization
        image_feat = self.dropout(image_feat)
        text_feat = self.dropout(text_feat)
        
        # Fuse features
        fused_feat = self.fusion_module(image_feat, text_feat)
        
        # Get predictions from each branch
        image_logits = self.image_classifier(image_feat)
        text_logits = self.text_classifier(text_feat)
        fusion_logits = self.fusion_classifier(fused_feat)
        
        if return_features:
            return {
                'image_logits': image_logits,
                'text_logits': text_logits,
                'fusion_logits': fusion_logits,
                'image_feat': image_feat,
                'text_feat': text_feat,
                'fused_feat': fused_feat
            }
        
        # During inference, return only fusion logits
        return fusion_logits

        """
    
    def forward(self, images=None, texts=None, input_ids=None, attention_mask=None, return_features=False, **kwargs):
        """
        Forward pass for the EAML model with flexible parameter handling.
        
        Args:
            images: Tensor of document images [batch_size, channels, height, width]
            texts: Dictionary or tensor containing text features
            input_ids: Text token IDs (used if texts is None)
            attention_mask: Attention mask for text tokens (used if texts is None)
            return_features: Whether to return intermediate features
            **kwargs: Additional arguments (ignored)
        
        Returns:
            logits or dictionary of features
        """

        # Input validation
        if images is None:
            raise ValueError("Images input cannot be None")
        # Handle different input formats for text features
        if texts is None and input_ids is not None:
            # If input_ids is provided directly
            text_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask if attention_mask is not None else torch.ones_like(input_ids)
            }
        elif texts is not None:
            # If texts is provided (either as dict or tensor)
            text_inputs = texts
        else:
            raise ValueError("Either 'texts' or 'input_ids' must be provided")
        
        # Extract features
        image_feat, text_feat = self._encode(images, text_inputs)
        
        # Apply dropout for regularization
        image_feat = self.dropout(image_feat)
        text_feat = self.dropout(text_feat)

        #Normalization
        image_feat = F.normalize(image_feat, p=2, dim=-1)
        text_feat = F.normalize(text_feat, p=2, dim=-1)
        #print("Features Shape:, image_feat.shape, text_feat.shape)

        # Stack along modality dimension for multi-head attention
        #fusion_input = torch.stack([image_feat, text_feat], dim=1)  
        # shape: [batch_size, 2, embed_dim]
        #fusion_module = EnhancedFusionModule(embed_dim=512, num_heads=8)
        
        # Fuse features
        fused_feat = self.fusion_module(image_feat, text_feat)
        #print("Fusion output shape:", fused_feat.shape)

        # Get predictions from each branch
        image_logits = self.image_classifier(image_feat)
        text_logits = self.text_classifier(text_feat)
        fusion_logits = self.fusion_classifier(fused_feat)
        
        if return_features:
            return {
                'image_logits': image_logits,
                'text_logits': text_logits,
                'fusion_logits': fusion_logits,
                'image_feat': image_feat,
                'text_feat': text_feat,
                'fused_feat': fused_feat
            }
        
        # During inference, return only fusion logits
        return fusion_logits

    def _encode(self, images, text_inputs):
        """Both branches with the joint attention block between their first and remaining stages."""
        image_map = self.image_encoder.stem(images)
        text_seq, ext_mask, text_mask = self.text_encoder.stem(text_inputs)
        image_map, text_seq = self.attention_block(image_map, text_seq, text_mask)
        return self.image_encoder.rest(image_map), self.text_encoder.rest(text_seq, ext_mask)

    def extract_features(self, images, texts, input_ids=None, attention_mask=None):
        """
        Extract fused features - used during exemplar herding.
        Args same as forward.
        Returns:
            fused features tensor
        """
        if images is None:
            raise ValueError("Images input cannot be None")

        if texts is None and input_ids is not None:
            text_inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask if attention_mask is not None else torch.ones_like(input_ids)
            }
        elif texts is not None:
            text_inputs = texts
        else:
            raise ValueError("Either 'texts' or 'input_ids' must be provided")

        image_feat, text_feat = self._encode(images, text_inputs)
        image_feat = F.normalize(image_feat, p=2, dim=-1)
        text_feat = F.normalize(text_feat, p=2, dim=-1)
        fused_feat = self.fusion_module(image_feat, text_feat)
        return fused_feat
