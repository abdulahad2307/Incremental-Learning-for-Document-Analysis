import torch
import torch.nn as nn
from transformers import BertModel
import timm

class VisionEncoder(nn.Module):
    def __init__(self, vision_model_name="vit_base_patch16_224"): #vit_tiny_patch16_224, vit_base_patch16_224
        super().__init__()
        self.vit = timm.create_model(vision_model_name, pretrained=True)
        self.hidden_size = self.vit.embed_dim
        self.patch_embed = self.vit.patch_embed
        self.cls_token = self.vit.cls_token
        self.pos_drop = self.vit.pos_drop
        self.pos_embed = self.vit.pos_embed
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm

    def forward(self, images):
        # images: (B, 3, H, W), expected float32 in [0, 1]
        x = self.patch_embed(images)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # prepend cls token
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # (B, num_patches+1, hidden_size)

class LayoutLMv3(nn.Module):
    def __init__(
        self,
        text_model_name="bert-base-uncased",
        vision_model_name="vit_base_patch16_224",
        num_labels=16,  # Change as per your dataset
        fusion_hidden_size=768,
        n_fusion_layers=2,
        fusion_nhead=8
    ):
        super().__init__()
        # Text encoder (BERT)
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        hidden_size = fusion_hidden_size

        # Layout embedding
        self.layout_embedding = nn.Linear(2, hidden_size)

        # Vision encoder (ViT via timm)
        self.vision_encoder = VisionEncoder(vision_model_name)
        assert self.vision_encoder.hidden_size == hidden_size

        # Simple fusion: after concatenating text+layout and vision embeddings
        fusion_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=fusion_nhead,batch_first=True)
        self.fusion_transformer = nn.TransformerEncoder(fusion_layer, num_layers=n_fusion_layers)

        self.classifier = nn.Linear(hidden_size, num_labels)

    def _check_token_ids(self, input_ids):
        vocab_size = self.text_encoder.config.vocab_size
        max_id = int(input_ids.max())
        if max_id >= vocab_size:
            raise ValueError(
                f"input_ids contain id {max_id} but the text encoder ({self.text_encoder.config.name_or_path}) "
                f"has vocab_size {vocab_size}. The OCR tensors were tokenized with a different tokenizer "
                f"(e.g. LayoutLMv3/RoBERTa ids fed to BERT); use a matching text encoder or re-tokenize."
            )

    def forward(self, input_ids, bbox, attention_mask, pixel_values):
        self._check_token_ids(input_ids)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_out.last_hidden_state  # (B, seq_len, hidden)
        
        # Layout: (B, seq_len, 4) -> (x_center, y_center) -> (B, seq_len, 2)
        bbox = bbox.float()
        x_c = ((bbox[..., 0] + bbox[..., 2]) / 2) / 1000.0
        y_c = ((bbox[..., 1] + bbox[..., 3]) / 2) / 1000.0
        layout_coords = torch.stack([x_c, y_c], dim=-1)
        layout_embeds = self.layout_embedding(layout_coords)

        fused_text = text_embeds + layout_embeds  # (B, seq_len, hidden)

        vision_embeds = self.vision_encoder(pixel_values)  # (B, num_patches+1, hidden)
        
        # Concatenate along sequence dim: [text+layout, vision]
        combined = torch.cat([fused_text, vision_embeds], dim=1)  # (B, seq_concat, hidden)
        # transformers expects (seq_len, B, hidden)
        #combined = combined.permute(1, 0, 2)
        fused_out = self.fusion_transformer(combined)#.permute(1, 0, 2)  # (B, seq_concat, hidden)
        cls_token = fused_out[:, 0]  # Use first token for classification

        logits = self.classifier(cls_token)
        return logits

    @torch.no_grad()
    def extract_features(self, input_ids, bbox, attention_mask, pixel_values):
        """Document-level features before the classifier layer, without gradients. Shape [batch_size, hidden_size]."""
        return self.forward_features(input_ids, bbox, attention_mask, pixel_values)

    def forward_features(self, input_ids, bbox, attention_mask, pixel_values):
        """
        Extract document-level features before the classifier layer (keeps the autograd graph, for losses on features).
        Returns:
            torch.FloatTensor: shape [batch_size, hidden_size]
        """
        # Text path
        self._check_token_ids(input_ids)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embeds = text_out.last_hidden_state  # (B, seq_len, hidden)

        # Layout path (assume bbox shape: [B, seq_len, 4])
        bbox = bbox.float()
        x_c = ((bbox[..., 0] + bbox[..., 2]) / 2) / 1000.0
        y_c = ((bbox[..., 1] + bbox[..., 3]) / 2) / 1000.0
        layout_coords = torch.stack([x_c, y_c], dim=-1)  # (B, seq_len, 2)
        layout_embeds = self.layout_embedding(layout_coords)  # (B, seq_len, hidden)

        # Fuse text embeddings with layout
        fused_text = text_embeds + layout_embeds  # (B, seq_len, hidden)

        # Visual path
        vision_embeds = self.vision_encoder(pixel_values)  # (B, num_patches + 1, hidden)

        # Concatenate [text+layout, vision]
        combined = torch.cat([fused_text, vision_embeds], dim=1)  # (B, seq_concat, hidden)
        # Transformer expects (seq_len, B, hidden)
        #combined = combined.permute(1, 0, 2)
        fused_out = self.fusion_transformer(combined)#.permute(1, 0, 2)  # (B, seq_concat, hidden)

        # Take the [CLS] token (first one) as the global document representation
        cls_token_embed = fused_out[:, 0]  # [B, hidden_size]
        return cls_token_embed
