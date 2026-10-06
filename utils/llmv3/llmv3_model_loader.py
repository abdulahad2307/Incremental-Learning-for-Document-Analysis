import os

import torch
import torch.nn as nn
from transformers import BertModel, LayoutLMv3Model
import timm

HF_LAYOUTLMV3 = "microsoft/layoutlmv3-base"
MODEL_TYPES = ("custom", "hf")

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

class CustomLayoutLMv3(nn.Module):
    """Custom LayoutLMv3-style model ("Custom LLMv3"), used in the thesis and the workshop paper: bert-base-uncased
    text encoder + box-centre layout embedding, a timm ViT, and a 2-layer fusion transformer, trained from these
    unimodal checkpoints (no LayoutLMv3 pre-training). For the pre-trained microsoft/layoutlmv3-base see HFLayoutLMv3."""

    model_type = "custom"

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


# Old name of CustomLayoutLMv3, kept so existing imports and checkpoints keep working.
LayoutLMv3 = CustomLayoutLMv3


class HFLayoutLMv3(nn.Module):
    """Pre-trained LayoutLMv3 (microsoft/layoutlmv3-base) with the same interface as CustomLayoutLMv3:
    forward(input_ids, bbox, attention_mask, pixel_values) -> logits, forward_features / extract_features -> document
    features, and a plain nn.Linear `classifier` (so classifier expansion, bias correction and the OOD detectors work
    unchanged).

    Classification head as in the LayoutLMv3 paper (an MLP on the [CLS] token) and Hugging Face's
    LayoutLMv3ForSequenceClassification: dropout -> dense -> tanh -> dropout -> linear. The tanh output is the
    document feature; `classifier` is the final linear layer.

    Inputs: LayoutLMv3 tokenizer ids (RoBERTa vocabulary, from tools/ocr/ocr_extraction_bbox_layoutlmv3.py), one
    [x0, y0, x1, y1] box per token in 0-1000, and pixel_values already normalised with mean = std = 0.5 by the
    data loaders (utils/image_transforms.py), as the LayoutLMv3 image processor does."""

    model_type = "hf"

    def __init__(self, num_labels=16, hf_model_name=HF_LAYOUTLMV3, dropout=0.1):
        super().__init__()
        self.hf_model_name = hf_model_name
        self.backbone = LayoutLMv3Model.from_pretrained(hf_model_name)
        hidden_size = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)  # MLP head on [CLS] (dense + tanh), then `classifier`
        self.classifier = nn.Linear(hidden_size, num_labels)

    def _check_inputs(self, input_ids, bbox):
        cfg = self.backbone.config
        if int(input_ids.max()) >= cfg.vocab_size or bool((input_ids[:, 0] != cfg.bos_token_id).any()):
            raise ValueError(
                f"input_ids do not look like {self.hf_model_name} tokens (sequences must start with <s> = "
                f"{cfg.bos_token_id}, ids < {cfg.vocab_size}). The OCR tensors were probably tokenized for the custom "
                f"model (bert-base-uncased); use the tensors from tools/ocr/ocr_extraction_bbox_layoutlmv3.py.")
        if bbox.shape[-1] != 4:
            raise ValueError(f"HFLayoutLMv3 needs one [x0, y0, x1, y1] box per token (bbox_style 'rect'), got {tuple(bbox.shape)}")

    def _encode(self, input_ids, bbox, attention_mask, pixel_values):
        self._check_inputs(input_ids, bbox)
        out = self.backbone(input_ids=input_ids, bbox=bbox.long().clamp(0, 1000), attention_mask=attention_mask,
                            pixel_values=pixel_values)
        return out.last_hidden_state  # (B, seq_len + 1 + num_patches, hidden)

    def forward(self, input_ids, bbox, attention_mask, pixel_values):
        return self.classifier(self.dropout(self.forward_features(input_ids, bbox, attention_mask, pixel_values)))

    @torch.no_grad()
    def extract_features(self, input_ids, bbox, attention_mask, pixel_values):
        """Document-level features before the classifier layer, without gradients. Shape [batch_size, hidden_size]."""
        return self.forward_features(input_ids, bbox, attention_mask, pixel_values)

    def forward_features(self, input_ids, bbox, attention_mask, pixel_values):
        """Document features before the classifier layer: tanh(dense(dropout([CLS]))) (keeps the autograd graph)."""
        cls_token = self._encode(input_ids, bbox, attention_mask, pixel_values)[:, 0]
        return torch.tanh(self.dense(self.dropout(cls_token)))


def build_llmv3_model(model_type="custom", num_labels=16, text_model_name="bert-base-uncased",
                      vision_model_name="vit_base_patch16_224", hf_model_name=HF_LAYOUTLMV3):
    """custom: CustomLayoutLMv3 (thesis / workshop model); hf: pre-trained LayoutLMv3 (HFLayoutLMv3)."""
    if model_type == "custom":
        return CustomLayoutLMv3(text_model_name=text_model_name, vision_model_name=vision_model_name, num_labels=num_labels)
    if model_type == "hf":
        return HFLayoutLMv3(num_labels=num_labels, hf_model_name=hf_model_name)
    raise ValueError(f"Unknown model_type '{model_type}'; choose from {MODEL_TYPES}")


def infer_model_type(state_dict):
    """Model type from the parameter names of a checkpoint (checkpoints written before model_type was stored)."""
    if any(k.startswith("backbone.") for k in state_dict):
        return "hf"
    if any(k.startswith("fusion_transformer.") for k in state_dict):
        return "custom"
    raise ValueError("Checkpoint is neither a CustomLayoutLMv3 nor an HFLayoutLMv3 state dict")


def checkpoint_meta(model):
    """Fields to store next to 'model_state_dict' so the checkpoint says which model it holds."""
    meta = {"model_type": model.model_type}
    if model.model_type == "hf":
        meta["hf_model_name"] = model.hf_model_name
    return meta


def load_llmv3_checkpoint(path, num_labels, device, model_type=None):
    """Build the model a checkpoint was saved from (custom or hf, read from the checkpoint) and load its weights.
    Returns (model on device, checkpoint dict)."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    model_type = model_type or checkpoint.get("model_type") or infer_model_type(state_dict)
    expected = os.environ.get("LLMV3_MODEL")  # variant selected in scripts/config.sh, if any
    if expected and expected != model_type:
        raise ValueError(f"{path} holds a '{model_type}' LayoutLMv3 but LLMV3_MODEL={expected}; "
                         f"submit with the matching --export=LLMV3_MODEL=... or use the matching checkpoint")
    model = build_llmv3_model(model_type, num_labels, hf_model_name=checkpoint.get("hf_model_name", HF_LAYOUTLMV3))
    model.load_state_dict(state_dict)
    print(f"Loaded {type(model).__name__} ({model_type}) from {path}")
    return model.to(device), checkpoint
