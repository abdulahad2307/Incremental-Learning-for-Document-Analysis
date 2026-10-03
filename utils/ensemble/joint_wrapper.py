import torch.nn as nn
import torch.nn.functional as F


class JointModelWrapper(nn.Module):
    """
    A unified wrapper for combining EAML and DocFormer models for ensembling.
    Supports both softmax fusion and score averaging.
    """
    def __init__(self, eaml_model, docformer_model, fusion="average", weights=None):
        """
        Args:
            eaml_model: Pre-loaded EAML model
            docformer_model: Pre-loaded DocFormer model
            fusion: One of ['average', 'weighted']
            weights: Optional weights for weighted fusion (tuple of 2 floats)
        """
        super().__init__()
        self.eaml = eaml_model
        self.docformer = docformer_model

        self.fusion = fusion
        if fusion == "weighted":
            assert weights is not None and len(weights) == 2, \
                "Weighted fusion requires two weights for EAML and DocFormer"
            self.weights = weights
        else:
            self.weights = (0.5, 0.5)

        self.eaml.eval()
        self.docformer.eval()

    def forward(self, batch):
        """
        Args:
            batch: Dict containing all fields required by both models
        Returns:
            logits: Combined logits after fusion
        """
        device = next(self.parameters()).device

        # ---- EAML forward ----
        images = batch['images'].to(device)
        eaml_input_ids = batch['texts']['input_ids'].to(device)
        eaml_attention_mask = batch['texts']['attention_mask'].to(device)

        eaml_logits = self.eaml(images, input_ids=eaml_input_ids, attention_mask=eaml_attention_mask)
        eaml_probs = F.softmax(eaml_logits, dim=1)

        # ---- DocFormer forward ----
        doc_images = batch['pixel_values'].to(device)
        doc_input_ids = batch['input_ids'].to(device)
        doc_attention_mask = batch['attention_mask'].to(device)
        bboxes = batch['bboxes'].to(device)

        doc_logits = self.docformer(
            doc_images,
            input_ids=doc_input_ids,
            attention_mask=doc_attention_mask,
            bboxes=bboxes
        )
        doc_probs = F.softmax(doc_logits, dim=1)

        # ---- Fusion ----
        fused_probs = self.weights[0] * eaml_probs + self.weights[1] * doc_probs
        return fused_probs
