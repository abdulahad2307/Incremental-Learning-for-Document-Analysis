import torch.nn as nn


class ElementwiseSumFusion(nn.Module):
    """EAML fusion (Bakkali et al., Eqs. 8-9): X_3 = X_1 + X_2, the element-wise sum of each document's image and text
    features; the fusion classifier is applied to X_3. Each sample is fused on its own (no mixing across the batch)."""

    def forward(self, image_feat, text_feat):
        return image_feat + text_feat
