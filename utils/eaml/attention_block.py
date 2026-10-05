"""Joint image-text self-attention block of EAML (Bakkali et al., 2021), applied inside the two branches.

As in the paper, it sits after the first residual stage of Inception-ResNet-V2 and after the first BERT transformer
block; the image features are global-average- and global-max-pooled, the text features global-max-pooled, and

    Q = FC_q(F'),  K = FC_k(F'),  V = FC_v(F'),   A = softmax(Q K^T / sqrt(d)),   F = A V,   M(F) = sigmoid(F) * F

is passed on to the following image and text blocks.

Deviation (documented choice): in the paper Q, K, V are m x d, i.e. the attention runs over the m documents of a
batch, so a document's prediction depends on the rest of its batch. Here F' are the three pooled descriptors of ONE
document (image-avg, image-max, text-max), so A is 3 x 3 per document and every document is processed on its own.

How M(F) is passed on is not specified in the paper; here it gates the branch features channel-wise,
X <- X * (1 + W M(F)), with W initialised to zero so the block starts as the identity and does not disturb the
pretrained backbones.
"""
import math

import torch
import torch.nn as nn


class EAMLAttentionBlock(nn.Module):
    def __init__(self, img_channels=320, txt_dim=768, d=256):
        super().__init__()
        self.d = d
        # one embedding per pooled descriptor, then shared Q/K/V projections
        self.embed_img_avg = nn.Linear(img_channels, d)
        self.embed_img_max = nn.Linear(img_channels, d)
        self.embed_txt_max = nn.Linear(txt_dim, d)
        self.fc_q = nn.Linear(d, d)
        self.fc_k = nn.Linear(d, d)
        self.fc_v = nn.Linear(d, d)
        # gates back to each branch's channels; zero-initialised -> identity at the start
        self.gate_img = nn.Linear(2 * d, img_channels)
        self.gate_txt = nn.Linear(d, txt_dim)
        for layer in (self.gate_img, self.gate_txt):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, image_map, text_seq, text_mask=None):
        """
        image_map: (B, C_img, H, W) features after the first Inception-ResNet-V2 residual stage
        text_seq:  (B, L, C_txt) hidden states after the first BERT block
        text_mask: (B, L) attention mask (1 = token, 0 = padding); padding is excluded from the max-pooling
        Returns the gated (image_map, text_seq).
        """
        img_avg = image_map.mean(dim=(2, 3))                                  # GlobalAvgPool2D
        img_max = image_map.amax(dim=(2, 3))                                  # GlobalMaxPool2D
        if text_mask is not None:
            text_seq_masked = text_seq.masked_fill(text_mask.unsqueeze(-1) == 0, float("-inf"))
            txt_max = text_seq_masked.amax(dim=1)                             # GlobalMaxPool1D over real tokens
            txt_max = torch.nan_to_num(txt_max, neginf=0.0)                   # documents without any token
        else:
            txt_max = text_seq.amax(dim=1)

        tokens = torch.stack([self.embed_img_avg(img_avg), self.embed_img_max(img_max),
                              self.embed_txt_max(txt_max)], dim=1)            # (B, 3, d) = F'
        q, k, v = self.fc_q(tokens), self.fc_k(tokens), self.fc_v(tokens)
        attn = torch.softmax(q @ k.transpose(1, 2) / math.sqrt(self.d), dim=-1)   # (B, 3, 3), per document
        f = attn @ v                                                          # (B, 3, d) = F
        m = torch.sigmoid(f) * f                                              # M(F) = sigmoid(F) * F

        g_img = self.gate_img(torch.cat([m[:, 0], m[:, 1]], dim=-1))          # (B, C_img)
        g_txt = self.gate_txt(m[:, 2])                                        # (B, C_txt)
        image_map = image_map * (1 + g_img)[:, :, None, None]
        text_seq = text_seq * (1 + g_txt)[:, None, :]
        return image_map, text_seq
