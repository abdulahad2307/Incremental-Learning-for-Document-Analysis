import torch
import torch.nn as nn

class FusionModule(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8):
        """
        Implementation of Self-attention-based fusion module for combining image and text features.
        
        Parameters:
            embed_dim (int): Feature dimension of image and text embeddings.
            num_heads (int): Number of attention heads.
        """
        super(FusionModule, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, image_feat, text_feat):
        fusion_input = torch.cat((image_feat.unsqueeze(0), text_feat.unsqueeze(0)), dim=0)
        attn_output, _ = self.self_attn(fusion_input, fusion_input, fusion_input)
        output = self.fc(attn_output.mean(dim=0))
        return self.norm(output)
