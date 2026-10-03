import torch
import torch.nn as nn
import timm  # Pretrained models

class ImageEncoder(nn.Module):
    def __init__(self, model_name="inception_resnet_v2", embed_dim=512):
        """
        Implementation Image feature extractor using a pretrained model (Inception-ResNet-V2).

        Parameters:
            model_name (str): Model name from TIMM library.
            embed_dim (int): Feature embedding dimension.
        """
        super(ImageEncoder, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True, num_classes=embed_dim)

    def forward(self, x):
        return self.model(x)
