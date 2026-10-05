import torch
import torch.nn as nn
import timm

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

        for param in self.model.parameters():
            param.requires_grad = True #False ##  # True for trainable, False for frozen

    def forward(self, x):
        return self.model(x)

    # Split at the first residual stage (EAML inserts its attention block there, after "Residual block 0")
    STEM = ("conv2d_1a", "conv2d_2a", "conv2d_2b", "maxpool_3a", "conv2d_3b", "conv2d_4a", "maxpool_5a",
            "mixed_5b", "repeat")
    REST = ("mixed_6a", "repeat_1", "mixed_7a", "repeat_2", "block8", "conv2d_7b")

    def stem(self, x):
        """Images -> feature maps after the first residual stage (B, 320, H', W')."""
        for name in self.STEM:
            x = getattr(self.model, name)(x)
        return x

    def rest(self, x):
        """Feature maps of stem() -> image embedding (B, embed_dim); stem + rest == forward."""
        for name in self.REST:
            x = getattr(self.model, name)(x)
        return self.model.forward_head(x)
