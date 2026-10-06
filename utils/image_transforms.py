"""Input normalisation of the pretrained image backbones, shared by every EAML and LayoutLMv3 data loader.

All three pretrained image models expect pixels scaled as (x - 0.5) / 0.5, i.e. to [-1, 1]:
  - Inception-ResNet-V2, EAML (timm 'tf_in1k' weights): mean = std = (0.5, 0.5, 0.5)
  - ViT-B/16, Custom LayoutLMv3 (timm 'augreg2_in21k_ft_in1k' weights): mean = std = (0.5, 0.5, 0.5)
  - microsoft/layoutlmv3-base (its image processor, preprocessor_config.json): image_mean = image_std = [0.5, 0.5, 0.5]
Base models and incremental steps must use the same normalisation, so it is defined only here.
"""
from torchvision import transforms

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def normalize():
    """The Normalize transform every loader applies after ToTensor()."""
    return transforms.Normalize(mean=MEAN, std=STD)
