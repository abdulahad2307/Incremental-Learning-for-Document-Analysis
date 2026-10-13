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


EAML_IMG_SIZE = 229   # EAML paper: images downscaled to 229 x 229


def eaml_eval_transform(img_size=EAML_IMG_SIZE):
    """EAML input without augmentation (validation, test, herding, EVM / Fisher features)."""
    return transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize()])


def eaml_train_transform(img_size=EAML_IMG_SIZE):
    """EAML paper (Sec. 5.3) training augmentation: horizontal / vertical shift of 0.1, shear of 0.1 (Keras
    shear_range, in degrees) and cutout (random erasing on the normalised tensor)."""
    return transforms.Compose([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0.1),
                               transforms.Resize((img_size, img_size)), transforms.ToTensor(), normalize(),
                               transforms.RandomErasing(p=0.5, value=0)])


def _image_datasets(obj):
    """The datasets with an image `transform` behind a DataLoader / Subset / ConcatDataset."""
    obj = getattr(obj, "dataset", obj) if hasattr(obj, "batch_size") else obj   # DataLoader -> dataset
    if hasattr(obj, "datasets"):                                                 # ConcatDataset
        return [d for sub in obj.datasets for d in _image_datasets(sub)]
    if hasattr(obj, "indices") and hasattr(obj, "dataset"):                      # Subset
        return _image_datasets(obj.dataset)
    return [obj] if hasattr(obj, "transform") else []


def set_train_augmentation(loader, on):
    """Switch the EAML training augmentation on (during a training epoch) or off (everything else) for the datasets
    of `loader`, so herding, EWC Fisher and EVM features always see unaugmented images. Takes effect for the next
    iteration over the loader (its worker processes are created per epoch)."""
    for ds in _image_datasets(loader):
        if not hasattr(ds, "_eval_transform"):
            ds._eval_transform = ds.transform
        ds.transform = eaml_train_transform() if on else ds._eval_transform
