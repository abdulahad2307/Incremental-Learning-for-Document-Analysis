import os
import gc
import torch
from torch.utils.data import Dataset, DataLoader,Subset
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as T
import random
import numpy as np

from utils.data_subset import split_indices

def poly8_to_bbox4(poly):
    xs = poly[0::2]
    ys = poly[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


class IncrementalOCRTensorsDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        ocr_tensor_file: str,
        included_classes: list,
        exemplar_samples: list = None,
        max_length: int = 512,
        bbox_style: str = "rect",
        images_per_class: int = None,
        seed: int = 42,
        dataset_name: str = "rvl_cdip",  # or "tobacco3482"
        split: str = "train",
        label_classes: list = None,
    ):
        self.image_dir = Path(image_dir)
        self.ocr_tensor_dir = Path(ocr_tensor_file)  # directory containing .pt files
        self.included_classes = included_classes
        # Labels index into label_classes (the model's full label space), not into the loaded subset
        self.label_classes = label_classes if label_classes is not None else included_classes
        missing = [c for c in included_classes if c not in self.label_classes]
        if missing:
            raise ValueError(f"Classes {missing} are not in label_classes {self.label_classes}")
        self.class2idx = {c: i for i, c in enumerate(self.label_classes)}
        self.max_length = max_length
        self.bbox_style = bbox_style
        # Same preprocessing as the base model (utils/llmv3/llmv3_data_loader.py): images in [0, 1]
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
        ])
        self.dataset_name = dataset_name

        # Collect samples: tuples of (image_path, None, class_name)
        random.seed(seed)
        self.samples = []
        print(f"Looking for classes: {included_classes}")
        for class_name in included_classes:
            # Dataset folder structure per dataset
            if dataset_name == "rvl_cdip":
                class_dir = self.image_dir / split / class_name
            elif dataset_name == "tobacco3482":
                class_dir = self.image_dir / class_name
            else:
                raise ValueError(f"Unknown dataset_name {dataset_name}")

            #print(f"Checking {class_dir} exists? {class_dir.exists()}")

            if not class_dir.exists():
                continue

            #images = list(class_dir.glob("*.*"))
            valid_extensions = ['.jpg', '.jpeg', '.png', '.tiff','.tif', '.bmp']
            images = [img for img in class_dir.iterdir() if img.suffix.lower() in valid_extensions]
            # Skip images without an OCR file (unreadable images get none), as the base-model loader does
            no_ocr = [img for img in images if not (self.ocr_tensor_dir / (img.stem + ".pt")).exists()]
            if no_ocr:
                print(f"Skipping {len(no_ocr)} image(s) in {class_dir} without OCR file, e.g. {no_ocr[0].name}")
                images = [img for img in images if img not in set(no_ocr)]
            if images_per_class:
                images = random.sample(images, min(images_per_class, len(images)))
            else:
                images = random.sample(images, len(images))

            for img_path in images:
                self.samples.append((img_path, None, class_name))

        # Append exemplars if provided
        if exemplar_samples:
            for ex in exemplar_samples:
                ex_img_path = Path(ex['image_path'])
                ex_img_path_full = self.image_dir / ex_img_path
                cls = ex.get('label', None)
                self.samples.append((ex_img_path_full, ex, cls))
        print(f"[DEBUG] Dataset '{dataset_name}' loaded {len(self.samples)} samples for classes: {included_classes}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ex_ocr_dict, cls = self.samples[idx]
        try:
            pil_image = Image.open(str(img_path)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            pil_image = Image.new("RGB", (224, 224), color="white")
        image = self.transform(pil_image)

        # Load OCR tensor on-demand if not an exemplar (exemplar OCR dict directly)
        if ex_ocr_dict is not None:
            ocr_dict = ex_ocr_dict
        else:
            ocr_tensor_path = self.ocr_tensor_dir / (img_path.stem + ".pt")
            ocr_dict = torch.load(str(ocr_tensor_path))

        input_ids = ocr_dict["input_ids"]
        input_ids = input_ids.clamp(min=0)  # token ids must not be clamped to seq length
        attention_mask = ocr_dict["attention_mask"]

        if self.bbox_style == "rect":
            if "bbox" in ocr_dict:
                bbox = ocr_dict["bbox"]
            elif "bboxes" in ocr_dict:
                bbox = torch.stack([
                    torch.tensor(poly8_to_bbox4(poly.tolist()), dtype=torch.float32)
                    for poly in ocr_dict["bboxes"]])
            else:
                raise ValueError(f"No bbox or bboxes in OCR dict for {img_path}")
        elif self.bbox_style == "poly":
            if "bboxes" in ocr_dict:
                bbox = ocr_dict["bboxes"].float()
            elif "bbox" in ocr_dict:
                N = ocr_dict["bbox"].shape[0]
                bbox = torch.zeros((N, 8), dtype=torch.float32)
                rects = ocr_dict["bbox"]
                for i in range(N):
                    x0, y0, x1, y1 = rects[i]
                    bbox[i] = torch.tensor([x0, y0, x1, y0, x1, y1, x0, y1])
            else:
                raise ValueError(f"No bbox or bboxes in OCR dict for {img_path}")
        else:
            raise ValueError(f"Unsupported bbox_style: {self.bbox_style}")

        input_ids = self._pad_truncate(input_ids, self.max_length)
        attention_mask = self._pad_truncate(attention_mask, self.max_length)
        bbox = self._pad_truncate(bbox, self.max_length)

        label = torch.tensor(self.class2idx[cls], dtype=torch.long)

        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "bbox": bbox,
            "attention_mask": attention_mask,
            "labels": label,
        }

    def _pad_truncate(self, tensor, max_len):
        N = tensor.shape[0]
        if N > max_len:
            return tensor[:max_len]
        elif N < max_len:
            pad_shape = (max_len - N,) + tensor.shape[1:]
            pad = torch.zeros(pad_shape, dtype=tensor.dtype)
            return torch.cat([tensor, pad], dim=0)
        else:
            return tensor


def get_incremental_dataloader(
    dataset_name: str,
    ocr_tensor_file: str,
    classes: list,
    image_dir: str,
    split: str = "train",
    batch_size: int = 8,
    max_length: int = 512,
    bbox_style: str = "rect",
    images_per_class: int = None,
    seed: int = 42,
    label_classes: list = None,
):
    base_path = Path(image_dir)
    split_path = base_path / split

    # Check if split folder exists (for datasets with explicit splits)
    if split_path.exists():
        data_path = base_path
        use_internal_split = False
        print("Taking All")
        print(f"Loading data from folder: {data_path}")
    else:
        data_path = base_path
        use_internal_split = True
        print("Splitting data 70/15/15 by document (utils/data_subset.split_documents, same as EAML)")
        print(f"Loading data from folder: {data_path}")

    
    full_dataset = IncrementalOCRTensorsDataset(
        image_dir=str(data_path),
        ocr_tensor_file=ocr_tensor_file,
        included_classes=classes,
        max_length=max_length,
        bbox_style=bbox_style,
        images_per_class=None if use_internal_split else images_per_class,
        seed=seed,
        dataset_name=dataset_name,
        split=split if not use_internal_split else "train",
        label_classes=label_classes,
    )

    if use_internal_split:
        # No split folders (Tobacco-3482): stratified 70/15/15 split by document, independent of the run seed and
        # identical to the EAML DIL loader, so both backbones and all seeds use the same documents per split
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        dataset = Subset(full_dataset, split_indices([s[0] for s in full_dataset.samples], str(data_path), split))
    else:
        dataset = full_dataset

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
    )

    return loader