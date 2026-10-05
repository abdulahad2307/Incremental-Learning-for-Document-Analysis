import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as T
import random

from utils.data_subset import select_per_class


def poly8_to_bbox4(poly):
    xs = poly[0::2]
    ys = poly[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]


class OCRTensorsDataset(Dataset):
    def __init__(self, image_dir, ocr_tensor_file, classes, max_length=512, bbox_style="rect",
                 images_per_class=None, seed=42):
        self.image_dir = Path(image_dir)
        self.classes = classes
        self.class2idx = {c: i for i, c in enumerate(classes)}
        self.max_length = max_length
        self.bbox_style = bbox_style
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        # OCR source: a directory of per-image <stem>.pt files, or one combined file (list of entries with image_path)
        self.ocr_dir = Path(ocr_tensor_file) if Path(ocr_tensor_file).is_dir() else None
        self.ocr_map = {}
        if self.ocr_dir is None:
            self.ocr_data = torch.load(ocr_tensor_file)
            for entry in self.ocr_data:
                im_path = Path(entry.get("image_path", ""))
                self.ocr_map[str(im_path).lower()] = entry

        self.samples = []
        for c in classes:
            class_dir = self.image_dir / c
            if not class_dir.exists():
                continue
            # reproducible per-class subset, the same documents as the EAML base model (utils/data_subset.py)
            images = [Path(p) for p in select_per_class(class_dir.glob("*.*"), images_per_class, seed, c)]
            for img_path in images:
                if self.ocr_dir is not None:
                    pt_path = self.ocr_dir / (img_path.stem + ".pt")
                    if pt_path.exists():
                        self.samples.append((img_path, pt_path, c))  # loaded in __getitem__
                    continue
                key = str(img_path).lower()
                if key in self.ocr_map:
                    self.samples.append((img_path, self.ocr_map[key], c))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ocr_dict, cls = self.samples[idx]
        if isinstance(ocr_dict, Path):
            ocr_dict = torch.load(ocr_dict, weights_only=False)
        try:
            pil_image = Image.open(str(img_path)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            pil_image = Image.new("RGB", (224, 224), color="white")  # fallback blank image
        image = self.transform(pil_image)

        input_ids = ocr_dict["input_ids"]
        input_ids = input_ids.clamp(min=0)  # token ids must not be clamped to seq length
        attention_mask = ocr_dict["attention_mask"]

        if self.bbox_style == "rect":
            if "bbox" in ocr_dict:
                bbox = ocr_dict["bbox"]
            elif "bboxes" in ocr_dict:
                bbox = torch.stack([torch.tensor(poly8_to_bbox4(poly.tolist()), dtype=torch.float32)
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
            "labels": label
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


def get_dataloaders(dataset_name, ocr_tensor_file, base_classes, image_dir, batch_size=8,
                    max_length=512, bbox_style="rect", images_per_class=None, seed=42, val_image_dir=None):
    """Train on image_dir (images_per_class per class); validate on val_image_dir (all its images) if given,
    otherwise on a seeded 80/20 split of image_dir."""
    if dataset_name == "rvl_cdip":
        all_classes = ['letter', 'form', 'email', 'handwritten', 'advertisement', 'scientific_report',
                       'scientific_publication', 'specification', 'file_folder', 'news_article',
                       'budget', 'invoice','presentation', 'questionnaire', 'resume', 'memo']
    elif dataset_name == "tobacco3482":
        all_classes = ['ad', 'letter', 'form', 'memo', 'report', 'resume', 'scientific', 'specification', 'news']
    elif dataset_name == "docbank":
        all_classes = ['abstract', 'caption', 'equation', 'figure', 'list', 'paragraph', 'reference', 'table', 'title']
    elif dataset_name == "publaynet":
        all_classes = ['text', 'title', 'list', 'table', 'figure']
    else:
        raise ValueError(f"Unsupported dataset {dataset_name}")

    if base_classes:
        classes = [c for c in all_classes if c in base_classes]
    else:
        classes = all_classes

    print(f"Using classes: {classes}")
    dataset = OCRTensorsDataset(image_dir, ocr_tensor_file, classes, max_length, bbox_style,
                               images_per_class=images_per_class, seed=seed)
    
    print(f"Total samples loaded: {len(dataset)}")
    print(f"Number of classes: {len(classes)}")

    from collections import Counter
    class_counts = Counter([sample[2] for sample in dataset.samples])
    print("Samples per class:")
    for c in classes:
        print(f"  {c}: {class_counts.get(c, 0)}")
    
    
    if val_image_dir is not None:
        train_ds = dataset
        val_ds = OCRTensorsDataset(val_image_dir, ocr_tensor_file, classes, max_length, bbox_style,
                                   images_per_class=None, seed=seed)
        print(f"Validation: {len(val_ds)} samples from {val_image_dir}")
    else:
        n_total = len(dataset)
        n_train = int(0.8 * n_total)
        n_val = n_total - n_train
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, len(classes)


class CILLayoutLMv3Dataset(Dataset):
    def __init__(self, image_dir, ocr_tensor_file, current_classes, max_length=512, bbox_style="rect",
                 images_per_class=None, seed=42):
        self.image_dir = Path(image_dir)
        self.current_classes = [str(c) for c in current_classes]
        self.class2idx = {c: i for i, c in enumerate(self.current_classes)}
        self.max_length = max_length
        self.bbox_style = bbox_style
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor()
        ])

        self.ocr_data = torch.load(ocr_tensor_file)
        self.ocr_map = {}
        for entry in self.ocr_data:
            im_path = Path(entry.get("image_path", ""))
            self.ocr_map[str(im_path).lower()] = entry

        self.samples = []
        random.seed(seed)
        for c in self.current_classes:
            class_dir = self.image_dir / c
            if not class_dir.exists():
                continue
            images = list(class_dir.glob("*.*"))
            if images_per_class:
                images = random.sample(images, min(images_per_class, len(images)))
            for img_path in images:
                key = str(img_path).lower()
                if key in self.ocr_map:
                    self.samples.append((img_path, self.ocr_map[key], c))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ocr_dict, cls = self.samples[idx]
        try:
            pil_image = Image.open(str(img_path)).convert("RGB")
        except (UnidentifiedImageError, OSError):
            pil_image = Image.new("RGB", (224, 224), color="white")
        image = self.transform(pil_image)
        input_ids = ocr_dict["input_ids"].clamp(min=0)  # token ids must not be clamped to seq length
        attention_mask = ocr_dict["attention_mask"]

        if self.bbox_style == "rect":
            if "bbox" in ocr_dict:
                bbox = ocr_dict["bbox"]
            elif "bboxes" in ocr_dict:
                bbox = torch.stack([torch.tensor(poly8_to_bbox4(poly.tolist()), dtype=torch.float32)
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
            "labels": label
        }


def layoutlmv3_cil_collate_fn(batch):
    # All keys should be stacked like your main script, pay attention to tensor shapes
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "bbox": torch.stack([x["bbox"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": torch.stack([x["labels"] for x in batch])
    }
