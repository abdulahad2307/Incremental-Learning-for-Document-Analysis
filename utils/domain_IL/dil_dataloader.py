import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from typing import Dict, List, Optional
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedShuffleSplit

common_transform = transforms.Compose([
    transforms.Resize((229, 229)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

ACCEPTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

_OCR_FILE_CACHE = {}


def _load_ocr_file(path):
    """Load a single OCR tensor file dict[img_path] -> {'input_ids','attention_mask'} once per process,
    plus a filename index for lookups when image paths differ."""
    if path not in _OCR_FILE_CACHE:
        print(f"Loading OCR tensor file: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, dict):
            raise ValueError(f"OCR tensor file must be dict[img_path] -> {{'input_ids','attention_mask'}}: {path}")
        by_name = {os.path.basename(k): v for k, v in data.items()}
        _OCR_FILE_CACHE[path] = (data, by_name)
    return _OCR_FILE_CACHE[path]

class EAMLDocumentDataset(datasets.ImageFolder):
    """ImageFolder with OCR tensor loading and flexible image support."""

    def __init__(self, root, transform=None, ocr_tensor_dir=None, class_to_idx=None):
        super().__init__(root, transform=transform)
        # Filter samples by accepted extensions
        self.samples = [
            (p, label) for p, label in self.samples if p.lower().endswith(ACCEPTED_EXTENSIONS)
        ]
        self.imgs = self.samples
        self.ocr_tensor_dir = ocr_tensor_dir
        self.class_to_idx = class_to_idx or self.class_to_idx
        # OCR source: a directory of per-image .pt files, or a single .pt file with a dict of all images
        self.ocr_file = None
        self._n_missing_ocr = 0
        if ocr_tensor_dir and ocr_tensor_dir != "None":
            if os.path.isfile(ocr_tensor_dir):
                self.ocr_file = _load_ocr_file(ocr_tensor_dir)
            elif not os.path.isdir(ocr_tensor_dir):
                raise FileNotFoundError(f"OCR tensor path not found: {ocr_tensor_dir}")
        else:
            print(f"Warning: no OCR tensors for {root}; text branch gets empty input.")

    def _empty_text(self):
        return {
            "input_ids": torch.zeros(128, dtype=torch.long),
            "attention_mask": torch.zeros(128, dtype=torch.long),
        }

    def __getitem__(self, index):
        max_attempts = 10
        attempts = 0
        while attempts < max_attempts:
            path, label = self.samples[index]
            try:
                # Robust image loading
                image = Image.open(path).convert("RGB")
                if self.transform:
                    image = self.transform(image)

                # Load OCR tensor if available and valid
                ocr_tensor = None
                if self.ocr_file is not None:
                    by_path, by_name = self.ocr_file
                    entry = by_path.get(path) or by_name.get(os.path.basename(path))
                    if entry is not None:
                        ocr_tensor = {
                            "input_ids": entry["input_ids"],
                            "attention_mask": entry["attention_mask"],
                        }
                    else:
                        self._n_missing_ocr += 1
                        if self._n_missing_ocr in (1, 100, 1000):
                            print(f"Warning: {self._n_missing_ocr} image(s) without OCR entry in {self.ocr_tensor_dir}, e.g. {path}")
                        ocr_tensor = self._empty_text()
                elif (
                    self.ocr_tensor_dir
                    and self.ocr_tensor_dir != "None"
                    and os.path.isdir(self.ocr_tensor_dir)
                ):
                    basename = os.path.splitext(os.path.basename(path))[0]
                    pt_path = os.path.join(self.ocr_tensor_dir, f"{basename}.pt")
                    if os.path.exists(pt_path):
                        loaded = torch.load(pt_path)
                        if (
                            isinstance(loaded, dict)
                            and "input_ids" in loaded
                            and "attention_mask" in loaded
                        ):
                            ocr_tensor = {
                                "input_ids": loaded["input_ids"],
                                "attention_mask": loaded["attention_mask"],
                            }
                        else:
                            # Wrap tensor as dict if needed
                            ocr_tensor = {
                                "input_ids": loaded,
                                "attention_mask": torch.ones_like(loaded),
                            }
                    else:
                        raise FileNotFoundError(f"OCR tensor file missing: {pt_path}")
                else:
                    # No OCR source: provide zeros as fallback
                    ocr_tensor = self._empty_text()

                # Map class label globally if mapping exists
                if self.class_to_idx:
                    #print("class_to_idx keys:", list(self.class_to_idx.keys()))
                    class_name = self.classes[label]
                    # If the class name contains a space, replace spaces with underscores
                    if ' ' in class_name:
                        class_name = class_name.replace(' ', '_')
                    # (Optional: also lowercase if you want to remove case sensitivity)
                    # class_name = class_name.lower()
                    label = self.class_to_idx[class_name]


                return {"image": image, "text": ocr_tensor, "label": label}

            except (UnidentifiedImageError, OSError, FileNotFoundError) as e:
                print(f"Warning: skipping sample at {path} due to error: {e}")
                index = (index + 1) % len(self.samples)
                attempts += 1

        # If all attempts fail
        raise RuntimeError(f"Failed to load a valid sample after {max_attempts} attempts.")

def split_dataset(dataset, splits=(0.7, 0.15, 0.15)):
    n = len(dataset)
    lengths = [int(r * n) for r in splits]
    lengths[-1] = n - sum(lengths[:-1])
    return random_split(dataset, lengths)


def stratified_split(dataset, splits=(0.7, 0.15, 0.15), random_state=42):
    labels = [target for _, target in dataset.samples]
    n = len(dataset)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=(1 - splits[0]), random_state=random_state)
    train_idx, temp_idx = next(sss.split(np.zeros(n), labels))
    temp_labels = [labels[i] for i in temp_idx]
    temp_n = len(temp_idx)
    val_frac = splits[1] / (1 - splits[0])
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=random_state+1)
    val_idx_rel, test_idx_rel = next(sss2.split(np.zeros(temp_n), temp_labels))
    val_idx = np.array(temp_idx)[val_idx_rel]
    test_idx = np.array(temp_idx)[test_idx_rel]
    return (
        torch.utils.data.Subset(dataset, train_idx.tolist()),
        torch.utils.data.Subset(dataset, val_idx.tolist()),
        torch.utils.data.Subset(dataset, test_idx.tolist()),
    )


def safe_collate(batch):
    # Filter out samples missing 'text' or with incomplete OCR tokens
    clean_batch = [
        sample for sample in batch
        if sample.get('text') is not None
        and isinstance(sample['text'], dict)
        and 'input_ids' in sample['text']
        and sample['text']['input_ids'] is not None
        and 'attention_mask' in sample['text']
        and sample['text']['attention_mask'] is not None
    ]
    if len(clean_batch) == 0:
        raise ValueError("All samples in batch have missing or invalid OCR/text data.")

    return {
        "images": torch.stack([x["image"] for x in clean_batch]),
        "texts": {
            "input_ids": torch.stack([x["text"]["input_ids"] for x in clean_batch]),
            "attention_mask": torch.stack([x["text"]["attention_mask"] for x in clean_batch]),
        },
        "labels": torch.tensor([x["label"] for x in clean_batch]),
    }

class DILDataLoader:
    def __init__(
        self,
        data_root,
        domain_list,
        batch_size=16,
        img_size=(229, 229),
        num_workers=4,
        ocr_tensor_dirs=None,
        class_to_idx=None,
    ):
        self.data_root = data_root
        self.domain_list = domain_list
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers
        self.ocr_tensor_dirs = ocr_tensor_dirs or {}
        self.class_to_idx = class_to_idx

        self.transform = common_transform
        self._validate_domains()

    def _validate_domains(self):
        for domain in self.domain_list:
            if not os.path.isdir(os.path.join(self.data_root, domain)):
                raise ValueError(f"Domain directory not found: {domain}")

    def get_domain_loaders(self, phase="train") -> Dict[str, DataLoader]:
        loaders = {}
        for domain in self.domain_list:
            domain_phase_path = os.path.join(self.data_root, domain, phase)
            ocr_dir = self.ocr_tensor_dirs.get(domain)

            if os.path.exists(domain_phase_path):
                ds = EAMLDocumentDataset(domain_phase_path, transform=self.transform, ocr_tensor_dir=ocr_dir, class_to_idx=self.class_to_idx)
                if len(ds) == 0:
                    continue
                loaders[domain] = DataLoader(
                    ds,
                    batch_size=self.batch_size,
                    shuffle=(phase == "train"),
                    num_workers=self.num_workers,
                    pin_memory=True,
                    drop_last=(phase == "train"),
                    collate_fn=safe_collate,
                )
            else:
                full_ds = EAMLDocumentDataset(os.path.join(self.data_root, domain), transform=self.transform, ocr_tensor_dir=ocr_dir, class_to_idx=self.class_to_idx)
                train_ds, val_ds, test_ds = stratified_split(full_ds)
                if phase == "train":
                    ds = train_ds
                    shuffle = True
                    drop_last = True
                elif phase == "val":
                    ds = val_ds
                    shuffle = False
                    drop_last = False
                elif phase == "test":
                    ds = test_ds
                    shuffle = False
                    drop_last = False
                else:
                    continue
                loaders[domain] = DataLoader(
                    ds,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    num_workers=self.num_workers,
                    drop_last=drop_last,
                    collate_fn=safe_collate,
                )

        return loaders

    def get_class_counts(self):
        class_counts = {}
        for domain in self.domain_list:
            train_path = os.path.join(self.data_root, domain, "train")
            if os.path.exists(train_path):
                dataset_path = train_path
            else:
                dataset_path = os.path.join(self.data_root, domain)

            try:
                dataset = datasets.ImageFolder(dataset_path)
                class_counts[domain] = len(dataset.classes)
                print(f"Domain {domain} classes found: {class_counts[domain]} at {dataset_path}")
            except Exception as e:
                class_counts[domain] = 0
                print(f"Error reading domain {domain} at {dataset_path}: {e}")
        return class_counts

    
    def get_full_dataset(self, domain: str) -> Dataset:
        """
        Returns the full EAMLDocumentDataset (without split) for the given domain.
        Useful for exemplar extraction.
        """
        ocr_dir = self.ocr_tensor_dirs.get(domain)
        domain_path = os.path.join(self.data_root, domain)

        # If domain has train/val/test subfolders, you might want to use train subset only for exemplars:
        train_path = os.path.join(domain_path, "train")
        if os.path.exists(train_path):
            return EAMLDocumentDataset(train_path, transform=self.transform, ocr_tensor_dir=ocr_dir, class_to_idx=self.class_to_idx)
        else:
            # Otherwise, return full domain dataset (all data)
            return EAMLDocumentDataset(domain_path, transform=self.transform, ocr_tensor_dir=ocr_dir, class_to_idx=self.class_to_idx)
