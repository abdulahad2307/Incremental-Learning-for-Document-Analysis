import os
import json
from typing import List, Dict, Optional
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import torch
from transformers import BertTokenizer

from utils.llmv3.llmv3_data_loader import get_dataloaders, CILLayoutLMv3Dataset, layoutlmv3_cil_collate_fn
from utils.image_transforms import normalize

common_transform = transforms.Compose([
    transforms.Resize((229, 229)),
    transforms.ToTensor(),
    normalize()
])

def _normalize_class_name(name):
    # Removing leading/trailing whitespace and quotes
    return name.strip().strip("'").strip('"')

# ========================== EAML Dataset for Class IL ==========================
class EAMLClassILDataset(Dataset):
    def __init__(self, data_dir, current_classes, ocr_data_path=None, transform=None, img_size=229, handle_empty_text="exclude", fallback_text="[EMPTY]"):
        self.data_dir = data_dir
        self.current_classes = [cls.strip().strip('"').strip("'") for cls in current_classes]
        self.transform = transform
        self.img_size = img_size
        self.handle_empty_text = handle_empty_text
        self.fallback_text = fallback_text

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.current_classes)}

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.ocr_tokenized = False
        self.ocr_data = self._load_ocr_data(ocr_data_path)
        self.samples = []
        self._load_samples()

    def _load_ocr_data(self, ocr_path):
        if ocr_path is None:
            return {}
        if not os.path.exists(ocr_path):
            raise FileNotFoundError(f"OCR data file not found: {ocr_path}")
        if ocr_path.endswith('.pt') or ocr_path.endswith('.pth'):
            loaded = torch.load(ocr_path, map_location='cpu')
            first_val = next(iter(loaded.values()))
            if isinstance(first_val, dict) and 'input_ids' in first_val and 'attention_mask' in first_val:
                self.ocr_tokenized = True
                return loaded
            else:
                raise ValueError("Tensor OCR file must be dict[img_path]->{'input_ids','attention_mask'}")
        elif ocr_path.endswith('.json'):
            import json
            with open(ocr_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported OCR data format: {ocr_path}")

    def _get_ocr_entry(self, img_path):
        if img_path in self.ocr_data:
            return self.ocr_data[img_path]
        filename = os.path.basename(img_path)
        if filename in self.ocr_data:
            return self.ocr_data[filename]
        for k in self.ocr_data.keys():
            if img_path.endswith(k) or k.endswith(filename):
                return self.ocr_data[k]
        return None

    def _load_samples(self):
        for class_name in self.current_classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory for class '{class_name}' not found at {class_dir}")
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                    img_path = os.path.join(class_dir, fname)
                    ocr_entry = self._get_ocr_entry(img_path)
                    if ocr_entry is None:
                        print(f"Warning: No OCR data found for {img_path}")
                        continue
                    if self.ocr_tokenized:
                        tokens = {
                            "input_ids": torch.tensor(ocr_entry["input_ids"]),
                            #"input_ids": ocr_entry["input_ids"].detach().clone(), <== use this for EVM +OOD
                            "attention_mask": torch.tensor(ocr_entry["attention_mask"])
                            #"attention_mask": ocr_entry["attention_mask"].detach().clone(), <== use this for EVM +OOD to ensures a safe copy of the tensor without gradient tracking.
                        }
                    else:
                        text = ocr_entry
                        if not text or (isinstance(text, str) and len(text.strip()) == 0):
                            if self.handle_empty_text == "exclude":
                                continue
                            elif self.handle_empty_text == "fallback":
                                text = self.fallback_text
                        tokens_raw = self.tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
                        tokens = {
                            "input_ids": tokens_raw["input_ids"].squeeze(0),
                            "attention_mask": tokens_raw["attention_mask"].squeeze(0)
                        }
                    self.samples.append((img_path, tokens, class_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        max_attempts = 10
        attempts = 0
        starting_idx = idx
        while attempts < max_attempts:
            img_path, tokens, label = self.samples[idx]
            try:
                image = Image.open(img_path).convert("RGB")
                break
            except (UnidentifiedImageError, OSError, IOError):
                print(f"Warning: Skipping corrupt or unreadable image: {img_path}")
                idx = (idx + 1) % len(self.samples)
                attempts += 1
        else:
            raise RuntimeError(f"Too many consecutive corrupt images encountered, starting from idx {starting_idx}")

        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "text": {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"]
            },
            "label": torch.tensor(self.class_to_idx[label])
        }


def eaml_collate_fn(batch):
    return {
        "images": torch.stack([x["image"] for x in batch]),
        "texts": {
            "input_ids": torch.stack([x["text"]["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["text"]["attention_mask"] for x in batch])
        },
        "labels": torch.stack([x["label"] for x in batch])
    }

# ========================== DocFormer Dataset for Class IL ==========================
class DocFormerClassILDataset(Dataset):
    def __init__(self, data_dir: str, current_classes: List[str], tokenizer_name="bert-base-uncased", max_seq_length=256):
        self.data_dir = data_dir
        self.current_classes = [_normalize_class_name(cls) for cls in current_classes]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.current_classes)}
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        for class_name in os.listdir(self.data_dir):
            norm_class = _normalize_class_name(class_name)
            if norm_class not in self.current_classes:
                continue
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            found = False
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.samples.append({
                        'image_path': os.path.join(class_dir, file),
                        'class_name': norm_class
                    })
                    found = True
            if not found:
                print(f"Warning: No valid images found for class '{norm_class}' in {class_dir}")
        if len(self.samples) == 0:
            print("Warning: No samples loaded in DocFormerClassILDataset.")

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert("RGB")
        pixel_values = common_transform(image)
        input_ids = torch.tensor([self.tokenizer.cls_token_id] +
                                 [self.tokenizer.pad_token_id] * (self.max_seq_length - 2) +
                                 [self.tokenizer.sep_token_id])[:self.max_seq_length]
        attention_mask = torch.tensor([1] + [0] * (self.max_seq_length - 2) + [1])[:self.max_seq_length]
        bboxes = torch.tensor([[0, 0, image.width, 0, image.width, image.height, 0, image.height]] * self.max_seq_length)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bboxes": bboxes,
            "label": torch.tensor(self.class_to_idx[sample["class_name"]])
        }

def docformer_collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "bboxes": torch.stack([x["bboxes"] for x in batch]),
        "labels": torch.stack([x["label"] for x in batch])
    }

# ========================== Wrapper Loader ==========================
def get_class_il_loader(
    model_type: str,
    data_dir: str,
    current_classes: List[str],
    batch_size: int = 16,
    num_workers: int = 0,
    ocr_data: Optional[Dict] = None,
    max_length: int = 512,         
    bbox_style: str = "rect"        
) -> DataLoader:
    """Get dataloader for class incremental learning
    Args:
        model_type: Either 'eaml' or 'docformer'
        data_dir: Root directory containing class folders
        current_classes: List of classes to include
        batch_size: Number of samples per batch
        num_workers: Number of workers for data loading
    Returns:
        Configured DataLoader for the specified model type
    """    
    if model_type == "eaml":
        if model_type == "eaml":
            if ocr_data is None:
                raise ValueError("OCR data must be provided for EAML model")
            dataset = EAMLClassILDataset(
                data_dir=data_dir,
                current_classes=current_classes,
                ocr_data_path=ocr_data,
                transform=common_transform
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=eaml_collate_fn
            )
    elif model_type == "docformer":
        dataset = DocFormerClassILDataset(
            data_dir=data_dir,
            current_classes=current_classes
        )
        if len(dataset) == 0:
            print("Warning: DocFormerClassILDataset is empty!")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=docformer_collate_fn
        )
    elif model_type == "layoutlmv3":
        dataset = CILLayoutLMv3Dataset(
            image_dir=data_dir,
            ocr_tensor_file=ocr_data,
            current_classes=current_classes,
            max_length=max_length,
            bbox_style=bbox_style
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=layoutlmv3_cil_collate_fn
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_class_loader_for_class(class_name: str, dataset, batch_size: int, device, ocr_tensor_path):
    """
    Returns a DataLoader that loads only samples belonging to the specified class.
    
    Args:
        class_name (str): The class to filter.
        dataset (torch.utils.data.Dataset): The full dataset.
        batch_size (int): Batch size for DataLoader.
        device: Computation device (for reference).
        ocr_tensor_path (str): Path for OCR data as needed by dataset.
    
    Returns:
        DataLoader: DataLoader yielding only samples of class `class_name`.
    """
    from torch.utils.data import Subset, DataLoader

    # Finding indices of samples belonging to class_name
    indices = [idx for idx, sample in enumerate(dataset.samples)
               if sample[2] == class_name]  # Adjust if class label is at a different index
    
    # Creating subset of dataset with only those indices
    subset = Subset(dataset, indices)
    
    # Creating DataLoader; use your standard collate_fn if needed
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,  # no shuffle for feature extraction
        num_workers=2,
        collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
    )
    return loader
