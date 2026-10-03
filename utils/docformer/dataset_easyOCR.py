# dataset.py

import os
import numpy as np
import torch
import easyocr
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer
from typing import List, Dict, Optional

class DocFormerDataset(Dataset):
    """
    DocFormer-compatible dataset using EasyOCR for word boxes.
    """
    def __init__(
        self,
        data_dir: str,
        tokenizer_name: str = "bert-base-uncased",
        max_seq_length: int = 512,
        split: str = "train",
        classes: Optional[List[str]] = None,
        languages: List[str] = ['en'],
        use_gpu_ocr: bool = True
    ):
        self.root = os.path.join(data_dir, split)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.classes = sorted(classes) if classes else None
        # EasyOCR reader
        self.reader = easyocr.Reader(languages, gpu=use_gpu_ocr)
        # Build sample list
        self.samples = []
        for cls in os.listdir(self.root):
            if self.classes and cls not in self.classes:
                continue
            cls_dir = os.path.join(self.root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for f in os.listdir(cls_dir):
                if f.lower().endswith((".png",".jpg",".jpeg",".tif",".tiff")):
                    self.samples.append({"image_path": os.path.join(cls_dir,f), "class": cls})
        if not self.samples:
            raise RuntimeError(f"No samples under {self.root}")
        self.class_to_idx = {c:i for i,c in enumerate(sorted({s["class"] for s in self.samples}))}

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        img = Image.open(sample["image_path"]).convert("RGB")
        pixel_values = self.transform(img)
        # OCR returns list of (bbox, text, conf)
        ocr = self.reader.readtext(np.array(img))
        tokens, bboxes = [], []
        # flatten words & bboxes
        for box, text, _ in ocr:
            # flatten 4 corners [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] -> [x1,y1,...,x4,y4]
            flat = [coord for pt in box for coord in pt]
            # tokenize subwords
            sub = self.tokenizer.tokenize(text)
            ids = self.tokenizer.convert_tokens_to_ids(sub)
            for _id in ids:
                tokens.append(_id)
                bboxes.append(flat)
            if len(tokens) >= self.max_seq_length - 2:
                break
        # add [CLS] and [SEP]
        cls_box = [0,0,img.width,0,img.width,img.height,0,img.height]
        tokens = [self.tokenizer.cls_token_id] + tokens[:self.max_seq_length-2] + [self.tokenizer.sep_token_id]
        bboxes = [cls_box] + bboxes[:self.max_seq_length-2] + [cls_box]
        # pad
        pad_len = self.max_seq_length - len(tokens)
        tokens += [self.tokenizer.pad_token_id]*pad_len
        bboxes += [[0]*8]*pad_len
        attention_mask = [1]*(len(tokens)-pad_len) + [0]*pad_len

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "bboxes": torch.tensor(bboxes, dtype=torch.long),
            "label": torch.tensor(self.class_to_idx[sample["class"]], dtype=torch.long),
        }

def docformer_collate_fn(batch: List[Dict]) -> Dict:
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "input_ids":    torch.stack([x["input_ids"]    for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "bboxes":       torch.stack([x["bboxes"]       for x in batch]),
        "labels":       torch.stack([x["label"]        for x in batch]),
    }

class DocFormerDataLoader:
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4, classes: Optional[List[str]] = None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes = classes

    def get_loader(self, split: str = "train", shuffle: bool = True):
        dataset = DocFormerDataset(
            data_dir=self.data_dir,
            split=split,
            classes=self.classes
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=docformer_collate_fn
        )

    def get_class_mappings(self) -> Dict:
        ds = DocFormerDataset(data_dir=self.data_dir, split="train", classes=self.classes)
        return {
            "class_to_idx": ds.class_to_idx,
            "idx_to_class": {i:c for c,i in ds.class_to_idx.items()},
            "classes":      sorted(ds.class_to_idx.keys())
        }
