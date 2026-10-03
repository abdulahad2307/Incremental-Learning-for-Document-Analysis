import os
import numpy as np
import json
from torchvision import transforms
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from transformers import BertTokenizer, TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import torch.nn as nn 
from typing import Optional, Dict, List
import time
from utils.eaml.ocr_manager import OCRManager

# ==================== EAML Components ====================
class EAML_Dataset(Dataset):
    def __init__(self, data_dir: str, transform=None, class_list: Optional[List[str]] = None, ocr_engine="trocr", **ocr_kwargs):
        self.data_dir = data_dir
        self.transform = transform
        self.class_list = sorted(class_list) if class_list else None
        self.samples = []
        self.ocr_engine_type = ocr_engine
        self.ocr_kwargs = ocr_kwargs
        
        # Initialize class mappings first
        self.class_to_idx = {}
        self.idx_to_class = {}
        self._build_class_mappings()
        
        # Initialize OCR components
        #self._initialize_ocr()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Load samples with strict class filtering
        self._load_samples()
        self._verify_labels()
        
        if not self.samples:
            raise ValueError(f"No valid samples found for specified classes: {class_list}")

    def _build_class_mappings(self):
        """Build class to index mappings only for specified classes"""
        if self.class_list is None:
            raise ValueError("Class list must be provided")
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_list)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.class_list)}

    def _initialize_ocr(self):
        """Initialize OCR components with proper weights"""
        start_time = time.time()
        print("Initializing OCR model...", end=" ")
        
        try:
            self.ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
            
            # Handle newly initialized weights
            if hasattr(self.ocr_model.encoder, 'pooler'):
                print("\nInitializing missing pooler weights...")
                nn.init.xavier_uniform_(self.ocr_model.encoder.pooler.dense.weight)
                nn.init.zeros_(self.ocr_model.encoder.pooler.dense.bias)
            
            # Freeze OCR model
            for param in self.ocr_model.parameters():
                param.requires_grad = False
                
            self.ocr_model.eval()
            print(f"done in {time.time()-start_time:.2f}s")
        except Exception as e:
            print(f"\nFailed to initialize OCR model: {str(e)}")
            raise

    def _verify_labels(self):
        """Verify all loaded samples have valid labels"""
        if not hasattr(self, 'class_to_idx'):
            raise AttributeError("Class mappings not initialized. Call _build_class_mappings() first.")
        
        valid_samples = []
        invalid_samples = 0
        
        for sample in self.samples:
            _, _, label = sample
            if label in self.class_to_idx:
                valid_samples.append(sample)
            else:
                invalid_samples += 1
        
        if invalid_samples > 0:
            print(f"Warning: Found {invalid_samples} samples with invalid labels")
        
        self.samples = valid_samples
    """
    def _extract_text_from_image(self, image: Image.Image) -> str:
        #Extract text from image using TrOCR
        self._initialize_ocr()
        
        try:
            # Process image
            pixel_values = self.ocr_processor(image, return_tensors="pt").pixel_values
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.ocr_model.generate(pixel_values)
            
            return self.ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"OCR failed for image: {str(e)}")
            return ""  # Return empty string if OCR fails
    """

    def _extract_text_from_image(self, image: Image.Image) -> str:
        """Extract text from image using the configured OCR engine"""
        ocr_manager = OCRManager()
        engine = ocr_manager.get_ocr_engine(self.ocr_engine_type, **self.ocr_kwargs)
        
        try:
            if self.ocr_engine_type == "trocr":
                # Process image
                pixel_values = engine["processor"](image, return_tensors="pt").pixel_values
                
                # Generate text
                with torch.no_grad():
                    generated_ids = engine["model"].generate(pixel_values)
                return engine["processor"].batch_decode(generated_ids, skip_special_tokens=True)[0]
                
            elif self.ocr_engine_type == "tesseract":
                return engine["engine"].image_to_string(image)
                
            elif self.ocr_engine_type == "easyocr":
                result = engine["reader"].readtext(np.array(image))
                return " ".join([text for _, text, _ in result])
                
        except Exception as e:
            print(f"OCR failed for image: {str(e)}")
            return ""  # Return empty string if OCR fails
        
    def _load_samples(self):
        print(f"Loading dataset from: {self.data_dir}")
        print(f"Filtering for classes: {self.class_list}")
        
        class_set = set(self.class_list) if self.class_list else set()
        valid_samples = 0
        skipped_samples = 0
        
        for root, _, files in os.walk(self.data_dir):
            label = os.path.basename(root)
            
            # Skip if not in our class list
            if label not in class_set:
                continue
                
            for img_file in files:
                if img_file.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(root, img_file)
                    
                    try:
                        # Load image
                        image = Image.open(img_path).convert("RGB")
                        
                        # Extract text
                        extracted_text = self._extract_text_from_image(image)
                        if not extracted_text.strip():
                            print(f"Empty text extracted from {img_path}")
                            skipped_samples += 1
                            continue
                            
                        # Tokenize text
                        tokenized_text = self.tokenizer(
                            extracted_text,
                            padding="max_length",
                            truncation=True,
                            max_length=128,
                            return_tensors="pt"
                        )
                        
                        self.samples.append((img_path, tokenized_text, label))
                        valid_samples += 1
                        
                    except Exception as e:
                        print(f"Error processing {img_path}: {str(e)}")
                        skipped_samples += 1
                        continue
        
        print(f"Loaded {valid_samples} valid samples")
        print(f"Skipped {skipped_samples} samples due to errors")
        if valid_samples % 100 == 0:
            print(f"Processed {valid_samples} samples...")

    def _build_class_mappings(self):
        """Build class mappings only for specified classes"""
        if not self.class_list:
            raise ValueError("Class list must be provided")
            
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.class_list))}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(sorted(self.class_list))}
        print(f"Class mappings created for {len(self.class_to_idx)} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        img_path, text_data, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
                
            return {
                'image': image,
                'text': {
                    'input_ids': text_data['input_ids'].squeeze(0),
                    'attention_mask': text_data['attention_mask'].squeeze(0)
                },
                'label': torch.tensor(self.class_to_idx[label])
            }
        except Exception as e:
            print(f"Error loading sample {img_path}: {str(e)}")
            raise

def eaml_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for EAML dataset
    """
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'texts': {
            'input_ids': torch.stack([item['text']['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['text']['attention_mask'] for item in batch])
        },
        'labels': torch.stack([item['label'] for item in batch])
    }

class EAML_DataLoader:
    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4,
                 img_size: int = 224, class_list: Optional[List[str]] = None, ocr_engine="trocr", **ocr_kwargs):
        if not class_list:
            raise ValueError("Class list cannot be empty")
            
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.class_list = sorted(class_list)

        self.ocr_engine = ocr_engine
        self.ocr_kwargs = ocr_kwargs
        
        # Build class mappings once for all splits
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_list)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(self.class_list)}

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def get_loader(self, split: str, shuffle: bool = True):
        """Get loader for specific split with class validation"""
        dataset_path = os.path.join(self.data_dir, split)
        if not os.path.exists(dataset_path):
            raise ValueError(f"Split directory does not exist: {dataset_path}")
        """
        dataset = EAML_Dataset(
            data_dir=dataset_path,
            transform=self.transform,
            class_list=self.class_list
        )
        """

        dataset = EAML_Dataset(
            data_dir=dataset_path,
            transform=self.transform,
            class_list=self.class_list,
            ocr_engine=self.ocr_engine,
            **self.ocr_kwargs
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=eaml_collate_fn
        )

    def get_class_stats(self):
        """Get statistics about class distribution across splits"""
        stats = {}
        for split in ['train', 'val', 'test']:
            try:
                loader = self.get_loader(split, shuffle=False)
                class_counts = {cls: 0 for cls in self.class_list}
                for _, _, label in loader.dataset.samples:
                    class_counts[label] += 1
                stats[split] = class_counts
            except ValueError as e:
                print(f"Skipping {split} split: {str(e)}")
                continue
        return stats

    def get_class_mappings(self) -> Dict:
        """
        Get class to index mappings
        """
        sample_dataset = EAML_Dataset(
            data_dir=os.path.join(self.data_dir, 'train'),
            transform=self.transform,
            class_list=self.class_list
        )
        return {
            'class_to_idx': sample_dataset.class_to_idx,
            'idx_to_class': sample_dataset.idx_to_class,
            'classes': sample_dataset.class_list
        }

def eaml_collate_fn(batch: List[Dict]) -> Dict:
    return {
        'images': torch.stack([item['image'] for item in batch]),
        'texts': {
            'input_ids': torch.stack([item['text']['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['text']['attention_mask'] for item in batch])
        },
        'labels': torch.stack([item['label'] for item in batch])
    }

# ==================== DocFormer Components ====================
class DocFormerDataset(Dataset):
    """
    DocFormer-compatible dataset
    """
    def __init__(self, data_dir, tokenizer_name="bert-base-uncased", max_seq_length=512, split="train"):
        self.data_dir = os.path.join(data_dir, split)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.split = split
        
        self.samples = []
        self.class_names = []
        self._load_samples()
        
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(set(self.class_names)))}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_samples(self):
        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
                        self.samples.append({
                            'image_path': os.path.join(class_dir, img_file),
                            'class_name': class_name
                        })
                        self.class_names.append(class_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        pixel_values = self.transform(image)
        
        input_ids = torch.tensor([self.tokenizer.cls_token_id] + 
                               [self.tokenizer.pad_token_id] * (self.max_seq_length - 2) +
                               [self.tokenizer.sep_token_id])[:self.max_seq_length]
        
        attention_mask = torch.tensor([1] + [0] * (self.max_seq_length - 2) + [1])[:self.max_seq_length]
        
        bboxes = torch.tensor([[0, 0, image.width, 0, image.width, image.height, 0, image.height]] * self.max_seq_length)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'bboxes': bboxes,
            'label': torch.tensor(self.class_to_idx[sample['class_name']])
        }

def docformer_collate_fn(batch):
    """
    Collate function for DocFormer dataset
    """
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'bboxes': torch.stack([x['bboxes'] for x in batch]),
        'labels': torch.stack([x['label'] for x in batch])
    }

class DataLoader:
    """
    Original DocFormer DataLoader implementation"""
    def __init__(self, data_dir, batch_size=32, num_workers=4, img_size=224):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

    def get_loader(self, split='train', shuffle=True):
        dataset = DocFormerDataset(
            data_dir=self.data_dir,
            split=split
        )
        
        return TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=docformer_collate_fn
        )

    def get_class_mappings(self):
        sample_dataset = DocFormerDataset(
            data_dir=self.data_dir,
            split='train'
        )
        return {
            'class_to_idx': sample_dataset.class_to_idx,
            'idx_to_class': sample_dataset.idx_to_class,
            'classes': list(sample_dataset.class_to_idx.keys())
        }

# ==================== Utility Functions ====================
def load_class_list(class_list_path: Optional[str] = None) -> Optional[List[str]]:
    """
    Load class list from JSON file
    
    Args:
        class_list_path: Path to JSON file containing class list
    Returns:
        List of class names or None if file not found
    """
    if class_list_path and os.path.exists(class_list_path):
        with open(class_list_path) as f:
            return json.load(f)
    return None