import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from torchvision import transforms
import pytesseract

class RVLCDIPDataset(Dataset):
    """
    Dataset class for RVL-CDIP documents
    Directory structure: data/{train,val,test}/class_name/image_name.tif
    """
    def __init__(self, data_dir, tokenizer_name="bert-base-uncased", max_seq_length=512, split="train"):
        self.data_dir = os.path.join(data_dir, split)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_length = max_seq_length
        self.split = split
        
        # Collect all samples
        self.samples = []
        self.class_names = []
        self._load_samples()
        
        # Create label mappings
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(set(self.class_names)))}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_samples(self):
        """Load all samples from the directory structure"""
        for class_name in os.listdir(self.data_dir):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                        self.samples.append({
                            'image_path': os.path.join(class_dir, img_file),
                            'class_name': class_name
                        })
                        self.class_names.append(class_name)

    def __len__(self):
        return len(self.samples)

    def _extract_text_and_bboxes(self, image):
        """Extract text and bounding boxes using Tesseract OCR"""
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words, bboxes = [], []
        
        for i in range(len(ocr_data['text'])):
            word = ocr_data['text'][i].strip()
            if word:  # Only consider non-empty words
                # Get bounding box (left, top, width, height)
                left = ocr_data['left'][i]
                top = ocr_data['top'][i]
                right = left + ocr_data['width'][i]
                bottom = top + ocr_data['height'][i]
                
                # Convert to quadrilateral format (8 coordinates)
                bbox = [
                    left, top,       # top-left
                    right, top,      # top-right
                    right, bottom,   # bottom-right
                    left, bottom     # bottom-left
                ]
                
                words.append(word)
                bboxes.append(bbox)
        
        return words, bboxes

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Process image
        pixel_values = self.transform(image)
        
        # Extract OCR data
        words, bboxes = self._extract_text_and_bboxes(image)
        
        # Tokenize and align bounding boxes
        input_ids = []
        bbox_tensors = []
        attention_mask = []
        
        # Add [CLS] token
        input_ids.append(self.tokenizer.cls_token_id)
        bbox_tensors.append([0]*8)  # Dummy bbox for special tokens
        attention_mask.append(1)
        
        # Process each word
        for word, bbox in zip(words, bboxes):
            word_tokens = self.tokenizer.tokenize(word)
            token_ids = self.tokenizer.convert_tokens_to_ids(word_tokens)
            
            # Extend bbox for each token
            for _ in word_tokens:
                bbox_tensors.append(bbox)
                attention_mask.append(1)
            input_ids.extend(token_ids)
        
        # Add [SEP] token
        input_ids.append(self.tokenizer.sep_token_id)
        bbox_tensors.append([0]*8)
        attention_mask.append(1)
        
        # Truncate/pad sequences
        input_ids = input_ids[:self.max_seq_length]
        attention_mask = attention_mask[:self.max_seq_length]
        bbox_tensors = bbox_tensors[:self.max_seq_length]
        
        padding_length = self.max_seq_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        bbox_tensors += [[0]*8] * padding_length
        
        return {
            'pixel_values': pixel_values,
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask),
            'bboxes': torch.tensor(bbox_tensors),
            'label': torch.tensor(self.class_to_idx[sample['class_name']])
        }

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'bboxes': torch.stack([x['bboxes'] for x in batch]),
        'labels': torch.stack([x['label'] for x in batch])
    }