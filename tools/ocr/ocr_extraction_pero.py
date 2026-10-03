import os
import torch
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
from pero_ocr.document_ocr import DocumentOCR

def ocr_extraction_tokenization(
    data_dir,
    output_pt,
    ocr_engine="pero",
    ocr_kwargs=None,
    max_size=640,
    offset=0,
    max_images=None,
    tokenizer=None,
    max_length=128,
    use_filename_key=True
):
    if ocr_kwargs is None:
        ocr_kwargs = {}
    ocr_dict = {}

    # Collect image paths
    all_images = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
                all_images.append(os.path.join(root, file))

    # Apply offset and max_images limit
    if offset > 0:
        all_images = all_images[offset:]
    if max_images is not None:
        all_images = all_images[:max_images]

    # Initialize PERO OCR
    from pero_ocr.ocr_engine import OCREngine
    if ocr_engine == "pero":
        try:
            from pero_ocr.document_ocr import DocumentOCR
            
            # Try custom config path first
            config_path = ocr_kwargs.get("config_path", None)
            
            if config_path and os.path.exists(config_path):
                print(f"Using custom PERO config: {config_path}")
                ocr_engine_pero = DocumentOCR(config_path=config_path)
            else:
                print("Using default PERO config")
                ocr_engine_pero = DocumentOCR()  # Uses default config
                
            # Enable GPU if available
            if torch.cuda.is_available():
                ocr_engine_pero.use_gpu()
                
        except Exception as e:
            print(f"Failed to initialize PERO OCR: {e}")
            return 0
    # Initialize tokenizer
    if tokenizer is None:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for img_path in tqdm(all_images, desc="OCR Extraction"):
        try:
            image = Image.open(img_path).convert("RGB")
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.LANCZOS)
            # PERO expects numpy array in RGB
            img_array = np.array(image)
            result = ocr_engine_pero.process_page(image)
            texts = []
            bboxes = []
            for line in result.lines:
                if line.transcription:
                    texts.append(line.transcription)
                    # Normalize bbox to [0, 1] range
                    coords = line.geometry
                    norm_bbox = []
                    for point in coords:
                        norm_bbox.extend([point[0] / image.width, point[1] / image.height])
                    bboxes.append(norm_bbox)
            text = " ".join(texts)
            # Tokenization
            tokenized = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            key = os.path.basename(img_path) if use_filename_key else img_path
            ocr_dict[key] = {
                "text": text,
                "bboxes": bboxes,
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0)
            }
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            ocr_dict[img_path] = {
                "text": "[UNK]",
                "bboxes": [],
                "input_ids": torch.zeros(max_length, dtype=torch.long),
                "attention_mask": torch.zeros(max_length, dtype=torch.long)
            }
    torch.save(ocr_dict, output_pt)
    print(f"Saved tokenized OCR for {len(ocr_dict)} images to {output_pt}")
    return len(ocr_dict)

def main():
    parser = argparse.ArgumentParser(description="OCR Extraction and Tokenization for DocFormer")
    parser.add_argument("--data_dir", required=True, help="Directory with images")
    parser.add_argument("--output_pt", required=True, help="Output .pt file path or directory")
    parser.add_argument("--ocr_engine", default="pero", choices=["tesseract", "trocr", "easyocr", "pero", "paddleocr"])
    parser.add_argument("--max_size", type=int, default=640, help="Max image dimension")
    parser.add_argument("--offset", type=int, default=0, help="Starting index for image processing")
    parser.add_argument("--max_images", type=int, default=None, help="Max number of images to process")
    parser.add_argument("--use_filename_key", action="store_true", help="Use filename as dictionary key")
    parser.add_argument("--ocr_config_path", type=str, default=None, help="Path to PERO OCR config.yaml (if needed)")
    args = parser.parse_args()

    ocr_kwargs = {}
    if args.ocr_engine == "pero" and args.ocr_config_path:
        ocr_kwargs["config_path"] = args.ocr_config_path

    ocr_extraction_tokenization(
        data_dir=args.data_dir,
        output_pt=args.output_pt,
        ocr_engine=args.ocr_engine,
        ocr_kwargs=ocr_kwargs,
        max_size=args.max_size,
        offset=args.offset,
        max_images=args.max_images,
        use_filename_key=args.use_filename_key
    )

if __name__ == "__main__":
    main()
