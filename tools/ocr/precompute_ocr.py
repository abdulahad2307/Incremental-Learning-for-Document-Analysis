import os
import json
import time
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

def precompute_ocr(data_dir, output_json, ocr_engine="trocr", ocr_kwargs=None, max_size=640):
    if ocr_kwargs is None:
        ocr_kwargs = {}
    ocr_results = {}
    start_time = time.time()
    
    # Collect image paths
    all_image_paths = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
                all_image_paths.append(os.path.join(root, fname))
    
    print(f"Found {len(all_image_paths)} images for OCR extraction.")
    
    # Initialize OCR engine
    if ocr_engine == "trocr":
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = TrOCRProcessor.from_pretrained(ocr_kwargs.get("model_name", "microsoft/trocr-base-handwritten"))
        model = VisionEncoderDecoderModel.from_pretrained(ocr_kwargs.get("model_name", "microsoft/trocr-base-handwritten")).to(device)
        model.eval()
    elif ocr_engine == "tesseract":
        import pytesseract
    elif ocr_engine == "easyocr":
        import easyocr
        reader = easyocr.Reader(ocr_kwargs.get("languages", ["en"]))
    
    # Process images one-by-one with aggressive memory cleanup
    for img_path in tqdm(all_image_paths, desc="OCR Extraction"):
        try:
            # Load and resize image
            image = Image.open(img_path).convert("RGB")
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.LANCZOS)
            
            # Process with selected OCR engine
            if ocr_engine == "trocr":
                pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
                with torch.no_grad():
                    generated_ids = model.generate(pixel_values)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            elif ocr_engine == "tesseract":
                text = pytesseract.image_to_string(image)
            elif ocr_engine == "easyocr":
                result = reader.readtext(np.array(image))
                text = " ".join([text for _, text, _ in result])
            else:
                text = ""
            
            ocr_results[img_path] = text
            
            # Aggressive memory cleanup
            del image, text
            if 'pixel_values' in locals(): del pixel_values
            if 'generated_ids' in locals(): del generated_ids
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            ocr_results[img_path] = ""
    
    # Save results
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, ensure_ascii=False, indent=2)
    
    print(f"OCR extraction completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Memory-Optimized OCR Precomputation")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--ocr_engine', type=str, default='trocr', choices=['trocr', 'tesseract', 'easyocr'])
    parser.add_argument('--ocr_model', type=str, default='microsoft/trocr-base-handwritten')
    parser.add_argument('--ocr_lang', nargs='+', default=['en'])
    parser.add_argument('--max_size', type=int, default=640, help='Max image dimension')
    args = parser.parse_args()
    
    ocr_kwargs = {}
    if args.ocr_engine == "trocr":
        ocr_kwargs["model_name"] = args.ocr_model
    if args.ocr_engine == "easyocr":
        ocr_kwargs["languages"] = args.ocr_lang
    
    precompute_ocr(
        args.data_dir,
        args.output_json,
        args.ocr_engine,
        ocr_kwargs,
        max_size=args.max_size
    )
