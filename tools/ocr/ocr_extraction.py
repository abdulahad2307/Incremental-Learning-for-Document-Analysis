import os
import json
import time
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

def precompute_ocr(data_dir, output_json, ocr_engine="tesseract", ocr_kwargs=None, 
                   max_size=640, offset=0, max_images=None):
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
    
    # Apply offset and limit
    if offset > 0:
        all_image_paths = all_image_paths[offset:]
    if max_images is not None:
        all_image_paths = all_image_paths[:max_images]
    
    print(f"Processing {len(all_image_paths)} images (offset: {offset}, limit: {max_images})")
    
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
    elif ocr_engine == "pero":
        from pero_ocr.document_ocr import DocumentOCR
        config_path = ocr_kwargs.get("config_path", None)
        if config_path and os.path.exists(config_path):
            ocr_engine_pero = DocumentOCR(config_path=config_path)
        else:
            ocr_engine_pero = DocumentOCR()
    
    # Process images
    for img_path in tqdm(all_image_paths, desc="OCR Extraction"):
        try:
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
            elif ocr_engine == "pero":
                img_array = np.array(image)
                img_bgr = img_array[:, :, ::-1]  # Convert to BGR
                result = ocr_engine_pero.run_ocr(img_bgr)
                text = " ".join([line.transcription for page in result.pages 
                                for region in page.regions for line in region.lines])
            else:
                text = ""
            
            ocr_results[img_path] = text
            
            # Memory cleanup
            del image
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
    
    duration = time.time() - start_time
    print(f"OCR completed: {len(ocr_results)} images in {duration:.2f} seconds ({duration/len(all_image_paths):.2f}s/image)")
    return len(ocr_results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Memory-Optimized OCR Precomputation")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_json', type=str, required=True)
    parser.add_argument('--ocr_engine', type=str, default='trocr', 
                        choices=['trocr', 'tesseract', 'easyocr', 'pero', 'paddleocr'])
    parser.add_argument('--ocr_model', type=str, default='microsoft/trocr-base-handwritten')
    parser.add_argument('--ocr_lang', nargs='+', default=['en'])
    parser.add_argument('--max_size', type=int, default=1024)
    parser.add_argument('--offset', type=int, default=0, help='Starting image index')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum images to process')
    parser.add_argument('--pero_config', type=str, default=None, help='Pero-OCR config path')
    args = parser.parse_args()
    
    ocr_kwargs = {}
    if args.ocr_engine == "trocr":
        ocr_kwargs["model_name"] = args.ocr_model
    if args.ocr_engine in ["easyocr", "paddleocr"]:
        ocr_kwargs["languages"] = args.ocr_lang
    if args.ocr_engine == "pero":
        ocr_kwargs["config_path"] = args.pero_config
    
    print(f"Using the OCR: {args.ocr_engine}")
    
    precompute_ocr(
        args.data_dir,
        args.output_json,
        args.ocr_engine,
        ocr_kwargs,
        max_size=args.max_size,
        offset=args.offset,
        max_images=args.max_images
    )
