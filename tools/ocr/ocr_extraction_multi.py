import os
import json
import time
import gc
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import torch
import cv2
from transformers import BertTokenizer, TrOCRProcessor, VisionEncoderDecoderModel
import argparse
import subprocess

def get_available_gpus(num_gpus):
    """Get top N available GPUs sorted by free memory"""
    try:
        # Get GPU memory info using nvidia-smi
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        
        # Parse output and calculate free memory
        gpu_info = []
        for line in output.strip().split('\n'):
            idx, mem = line.split(',')
            gpu_info.append((int(idx.strip()), int(mem.strip())))
        
        # Sort by free memory (descending)
        gpu_info.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N GPUs
        return [idx for idx, _ in gpu_info[:num_gpus]]
    
    except Exception as e:
        print(f"Error getting GPU info: {e}. Using CPU-only mode.")
        return []

def preprocess_image(img_path, max_size=640):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = thresh.shape
    scale = max(h, w) / max_size if max(h, w) > max_size else 1
    if scale > 1:
        new_h, new_w = int(h / scale), int(w / scale)
        thresh = cv2.resize(thresh, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    return thresh, rgb

def run_ocr(img_gray, img_rgb, engine, trocr_model=None, trocr_processor=None, easyocr_reader=None, pero_ocr=None):
    if engine == "tesseract":
        import pytesseract
        return pytesseract.image_to_string(img_gray, config='--oem 1 --psm 6')
    elif engine == "easyocr":
        import easyocr
        result = easyocr_reader.readtext(img_gray)
        return " ".join([t[1] for t in result])
    elif engine == "trocr":
        from pero_ocr.document_ocr import DocumentOCR
        from PIL import Image
        pil_img = Image.fromarray(img_rgb)
        pixel_values = trocr_processor(pil_img, return_tensors="pt").pixel_values
        
        # Use model's current device
        device = next(trocr_model.parameters()).device
        pixel_values = pixel_values.to(device)
        
        with torch.no_grad():
            generated_ids = trocr_model.generate(pixel_values)
        return trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    elif engine == "pero":
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        result = pero_ocr.run_ocr(img_bgr)
        return " ".join([line.transcription for page in result.pages
                         for region in page.regions for line in region.lines])
    return ""

def process_image(args):
    img_path, gpu_id, ocr_engines_list, ocr_kwargs, max_size, log_every, tokenizer, max_length, idx = args
    try:
        # Initialize OCR engines per process
        easyocr_reader = None
        trocr_processor, trocr_model = None, None
        pero_ocr = None
        
        # Set GPU for this process if available
        device = f"cuda:{gpu_id}" if gpu_id is not None and torch.cuda.is_available() else "cpu"
        
        if "easyocr" in ocr_engines_list:
            import easyocr
            easyocr_reader = easyocr.Reader(ocr_kwargs.get("languages", ["en"]))
        if "trocr" in ocr_engines_list:
            trocr_processor = TrOCRProcessor.from_pretrained(
                ocr_kwargs.get("model_name", "microsoft/trocr-base-handwritten")
            )
            trocr_model = VisionEncoderDecoderModel.from_pretrained(
                ocr_kwargs.get("model_name", "microsoft/trocr-base-handwritten")
            )
            trocr_model = trocr_model.to(device)
            trocr_model.eval()
        if "pero" in ocr_engines_list:
            from pero_ocr.document_ocr import DocumentOCR
            config_path = ocr_kwargs.get("config_path", None)
            if config_path and os.path.exists(config_path):
                pero_ocr = DocumentOCR(config_path=config_path)
            else:
                pero_ocr = DocumentOCR()
        
        # Preprocess and run OCR
        img_gray, img_rgb = preprocess_image(img_path, max_size)
        best_text = ""
        for engine in ocr_engines_list:
            try:
                text_candidate = run_ocr(
                    img_gray, img_rgb, engine,
                    trocr_model=trocr_model,
                    trocr_processor=trocr_processor,
                    easyocr_reader=easyocr_reader,
                    pero_ocr=pero_ocr
                )
                if idx % log_every == 0:
                    print(f"[GPU:{gpu_id}][{engine}] OCR sample: {text_candidate[:100]}")
                if len(text_candidate.strip()) > len(best_text):
                    best_text = text_candidate.strip()
            except Exception as e:
                print(f"OCR failed: {e}")
        
        # Handle empty text
        if len(best_text) < 10 or best_text.isspace():
            text = "Empty Text"
        else:
            text = best_text
        
        # Tokenize if needed
        input_ids, attn_mask = None, None
        if tokenizer:
            tokens = tokenizer(
                text if text != "Empty Text" else "[PAD]",
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            input_ids = tokens['input_ids'].squeeze(0)
            attn_mask = tokens['attention_mask'].squeeze(0)
        
        return {
            "img_path": img_path,
            "text": text,
            "input_ids": input_ids,
            "attn_mask": attn_mask
        }
    except Exception as e:
        print(f"Processing failed: {e}")
        return {
            "img_path": img_path,
            "text": "Empty Text",
            "input_ids": None,
            "attn_mask": None
        }
    finally:
        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description="Parallel OCR Extraction with Auto GPU Selection")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_pt', type=str, default=None)
    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument('--ocr_engine', type=str, default='trocr', choices=['trocr', 'tesseract', 'easyocr', 'pero'])
    parser.add_argument('--ocr_engines', nargs='+', default=None)
    parser.add_argument('--ocr_model', type=str, default='microsoft/trocr-base-handwritten')
    parser.add_argument('--ocr_lang', nargs='+', default=['en'])
    parser.add_argument('--max_size', type=int, default=640)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--max_images', type=int, default=None)
    parser.add_argument('--pero_config', type=str, default=None)
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--tokenizer_name', type=str, default='bert-base-uncased')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_processes', type=int, default=4)
    parser.add_argument('--num_gpus', type=int, default=0, help='Number of GPUs to use (0 for CPU-only)')
    args = parser.parse_args()
    
    # Configure OCR
    ocr_kwargs = {}
    engines = args.ocr_engines if args.ocr_engines else [args.ocr_engine]
    if "trocr" in engines:
        ocr_kwargs["model_name"] = args.ocr_model
    if "easyocr" in engines:
        ocr_kwargs["languages"] = args.ocr_lang
    if "pero" in engines:
        ocr_kwargs["config_path"] = args.pero_config
    
    print(f"Using OCR engines: {engines}")
    
    # Collect image paths
    all_image_paths = []
    for root, _, files in os.walk(args.data_dir):
        for fname in files:
            if fname.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
                all_image_paths.append(os.path.join(root, fname))
    
    if args.offset > 0:
        all_image_paths = all_image_paths[args.offset:]
    if args.max_images:
        all_image_paths = all_image_paths[:args.max_images]
    
    print(f"Processing {len(all_image_paths)} images")
    
    # Automatic GPU selection
    if args.num_gpus > 0:
        gpu_ids = get_available_gpus(args.num_gpus)
        print(f"Selected GPUs: {gpu_ids}")
        
        # Set visible GPUs for this process
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
        
        # Map worker IDs to GPU indices
        worker_gpu_ids = [i % len(gpu_ids) for i in range(args.num_processes)]
    else:
        print("Using CPU-only mode")
        gpu_ids = []
        worker_gpu_ids = [None] * args.num_processes
    
    # Tokenizer initialization
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name) if args.output_pt else None
    
    # Prepare arguments
    args_list = []
    for idx, img_path in enumerate(all_image_paths):
        worker_id = idx % args.num_processes
        gpu_id = worker_gpu_ids[worker_id] if worker_id < len(worker_gpu_ids) else None
        args_list.append((
            img_path,
            gpu_id,
            engines,
            ocr_kwargs,
            args.max_size,
            args.log_every,
            tokenizer,
            args.max_length,
            idx
        ))
    
    # Parallel processing
    start_time = time.time()
    with mp.get_context('spawn').Pool(processes=args.num_processes) as pool:
        results = list(tqdm(pool.imap(process_image, args_list), total=len(all_image_paths), desc="OCR Extraction"))
    
    # Process results
    ocr_json = {res["img_path"]: res["text"] for res in results}
    empty_count = sum(1 for res in results if res["text"] == "Empty Text")
    print(f"Empty texts: {empty_count}/{len(results)}")
    
    # Save outputs
    if args.output_pt:
        input_ids = [res["input_ids"] for res in results if res["input_ids"] is not None]
        attn_mask = [res["attn_mask"] for res in results if res["attn_mask"] is not None]
        if input_ids and attn_mask:
            tensor_data = {
                "input_ids": torch.stack(input_ids),
                "attention_mask": torch.stack(attn_mask),
                "image_paths": [res["img_path"] for res in results],
                "texts": [res["text"] for res in results]
            }
            torch.save(tensor_data, args.output_pt)
            print(f"Saved tensor file: {args.output_pt}")
    
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(ocr_json, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON file: {args.output_json}")
    
    duration = time.time() - start_time
    print(f"Processed {len(results)} images in {duration:.2f}s")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
