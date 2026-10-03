#!/usr/bin/env python3
"""
Offset VLM Extractor - Raw Images with Offset/Max Control
No preprocessing needed - directly processes images from folder
"""

import os
import torch
import argparse
import glob
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings("ignore")

def extract_offset_images(images_dir, output_pt, offset=0, max_images=None, model_name="Qwen/Qwen2-VL-2B-Instruct", max_new_tokens=200, device=None):
    """Extract content/layout from raw images with offset/max_images slicing"""
    
    # Find all images (case insensitive)
    image_extensions = ['*.jpg', '*.jpeg', '*.png','*.tif' , '*.tiff', '*.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(images_dir, ext)))
        all_images.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    all_images = sorted(set(all_images))  # Unique + sorted
    
    print(f"Found {len(all_images)} images in {images_dir}")
    
    # Apply offset and max_images (exact match to your original logic)
    if offset > 0:
        all_images = all_images[offset:]
        print(f"Applied offset {offset}, remaining: {len(all_images)}")
    if max_images is not None:
        all_images = all_images[:max_images]
        print(f"Applied max_images {max_images}, processing: {len(all_images)}")
    
    if not all_images:
        print("No images to process!")
        torch.save({}, output_pt)
        return {}
    
    # Setup device and model
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading model...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(
        model_name, 
        min_pixels=256*28*28, 
        max_pixels=1280*28*28
    )
    
    # Prompt (content + layout like your "both" mode)
    prompt = (
        "Extract all visible text content from this document image accurately. "
        "Also describe the layout, structure, content type, and visual elements. "
        "Maintain original formatting where possible."
    )
    
    results = {}
    for img_path in all_images:
        try:
            filename = os.path.basename(img_path)
            print(f"Processing: {filename}")
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": prompt}
                ]
            }]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt"
            ).to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=0.1,
                    num_beams=1
                )
            
            generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
            extracted_text = processor.batch_decode(
                [generated_ids_trimmed], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            results[filename] = {
                "image_path": img_path,
                "text": extracted_text
            }
            print(f"{len(extracted_text)} chars")
            
        except Exception as e:
            print(f"  ✗ Error {filename}: {e}")
            results[filename] = {"image_path": img_path, "text": ""}
    
    torch.save(results, output_pt)
    print(f"Saved {len(results)} results to {output_pt}")

    txt_path = output_pt.replace('.pt', '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"VLM Extraction Results\n")
        f.write(f"Images processed: {len(results)}\n")
        f.write(f"Source dir: {images_dir}\n")
        f.write("="*80 + "\n\n")
        
        for filename, data in results.items():
            f.write(f"--- {filename} ---\n")
            f.write(f"Path: {data['image_path']}\n")
            f.write(f"Text: {data['text']}\n")
            f.write("-" * 80 + "\n\n")

    print(f"Saved readable TXT to {txt_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offset VLM extraction from raw images")
    parser.add_argument("--images_dir", required=True, help="Directory with raw images")
    parser.add_argument("--output_pt", required=True, help="Output .pt file")
    parser.add_argument("--offset", type=int, default=0, help="Starting image index")
    parser.add_argument("--max_images", type=int, default=None, help="Max images to process")
    parser.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct", help="Model name/path")
    parser.add_argument("--max_tokens", type=int, default=200, help="Max generation tokens")
    parser.add_argument("--device", help="Force device (cuda/cpu)")
    
    args = parser.parse_args()
    
    # FIXED - EXPLICIT ARGS (no **vars)
    extract_offset_images(
        images_dir=args.images_dir,
        output_pt=args.output_pt,
        offset=args.offset,
        max_images=args.max_images,
        model_name=args.model,           # CLI --model → func model_name
        max_new_tokens=args.max_tokens,
        device=args.device
    )
