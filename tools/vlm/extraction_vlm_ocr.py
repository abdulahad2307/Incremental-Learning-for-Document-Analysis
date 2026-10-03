import os
import json
import pickle
import gc
import psutil
from tqdm import tqdm
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from transformers import Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings("ignore")

def aggressive_memory_cleanup():
    """Aggressive memory cleanup for HPC environments"""
    import gc
    import torch
    
    # Clearing Python garbage
    for _ in range(3):
        gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def log_memory_usage(step_name=""):
    """Log current memory usage"""
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_cached = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"[{step_name}] GPU Memory: {gpu_memory:.2f}GB allocated, {gpu_cached:.2f}GB cached")
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    ram_usage = memory_info.rss / 1024**3  # GB
    print(f"[{step_name}] RAM Usage: {ram_usage:.2f}GB")

def load_preprocessed_data(preprocessed_dir):
    """Load preprocessed images and metadata"""
    print("Loading preprocessed data...")
    
    metadata_path = os.path.join(preprocessed_dir, "image_metadata.json")
    preprocessed_path = os.path.join(preprocessed_dir, "preprocessed_data.pkl")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    with open(preprocessed_path, 'rb') as f:
        preprocessed_data = pickle.load(f)
    
    print(f"Loaded metadata for {len(metadata)} images")
    print(f"Loaded preprocessed data for {len(preprocessed_data)} images")
    
    return metadata, preprocessed_data

def process_microbatch_vlm(
    metadata_batch,
    preprocessed_data,
    model,
    tokenizer,
    processor,
    vlm_model,
    prompt,
    generation_config,
    max_length,
    use_filename_key,
    device
):
    """Process a single microbatch of images"""
    batch_results = {}
    
    microbatch_pbar = tqdm(metadata_batch, desc=f"Processing Microbatch", leave=False)
    
    for img_meta in microbatch_pbar:
        try:
            filename = img_meta["filename"]
            microbatch_pbar.set_postfix({"Current": filename[:20] + "..."})
            
            if filename not in preprocessed_data:
                print(f"Preprocessed data not found for {filename}")
                continue

            extracted_text = ""
            
            # VLM models
            if vlm_model == "internvl2-2b":
                pixel_values = preprocessed_data[filename].to(torch.bfloat16).to(device)
                question = f"<image>\n{prompt}"
                
                with torch.no_grad():
                    response = model.chat(tokenizer, pixel_values, question, generation_config)
                extracted_text = response
                
            elif vlm_model == "qwen2-vl":
                image = preprocessed_data[filename]
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                # Preparing for inference using the updated approach
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                
                # Inference- Generation of the output
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, **generation_config)
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                extracted_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                
            elif vlm_model == "minicpm-v":
                image = preprocessed_data[filename]
                msgs = [{'role': 'user', 'content': [image, prompt]}]
                
                with torch.no_grad():
                    res = model.chat(
                        image=None,
                        msgs=msgs,
                        tokenizer=tokenizer,
                        **generation_config
                    )
                extracted_text = res
            
            # Tokenizing the extracted text
            tokenized = tokenizer(
                extracted_text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Storing results
            key = filename if use_filename_key else img_meta["full_path"]
            batch_results[key] = {
                "text": extracted_text,
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0)
            }
            
            # Cleanup variables
            if 'pixel_values' in locals():
                del pixel_values
            if 'inputs' in locals():
                del inputs
            if 'generated_ids' in locals():
                del generated_ids
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            key = filename if use_filename_key else img_meta["full_path"]
            batch_results[key] = {
                "text": "",
                "input_ids": torch.tensor([]),
                "attention_mask": torch.tensor([])
            }
    
    microbatch_pbar.close()
    return batch_results

def extract_text_microbatch(
    preprocessed_dir,
    output_pt,
    vlm_model="qwen2-vl",
    model_path=None,
    offset=0,
    max_images=None,
    max_length=200,
    extract_mode="description",
    microbatch_size=50,
    use_filename_key=False,
    save_intermediate=True,
    intermediate_save_freq=10
):
    """
    Extract text from preprocessed images using micro-batch processing with CUDA GPU
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available! This script requires GPU.")
    
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    log_memory_usage("Initial")
    
    metadata, preprocessed_data = load_preprocessed_data(preprocessed_dir)
    log_memory_usage("After loading data")
    
    # Applying offset and limit
    if offset > 0:
        metadata = metadata[offset:]
        print(f"Applied offset {offset}, remaining images: {len(metadata)}")
    if max_images is not None:
        metadata = metadata[:max_images]
        print(f"Applied limit {max_images}, processing images: {len(metadata)}")
    
    print(f"Processing {len(metadata)} images with {vlm_model} using micro-batches of {microbatch_size}")
    
    # Initializing VLM model on GPU
    print(f"Loading {vlm_model} model...")
    processor = None
    
    if vlm_model == "internvl2-2b":
        model_path = model_path or "OpenGVLab/InternVL2-2B"
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
    elif vlm_model == "qwen2-vl":
        model_path = model_path or "Qwen/Qwen2-VL-2B-Instruct"
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map="auto"
        )
        
        # Configuring processor with pixel limits for memory efficiency
        min_pixels = 256 * 28 * 28
        max_pixels = 1280 * 28 * 28
        processor = AutoProcessor.from_pretrained(
            model_path, 
            min_pixels=min_pixels, 
            max_pixels=max_pixels
        )
        tokenizer = processor.tokenizer
        
    elif vlm_model == "minicpm-v":
        model_path = model_path or "openbmb/MiniCPM-V-2_6"
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval().to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    log_memory_usage("After model loading")
    
    # Propmt based on extraction mode
    if extract_mode == "ocr":
        prompt = "Please extract all the text content from this document image. Focus on accurately transcribing all visible text, maintaining the original structure and formatting as much as possible."
    elif extract_mode == "description":
        prompt = "Please provide a detailed description of this document image, including its layout, structure, content type, and any notable visual elements."
    elif extract_mode == "both":
        prompt = "Please extract all the text content from this document image and also provide a brief description of the document's layout and structure."
    
    # Generation configuration
    generation_config = {
        "num_beams": 1,
        "max_new_tokens": max_length,
        "do_sample": False,
        "temperature": 0.1,
        "top_p": 0.9
    }
    
    total_microbatches = (len(metadata) + microbatch_size - 1) // microbatch_size
    print(f"Total microbatches to process: {total_microbatches}")
    
    vlm_dict = {}
    processed_images = 0
    
    main_pbar = tqdm(total=len(metadata), desc="Overall Progress", position=0)
    
    for batch_idx in range(total_microbatches):
        start_idx = batch_idx * microbatch_size
        end_idx = min(start_idx + microbatch_size, len(metadata))
        metadata_batch = metadata[start_idx:end_idx]
        
        print(f"\nProcessing microbatch {batch_idx + 1}/{total_microbatches} (images {start_idx} to {end_idx-1})")
        log_memory_usage(f"Microbatch {batch_idx + 1} start")
        
        batch_results = process_microbatch_vlm(
            metadata_batch=metadata_batch,
            preprocessed_data=preprocessed_data,
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            vlm_model=vlm_model,
            prompt=prompt,
            generation_config=generation_config,
            max_length=max_length,
            use_filename_key=use_filename_key,
            device=device
        )
        
        vlm_dict.update(batch_results)
        processed_images += len(batch_results)
        
        main_pbar.update(len(metadata_batch))
        main_pbar.set_postfix({
            "Processed": processed_images,
            "GPU Mem": f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB"
        })
        
        aggressive_memory_cleanup()
        log_memory_usage(f"Microbatch {batch_idx + 1} end")
        
        if save_intermediate and (batch_idx + 1) % intermediate_save_freq == 0:
            intermediate_file = output_pt.replace(".pt", f"_intermediate_{batch_idx + 1}.pt")
            torch.save(vlm_dict, intermediate_file)
            print(f"Saved intermediate results to {intermediate_file}")
    
    main_pbar.close()
    
    torch.save(vlm_dict, output_pt)
    print(f"\nCompleted! Saved VLM extracted text for {len(vlm_dict)} images to {output_pt}")
    log_memory_usage("Final")
    
    return len(vlm_dict)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text from preprocessed images using micro-batch processing")
    parser.add_argument("--preprocessed_dir", required=True, help="Directory with preprocessed data")
    parser.add_argument("--output_pt", required=True, help="Output .pt file path")
    parser.add_argument("--vlm_model", default="qwen2-vl", 
                        choices=["internvl2-2b", "qwen2-vl", "minicpm-v"],
                        help="VLM model to use")
    parser.add_argument("--model_path", default=None, help="Custom model path")
    parser.add_argument("--offset", type=int, default=0, help="Starting index")
    parser.add_argument("--max_images", type=int, default=None, help="Max images to process")
    parser.add_argument("--max_length", type=int, default=200, help="Max token length")
    parser.add_argument("--extract_mode", default="description",
                        choices=["ocr", "description", "both"],
                        help="Extraction mode")
    parser.add_argument("--microbatch_size", type=int, default=50, help="Size of each microbatch")
    parser.add_argument("--use_filename_key", action="store_true", help="Use filename as key")
    parser.add_argument("--save_intermediate", action="store_true", help="Save intermediate results")
    parser.add_argument("--intermediate_save_freq", type=int, default=10, help="Save intermediate results every N microbatches")
    
    args = parser.parse_args()
    
    extract_text_microbatch(
        preprocessed_dir=args.preprocessed_dir,
        output_pt=args.output_pt,
        vlm_model=args.vlm_model,
        model_path=args.model_path,
        offset=args.offset,
        max_images=args.max_images,
        max_length=args.max_length,
        extract_mode=args.extract_mode,
        microbatch_size=args.microbatch_size,
        use_filename_key=args.use_filename_key,
        save_intermediate=args.save_intermediate,
        intermediate_save_freq=args.intermediate_save_freq
    )
