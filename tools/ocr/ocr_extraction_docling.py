import os
import json
import time
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True

def process_batch_docling(images_batch, processor, model, device, ocr_kwargs):
    """Process multiple images in a single batch for Docling OCR"""
    batch_results = []
    
    with torch.amp.autocast('cuda'):
        for image in images_batch:
            try:
                instruction = ocr_kwargs.get("instruction", "Convert this page to docling.")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": instruction}
                        ]
                    }
                ]
                
                # Processing the image and text
                input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(
                    text=input_text,
                    images=[image],
                    return_tensors="pt",
                    truncation=False  # False to avoid token mismatch
                ).to(device)
                
                # Text generation with the model
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=ocr_kwargs.get("max_new_tokens", 2048),
                        do_sample=False
                    )
                
                # Decoding the generated text
                generated_text = processor.batch_decode(
                    generated_ids[:, inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )[0]
                
                # Extracting text from DocTags format
                try:
                    from docling_core.types.doc import DocTagsDocument
                    doc_tags = DocTagsDocument.model_validate_json(generated_text)
                    docling_doc = doc_tags.export_to_document()
                    text = docling_doc.export_to_markdown()
                except:
                    text = generated_text # Raw text
                
                batch_results.append(text)
                
                # Cleaning up intermediate variables
                del inputs, generated_ids
                
            except Exception as e:
                print(f"Error in batch processing: {str(e)}")
                batch_results.append("")
    
    return batch_results

def ocr_extraction_tokenization(
    data_dir,
    output_pt,
    ocr_engine="tesseract",
    ocr_kwargs=None,
    max_size=1024,
    offset=0,
    max_images=None,
    tokenizer=None,
    max_length=4096,
    use_filename_key=True,
    batch_size=4
):
    if ocr_kwargs is None:
        ocr_kwargs = {}
    ocr_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    all_images = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
                all_images.append(os.path.join(root, file))

    if offset > 0:
        all_images = all_images[offset:]
    if max_images is not None:
        all_images = all_images[:max_images]
    
    print(f"Processing {len(all_images)} images")

    # Initializing OCR engine with optimizations
    if ocr_engine == "trocr":
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        processor = TrOCRProcessor.from_pretrained(ocr_kwargs.get("model_name", "microsoft/trocr-base-handwritten"))
        model = VisionEncoderDecoderModel.from_pretrained(ocr_kwargs.get("model_name", "microsoft/trocr-base-handwritten")).to(device)
        model.eval()
        
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model)
            
    elif ocr_engine == "tesseract":
        import pytesseract
        
    elif ocr_engine == "easyocr":
        import easyocr
        reader = easyocr.Reader(ocr_kwargs.get("languages", ["en"]), gpu=True)
        
    elif ocr_engine == "pero":
        from pero_ocr.document_ocr import DocumentOCR
        config_path = ocr_kwargs.get("config_path", None)
        ocr_engine_pero = DocumentOCR(config_path=config_path) if config_path else DocumentOCR()
        
    elif ocr_engine == "docling_ocr":
        from transformers import AutoProcessor, AutoModelForVision2Seq
        from docling_core.types.doc import DoclingDocument
        
        model_name = ocr_kwargs.get("model_name", "ds4sd/SmolDocling-256M-preview")
        processor = AutoProcessor.from_pretrained(model_name)
        
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        ).to(device)
        model.eval()
        
        # Compiling model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model)

    # Initializing tokenizer
    if tokenizer is None:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if ocr_engine == "docling_ocr" and batch_size > 1:
        for i in tqdm(range(0, len(all_images), batch_size), desc="OCR Extraction (Batched)"):
            batch_paths = all_images[i:i + batch_size]
            batch_images = []
            
            try:
                for img_path in batch_paths:
                    image = Image.open(img_path).convert("RGB")
                    if max(image.size) > max_size:
                        image.thumbnail((max_size, max_size), Image.LANCZOS)
                    batch_images.append(image)
                
                batch_texts = process_batch_docling(batch_images, processor, model, device, ocr_kwargs)
                
                for img_path, text in zip(batch_paths, batch_texts):
                    try:
                        tokenized = tokenizer(
                            text,
                            padding="max_length",
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
                        )
                        
                        key = os.path.basename(img_path) if use_filename_key else img_path
                        ocr_dict[key] = {
                            "input_ids": tokenized["input_ids"].squeeze(0),
                            "attention_mask": tokenized["attention_mask"].squeeze(0)
                        }
                    except Exception as e:
                        print(f"Error tokenizing {img_path}: {str(e)}")
                        ocr_dict[img_path] = {
                            "input_ids": torch.tensor([]),
                            "attention_mask": torch.tensor([])
                        }
                
                # memory cleanup after batch
                del batch_images, batch_texts
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                gc.collect()
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {str(e)}")
                for img_path in batch_paths:
                    ocr_dict[img_path] = {
                        "input_ids": torch.tensor([]),
                        "attention_mask": torch.tensor([])
                    }
    else:
        # Single image processing for other engines
        for img_path in tqdm(all_images, desc="OCR Extraction"):
            try:
                # Load and resize image
                image = Image.open(img_path).convert("RGB")
                if max(image.size) > max_size:
                    image.thumbnail((max_size, max_size), Image.LANCZOS)
                
                # OCR extraction
                if ocr_engine == "trocr":
                    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
                    with torch.no_grad():
                        generated_ids = model.generate(pixel_values)
                    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                elif ocr_engine == "tesseract":
                    text = pytesseract.image_to_string(image)
                    
                elif ocr_engine == "easyocr":
                    result = reader.readtext(np.array(image), detail=0)
                    text = " ".join(result)
                    
                elif ocr_engine == "pero":
                    img_array = np.array(image)
                    img_bgr = img_array[:, :, ::-1]  # Converting to BGR
                    result = ocr_engine_pero.run_ocr(img_bgr)
                    text = " ".join(line.transcription for page in result.pages 
                                    for region in page.regions for line in region.lines)
                    
                elif ocr_engine == "docling_ocr":
                    # Single image Docling processing
                    instruction = ocr_kwargs.get("instruction", "Convert this page to docling.")
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": instruction}
                            ]
                        }
                    ]
                    
                    with torch.cuda.amp.autocast():
                        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
                        inputs = processor(
                            text=input_text,
                            images=[image],
                            return_tensors="pt",
                            truncation=False
                        ).to(device)
                        
                        with torch.no_grad():
                            generated_ids = model.generate(
                                **inputs,
                                max_new_tokens=ocr_kwargs.get("max_new_tokens", 2048),
                                do_sample=False
                            )
                        
                        generated_text = processor.batch_decode(
                            generated_ids[:, inputs.input_ids.shape[1]:],
                            skip_special_tokens=True
                        )[0]
                        
                        try:
                            from docling_core.types.doc import DocTagsDocument
                            doc_tags = DocTagsDocument.model_validate_json(generated_text)
                            docling_doc = doc_tags.export_to_document()
                            text = docling_doc.export_to_markdown()
                        except:
                            text = generated_text
                else:
                    text = ""
                
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
                    "input_ids": tokenized["input_ids"].squeeze(0),
                    "attention_mask": tokenized["attention_mask"].squeeze(0)
                }

                # Memory cleanup
                del image
                if 'pixel_values' in locals(): 
                    del pixel_values
                if 'generated_ids' in locals(): 
                    del generated_ids
                if 'inputs' in locals():
                    del inputs
                    
                #memory cleanup every 10 images
                if (len(ocr_dict) % 10 == 0) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
                
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                ocr_dict[img_path] = {
                    "input_ids": torch.tensor([]),
                    "attention_mask": torch.tensor([])
                }

    # Last memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

    torch.save(ocr_dict, output_pt)
    print(f"Saved tokenized OCR for {len(ocr_dict)} images to {output_pt}")
    return len(ocr_dict)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OCR Extraction and Tokenization")
    parser.add_argument("--data_dir", required=True, help="Directory with images")
    parser.add_argument("--output_pt", required=True, help="Output .pt file path")
    parser.add_argument("--ocr_engine", default="tesseract", 
                        choices=["tesseract", "trocr", "easyocr", "pero", "docling_ocr"])
    parser.add_argument("--max_size", type=int, default=1024, 
                        help="Max image dimension")
    parser.add_argument("--offset", type=int, default=0, 
                        help="Starting index for image processing")
    parser.add_argument("--max_images", type=int, default=None, 
                        help="Max number of images to process")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Max tokenization length")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for Docling processing")
    parser.add_argument("--use_filename_key", action="store_true", 
                        help="Use filename as dictionary key")
    
    args = parser.parse_args()
    
    ocr_extraction_tokenization(
        data_dir=args.data_dir,
        output_pt=args.output_pt,
        ocr_engine=args.ocr_engine,
        max_size=args.max_size,
        offset=args.offset,
        max_images=args.max_images,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_filename_key=args.use_filename_key
    )
