import os
import json
import time
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

def ocr_extraction_tokenization(
    data_dir,
    output_pt,
    ocr_engine="tesseract",
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_images = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
                all_images.append(os.path.join(root, file))
    all_images.sort()  # same order in every job, so --offset shards neither overlap nor miss images

    # Apply offset and max_images limit
    if offset > 0:
        all_images = all_images[offset:]
    if max_images is not None:
        all_images = all_images[:max_images]

    # Initialize OCR engine
    if ocr_engine == "trocr":
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        print("OCR Engine:", ocr_engine)

        processor = TrOCRProcessor.from_pretrained(ocr_kwargs.get("model_name", "microsoft/trocr-base-handwritten"))
        model = VisionEncoderDecoderModel.from_pretrained(ocr_kwargs.get("model_name", "microsoft/trocr-base-handwritten")).to(device)
        model.eval()
    elif ocr_engine == "tesseract":
        import pytesseract
        print("OCR Engine:", ocr_engine)
    elif ocr_engine == "easyocr":
        import easyocr
        reader = easyocr.Reader(ocr_kwargs.get("languages", ["en"]))
    elif ocr_engine == "pero":
        from pero_ocr.document_ocr import DocumentOCR
        config_path = ocr_kwargs.get("config_path", None)
        ocr_engine_pero = DocumentOCR(config_path=config_path) if config_path else DocumentOCR()

    # Initialize tokenizer
    if tokenizer is None:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            ocr_dict[img_path] = {
                "input_ids": torch.tensor([]),
                "attention_mask": torch.tensor([])
            }

    torch.save(ocr_dict, output_pt)
    print(f"Saved tokenized OCR for {len(ocr_dict)} images to {output_pt}")
    return len(ocr_dict)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="OCR Extraction and Tokenization")
    parser.add_argument("--data_dir", required=True, help="Directory with images")
    parser.add_argument("--output_pt", required=True, help="Output .pt file path")
    parser.add_argument("--ocr_engine", default="tesseract", 
                        choices=["tesseract", "trocr", "easyocr", "pero"])
    parser.add_argument("--max_size", type=int, default=640, 
                        help="Max image dimension")
    parser.add_argument("--offset", type=int, default=0, 
                        help="Starting index for image processing")
    parser.add_argument("--max_images", type=int, default=None, 
                        help="Max number of images to process")
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
        use_filename_key=args.use_filename_key
    )
