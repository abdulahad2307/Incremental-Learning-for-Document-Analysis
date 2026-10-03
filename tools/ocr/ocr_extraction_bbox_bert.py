"""OCR extraction for the LayoutLMv3 backbone with bert-base-uncased tokens (the model's text encoder).

Same OCR engines and per-image output format as ocr_extraction_bbox_single.py (one <image stem>.pt per image with
input_ids, attention_mask, bbox, image_path), but:
  - tokens come from the bert-base-uncased WordPiece tokenizer, matching the model's BERT text encoder;
  - every word-piece gets the box of the word it belongs to ([CLS]/padding: [0,0,0,0], [SEP]: [1000,1000,1000,1000]);
  - boxes are normalised to 0-1000 and clipped.
"""
import os
import gc
import argparse

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizerFast

from tools.ocr.ocr_extraction_bbox_single import (
    get_ocr_results_tesseract, get_ocr_results_easyocr, get_ocr_results_pero, normalize_bbox_layoutlm,
)

CLS_BOX, SEP_BOX, PAD_BOX = [0, 0, 0, 0], [1000, 1000, 1000, 1000], [0, 0, 0, 0]


def _clip_box(box):
    x0, y0, x1, y1 = (min(max(int(v), 0), 1000) for v in box)
    return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]


def tokenize_bert_with_boxes(words, word_boxes, tokenizer, img_size, max_len=512):
    """WordPiece-tokenize OCR words and give each piece the normalised box of its word."""
    img_w, img_h = img_size
    boxes = [_clip_box(normalize_bbox_layoutlm(b, img_w, img_h)) for b in word_boxes]
    enc = tokenizer(list(words), is_split_into_words=True, truncation=True, max_length=max_len,
                    padding="max_length", return_attention_mask=True)
    bbox = []
    for idx, word_id in enumerate(enc.word_ids()):
        if word_id is not None:
            bbox.append(boxes[word_id])
        elif enc["input_ids"][idx] == tokenizer.sep_token_id:
            bbox.append(SEP_BOX)
        elif enc["input_ids"][idx] == tokenizer.cls_token_id:
            bbox.append(CLS_BOX)
        else:
            bbox.append(PAD_BOX)
    return {
        "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        "bbox": torch.tensor(bbox, dtype=torch.long),
        "tokenizer": "bert-base-uncased",
    }


OCR_ENGINES = {"tesseract": get_ocr_results_tesseract, "easyocr": get_ocr_results_easyocr, "pero": get_ocr_results_pero}


def precompute_ocr_bert(data_dir, output_dir, ocr_engine="tesseract", max_size=1024, max_len=512, offset=0,
                        max_images=None, skip_existing=False):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    run_ocr = OCR_ENGINES[ocr_engine]

    image_paths = sorted(
        os.path.join(root, f) for root, _, files in os.walk(data_dir)
        for f in files if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
    )
    image_paths = image_paths[offset:]
    if max_images is not None:
        image_paths = image_paths[:max_images]
    print(f"Processing {len(image_paths)} images with OCR engine {ocr_engine} (bert-base-uncased tokens)")

    processed = skipped = 0
    for img_path in tqdm(image_paths):
        out_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + ".pt")
        if skip_existing and os.path.exists(out_path):
            skipped += 1
            continue
        try:
            image = Image.open(img_path).convert("RGB")
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.LANCZOS)
            words, word_boxes = run_ocr(image)
            out = tokenize_bert_with_boxes(words, word_boxes, tokenizer, image.size, max_len)
            out["image_path"] = img_path
            torch.save(out, out_path)
            processed += 1
            del image
            gc.collect()
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    print(f"OCR extraction completed: {processed} written, {skipped} skipped (existing), "
          f"{len(image_paths) - processed - skipped} failed. Output: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LayoutLMv3 OCR extraction with bert-base-uncased tokens")
    parser.add_argument("--data_dir", required=True, help="Image dataset root (walked recursively)")
    parser.add_argument("--output_dir", required=True, help="Directory for the per-image .pt files")
    parser.add_argument("--ocr_engine", default="tesseract", choices=list(OCR_ENGINES))
    parser.add_argument("--max_size", type=int, default=1024, help="Maximum image side before OCR")
    parser.add_argument("--max_len", type=int, default=512, help="Token sequence length")
    parser.add_argument("--offset", type=int, default=0, help="Start index in the sorted image list (for job shards)")
    parser.add_argument("--max_images", type=int, default=None, help="Number of images for this job")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images whose .pt already exists")
    args = parser.parse_args()
    precompute_ocr_bert(args.data_dir, args.output_dir, args.ocr_engine, args.max_size, args.max_len,
                        args.offset, args.max_images, args.skip_existing)
