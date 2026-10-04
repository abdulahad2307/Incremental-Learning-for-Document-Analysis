"""OCR extraction for the pre-trained LayoutLMv3 (microsoft/layoutlmv3-base, --model_type hf).

Same OCR engines, image resizing and per-image output format as ocr_extraction_bbox_bert.py (one <image stem>.pt per
image with input_ids, attention_mask, bbox, image_path), but:
  - tokens come from the LayoutLMv3 tokenizer (RoBERTa BPE, cased), which also assigns each token its word's box
    (special tokens and padding: [0, 0, 0, 0]);
  - the OCR words and their normalised boxes are stored as well ("words", "word_boxes"), so the tensors can be
    re-tokenized later without running OCR again.
"""
import os
import gc
import argparse

import torch
from PIL import Image
from tqdm import tqdm
from transformers import LayoutLMv3TokenizerFast

from tools.ocr.ocr_extraction_bbox_bert import OCR_ENGINES, _clip_box
from tools.ocr.ocr_extraction_bbox_single import normalize_bbox_layoutlm

HF_LAYOUTLMV3 = "microsoft/layoutlmv3-base"


def tokenize_layoutlmv3_with_boxes(words, word_boxes, tokenizer, img_size, max_len=512):
    """LayoutLMv3-tokenize OCR words; every token gets the normalised (0-1000) box of its word."""
    img_w, img_h = img_size
    words = [str(w) for w in words]
    boxes = [_clip_box(normalize_bbox_layoutlm(b, img_w, img_h)) for b in word_boxes]
    enc = tokenizer(words, boxes=boxes, truncation=True, max_length=max_len, padding="max_length",
                    return_attention_mask=True)
    return {
        "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
        "bbox": torch.tensor(enc["bbox"], dtype=torch.long),
        "tokenizer": tokenizer.name_or_path,
        "words": words,
        "word_boxes": boxes,
    }


def precompute_ocr_layoutlmv3(data_dir, output_dir, ocr_engine="tesseract", max_size=1024, max_len=512, offset=0,
                              max_images=None, skip_existing=False, tokenizer_name=HF_LAYOUTLMV3):
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(tokenizer_name)
    run_ocr = OCR_ENGINES[ocr_engine]

    image_paths = sorted(
        os.path.join(root, f) for root, _, files in os.walk(data_dir)
        for f in files if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
    )
    image_paths = image_paths[offset:]
    if max_images is not None:
        image_paths = image_paths[:max_images]
    print(f"Processing {len(image_paths)} images with OCR engine {ocr_engine} ({tokenizer_name} tokens)")

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
            out = tokenize_layoutlmv3_with_boxes(words, word_boxes, tokenizer, image.size, max_len)
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
    parser = argparse.ArgumentParser(description="OCR extraction with LayoutLMv3 tokens (pre-trained LayoutLMv3)")
    parser.add_argument("--data_dir", required=True, help="Image dataset root (walked recursively)")
    parser.add_argument("--output_dir", required=True, help="Directory for the per-image .pt files")
    parser.add_argument("--ocr_engine", default="tesseract", choices=list(OCR_ENGINES))
    parser.add_argument("--max_size", type=int, default=1024, help="Maximum image side before OCR")
    parser.add_argument("--max_len", type=int, default=512, help="Token sequence length")
    parser.add_argument("--offset", type=int, default=0, help="Start index in the sorted image list (for job shards)")
    parser.add_argument("--max_images", type=int, default=None, help="Number of images for this job")
    parser.add_argument("--skip_existing", action="store_true", help="Skip images whose .pt already exists")
    parser.add_argument("--tokenizer", default=HF_LAYOUTLMV3, help="LayoutLMv3 tokenizer to use")
    args = parser.parse_args()
    precompute_ocr_layoutlmv3(args.data_dir, args.output_dir, args.ocr_engine, args.max_size, args.max_len,
                              args.offset, args.max_images, args.skip_existing, args.tokenizer)
