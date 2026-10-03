import os
import gc
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from transformers import LayoutLMv3TokenizerFast

def normalize_bbox_layoutlm(bbox, img_width, img_height):
    return [
        int(1000 * bbox[0] / img_width),
        int(1000 * bbox[1] / img_height),
        int(1000 * bbox[2] / img_width),
        int(1000 * bbox[3] / img_height),
    ]

def normalize_bbox_polygon8(bbox8, img_width, img_height):
    return [b / img_width if i % 2 == 0 else b / img_height for i, b in enumerate(bbox8)]

def polygon8_to_rect4(bbox_8):
    xs = bbox_8[0::2]
    ys = bbox_8[1::2]
    return [min(xs), min(ys), max(xs), max(ys)]

def get_ocr_results_tesseract(image):
    import pytesseract
    from pytesseract import Output
    d = pytesseract.image_to_data(image, output_type=Output.DICT)
    texts = []
    bboxes_rect = []
    bboxes_poly = []
    img_width, img_height = image.size
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        text = d['text'][i].strip()
        if text == '' or int(d['conf'][i]) < 50:
            continue
        x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        bbox_8 = [x, y, x + w, y, x + w, y + h, x, y + h]
        rect4 = polygon8_to_rect4(bbox_8)
        texts.append(text)
        bboxes_rect.append(rect4)
        bboxes_poly.append(bbox_8)
    return texts, bboxes_rect, bboxes_poly

def get_ocr_results_easyocr(image):
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(np.array(image))
    texts = []
    bboxes_rect = []
    bboxes_poly = []
    img_width, img_height = image.size
    for bbox, text, conf in results:
        if conf < 0.5:
            continue
        bbox_8 = [coord for pt in bbox for coord in pt]
        rect4 = polygon8_to_rect4(bbox_8)
        texts.append(text)
        bboxes_rect.append(rect4)
        bboxes_poly.append(bbox_8)
    return texts, bboxes_rect, bboxes_poly

def get_ocr_results_pero(image):
    try:
        from pero_ocr.document_ocr import DocumentOCR
        doc_ocr = DocumentOCR()
        result = doc_ocr.run_ocr(np.array(image)[:, :, ::-1])  # RGB to BGR
        texts = []
        bboxes_rect = []
        bboxes_poly = []
        for page in result.pages:
            for region in page.regions:
                for line in region.lines:
                    texts.append(line.transcription)
                    polygon = line.polygon  # list of 4 points (x, y)
                    bbox_8 = [point[0] for point in polygon] + [point[1] for point in polygon]
                    rect4 = polygon8_to_rect4(bbox_8)
                    bboxes_rect.append(rect4)
                    bboxes_poly.append(bbox_8)
        return texts, bboxes_rect, bboxes_poly
    except Exception as e:
        print(f"Pero OCR extraction failed: {e}")
        return [], [], []

def tokenize_layoutlm(texts, bboxes_rect, bboxes_poly, tokenizer, img_size, max_len, out_bbox_style="rect"):
    img_width, img_height = img_size
    words = list(texts)
    if out_bbox_style == "rect":
        boxes = [normalize_bbox_layoutlm(b, img_width, img_height) for b in bboxes_rect]
        special_box = [0, 0, 1000, 1000]
    else:
        boxes = [normalize_bbox_polygon8(b, img_width, img_height) for b in bboxes_poly]
        special_box = [0, 0, 1, 0, 1, 1, 0, 1]

    words = ["[CLS]"] + words + ["[SEP]"]
    boxes = [special_box] + boxes + [special_box]

    words = words[:max_len]
    boxes = boxes[:max_len]

    pad_len = max_len - len(words)
    if pad_len > 0:
        words += ["[PAD]"] * pad_len
        boxes += [special_box] * pad_len

    encoding = tokenizer(
        text=words,
        boxes=boxes,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)
    if out_bbox_style == "rect":
        out_dict = {"input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "bbox": torch.tensor(boxes, dtype=torch.long)}
    else:  # poly
        out_dict = {"input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "bboxes": torch.tensor(boxes, dtype=torch.float)}
    return out_dict

def precompute_ocr_layoutlm(data_dir, output_dir, ocr_engine="tesseract", max_size=1024, max_len=512, offset=0, max_images=None, bbox_style="rect"):
    #os.makedirs(output_dir, exist_ok=True)
    all_image_paths = []
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.lower().endswith((".tif", ".png", ".jpg", ".jpeg")):
                all_image_paths.append(os.path.join(root, fname))
    if offset > 0:
        all_image_paths = all_image_paths[offset:]
    if max_images is not None:
        all_image_paths = all_image_paths[:max_images]
    print(f"Processing {len(all_image_paths)} images with OCR engine {ocr_engine}")
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")
    processed = 0
    all_outputs = []
    for img_path in tqdm(all_image_paths):
        try:
            image = Image.open(img_path).convert("RGB")
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.LANCZOS)
            if ocr_engine == "tesseract":
                texts, bboxes_rect, bboxes_poly = get_ocr_results_tesseract(image)
            elif ocr_engine == "easyocr":
                texts, bboxes_rect, bboxes_poly = get_ocr_results_easyocr(image)
            elif ocr_engine == "pero":
                texts, bboxes_rect, bboxes_poly = get_ocr_results_pero(image)
            else:
                texts, bboxes_rect, bboxes_poly = [], [], []
            out_dict = tokenize_layoutlm(texts, bboxes_rect, bboxes_poly,
                                         tokenizer, image.size, max_len, bbox_style)
            out_dict["image_path"] = img_path
            all_outputs.append(out_dict)
            processed += 1
            del image
            gc.collect()
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
    #save_path = os.path.join(output_dir, "all_ocr.pt")
    save_path = output_dir
    torch.save(all_outputs, save_path)
    print(f"OCR extraction completed for {processed} images out of {len(all_image_paths)}. All outputs saved to {save_path}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LayoutLMv3-compatible OCR Extraction")
    parser.add_argument('--data_dir', required=True, help='Path to the image dataset root')
    parser.add_argument('--output_dir', required=True, help='Directory to save extracted OCR token files')
    parser.add_argument('--ocr_engine', default='tesseract', choices=['tesseract', 'easyocr', 'pero'], help='OCR engine to use')
    parser.add_argument('--max_size', type=int, default=1024, help='Maximum image dimension size')
    parser.add_argument('--max_len', type=int, default=512, help='Maximum token length for OCR tokens')
    parser.add_argument('--offset', type=int, default=0, help='Starting index in image list')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images to process')
    parser.add_argument('--bbox_style', default='rect', choices=['rect', 'poly'], help="'rect' for LayoutLMv3, 'poly' for DocFormer")
    args = parser.parse_args()
    precompute_ocr_layoutlm(args.data_dir, args.output_dir, args.ocr_engine, args.max_size, args.max_len, args.offset, args.max_images, args.bbox_style)