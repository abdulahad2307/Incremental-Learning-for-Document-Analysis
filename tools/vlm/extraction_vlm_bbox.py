"""
VLM-based OCR replacement: content-only + content+layout extraction, HPC-optimized.

Outputs (under --output_root):
  content/vlm_content_all.pt   dict[filename] = {input_ids, attention_mask}        (BERT tokens, for EAML)
  layout/vlm_layout_all.pt     list[{input_ids, attention_mask, bbox, image_path}] (LayoutLMv3 tokens, base training)
  layout/<image_stem>.pt       per-image LayoutLMv3 tensors (required by the CIL/DIL incremental dataloaders)
  viz/<class>/<image_stem>.png + .txt   a few annotated samples per class for manual QA
  shards/*.pt                  per-job/per-rank partial results; merged into the two combined files above

HPC efficiency:
  - Batched VLM inference (auto batch size from free GPU memory, OOM-safe fallback to per-image)
  - Parallel image decode/resize via DataLoader workers, overlapped with GPU compute
  - Async CPU post-processing (parsing, BERT + LayoutLMv3 tokenization, per-image .pt write, viz) on background
    threads while the GPU works on the next batch
  - Tokenizers loaded once (not per-image, which the previous version did for LayoutLMv3TokenizerFast)
  - Periodic checkpointing (under --output_root, not node-local TMPDIR) so a SLURM walltime kill doesn't lose progress
  - Shard-based output naming: every job/rank writes uniquely named shards and the combined files are rebuilt
    from ALL shards present, so chunked/array jobs never clobber each other's output
  - Optional multi-GPU via torchrun (auto-detected from RANK/WORLD_SIZE); plain `python` still works single-GPU

Usage:
    python extraction_vlm_bbox.py --data_dir ... --output_root ...                    # single GPU
    torchrun --nproc_per_node=N extraction_vlm_bbox.py --data_dir ... --output_root .. # multi-GPU
    python extraction_vlm_bbox.py --merge_only --output_root ...                       # rebuild combined files only
"""

import argparse
import gc
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    LayoutLMv3TokenizerFast,
    BertTokenizer,
)
from qwen_vl_utils import process_vision_info
import warnings

warnings.filterwarnings("ignore")

IMAGE_EXT = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


# ---------------------------------------------------------------------------
#  Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> Tuple[int, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def cleanup_distributed(is_dist: bool):
    if is_dist:
        dist.destroy_process_group()


def print_rank0(msg: str, rank: int):
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
#  Atomic save (avoid corrupt files if the job is killed mid-write at the
#  SLURM walltime boundary)
# ---------------------------------------------------------------------------

def atomic_torch_save(obj, path: str):
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
#  Image collection: group by class, optionally cap per class
# ---------------------------------------------------------------------------

def collect_images_by_class(data_dir: str) -> Dict[str, List[str]]:
    by_class = defaultdict(list)
    for root, _, files in os.walk(data_dir):
        imgs = [f for f in files if f.lower().endswith(IMAGE_EXT)]
        if not imgs:
            continue
        class_name = os.path.basename(root) or "unknown"
        for fname in imgs:
            by_class[class_name].append(os.path.join(root, fname))
    for c in by_class:
        by_class[c].sort()
    return by_class


def build_image_list(data_dir: str, images_per_class: Optional[int]) -> Tuple[List[str], Dict[str, int]]:
    """Group images by class (immediate parent folder name) and optionally cap
    each class to `images_per_class` images. None (default) = full dataset."""
    by_class = collect_images_by_class(data_dir)
    counts_before = {c: len(v) for c, v in by_class.items()}
    all_images = []
    for class_name in sorted(by_class):
        imgs = by_class[class_name]
        if images_per_class is not None:
            imgs = imgs[:images_per_class]
        all_images.extend(imgs)
    return all_images, counts_before


def shard_list(lst: list, rank: int, world_size: int) -> list:
    return [lst[i] for i in range(rank, len(lst), world_size)]


# ---------------------------------------------------------------------------
#  Dataset — workers pre-load + resize images in parallel, overlapped with GPU work
# ---------------------------------------------------------------------------

class ImagePathDataset(Dataset):
    def __init__(self, paths: List[str], max_size: int = 1024):
        self.paths = paths
        self.max_size = max_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            if max(img.size) > self.max_size:
                img.thumbnail((self.max_size, self.max_size), Image.LANCZOS)
            return path, img
        except Exception:
            return path, None


def collate_batch(batch):
    paths, imgs = zip(*batch)
    return list(paths), list(imgs)


# ---------------------------------------------------------------------------
#  Auto batch-size from free GPU memory
# ---------------------------------------------------------------------------

def auto_batch_size(device, model_gb: float = 5.0, per_img_gb: float = 2.0, user_bs: int = 0) -> int:
    if user_bs > 0:
        return user_bs
    if not torch.cuda.is_available():
        return 1
    free_gb = torch.cuda.mem_get_info(device)[0] / (1024 ** 3)
    usable = free_gb - model_gb - 1.5
    bs = max(1, int(usable / per_img_gb))
    name = torch.cuda.get_device_name(device).lower()
    if "a100" in name:
        bs = min(bs, 6)
    elif "v100" in name:
        bs = min(bs, 3)
    else:
        bs = min(bs, 2)
    return max(1, bs)


# ---------------------------------------------------------------------------
#  VLM prompt + batched inference
# ---------------------------------------------------------------------------

VLM_PROMPT = (
    "Extract all visible text elements from this document image. "
    "For each distinct text block/word/line, output a JSON list like: "
    "[{\"text\": \"extracted text\", \"bbox\": [x1,y1,x2,y2]}] "
    "where bbox is normalized coordinates [0-1] relative to image width/height (top-left=0,0). "
    "Include layout description after the list. Be precise and comprehensive."
)


def extract_texts_batched(model, processor, images: List[Image.Image], device, max_new_tokens: int) -> List[str]:
    """Run Qwen2-VL on a BATCH of images. Returns raw text outputs in the same order."""
    batch_messages = [[{
        "role": "user",
        "content": [{"type": "image", "image": img}, {"type": "text", "text": VLM_PROMPT}],
    }] for img in images]

    text_prompts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in batch_messages
    ]
    all_imgs = []
    for m in batch_messages:
        im_in, _ = process_vision_info(m)
        all_imgs.extend(im_in)

    inputs = processor(
        text=text_prompts, images=all_imgs, videos=None,
        padding=True, return_tensors="pt",
    ).to(device, non_blocking=True)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        gen_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
        )

    input_len = inputs.input_ids.shape[1]
    results = []
    for i in range(len(images)):
        trimmed = gen_ids[i][input_len:]
        text = processor.decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        results.append(text.strip())
    return results


def extract_single_safe(model, processor, img, device, max_new_tokens) -> str:
    """Single-image fallback used after a batch OOMs."""
    try:
        return extract_texts_batched(model, processor, [img], device, max_new_tokens)[0]
    except Exception:
        torch.cuda.empty_cache()
        return ""


# ---------------------------------------------------------------------------
#  Parse VLM output into texts + pixel bboxes
# ---------------------------------------------------------------------------

def parse_vlm_bbox_output(text_output, img_width, img_height):
    """
    Parse VLM structured output to texts + pixel bboxes like OCR.
    Expects JSON list: [{"text": "...", "bbox": [x1,y1,x2,y2]}, ...] with bbox in 0-1.
    """
    texts, bboxes_rect = [], []
    json_match = re.search(r'\[\s*\{[^\}]+\}\s*(,\s*\{[^\}]+\}\s*)*\]', text_output, re.DOTALL)
    if json_match:
        try:
            items = json.loads(json_match.group(0))
            for item in items:
                if isinstance(item, dict) and "text" in item and "bbox" in item:
                    text = str(item["text"]).strip()
                    if not text:
                        continue
                    b = item["bbox"]
                    if isinstance(b, (list, tuple)) and len(b) == 4:
                        x1 = max(0, min(img_width, int(float(b[0]) * img_width)))
                        y1 = max(0, min(img_height, int(float(b[1]) * img_height)))
                        x2 = max(0, min(img_width, int(float(b[2]) * img_width)))
                        y2 = max(0, min(img_height, int(float(b[3]) * img_height)))
                        texts.append(text)
                        bboxes_rect.append([x1, y1, x2, y2])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    if not texts:
        texts = [text_output.strip() or "[EMPTY]"]
        bboxes_rect = [[0, 0, img_width, img_height]]

    return texts, bboxes_rect


# ---------------------------------------------------------------------------
#  LayoutLMv3 tokenization helpers
# ---------------------------------------------------------------------------

def normalize_bbox_layoutlm(bbox, img_width, img_height):
    return [
        int(1000 * bbox[0] / img_width),
        int(1000 * bbox[1] / img_height),
        int(1000 * bbox[2] / img_width),
        int(1000 * bbox[3] / img_height),
    ]


def tokenize_layoutlm_from_vlm(tokenizer, texts, bboxes_rect, img_size, max_len=512):
    """`tokenizer` must be passed in (loaded once in run()) — instantiating
    LayoutLMv3TokenizerFast per image, as the previous version did, is a large
    unnecessary per-image cost."""
    img_width, img_height = img_size
    words = list(texts)
    boxes = [normalize_bbox_layoutlm(b, img_width, img_height) for b in bboxes_rect]
    special_box = [0, 0, 1000, 1000]

    words = ["[CLS]"] + words + ["[SEP]"]
    boxes = [special_box] + boxes + [special_box]
    words, boxes = words[:max_len], boxes[:max_len]

    pad_len = max_len - len(words)
    if pad_len > 0:
        words += ["[PAD]"] * pad_len
        boxes += [special_box] * pad_len

    encoding = tokenizer(
        text=words, boxes=boxes, padding="max_length",
        truncation=True, max_length=max_len, return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
        "bbox": torch.tensor(boxes, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
#  Visualization helpers (manual QA of VLM extraction)
# ---------------------------------------------------------------------------

def _load_font(size=14):
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def save_bbox_visualization(image, texts, bboxes_rect, raw_text, out_image_path, out_text_path):
    """Draw numbered boxes over `image` and write the matching texts to a .txt file
    so extraction quality (boxes + text) can be checked by eye."""
    vis = image.convert("RGB").copy()
    draw = ImageDraw.Draw(vis)
    font = _load_font()

    for i, box in enumerate(bboxes_rect, start=1):
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        label = str(i)
        tx1, ty1, tx2, ty2 = draw.textbbox((0, 0), label, font=font)
        tw, th = tx2 - tx1, ty2 - ty1
        draw.rectangle([x1, y1, x1 + tw + 4, y1 + th + 4], fill="red")
        draw.text((x1 + 2, y1 + 2), label, fill="white", font=font)

    os.makedirs(os.path.dirname(out_image_path), exist_ok=True)
    vis.save(out_image_path)

    with open(out_text_path, "w", encoding="utf-8") as f:
        f.write(f"boxes_found: {len(bboxes_rect)}\n\n")
        for i, text in enumerate(texts, start=1):
            f.write(f"{i}: {text}\n")
        f.write("\n--- raw VLM output ---\n")
        f.write(raw_text)


# ---------------------------------------------------------------------------
#  Async post-processing: parse -> BERT content tokenize -> LayoutLMv3
#  tokenize -> per-image .pt write -> optional viz. Runs on CPU threads while
#  the GPU works on the next batch.
# ---------------------------------------------------------------------------

class AsyncPostProcessor:
    def __init__(self, bert_tok, layout_tok, layout_dir, viz_dir,
                 max_length_content, max_len_layout, workers=4):
        self.bert_tok = bert_tok
        self.layout_tok = layout_tok
        self.layout_dir = layout_dir
        self.viz_dir = viz_dir
        self.max_length_content = max_length_content
        self.max_len_layout = max_len_layout
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.futures: List[Future] = []

    def submit(self, img_path: str, raw_text: str, img_size: Tuple[int, int], viz_image=None):
        self.futures.append(self.pool.submit(self._run, img_path, raw_text, img_size, viz_image))

    def _run(self, img_path, raw_text, img_size, viz_image):
        img_width, img_height = img_size
        texts, bboxes_rect = parse_vlm_bbox_output(raw_text, img_width, img_height)
        key = os.path.basename(img_path)
        base_name = os.path.splitext(key)[0]

        content_tok = self.bert_tok(
            " ".join(texts), padding="max_length", truncation=True,
            max_length=self.max_length_content, return_tensors="pt",
        )
        content_entry = {
            "input_ids": content_tok["input_ids"].squeeze(0),
            "attention_mask": content_tok["attention_mask"].squeeze(0),
        }

        layout_entry = tokenize_layoutlm_from_vlm(
            self.layout_tok, texts, bboxes_rect, img_size, self.max_len_layout,
        )
        layout_entry["image_path"] = img_path
        # Per-image file: required by the CIL/DIL incremental dataloaders, which
        # load OCR/layout tensors on demand by image stem rather than from the
        # combined list.
        atomic_torch_save(layout_entry, os.path.join(self.layout_dir, f"{base_name}.pt"))

        if viz_image is not None:
            class_name = os.path.basename(os.path.dirname(img_path)) or "unknown"
            save_bbox_visualization(
                viz_image, texts, bboxes_rect, raw_text,
                out_image_path=os.path.join(self.viz_dir, class_name, f"{base_name}.png"),
                out_text_path=os.path.join(self.viz_dir, class_name, f"{base_name}.txt"),
            )

        return key, content_entry, layout_entry

    def collect(self):
        results = [f.result() for f in self.futures]
        self.futures.clear()
        return results

    def shutdown(self):
        self.pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
#  Shard merge — fixes the hard-coded-filename overwrite issue: every job/rank
#  writes its own uniquely named shard; the combined canonical files are
#  rebuilt from ALL shards present, so re-running or chunking never loses data.
# ---------------------------------------------------------------------------

def merge_shards(output_root, content_dir, layout_dir, max_len_layout):
    shards_dir = os.path.join(output_root, "shards")
    os.makedirs(shards_dir, exist_ok=True)

    merged_content: Dict[str, dict] = {}
    for f in sorted(Path(shards_dir).glob("content_*.pt")):
        try:
            merged_content.update(torch.load(f, map_location="cpu"))
        except Exception as e:
            print(f"[merge] skipping unreadable shard {f}: {e}")

    merged_layout: Dict[str, dict] = {}
    for f in sorted(Path(shards_dir).glob("layout_*.pt")):
        try:
            merged_layout.update(torch.load(f, map_location="cpu"))
        except Exception as e:
            print(f"[merge] skipping unreadable shard {f}: {e}")

    os.makedirs(content_dir, exist_ok=True)
    os.makedirs(layout_dir, exist_ok=True)

    content_pt_path = os.path.join(content_dir, "vlm_content_all.pt")
    atomic_torch_save(merged_content, content_pt_path)

    vlm_layout_list = []
    for key in merged_content:
        entry = merged_layout.get(key)
        if entry is None:
            entry = {
                "input_ids": torch.zeros(max_len_layout, dtype=torch.long),
                "attention_mask": torch.zeros(max_len_layout, dtype=torch.long),
                "bbox": torch.zeros(max_len_layout, 4, dtype=torch.long),
                "image_path": f"unknown_{key}",
            }
        vlm_layout_list.append(entry)

    layout_pt_path = os.path.join(layout_dir, "vlm_layout_all.pt")
    atomic_torch_save(vlm_layout_list, layout_pt_path)

    return content_pt_path, layout_pt_path, len(merged_content), len(vlm_layout_list)


# ---------------------------------------------------------------------------
#  Offline model check (HPC compute nodes have no internet)
# ---------------------------------------------------------------------------

def preflight_check(model_name: str, rank: int):
    if rank != 0:
        return
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or \
              os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    hf_home = os.environ.get("HF_HOME", "")
    if offline:
        print(f"[Preflight] Offline mode ON | HF_HOME={hf_home}", flush=True)
        if not hf_home or not os.path.isdir(hf_home):
            print(
                "\n!!! ERROR: HF_HOME is not set or does not exist. !!!\n"
                "Compute nodes have no internet — pre-download models first "
                "(see vlm_model_download.sh).\n",
                flush=True,
            )
            raise SystemExit(1)
    else:
        print(
            "[Preflight] WARNING: HF_HUB_OFFLINE is not set.\n"
            "  If this is a compute node, model download WILL fail.\n"
            "  Set HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, HF_HOME in your SLURM script.\n",
            flush=True,
        )


# ---------------------------------------------------------------------------
#  Checkpointing (persisted under --output_root, NOT node-local TMPDIR, so a
#  requeue to a different node can still resume)
# ---------------------------------------------------------------------------

def checkpoint_path(output_root, offset, max_images, rank) -> str:
    ckpt_dir = os.path.join(output_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    tag = f"o{offset}_m{max_images if max_images is not None else 'all'}_r{rank}"
    return os.path.join(ckpt_dir, f"ckpt_{tag}.pt")


def shard_tag(offset, max_images, rank) -> str:
    return f"o{offset}_m{max_images if max_images is not None else 'all'}_r{rank}"


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def run(args):
    content_dir = os.path.join(args.output_root, "content")
    layout_dir = os.path.join(args.output_root, "layout")
    viz_dir = args.viz_dir or os.path.join(args.output_root, "viz")

    if args.merge_only:
        os.makedirs(content_dir, exist_ok=True)
        os.makedirs(layout_dir, exist_ok=True)
        path_c, path_l, n_c, n_l = merge_shards(args.output_root, content_dir, layout_dir, args.max_len_layout)
        print(f"Merged {n_c} content entries -> {path_c}")
        print(f"Merged {n_l} layout entries -> {path_l}")
        return

    rank, world_size, local_rank, is_dist = setup_distributed()
    device = torch.device(args.device) if args.device else \
        torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print_rank0("=== VLM Extraction: content + content+layout (HPC-optimized) ===", rank)
    print_rank0(f"World size: {world_size} | Device: {device}", rank)

    preflight_check(args.model_name, rank)

    for d in (content_dir, layout_dir, viz_dir):
        os.makedirs(d, exist_ok=True)

    all_images, class_counts = build_image_list(args.data_dir, args.images_per_class)
    cap_note = "full dataset" if args.images_per_class is None else f"capped at {args.images_per_class}/class"
    print_rank0(
        f"Classes found: {len(class_counts)} | Images available: {sum(class_counts.values())} ({cap_note})", rank)
    print_rank0(f"Images selected after per-class cap: {len(all_images)}", rank)

    if args.offset > 0:
        all_images = all_images[args.offset:]
    if args.max_images is not None:
        all_images = all_images[:args.max_images]

    my_images = shard_list(all_images, rank, world_size)
    print(f"[Rank {rank}] This job's shard: {len(my_images)} images", flush=True)

    if not my_images:
        cleanup_distributed(is_dist)
        return

    # ---- Resume: skip images already checkpointed for this (offset, max_images, rank) ----
    ckpt_path = checkpoint_path(args.output_root, args.offset, args.max_images, rank)
    content_dict: Dict[str, dict] = {}
    layout_dict: Dict[str, dict] = {}
    if not args.no_resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        content_dict = ckpt.get("content", {})
        layout_dict = ckpt.get("layout", {})
        print(f"[Rank {rank}] Resumed {len(content_dict)} already-processed images from {ckpt_path}", flush=True)

    done_keys = set(content_dict.keys())
    remaining = [p for p in my_images if os.path.basename(p) not in done_keys]
    print(f"[Rank {rank}] Remaining to process: {len(remaining)}", flush=True)

    if not remaining:
        _finalize_shard(args, rank, content_dict, layout_dict, is_dist, content_dir, layout_dir)
        cleanup_distributed(is_dist)
        return

    # ---- Load model ----
    print(f"[Rank {rank}] Loading {args.model_name} ...", flush=True)
    mkw = {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32, "device_map": device}
    if torch.cuda.is_available() and torch.cuda.get_device_capability(local_rank)[0] >= 8:
        try:
            mkw["attn_implementation"] = "flash_attention_2"
            print(f"[Rank {rank}] Flash-Attention 2 enabled", flush=True)
        except Exception:
            pass
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_name, **mkw)
    model.eval()

    if hasattr(torch, "compile") and args.use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"[Rank {rank}] torch.compile enabled", flush=True)
        except Exception:
            pass

    processor = AutoProcessor.from_pretrained(
        args.model_name, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    # Loaded once here (not per-image, unlike the previous version) — this alone
    # removes a large redundant cost from the hot loop.
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    layout_tok = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")

    bs = auto_batch_size(device, user_bs=args.batch_size)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"[Rank {rank}] GPU: {gpu_name} ({gpu_mem:.1f} GB) | batch_size={bs}", flush=True)

    async_proc = AsyncPostProcessor(
        bert_tok, layout_tok, layout_dir, viz_dir,
        args.max_length_content, args.max_len_layout, workers=args.num_workers,
    )

    ds = ImagePathDataset(remaining, max_size=args.max_size)
    loader = DataLoader(
        ds, batch_size=bs, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_batch, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    viz_counts = defaultdict(int)
    t0 = time.time()
    processed = 0
    oom_count = 0

    for paths, images in tqdm(loader, desc=f"[R{rank}] VLM Extraction", disable=(rank != 0)):
        valid_p, valid_im = [], []
        for p, im in zip(paths, images):
            if im is None:
                key = os.path.basename(p)
                content_dict[key] = {
                    "input_ids": torch.zeros(args.max_length_content, dtype=torch.long),
                    "attention_mask": torch.zeros(args.max_length_content, dtype=torch.long),
                }
                continue
            valid_p.append(p)
            valid_im.append(im)

        if not valid_im:
            continue

        try:
            raw_texts = extract_texts_batched(model, processor, valid_im, device, args.max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            torch.cuda.empty_cache()
            gc.collect()
            raw_texts = [extract_single_safe(model, processor, im, device, args.max_new_tokens)
                         for im in valid_im]

        for p, im, raw in zip(valid_p, valid_im, raw_texts):
            class_name = os.path.basename(os.path.dirname(p)) or "unknown"
            viz_image = None
            if viz_counts[class_name] < args.viz_per_class:
                viz_image = im
                viz_counts[class_name] += 1
            async_proc.submit(p, raw, im.size, viz_image)

        processed += len(valid_p)

        if processed % args.save_every < bs * 2:
            for key, content_entry, layout_entry in async_proc.collect():
                content_dict[key] = content_entry
                layout_dict[key] = layout_entry
            atomic_torch_save({"content": content_dict, "layout": layout_dict}, ckpt_path)

        del valid_im, raw_texts
        if processed % (bs * 20) == 0:
            gc.collect()
            torch.cuda.empty_cache()

    for key, content_entry, layout_entry in async_proc.collect():
        content_dict[key] = content_entry
        layout_dict[key] = layout_entry
    async_proc.shutdown()
    atomic_torch_save({"content": content_dict, "layout": layout_dict}, ckpt_path)

    elapsed = time.time() - t0
    print(f"[Rank {rank}] {processed} new images, {elapsed:.1f}s, "
          f"{processed / max(elapsed, 1):.2f} img/s, OOM fallbacks: {oom_count}", flush=True)

    _finalize_shard(args, rank, content_dict, layout_dict, is_dist, content_dir, layout_dir)
    cleanup_distributed(is_dist)


def _finalize_shard(args, rank, content_dict, layout_dict, is_dist, content_dir, layout_dir):
    shards_dir = os.path.join(args.output_root, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    tag = shard_tag(args.offset, args.max_images, rank)
    atomic_torch_save(content_dict, os.path.join(shards_dir, f"content_{tag}.pt"))
    atomic_torch_save(layout_dict, os.path.join(shards_dir, f"layout_{tag}.pt"))
    print(f"[Rank {rank}] Wrote shard '{tag}': {len(content_dict)} images", flush=True)

    if is_dist:
        dist.barrier()
    if rank == 0:
        path_c, path_l, n_c, n_l = merge_shards(args.output_root, content_dir, layout_dir, args.max_len_layout)
        print(f"=== Merged {n_c} content entries -> {path_c}")
        print(f"=== Merged {n_l} layout entries -> {path_l}")


def parse_args():
    p = argparse.ArgumentParser(description="VLM Extraction: content + content+layout (HPC-optimized)")
    p.add_argument("--data_dir", required=True, help="Directory with images (walked recursively)")
    p.add_argument("--output_root", required=True, help="Root directory for VLM outputs")
    p.add_argument("--max_size", type=int, default=1024, help="Max image dimension for VLM input")
    p.add_argument("--max_len_layout", type=int, default=512, help="Max tokens for LayoutLM (layout)")
    p.add_argument("--max_length_content", type=int, default=128, help="Max tokens for BERT (content)")
    p.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct", help="VLM model name/path")
    p.add_argument("--max_new_tokens", type=int, default=400, help="Max new tokens for VLM generation")
    p.add_argument("--device", default=None, help="Force device (cuda/cuda:0/cpu); default = auto")

    p.add_argument("--images_per_class", type=int, default=None,
                    help="Cap the number of images extracted PER CLASS. Omit for the full dataset.")
    p.add_argument("--offset", type=int, default=0,
                    help="Starting index into the (capped) image list — for array-job chunking")
    p.add_argument("--max_images", type=int, default=None,
                    help="Max images for THIS job/chunk — for array-job chunking")

    p.add_argument("--batch_size", type=int, default=0, help="VLM batch size; 0 = auto-detect from GPU memory")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers for parallel image decode/resize")
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--save_every", type=int, default=200, help="Checkpoint every N processed images")
    p.add_argument("--no_resume", action="store_true", help="Ignore any existing checkpoint and start fresh")
    p.add_argument("--use_compile", action="store_true", help="torch.compile the model (experimental)")

    p.add_argument("--viz_dir", default=None, help="Output dir for annotated QA images (default: {output_root}/viz)")
    p.add_argument("--viz_per_class", type=int, default=3, help="Number of annotated samples to save per class")

    p.add_argument("--merge_only", action="store_true",
                    help="Skip extraction; just rebuild the combined .pt files from all shards in "
                         "{output_root}/shards (use this once after all array-job chunks finish)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
