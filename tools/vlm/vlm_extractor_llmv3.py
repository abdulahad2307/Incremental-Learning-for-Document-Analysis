"""
VLM Content+Layout Extraction for LayoutLMv3  (v2 — GPU-optimised)
=====================================================================
Replaces OCR layout extraction with Qwen2-VL based text + bbox extraction.

Outputs:
  1) Combined  .pt  →  list[dict] for batch training
  2) Per-image  .pt  →  one file per image for incremental learning

GPU optimisations over v1:
  • Batched VLM inference  — multiple images per forward pass
  • Async CPU post-processing  — LayoutLMv3 tokenisation overlaps next GPU batch
  • Auto batch-size selection  — probes GPU memory at startup
  • CUDA pin_memory + non-blocking transfers
  • OOM-safe: graceful fallback to bs=1 per image
  • torch.compile support (optional)

Optimised for FAU TinyGPU (V100 32 GB / A100 40 GB) with torch.distributed.

Usage:
    torchrun --nproc_per_node=NUM_GPUS vlm_extract_layout.py [args]
    python vlm_extract_layout.py [args]
"""

import os
import gc
import json
import re
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, Future

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    LayoutLMv3TokenizerFast,
)
from qwen_vl_utils import process_vision_info
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
#  Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed() -> Tuple[int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world_size, True
    return 0, 1, False


def cleanup_distributed(is_dist: bool):
    if is_dist:
        dist.destroy_process_group()


def print_rank0(msg: str, rank: int = 0):
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
#  Image collection & sharding
# ---------------------------------------------------------------------------

IMAGE_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def collect_images(data_dir: str, offset: int = 0,
                   max_images: Optional[int] = None) -> List[str]:
    imgs = sorted(str(p) for p in Path(data_dir).rglob("*")
                  if p.suffix.lower() in IMAGE_EXT)
    if offset > 0:
        imgs = imgs[offset:]
    if max_images is not None:
        imgs = imgs[:max_images]
    return imgs


def shard_list(lst: list, rank: int, world_size: int) -> list:
    return [lst[i] for i in range(rank, len(lst), world_size)]


# ---------------------------------------------------------------------------
#  Dataset — workers pre-load + resize images
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
#  Auto batch-size
# ---------------------------------------------------------------------------

def auto_batch_size(device: torch.device, model_gb: float = 5.0,
                    per_img_gb: float = 2.0, user_bs: int = 0) -> int:
    """Layout extraction uses more tokens → slightly larger per-image cost."""
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
#  VLM prompt (structured JSON output with bboxes)
# ---------------------------------------------------------------------------

LAYOUT_PROMPT = (
    "Extract all visible text elements from this document image. "
    "For each distinct text block, word, or line, output a JSON list like:\n"
    '[{"text": "extracted text", "bbox": [x1, y1, x2, y2]}]\n'
    "where bbox values are normalised coordinates between 0 and 1 relative to "
    "image width and height (top-left = 0,0, bottom-right = 1,1). "
    "Be precise and comprehensive. Output ONLY the JSON list."
)


# ---------------------------------------------------------------------------
#  Batched VLM inference (shared with content script pattern)
# ---------------------------------------------------------------------------

def extract_batch_vlm(
    model, processor, images: List[Image.Image],
    device: torch.device, max_new_tokens: int = 600,
) -> List[str]:
    """Run VLM on batch of images, return raw text outputs."""
    batch_msgs = []
    for img in images:
        batch_msgs.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": LAYOUT_PROMPT},
            ],
        }])

    prompts = [processor.apply_chat_template(m, tokenize=False,
               add_generation_prompt=True) for m in batch_msgs]

    all_img_inputs = []
    for m in batch_msgs:
        im_in, _ = process_vision_info(m)
        all_img_inputs.extend(im_in)

    inputs = processor(
        text=prompts, images=all_img_inputs, videos=None,
        padding=True, return_tensors="pt",
    ).to(device, non_blocking=True)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
        gen_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1,
        )

    input_len = inputs.input_ids.shape[1]
    results = []
    for i in range(len(images)):
        trimmed = gen_ids[i][input_len:]
        text = processor.decode(trimmed, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        results.append(text.strip())
    return results


def extract_single_safe(model, processor, img, device, max_new_tokens):
    try:
        return extract_batch_vlm(model, processor, [img], device, max_new_tokens)[0]
    except Exception:
        torch.cuda.empty_cache()
        return ""


# ---------------------------------------------------------------------------
#  Parse VLM output → (texts, bboxes_pixel)
# ---------------------------------------------------------------------------

def parse_vlm_layout(raw: str, img_w: int, img_h: int
                     ) -> Tuple[List[str], List[List[int]]]:
    texts, bboxes = [], []
    m = re.search(r"\[\s*\{[^\}]+\}\s*(?:,\s*\{[^\}]+\}\s*)*\]", raw, re.DOTALL)
    if m:
        try:
            items = json.loads(m.group(0))
            for it in items:
                if isinstance(it, dict) and "text" in it and "bbox" in it:
                    t = str(it["text"]).strip()
                    b = it["bbox"]
                    if t and isinstance(b, (list, tuple)) and len(b) == 4:
                        x1 = max(0., min(1., float(b[0])))
                        y1 = max(0., min(1., float(b[1])))
                        x2 = max(0., min(1., float(b[2])))
                        y2 = max(0., min(1., float(b[3])))
                        texts.append(t)
                        bboxes.append([int(x1*img_w), int(y1*img_h),
                                       int(x2*img_w), int(y2*img_h)])
        except json.JSONDecodeError:
            pass
    if not texts:
        cleaned = raw.strip() or "[EMPTY]"
        texts = [cleaned]
        bboxes = [[0, 0, img_w, img_h]]
    return texts, bboxes


# ---------------------------------------------------------------------------
#  LayoutLMv3 tokenisation helpers
# ---------------------------------------------------------------------------

def norm_bbox_1000(bb: List[int], w: int, h: int) -> List[int]:
    return [int(1000*bb[0]/w), int(1000*bb[1]/h),
            int(1000*bb[2]/w), int(1000*bb[3]/h)]


def tokenize_layout(tokenizer, texts, bboxes_px, img_size, max_len=512):
    w, h = img_size
    words = list(texts)
    boxes = [norm_bbox_1000(b, w, h) for b in bboxes_px]
    sp = [0, 0, 1000, 1000]

    words = ["[CLS]"] + words + ["[SEP]"]
    boxes = [sp] + boxes + [sp]
    words = words[:max_len]
    boxes = boxes[:max_len]

    pad = max_len - len(words)
    if pad > 0:
        words += ["[PAD]"] * pad
        boxes += [sp] * pad

    enc = tokenizer(text=words, boxes=boxes, padding="max_length",
                    truncation=True, max_length=max_len, return_tensors="pt")
    return {
        "input_ids": enc["input_ids"].squeeze(0).cpu(),
        "attention_mask": enc["attention_mask"].squeeze(0).cpu(),
        "bbox": torch.tensor(boxes, dtype=torch.long),
    }


def empty_layout(max_len: int) -> dict:
    return {
        "input_ids": torch.zeros(max_len, dtype=torch.long),
        "attention_mask": torch.zeros(max_len, dtype=torch.long),
        "bbox": torch.zeros(max_len, 4, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
#  Async LayoutLMv3 tokeniser (CPU thread pool)
# ---------------------------------------------------------------------------

class AsyncLayoutTokeniser:
    """Parse VLM output + LayoutLMv3-tokenise on background threads."""

    def __init__(self, tokenizer, max_len: int, workers: int = 4):
        self.tok = tokenizer
        self.max_len = max_len
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.futures: List[Future] = []

    def submit(self, img_path: str, raw_vlm: str, img_size: Tuple[int, int]):
        self.futures.append(
            self.pool.submit(self._run, img_path, raw_vlm, img_size))

    def _run(self, img_path, raw_vlm, img_size):
        w, h = img_size
        texts, bboxes = parse_vlm_layout(raw_vlm, w, h)
        entry = tokenize_layout(self.tok, texts, bboxes, img_size, self.max_len)
        entry["image_path"] = img_path
        return entry

    def collect(self) -> List[dict]:
        out = [f.result() for f in self.futures]
        self.futures.clear()
        return out

    def shutdown(self):
        self.pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
#  Offline model check  (HPC compute nodes have no internet)
# ---------------------------------------------------------------------------

def preflight_check(model_name: str, rank: int):
    if rank != 0:
        return
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or \
              os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    hf_home = os.environ.get("HF_HOME", "")
    if offline:
        print(f"[Preflight] Offline mode ON  |  HF_HOME={hf_home}", flush=True)
        if not hf_home or not os.path.isdir(hf_home):
            print(
                "\n!!! ERROR: HF_HOME is not set or does not exist. !!!\n"
                "Compute nodes have no internet — you must pre-download models.\n"
                "Run on the frontend node:  bash download_models.sh\n",
                flush=True,
            )
            raise SystemExit(1)
    else:
        print(
            "[Preflight] WARNING: HF_HUB_OFFLINE is not set.\n"
            "  If this is a compute node, model download WILL fail.\n"
            "  Set these in your SLURM script:\n"
            "    export HF_HUB_OFFLINE=1\n"
            "    export TRANSFORMERS_OFFLINE=1\n"
            "    export HF_HOME=$WORK/model_cache\n",
            flush=True,
        )


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def run(args):
    rank, world_size, is_dist = setup_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print_rank0("=== VLM Content+Layout (LayoutLMv3) — v2 GPU-optimised ===", rank)
    print_rank0(f"World size: {world_size}  |  Device: {device}", rank)

    # Pre-flight: verify models are cached locally
    preflight_check(args.model_name, rank)

    # Collect & shard
    all_imgs = collect_images(args.data_dir, args.offset, args.max_images)
    print_rank0(f"Total images: {len(all_imgs)}", rank)
    my_imgs = shard_list(all_imgs, rank, world_size)
    print(f"[Rank {rank}] Shard: {len(my_imgs)} images", flush=True)

    if not my_imgs:
        cleanup_distributed(is_dist)
        return

    # Output dirs
    per_img_dir = os.path.join(args.output_dir, "per_image")
    os.makedirs(per_img_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Resume
    done_stems: set = set()
    if args.resume:
        done_stems = {p.stem for p in Path(per_img_dir).glob("*.pt")}
        if done_stems:
            print(f"[Rank {rank}] Resume: {len(done_stems)} already done", flush=True)

    # Load VLM
    print(f"[Rank {rank}] Loading {args.model_name} …", flush=True)
    mkw = {"torch_dtype": torch.float16, "device_map": device}
    if torch.cuda.is_available() and torch.cuda.get_device_capability(local_rank)[0] >= 8:
        try:
            mkw["attn_implementation"] = "flash_attention_2"
            print(f"[Rank {rank}] Flash-Attention 2 ✓", flush=True)
        except Exception:
            pass

    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_name, **mkw)
    model.eval()

    if hasattr(torch, "compile") and args.use_compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print(f"[Rank {rank}] torch.compile ✓", flush=True)
        except Exception:
            pass

    processor = AutoProcessor.from_pretrained(
        args.model_name, min_pixels=256*28*28, max_pixels=1280*28*28)

    # LayoutLMv3 tokeniser
    layout_tok = LayoutLMv3TokenizerFast.from_pretrained("microsoft/layoutlmv3-base")

    # Auto batch size
    bs = auto_batch_size(device, user_bs=args.batch_size)
    gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"
    gpu_mem = torch.cuda.get_device_properties(device).total_mem / 1e9 if torch.cuda.is_available() else 0
    print(f"[Rank {rank}] GPU: {gpu_name} ({gpu_mem:.1f} GB)  |  batch_size={bs}", flush=True)

    # Async layout tokeniser
    async_tok = AsyncLayoutTokeniser(layout_tok, args.max_len, workers=4)

    # DataLoader
    ds = ImagePathDataset(my_imgs, max_size=args.max_size)
    loader = DataLoader(
        ds, batch_size=bs, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        collate_fn=collate_batch, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # Process
    t0 = time.time()
    shard_entries: List[dict] = []
    processed = 0
    skipped = 0
    oom_count = 0

    # If resuming, preload existing entries for combined list
    if done_stems:
        for stem in tqdm(done_stems, desc=f"[R{rank}] Loading resumed", disable=(rank != 0)):
            pt = os.path.join(per_img_dir, f"{stem}.pt")
            if os.path.exists(pt):
                shard_entries.append(torch.load(pt, map_location="cpu"))

    for paths, images in tqdm(loader, desc=f"[R{rank}] VLM→Layout", disable=(rank != 0)):
        valid_p, valid_im, valid_sizes = [], [], []
        for p, im in zip(paths, images):
            stem = Path(p).stem
            if stem in done_stems:
                skipped += 1
                continue
            if im is None:
                entry = empty_layout(args.max_len)
                entry["image_path"] = p
                torch.save(entry, os.path.join(per_img_dir, f"{stem}.pt"))
                shard_entries.append(entry)
                processed += 1
                continue
            valid_p.append(p)
            valid_im.append(im)
            valid_sizes.append(im.size)   # (w, h)

        if not valid_im:
            continue

        # ---- Batched VLM with OOM fallback ----
        try:
            raw_outputs = extract_batch_vlm(
                model, processor, valid_im, device, args.max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            torch.cuda.empty_cache()
            gc.collect()
            raw_outputs = [
                extract_single_safe(model, processor, im, device, args.max_new_tokens)
                for im in valid_im
            ]

        # ---- Async CPU tokenisation (overlaps with next GPU batch) ----
        for p, raw, sz in zip(valid_p, raw_outputs, valid_sizes):
            async_tok.submit(p, raw, sz)
        processed += len(valid_p)

        # ---- Drain periodically + save per-image .pt ----
        if len(async_tok.futures) >= bs * 4:
            entries = async_tok.collect()
            for e in entries:
                stem = Path(e["image_path"]).stem
                torch.save(e, os.path.join(per_img_dir, f"{stem}.pt"))
                shard_entries.append(e)

        del valid_im, raw_outputs
        if processed % (bs * 20) == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Drain remaining
    entries = async_tok.collect()
    for e in entries:
        stem = Path(e["image_path"]).stem
        torch.save(e, os.path.join(per_img_dir, f"{stem}.pt"))
        shard_entries.append(e)
    async_tok.shutdown()

    elapsed = time.time() - t0
    print(f"[Rank {rank}] {processed} new + {skipped} skipped, "
          f"{elapsed:.1f}s, {processed/max(elapsed,1):.2f} img/s, "
          f"OOM: {oom_count}", flush=True)

    # Save shard for merging
    tmp_dir = os.environ.get("TMPDIR", "/tmp")
    shard_path = os.path.join(tmp_dir, f"vlm_layout_shard_rank{rank}.pt")
    torch.save(shard_entries, shard_path)

    # Merge on rank 0
    if is_dist:
        dist.barrier()
    if rank == 0:
        print_rank0("Merging …", rank)
        combined = []
        for r in range(world_size):
            sp = os.path.join(tmp_dir, f"vlm_layout_shard_rank{r}.pt")
            if os.path.exists(sp):
                s = torch.load(sp, map_location="cpu")
                combined.extend(s)
                print(f"  rank {r}: {len(s)} entries", flush=True)

        combined.sort(key=lambda x: x.get("image_path", ""))
        out = os.path.join(args.output_dir, args.output_filename)
        torch.save(combined, out)
        print(f"=== Combined ({len(combined)} images) → {out}")
        print(f"=== Per-image → {per_img_dir}/")

    cleanup_distributed(is_dist)


def parse_args():
    p = argparse.ArgumentParser(description="VLM Layout Extraction (LayoutLMv3) v2")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_filename", default="vlm_layout_all.pt")
    p.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--max_size", type=int, default=1024)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=600)
    p.add_argument("--batch_size", type=int, default=0,
                   help="0 = auto-detect from GPU memory")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--use_compile", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())