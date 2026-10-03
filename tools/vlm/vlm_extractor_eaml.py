"""
VLM Content-Only Extraction for EAML Model  (v2 — GPU-optimised)
===================================================================
Replaces OCR content extraction with Qwen2-VL based text extraction.
Output: single .pt file  →  dict[filename] = {"input_ids": tensor, "attention_mask": tensor}
Tokenizer: BertTokenizer (bert-base-uncased), matching existing EAML pipeline.

GPU optimisations over v1:
  • Batched VLM inference  — multiple images per forward pass
  • Async CPU post-processing  — BERT tokenisation overlaps with next GPU batch
  • Auto batch-size selection  — probes GPU memory at startup
  • CUDA pin_memory + non-blocking transfers
  • OOM-safe: graceful fallback to bs=1 if a batch causes OOM
  • torch.compile support (PyTorch ≥ 2.1, optional)

Optimised for FAU TinyGPU (V100 32 GB / A100 40 GB) with torch.distributed.

Usage:
    torchrun --nproc_per_node=NUM_GPUS vlm_extract_content.py [args]
    python vlm_extract_content.py [args]          # single-GPU fallback
"""

import os
import gc
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BertTokenizer,
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
#  Dataset — workers pre-load + resize images in parallel
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
#  Auto batch-size from GPU memory
# ---------------------------------------------------------------------------

def auto_batch_size(device: torch.device, model_gb: float = 5.0,
                    per_img_gb: float = 1.8, user_bs: int = 0) -> int:
    """
    Estimate safe batch size.
    Qwen2-VL-2B fp16 ≈ 5 GB.   Each image in forward ≈ 1.5-2 GB.
    """
    if user_bs > 0:
        return user_bs
    if not torch.cuda.is_available():
        return 1

    free_gb = torch.cuda.mem_get_info(device)[0] / (1024 ** 3)
    usable = free_gb - model_gb - 1.5          # safety margin
    bs = max(1, int(usable / per_img_gb))

    name = torch.cuda.get_device_name(device).lower()
    if "a100" in name:
        bs = min(bs, 6)
    elif "v100" in name:
        bs = min(bs, 4)
    else:
        bs = min(bs, 2)
    return max(1, bs)


# ---------------------------------------------------------------------------
#  VLM prompt
# ---------------------------------------------------------------------------

VLM_PROMPT = (
    "Extract all visible text from this document image. "
    "Return only the extracted text, nothing else. Be precise and comprehensive."
)


# ---------------------------------------------------------------------------
#  Batched VLM inference
# ---------------------------------------------------------------------------

def extract_texts_batched(
    model, processor, images: List[Image.Image],
    device: torch.device, max_new_tokens: int = 512,
) -> List[str]:
    """Run Qwen2-VL on a BATCH of images.  Returns texts in same order."""
    batch_messages = []
    for img in images:
        batch_messages.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": VLM_PROMPT},
            ],
        }])

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
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1,
        )

    input_len = inputs.input_ids.shape[1]
    results = []
    for i in range(len(images)):
        trimmed = gen_ids[i][input_len:]
        text = processor.decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        results.append(text.strip())
    return results


def extract_single_safe(model, processor, img, device, max_new_tokens):
    """Single-image fallback (used after OOM)."""
    try:
        return extract_texts_batched(model, processor, [img], device, max_new_tokens)[0]
    except Exception:
        torch.cuda.empty_cache()
        return ""


# ---------------------------------------------------------------------------
#  Async BERT tokeniser (runs on CPU threads while GPU works on next batch)
# ---------------------------------------------------------------------------

class AsyncBERTTokeniser:
    def __init__(self, tokenizer: BertTokenizer, max_len: int, workers: int = 4):
        self.tok = tokenizer
        self.max_len = max_len
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.futures = []

    def submit(self, key: str, text: str):
        self.futures.append(self.pool.submit(self._run, key, text))

    def _run(self, key: str, text: str) -> Tuple[str, dict]:
        enc = self.tok(text, padding="max_length", truncation=True,
                       max_length=self.max_len, return_tensors="pt")
        return key, {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

    def collect(self) -> Dict[str, dict]:
        out = {}
        for f in self.futures:
            k, v = f.result()
            out[k] = v
        self.futures.clear()
        return out

    def shutdown(self):
        self.pool.shutdown(wait=True)


# ---------------------------------------------------------------------------
#  Offline model check  (HPC compute nodes have no internet)
# ---------------------------------------------------------------------------

def preflight_check(model_name: str, rank: int):
    """Verify models are cached locally before attempting to load.
    If HF_HUB_OFFLINE=1 and models aren't cached, from_pretrained() hangs
    or gives a cryptic error. This gives a clear message instead.
    """
    if rank != 0:
        return  # only check on rank 0

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

    print_rank0("=== VLM Content Extraction (EAML) — v2 GPU-optimised ===", rank)
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

    # Resume
    tmp_dir = os.environ.get("TMPDIR", "/tmp")
    ckpt_path = os.path.join(tmp_dir, f"vlm_content_rank{rank}.pt")
    done_keys: set = set()
    partial: Dict[str, dict] = {}
    if os.path.exists(ckpt_path):
        partial = torch.load(ckpt_path, map_location="cpu")
        done_keys = set(partial.keys())
        print(f"[Rank {rank}] Resumed {len(done_keys)}", flush=True)

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

    # Auto batch size
    bs = auto_batch_size(device, user_bs=args.batch_size)
    gpu_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"
    gpu_mem = torch.cuda.get_device_properties(device).total_mem / 1e9 if torch.cuda.is_available() else 0
    print(f"[Rank {rank}] GPU: {gpu_name} ({gpu_mem:.1f} GB)  |  batch_size={bs}", flush=True)

    # Async BERT tokeniser
    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    async_tok = AsyncBERTTokeniser(bert_tok, args.max_length, workers=4)

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
    processed = 0
    oom_count = 0
    empty_tensor = lambda: {
        "input_ids": torch.zeros(args.max_length, dtype=torch.long),
        "attention_mask": torch.zeros(args.max_length, dtype=torch.long),
    }

    for paths, images in tqdm(loader, desc=f"[R{rank}] VLM→BERT", disable=(rank != 0)):
        valid_p, valid_im = [], []
        for p, im in zip(paths, images):
            k = os.path.basename(p)
            if k in done_keys:
                continue
            if im is None:
                partial[k] = empty_tensor()
                processed += 1
                continue
            valid_p.append(p)
            valid_im.append(im)

        if not valid_im:
            continue

        # ---- Batched VLM with OOM fallback ------------------------------------
        try:
            texts = extract_texts_batched(
                model, processor, valid_im, device, args.max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            torch.cuda.empty_cache()
            gc.collect()
            texts = [extract_single_safe(model, processor, im, device, args.max_new_tokens)
                     for im in valid_im]

        # ---- Async BERT tokenisation (CPU, overlaps with next GPU batch) ------
        for p, txt in zip(valid_p, texts):
            async_tok.submit(os.path.basename(p), txt)
        processed += len(valid_p)

        # Periodically drain async results + checkpoint
        if processed % args.save_every < bs * 2:
            partial.update(async_tok.collect())
            torch.save(partial, ckpt_path)

        # Memory housekeeping
        del valid_im, texts
        if processed % (bs * 20) == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Drain remaining
    partial.update(async_tok.collect())
    async_tok.shutdown()

    # Save shard
    shard_path = os.path.join(tmp_dir, f"vlm_content_shard_rank{rank}.pt")
    torch.save(partial, shard_path)
    elapsed = time.time() - t0
    print(f"[Rank {rank}] {processed} imgs, {elapsed:.1f}s, "
          f"{processed/max(elapsed,1):.2f} img/s, OOM fallbacks: {oom_count}", flush=True)

    # Merge on rank 0
    if is_dist:
        dist.barrier()
    if rank == 0:
        print_rank0("Merging …", rank)
        merged = {}
        for r in range(world_size):
            sp = os.path.join(tmp_dir, f"vlm_content_shard_rank{r}.pt")
            if os.path.exists(sp):
                s = torch.load(sp, map_location="cpu")
                merged.update(s)
                print(f"  rank {r}: {len(s)} entries", flush=True)
        os.makedirs(args.output_dir, exist_ok=True)
        out = os.path.join(args.output_dir, args.output_filename)
        torch.save(merged, out)
        print(f"=== Saved ({len(merged)} images) → {out}")

    cleanup_distributed(is_dist)


def parse_args():
    p = argparse.ArgumentParser(description="VLM Content Extraction (EAML) v2")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_filename", default="vlm_content_all.pt")
    p.add_argument("--model_name", default="Qwen/Qwen2-VL-2B-Instruct")
    p.add_argument("--max_size", type=int, default=1024)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=0,
                   help="0 = auto-detect from GPU memory")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--use_compile", action="store_true",
                   help="torch.compile (experimental, needs PyTorch ≥ 2.1)")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())