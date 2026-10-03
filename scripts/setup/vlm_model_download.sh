#!/bin/bash
###############################################################################
#  Pre-download all models to local disk  (run on frontend node tinyx.nhr.fau.de)
#
#  TinyGPU compute nodes have NO internet access.
#  This script downloads everything needed ONCE on the frontend,
#  then SLURM jobs load from the local cache.
#
#  Usage:
#      bash vlm_model_download.sh
#
#  Run this from the frontend (tinyx.nhr.fau.de), NOT inside a SLURM job.
###############################################################################

set -euo pipefail

# ---------- Where to store models ------------------------------------------
# Using $WORK is recommended: persistent, shared across nodes, large quota.
# $HOME has limited quota and can fill up with large models.
MODEL_CACHE="/home/woody/iwi5/iwi5280h/model_cache"
mkdir -p "${MODEL_CACHE}"

echo "============================================="
echo "  Model Downloader for VLM Extraction"
echo "  Cache directory: ${MODEL_CACHE}"
echo "============================================="

# ---------- Activate your conda env ----------------------------------------
module load python
conda activate ocr_env    # <-- your conda env name

# ---------- Set HuggingFace cache to our persistent directory ---------------
export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}/transformers"
export HF_HUB_CACHE="${MODEL_CACHE}/hub"

# ---------- Download models ------------------------------------------------

echo ""
echo "[1/4] Downloading Qwen2-VL-2B-Instruct (VLM model) ..."
python3 -c "
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
print('  Downloading model weights ...')
Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
print('  Downloading processor ...')
AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
print('  Done.')
"

echo ""
echo "[2/4] Downloading LayoutLMv3 tokenizer ..."
python3 -c "
from transformers import LayoutLMv3TokenizerFast
LayoutLMv3TokenizerFast.from_pretrained('microsoft/layoutlmv3-base')
print('  Done.')
"

echo ""
echo "[3/4] Downloading BERT tokenizer ..."
python3 -c "
from transformers import BertTokenizer
BertTokenizer.from_pretrained('bert-base-uncased')
print('  Done.')
"

echo ""
echo "[4/4] Verifying downloads ..."
python3 -c "
import os
hf_home = os.environ['HF_HOME']
total = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, dn, filenames in os.walk(hf_home)
    for f in filenames
) / (1024**3)
print(f'  Total cache size: {total:.2f} GB')
print(f'  Cache location:   {hf_home}')
"

echo ""
echo "============================================="
echo "  All models downloaded successfully!"
echo ""
echo "  Add these lines to your SLURM scripts:"
echo ""
echo "    export HF_HOME=${MODEL_CACHE}"
echo "    export TRANSFORMERS_CACHE=${MODEL_CACHE}/transformers"
echo "    export HF_HUB_CACHE=${MODEL_CACHE}/hub"
echo "    export HF_HUB_OFFLINE=1"
echo ""
echo "============================================="