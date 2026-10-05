#!/bin/bash -l

#SBATCH --job-name=2_base_llmv3_16cls
#SBATCH --output=logs/2_base/llmv3/%x_%j.out
#SBATCH --error=logs/2_base/llmv3/%x_%j.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:59:00
#SBATCH --export=NONE

# Step 2 - base model: LayoutLMv3 on all 16 RVL-CDIP classes (12,500 images per class, bert-base-uncased OCR).
# Setup: 12,500 training images per class (the same documents as the EAML base, utils/data_subset.py),
# official val split; AdamW with a fixed lr of 2e-5, effective batch 64 (8 x 8 accumulation), 20,000 steps (LayoutLMv3 paper).
# Output: $LLMV3_BASE_16 (read by the LayoutLMv3 domain-incremental scripts (step 4)).
# Submit from the repo root (paths and classes: scripts/config.sh):
#   sbatch scripts/base_models/run_llmv3.sh                          # Custom LayoutLMv3
#   sbatch --export=LLMV3_MODEL=hf scripts/base_models/run_llmv3.sh   # pre-trained LayoutLMv3 (hf OCR tensors)

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

OUTPUT_DIR=$(dirname "$LLMV3_BASE_16")
RESUME_CKPT=""                           # resume: "$OUTPUT_DIR/layoutlmv3_rvl_cdip_best.pt"
RESUME_EPOCH=0                           # ... and the epoch it was saved at
mkdir -p "$OUTPUT_DIR"

python src/base_models/sota_llmv3_model.py \
  --model_type "$LLMV3_MODEL" \
  --dataset rvl_cdip \
  --image_dir "$RVL_DIR" \
  --ocr_tensor_file "$LLMV3_OCR_RVL" \
  --save_dir "$OUTPUT_DIR" \
  --images_per_class 12500 \
  --batch_size 8 \
  --grad_accum_steps 8 \
  --max_steps 20000 \
  --epochs 50 \
  --lr 2e-5 \
  --max_length 512 \
  --device cuda \
  --bbox_style rect \
  --patience 10 \
  --seed 42 \
  ${RESUME_CKPT:+--resume "$RESUME_CKPT" --resume_epoch "$RESUME_EPOCH"}

echo "Finished. Best model: $LLMV3_BASE_16"
