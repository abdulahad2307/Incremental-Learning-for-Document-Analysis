#!/bin/bash -l

#SBATCH --job-name=2_base_llmv3_11cls
#SBATCH --output=logs/2_base/llmv3/%x_%j.out
#SBATCH --error=logs/2_base/llmv3/%x_%j.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:59:00
#SBATCH --export=NONE

# Step 2 - base model: LayoutLMv3 on the 11 CIL base classes of RVL-CDIP (12,500 images per class, bert-base-uncased OCR).
# Output: $LLMV3_BASE_11 (read by the LayoutLMv3 class-incremental scripts (step 3)).
# Submit from the repo root (paths and classes: scripts/config.sh):
#   sbatch scripts/base_models/run_llmv3_11class.sh                          # Custom LayoutLMv3
#   sbatch --export=LLMV3_MODEL=hf scripts/base_models/run_llmv3_11class.sh   # pre-trained LayoutLMv3 (hf OCR tensors)

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

OUTPUT_DIR=$(dirname "$LLMV3_BASE_11")
RESUME_CKPT=""                           # resume: "$OUTPUT_DIR/layoutlmv3_rvl_cdip_best.pt"
RESUME_EPOCH=0                           # ... and the epoch it was saved at
mkdir -p "$OUTPUT_DIR"

python src/base_models/sota_llmv3_model.py \
  --model_type "$LLMV3_MODEL" \
  --dataset rvl_cdip \
  --image_dir "$RVL_DIR" \
  --ocr_tensor_file "$LLMV3_OCR_RVL" \
  --base_classes "$CIL_BASE_CLASSES" \
  --save_dir "$OUTPUT_DIR" \
  --images_per_class 12500 \
  --batch_size 8 \
  --epochs 50 \
  --lr 2e-5 \
  --max_length 512 \
  --device cuda \
  --bbox_style rect \
  --patience 10 \
  --seed 42 \
  ${RESUME_CKPT:+--resume "$RESUME_CKPT" --resume_epoch "$RESUME_EPOCH"}

echo "Finished. Best model: $LLMV3_BASE_11"
