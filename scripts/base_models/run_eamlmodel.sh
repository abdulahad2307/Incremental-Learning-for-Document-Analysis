#!/bin/bash -l

#SBATCH --job-name=2_base_eaml_11cls
#SBATCH --output=logs/2_base/eaml/%x_%j.out
#SBATCH --error=logs/2_base/eaml/%x_%j.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:59:00
#SBATCH --export=NONE

# Step 2 - base model: EAML on the 11 CIL base classes of RVL-CDIP (Tesseract OCR).
# Output: $EAML_BASE_11 (read by the EAML class-incremental scripts (step 3)).
# Submit from the repo root (paths and classes: scripts/config.sh):
#   sbatch scripts/base_models/run_eamlmodel.sh

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

OUTPUT_DIR=$(dirname "$EAML_BASE_11")
RESUME_CKPT=""                           # resume: "$OUTPUT_DIR/eaml_checkpoint_ep<N>.pt"
mkdir -p "$OUTPUT_DIR"

python src/base_models/sota_eaml_model.py \
  --data_dir "$RVL_DIR" \
  --ocr_data_path "$EAML_OCR_RVL" \
  --output_dir "$OUTPUT_DIR" \
  --classes "$CIL_BASE_CLASSES" \
  --class_mapping_path "$REPO/configs/class_mapping.json" \
  --num_epochs 100 \
  --batch_size 32 \
  --learning_rate 1e-3 \
  --weight_decay 0.01 \
  --device cuda \
  --patience 10 \
  --keep_checkpoints 2 \
  --cls_weight 1.0 \
  --kld_weight 0.5 \
  --kld_threshold 0.1 \
  --embed_dim 512 \
  --dropout_rate 0.5 \
  ${RESUME_CKPT:+--resume "$RESUME_CKPT"}

echo "Finished. Best model: $EAML_BASE_11"
