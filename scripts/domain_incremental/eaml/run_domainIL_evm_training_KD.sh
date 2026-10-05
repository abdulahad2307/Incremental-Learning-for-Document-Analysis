#!/bin/bash -l

#SBATCH --job-name=4_dil_eaml_evm_kd
#SBATCH --output=logs/4_dil/eaml/%x_%j.out
#SBATCH --error=logs/4_dil/eaml/%x_%j.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:00:00
#SBATCH --export=NONE

# Step 4 - domain-incremental learning RVL-CDIP -> Tobacco-3482, EAML: EVM (L_EVM), distillation-based IL.
# Submit from the repo root (paths and classes: scripts/config.sh):
#   sbatch scripts/domain_incremental/eaml/run_domainIL_evm_training_KD.sh

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

CKPT_DIR=$DIL_ROOT/eaml/evm_kd
require_file "$EAML_BASE_16" "train the base model first"
require_file "$DATA_ROOT/$TOB_DOMAIN"
RESUME_CKPT=""                           # resume: a checkpoint in $CKPT_DIR
mkdir -p "$CKPT_DIR"

python src/domain_incremental/eaml/domain_incremental_evm_training.py \
  --data_dir "$DATA_ROOT" \
  --ocr_tensor_dirs "$EAML_OCR_RVL" "$EAML_OCR_TOB" \
  --domains "$RVL_DOMAIN,$TOB_DOMAIN" \
  --global_classes "$DIL_CLASSES" \
  --eaml_ckpt_path "$EAML_BASE_16" \
  --checkpoint_dir "$CKPT_DIR" \
  --batch_size 32 \
  --lr 1e-4 \
  --num_epochs 100 \
  --strategy distillation \
  --temperature 2.0 \
  --lambda_distill 1.0 \
  --lambda_ewc 5000 \
  --use_ewc \
  --use_exemplars \
  --max_exemplars 320 \
  --use_bias_correction \
  --finetune_mode last_layer \
  --unfreeze_depth 2 \
  --evm_tailsize 0.3 \
  --evm_threshold 0.7 \
  --lambda_evm 0.1 \
  ${RESUME_CKPT:+--resume --resume_ckpt_path "$RESUME_CKPT"}

echo "Finished. Checkpoints: $CKPT_DIR"
