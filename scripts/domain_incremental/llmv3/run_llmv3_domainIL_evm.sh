#!/bin/bash -l

#SBATCH --job-name=4_dil_llmv3_evm_std
#SBATCH --output=logs/4_dil/llmv3/%x_%j.out
#SBATCH --error=logs/4_dil/llmv3/%x_%j.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:55:00
#SBATCH --export=NONE

# Step 4 - domain-incremental learning RVL-CDIP -> Tobacco-3482, LayoutLMv3: EVM (L_EVM); standard IL (default) or distillation-based IL.
# Submit from the repo root (paths and classes: scripts/config.sh):
#   sbatch scripts/domain_incremental/llmv3/run_llmv3_domainIL_evm.sh   # standard IL
#   sbatch -J 4_dil_llmv3_evm_kd scripts/domain_incremental/llmv3/run_llmv3_domainIL_evm.sh distillation   # distillation-based IL

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad-Extension/scripts/config.sh

STRATEGY=${1:-standard}                  # standard | distillation (KD keeps a teacher copy in GPU memory)
case "$STRATEGY" in standard) STRAT_TAG=std ;; distillation) STRAT_TAG=kd ;; *) echo "ERROR: strategy must be standard|distillation" >&2; exit 1 ;; esac
CKPT_DIR=$DIL_ROOT/llmv3/evm_$STRAT_TAG
require_file "$LLMV3_BASE_16" "train the base model first"
RESUME_CKPT=""                           # resume: a checkpoint in $CKPT_DIR
mkdir -p "$CKPT_DIR"

python src/domain_incremental/llmv3/llmv3_domain_incremental_evm_training.py \
  --data_dir "$DATA_ROOT" \
  --ocr_tensor_path_base "$LLMV3_OCR_RVL" \
  --ocr_tensor_path_inc "$LLMV3_OCR_TOB" \
  --all_classes "$DIL_CLASSES" \
  --dataset_base rvl_cdip \
  --dataset_inc tobacco3482 \
  --base_model_path "$LLMV3_BASE_16" \
  --checkpoint_dir "$CKPT_DIR" \
  --batch_size 32 \
  --lr 1e-3 \
  --num_epochs 30 \
  --use_ewc \
  --lambda_ewc 5000.0 \
  --strategy "$STRATEGY" \
  --use_bias_correction \
  --max_exemplars 16 \
  --exemplar_selection herding \
  --training_mode last_layer \
  --full_model_acc "$LLMV3_BASE_16_ACC" \
  ${RESUME_CKPT:+--resume --resume_checkpoint "$RESUME_CKPT"}

echo "Finished. Checkpoints: $CKPT_DIR"
