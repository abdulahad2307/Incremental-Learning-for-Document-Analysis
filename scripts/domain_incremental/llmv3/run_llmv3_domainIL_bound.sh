#!/bin/bash -l

#SBATCH --job-name=4_dil_llmv3_finetune
#SBATCH --output=logs/4_dil/llmv3/%x_%j.out
#SBATCH --error=logs/4_dil/llmv3/%x_%j.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:55:00
#SBATCH --export=NONE

# Step 4 - domain-incremental learning RVL-CDIP -> Tobacco-3482, LayoutLMv3: lower and upper bound (plan item A.6).
#   finetune  lower bound: Tobacco-3482 only; no replay, no EWC, no KD, no bias correction
#   joint     upper bound: Tobacco-3482 and the base model's RVL-CDIP training documents, shuffled together
#             (an epoch covers RVL-CDIP's training set; resume with RESUME_CKPT if it hits the time limit)
# Submit from the repo root (paths and classes: scripts/config.sh; pre-trained LayoutLMv3: --export=LLMV3_MODEL=hf):
#   sbatch scripts/domain_incremental/llmv3/run_llmv3_domainIL_bound.sh finetune
#   sbatch -J 4_dil_llmv3_joint scripts/domain_incremental/llmv3/run_llmv3_domainIL_bound.sh joint

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

BOUND=${1:?usage: sbatch scripts/domain_incremental/llmv3/run_llmv3_domainIL_bound.sh finetune|joint}
case "$BOUND" in
    finetune) BOUND_ARGS=(--max_exemplars 0) ;;
    joint)    BOUND_ARGS=(--joint_training) ;;
    *) echo "ERROR: bound must be finetune or joint" >&2; exit 1 ;;
esac
CKPT_DIR=$DIL_ROOT/$LLMV3_TAG/${BOUND}_std/seed$SEED
require_file "$LLMV3_BASE_16" "train the base model first"
RESUME_CKPT=""                           # resume: a checkpoint in $CKPT_DIR
mkdir -p "$CKPT_DIR"

python src/domain_incremental/llmv3/llmv3_domain_incremental.py \
  --seed "$SEED" \
  --data_dir "$DATA_ROOT" \
  --ocr_tensor_path_base "$LLMV3_OCR_RVL" \
  --ocr_tensor_path_inc "$LLMV3_OCR_TOB" \
  --all_classes "$DIL_CLASSES" \
  --dataset_base rvl_cdip \
  --dataset_inc tobacco3482 \
  --base_model_path "$LLMV3_BASE_16" \
  --checkpoint_dir "$CKPT_DIR" \
  --batch_size 64 \
  --lr 2e-5 \
  --num_epochs 100 \
  --strategy standard \
  --training_mode last_layer \
  --full_model_acc "$LLMV3_BASE_16_ACC" \
  "${BOUND_ARGS[@]}" \
  ${RESUME_CKPT:+--resume --resume_checkpoint "$RESUME_CKPT"}

echo "Finished. Checkpoints: $CKPT_DIR"
