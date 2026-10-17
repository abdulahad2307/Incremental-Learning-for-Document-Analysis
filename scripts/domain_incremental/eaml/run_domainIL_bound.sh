#!/bin/bash -l

#SBATCH --job-name=4_dil_eaml_finetune
#SBATCH --output=logs/4_dil/eaml/%x_%j.out
#SBATCH --error=logs/4_dil/eaml/%x_%j.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:00:00
#SBATCH --export=NONE

# Step 4 - domain-incremental learning RVL-CDIP -> Tobacco-3482, EAML: lower and upper bound (plan item A.6).
#   finetune  lower bound: Tobacco-3482 only; no replay, no EWC, no KD, no bias correction
#   joint     upper bound: Tobacco-3482 and the base model's RVL-CDIP training documents, shuffled together
#             (an epoch covers RVL-CDIP's training set; resume with RESUME_CKPT if it hits the time limit)
# Submit from the repo root (paths and classes: scripts/config.sh):
#   sbatch scripts/domain_incremental/eaml/run_domainIL_bound.sh finetune
#   sbatch -J 4_dil_eaml_joint scripts/domain_incremental/eaml/run_domainIL_bound.sh joint

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

BOUND=${1:?usage: sbatch scripts/domain_incremental/eaml/run_domainIL_bound.sh finetune|joint}
case "$BOUND" in
    finetune) BOUND_ARGS=() ;;
    joint)    BOUND_ARGS=(--joint_training) ;;
    *) echo "ERROR: bound must be finetune or joint" >&2; exit 1 ;;
esac
CKPT_DIR=$DIL_ROOT/eaml/${BOUND}_std/seed$SEED
require_file "$EAML_BASE_16" "train the base model first"
require_file "$DATA_ROOT/$TOB_DOMAIN"
RESUME_CKPT=""                           # resume: a checkpoint in $CKPT_DIR
mkdir -p "$CKPT_DIR"

python src/domain_incremental/eaml/domain_incremental.py \
  --seed "$SEED" \
  --data_dir "$DATA_ROOT" \
  --ocr_tensor_dirs "$EAML_OCR_RVL" "$EAML_OCR_TOB" \
  --domains "$RVL_DOMAIN,$TOB_DOMAIN" \
  --global_classes "$DIL_CLASSES" \
  --eaml_ckpt_path "$EAML_BASE_16" \
  --checkpoint_dir "$CKPT_DIR" \
  --batch_size 16 \
  --lr 1e-3 \
  --num_epochs 100 \
  --strategy standard \
  --finetune_mode last_layer \
  --unfreeze_depth 2 \
  "${BOUND_ARGS[@]}" \
  ${RESUME_CKPT:+--resume --resume_ckpt_path "$RESUME_CKPT"}

echo "Finished. Checkpoints: $CKPT_DIR"
