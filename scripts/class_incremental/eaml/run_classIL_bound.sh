#!/bin/bash -l

#SBATCH --job-name=3_cil_eaml_finetune
#SBATCH --output=logs/3_cil/eaml/%x_step%a_%A.out
#SBATCH --error=logs/3_cil/eaml/%x_step%a_%A.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:00:00
#SBATCH --export=NONE

# Step 3 - class-incremental learning, EAML: lower and upper bound (plan item A.6), standard IL.
#   finetune  lower bound: the new class's training data only; no exemplars, no EWC, no bias correction
#   joint     upper bound: all training data of all classes seen so far (as expensive as a base model per step;
#             step 5 equals the 16-class base model, so only steps 1-4 would be new)
# Submit from the repo root (paths and classes: scripts/config.sh):
#   sbatch --array=1 scripts/class_incremental/eaml/run_classIL_bound.sh finetune   # then --array=2 ... 5
#   sbatch -J 3_cil_eaml_joint --array=1 scripts/class_incremental/eaml/run_classIL_bound.sh joint

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

STEP=${SLURM_ARRAY_TASK_ID:?submit with: sbatch --array=<step> <script> finetune|joint}
BOUND=${1:?usage: sbatch --array=<step> scripts/class_incremental/eaml/run_classIL_bound.sh finetune|joint}
case "$BOUND" in
    finetune) BOUND_ARGS=() ;;
    joint)    BOUND_ARGS=(--joint_training) ;;
    *) echo "ERROR: bound must be finetune or joint" >&2; exit 1 ;;
esac
CKPT_DIR=$CIL_ROOT/eaml/${BOUND}_std/seed$SEED
cil_step "$STEP" "$CKPT_DIR" "$EAML_BASE_11" "best_model_{class}.pth"
RESUME_CKPT=""                           # resume this step: "$CKPT_DIR/epoch<N>_$UNSEEN_CLASSES.pth"
mkdir -p "$CKPT_DIR"

python src/class_incremental/eaml/class_incremental.py \
  --seed "$SEED" \
  --data_dir "$RVL_DIR" \
  --ocr_tensor_path "$EAML_OCR_RVL" \
  --all_classes "$ALL_CLASSES" \
  --base_classes "$BASE_CLASSES" \
  --unseen_classes "$UNSEEN_CLASSES" \
  --base_model_path "$BASE_MODEL" \
  --model_name eaml \
  --checkpoint_dir "$CKPT_DIR" \
  --batch_size 16 \
  --lr 1e-3 \
  --num_epochs 100 \
  --strategy standard \
  --training_mode last_layer \
  --full_model_acc "$EAML_BASE_11_ACC" \
  --patience 10 \
  "${BOUND_ARGS[@]}" \
  ${RESUME_CKPT:+--resume --resume_checkpoint "$RESUME_CKPT"}

echo "Finished. Checkpoints: $CKPT_DIR"
