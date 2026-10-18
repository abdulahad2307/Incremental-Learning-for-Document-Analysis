#!/bin/bash -l

#SBATCH --job-name=3_cil_llmv3_derpp_std
#SBATCH --output=logs/3_cil/llmv3/%x_step%a_%A.out
#SBATCH --error=logs/3_cil/llmv3/%x_step%a_%A.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:00:00
#SBATCH --export=NONE

# Step 3 - class-incremental learning, LayoutLMv3: DER++ (Buzzega et al., 2020; plan item 15). Exemplars (20 per
# class, herding) are replayed with the logits stored when they entered the memory; the memory is saved next to
# each step's best checkpoint and loaded by the next step. No EWC, bias correction, balanced sampler or KD.
# Submit from the repo root (paths and classes: scripts/config.sh):
#   sbatch --array=1 scripts/class_incremental/llmv3/run_llmv3_classIL_derpp.sh   # step 1, standard IL; then --array=2 ... 5

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

STEP=${SLURM_ARRAY_TASK_ID:?submit with: sbatch --array=<step> <script>}
CKPT_DIR=$CIL_ROOT/$LLMV3_TAG/derpp_std/seed$SEED
cil_step "$STEP" "$CKPT_DIR" "$LLMV3_BASE_11" "layoutlmv3_cil_incremental_{class}_best.pt"
RESUME_CKPT=""                           # resume this step: a checkpoint of this step in $CKPT_DIR
mkdir -p "$CKPT_DIR"

python src/class_incremental/llmv3/llmv3_class_incremental.py \
  --seed "$SEED" \
  --data_dir "$RVL_DIR" \
  --ocr_tensor_path "$LLMV3_OCR_RVL" \
  --all_classes "$ALL_CLASSES" \
  --base_classes "$BASE_CLASSES" \
  --unseen_classes "$UNSEEN_CLASSES" \
  --base_model_path "$BASE_MODEL" \
  --checkpoint_dir "$CKPT_DIR" \
  --batch_size 64 \
  --lr 2e-5 \
  --num_epochs 100 \
  --strategy standard \
  --max_exemplars 20 \
  --derpp \
  --derpp_alpha 0.1 \
  --derpp_beta 0.5 \
  --exemplar_selection herding \
  --training_mode last_layer \
  --base_model_acc "$LLMV3_BASE_11_ACC" \
  --full_model_acc "$LLMV3_BASE_11_ACC" \
  ${RESUME_CKPT:+--resume --resume_checkpoint "$RESUME_CKPT"}

echo "Finished. Checkpoints: $CKPT_DIR"
