#!/bin/bash -l

#SBATCH --job-name=3_cil_llmv3_evm_std
#SBATCH --output=logs/3_cil/llmv3/%x_step%a_%A.out
#SBATCH --error=logs/3_cil/llmv3/%x_step%a_%A.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:00:00
#SBATCH --export=NONE

# Step 3 - class-incremental learning, LayoutLMv3: EVM (L_EVM); standard IL (default) or distillation-based IL.
# Submit from the repo root (paths and classes: scripts/config.sh):
#   sbatch --array=1 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_training.sh   # step 1, standard IL; then --array=2 ... 5
#   sbatch -J 3_cil_llmv3_evm_kd --array=1 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_training.sh distillation   # distillation-based IL

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

STEP=${SLURM_ARRAY_TASK_ID:?submit with: sbatch --array=<step> <script>}
STRATEGY=${1:-standard}                  # standard | distillation (KD keeps a teacher copy in GPU memory)
case "$STRATEGY" in standard) STRAT_TAG=std ;; distillation) STRAT_TAG=kd ;; *) echo "ERROR: strategy must be standard|distillation" >&2; exit 1 ;; esac
CKPT_DIR=$CIL_ROOT/llmv3/evm_$STRAT_TAG
cil_step "$STEP" "$CKPT_DIR" "$LLMV3_BASE_11" "layoutlmv3_cil_incremental_evm_{class}_best.pt"
RESUME_CKPT=""                           # resume this step: a checkpoint of this step in $CKPT_DIR
mkdir -p "$CKPT_DIR"

python src/class_incremental/llmv3/llmv3_class_incremental_evm_training.py \
  --data_dir "$RVL_DIR" \
  --ocr_tensor_path "$LLMV3_OCR_RVL" \
  --all_classes "$ALL_CLASSES" \
  --base_classes "$BASE_CLASSES" \
  --unseen_classes "$UNSEEN_CLASSES" \
  --base_model_path "$BASE_MODEL" \
  --checkpoint_dir "$CKPT_DIR" \
  --batch_size 64 \
  --lr 1e-3 \
  --num_epochs 100 \
  --use_ewc \
  --lambda_ewc 5000.0 \
  --strategy "$STRATEGY" \
  --use_bias_correction \
  --max_exemplars 32 \
  --exemplar_selection herding \
  --training_mode last_layer \
  --base_model_acc "$LLMV3_BASE_11_ACC" \
  --full_model_acc "$LLMV3_BASE_11_ACC" \
  --evm_persist \
  ${RESUME_CKPT:+--resume --resume_checkpoint "$RESUME_CKPT"}

echo "Finished. Checkpoints: $CKPT_DIR"
