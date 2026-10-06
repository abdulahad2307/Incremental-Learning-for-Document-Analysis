#!/bin/bash -l

#SBATCH --job-name=3_cil_eaml_evm_std
#SBATCH --output=logs/3_cil/eaml/%x_step%a_%A.out
#SBATCH --error=logs/3_cil/eaml/%x_step%a_%A.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:58:00
#SBATCH --export=NONE

# Step 3 - class-incremental learning, EAML: EVM (L_EVM), standard IL.
# Submit from the repo root (paths and classes: scripts/config.sh):
#   sbatch --array=1 scripts/class_incremental/eaml/run_classIL_evm_training.sh   # step 1 (+scientific_publication)
#   then --array=2 ... --array=5, each after the previous step has finished

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

STEP=${SLURM_ARRAY_TASK_ID:?submit with: sbatch --array=<step> <script>}
CKPT_DIR=$CIL_ROOT/eaml/evm_std/seed$SEED
cil_step "$STEP" "$CKPT_DIR" "$EAML_BASE_11" "best_model_{class}.pth"
RESUME_CKPT=""                           # resume this step: "$CKPT_DIR/epoch<N>_$UNSEEN_CLASSES.pth"
mkdir -p "$CKPT_DIR"

python src/class_incremental/eaml/class_incremental_evm_training.py \
  --seed "$SEED" \
  --data_dir "$RVL_DIR" \
  --ocr_tensor_path "$EAML_OCR_RVL" \
  --checkpoint_dir "$CKPT_DIR" \
  --model_name eaml \
  --all_classes "$ALL_CLASSES" \
  --base_classes "$BASE_CLASSES" \
  --unseen_classes "$UNSEEN_CLASSES" \
  --batch_size 64 \
  --lr 1e-3 \
  --num_epochs 100 \
  --strategy standard \
  --temperature 2.0 \
  --lambda_distill 1.0 \
  --lambda_ewc 5000.0 \
  --use_ewc \
  --use_exemplars \
  --use_balanced_sampler \
  --use_bias_correction \
  --max_exemplars 320 \
  --exemplar_selection herding \
  --training_mode last_layer \
  --base_model_path "$BASE_MODEL" \
  --evm_tailsize 0.3 \
  --evm_threshold 0.7 \
  --full_model_acc "$EAML_BASE_11_ACC" \
  --patience 10 \
  --evm_persist \
  ${RESUME_CKPT:+--resume --resume_checkpoint "$RESUME_CKPT"}

echo "Finished. Checkpoints: $CKPT_DIR"
