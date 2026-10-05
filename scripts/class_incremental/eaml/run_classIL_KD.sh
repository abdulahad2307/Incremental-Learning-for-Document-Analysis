#!/bin/bash -l

#SBATCH --job-name=3_cil_eaml_noevm_kd
#SBATCH --output=logs/3_cil/eaml/%x_step%a_%A.out
#SBATCH --error=logs/3_cil/eaml/%x_step%a_%A.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:00:00
#SBATCH --export=NONE

# Step 3 - class-incremental learning, EAML: No EVM (ER, EWC, BC), distillation-based IL.
# Submit from the repo root (paths and classes: scripts/config.sh):
#   sbatch --array=1 scripts/class_incremental/eaml/run_classIL_KD.sh   # step 1 (+scientific_publication)
#   then --array=2 ... --array=5, each after the previous step has finished

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

STEP=${SLURM_ARRAY_TASK_ID:?submit with: sbatch --array=<step> <script>}
CKPT_DIR=$CIL_ROOT/eaml/noevm_kd
cil_step "$STEP" "$CKPT_DIR" "$EAML_BASE_11" "best_model_{class}.pth"
RESUME_CKPT=""                           # resume this step: "$CKPT_DIR/epoch<N>_$UNSEEN_CLASSES.pth"
mkdir -p "$CKPT_DIR"

python src/class_incremental/eaml/class_incremental.py \
  --data_dir "$RVL_DIR" \
  --ocr_tensor_path "$EAML_OCR_RVL" \
  --all_classes "$ALL_CLASSES" \
  --base_classes "$BASE_CLASSES" \
  --unseen_classes "$UNSEEN_CLASSES" \
  --base_model_path "$BASE_MODEL" \
  --model_name eaml \
  --checkpoint_dir "$CKPT_DIR" \
  --batch_size 64 \
  --lr 1e-3 \
  --num_epochs 100 \
  --strategy distillation \
  --temperature 2.0 \
  --lambda_distill 1.0 \
  --lambda_ewc 5000.0 \
  --use_ewc \
  --use_exemplars \
  --use_bias_correction \
  --use_balanced_sampler \
  --max_exemplars 320 \
  --exemplar_selection herding \
  --training_mode last_layer \
  --full_model_acc "$EAML_BASE_11_ACC" \
  --weight_decay 0.01 \
  --patience 10 \
  ${RESUME_CKPT:+--resume --resume_checkpoint "$RESUME_CKPT"}

echo "Finished. Checkpoints: $CKPT_DIR"
