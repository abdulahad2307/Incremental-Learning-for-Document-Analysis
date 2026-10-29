#!/bin/bash -l

#SBATCH --job-name=3_cil_eaml_adjrep_std
#SBATCH --output=logs/3_cil/eaml/%x_step%a_%A.out
#SBATCH --error=logs/3_cil/eaml/%x_step%a_%A.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:00:00
#SBATCH --export=NONE

# Step 3 - class-incremental learning, EAML: adjacency-aware replay (utils/class_IL/adjacency.py). No EVM (ER, EWC,
# BC) with the exemplar budget (20 per old class) split by each old class's similarity to the new class.
# Submit from the repo root (paths and classes: scripts/config.sh):
#   sbatch --array=1 scripts/class_incremental/eaml/run_classIL_adjacency.sh                 # standard IL, step 1
#   sbatch -J 3_cil_eaml_adjrep_kd --array=1 scripts/class_incremental/eaml/run_classIL_adjacency.sh distillation
#   then --array=2 ... --array=5, each after the previous step has finished
# REPLAY_WEIGHTING=allocation: old classes drawn in proportion to their exemplars instead of equally often (output
# adjrepw_<std|kd>; utils/class_IL/adjacency.py, replay_weights):
#   sbatch -J 3_cil_eaml_adjrepw_std --array=1 --export=REPLAY_WEIGHTING=allocation scripts/class_incremental/eaml/run_classIL_adjacency.sh

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

STEP=${SLURM_ARRAY_TASK_ID:?submit with: sbatch --array=<step> <script>}
STRATEGY=${1:-standard}
case "$STRATEGY" in standard) STRAT_TAG=std ;; distillation) STRAT_TAG=kd ;; *) echo "ERROR: strategy must be standard|distillation" >&2; exit 1 ;; esac
REPLAY_WEIGHTING=${REPLAY_WEIGHTING:-balanced}   # balanced | allocation
case "$REPLAY_WEIGHTING" in balanced) RUN=adjrep ;; allocation) RUN=adjrepw ;; *) echo "ERROR: REPLAY_WEIGHTING must be balanced|allocation" >&2; exit 1 ;; esac
CKPT_DIR=$CIL_ROOT/eaml/${RUN}_$STRAT_TAG/seed$SEED
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
  --strategy "$STRATEGY" \
  --temperature 2.0 \
  --lambda_distill 1.0 \
  --lambda_ewc 5000.0 \
  --use_ewc \
  --use_exemplars \
  --use_bias_correction \
  --use_balanced_sampler \
  --max_exemplars 320 \
  --exemplar_selection herding \
  --exemplar_allocation adjacency \
  --replay_weighting "$REPLAY_WEIGHTING" \
  --alloc_tau 0.2 \
  --alloc_min 10 \
  --training_mode last_layer \
  --full_model_acc "$EAML_BASE_11_ACC" \
  --patience 10 \
  ${RESUME_CKPT:+--resume --resume_checkpoint "$RESUME_CKPT"}

echo "Finished. Checkpoints: $CKPT_DIR"
