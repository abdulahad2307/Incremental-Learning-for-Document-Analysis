#!/bin/bash -l

#SBATCH --job-name=3_cil_eaml_sweep
#SBATCH --output=logs/3_cil/eaml/%x_step%a_%A.out
#SBATCH --error=logs/3_cil/eaml/%x_step%a_%A.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:v100:1
#SBATCH --time=23:59:00
#SBATCH --export=NONE

# Step 3 - class-incremental learning, EAML: training sweep (plan item 19), standard IL, steps 1-2.
# The run_classIL.sh (noevm) / run_classIL_evm_ood.sh (evmood) setup with one hyperparameter changed by environment:
#   LAMBDA_EVM (0.1), LAMBDA_OOD (0.1)   loss weights of L_EVM and L_OOD (evmood only)
#   MAX_EXEMPLARS (320)                  replay memory size (320 = 20 per class of the 16)
# Output: $RUN_ROOT/sweep/cil/eaml/<noevm|evmood>_std_<changed values>/seed<SEED>, e.g. evmood_std_le1 or noevm_std_m1280.
# The default runs are linked in, so one table compares all (tools/eval/build_results.py --run_root $RUN_ROOT/sweep
# --label_by_run --max_steps 2). Submit from the repo root, each step after the previous one:
#   sbatch --array=1 --export=LAMBDA_EVM=1 scripts/class_incremental/eaml/run_classIL_sweep.sh evmood
#   sbatch --array=2 --dependency=afterok:<step-1 job> --export=LAMBDA_EVM=1 scripts/class_incremental/eaml/run_classIL_sweep.sh evmood

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

METHOD=${1:?usage: sbatch --array=<step> scripts/class_incremental/eaml/run_classIL_sweep.sh noevm|evmood}
LAMBDA_EVM=${LAMBDA_EVM:-0.1}
LAMBDA_OOD=${LAMBDA_OOD:-0.1}
MAX_EXEMPLARS=${MAX_EXEMPLARS:-320}
case "$METHOD" in
    noevm)  SCRIPT=class_incremental.py
            METHOD_ARGS=() ;;
    evmood) SCRIPT=class_incremental_evm_ood.py
            METHOD_ARGS=(--ood_method vim --ood_tpr 0.95 --lambda_evm "$LAMBDA_EVM" --lambda_ood "$LAMBDA_OOD" --evm_persist) ;;
    *) echo "ERROR: method must be noevm or evmood" >&2; exit 1 ;;
esac
RUN=${METHOD}_std
[ "$LAMBDA_EVM" != 0.1 ] && RUN+=_le$LAMBDA_EVM
[ "$LAMBDA_OOD" != 0.1 ] && RUN+=_lo$LAMBDA_OOD
[ "$MAX_EXEMPLARS" != 320 ] && RUN+=_m$MAX_EXEMPLARS
[ "$RUN" = "${METHOD}_std" ] && { echo "ERROR: no hyperparameter changed; the default run is $CIL_ROOT/eaml/$RUN" >&2; exit 1; }

SWEEP_ROOT=$RUN_ROOT/sweep
mkdir -p "$SWEEP_ROOT/cil/eaml"
[ -e "$SWEEP_ROOT/base" ] || ln -s "$RUN_ROOT/base" "$SWEEP_ROOT/base"
for d in noevm_std evmood_std; do
    [ -e "$SWEEP_ROOT/cil/eaml/$d" ] || ln -s "$CIL_ROOT/eaml/$d" "$SWEEP_ROOT/cil/eaml/$d"
done

STEP=${SLURM_ARRAY_TASK_ID:?submit with: sbatch --array=<step> <script>}
CKPT_DIR=$SWEEP_ROOT/cil/eaml/$RUN/seed$SEED
cil_step "$STEP" "$CKPT_DIR" "$EAML_BASE_11" "best_model_{class}.pth"
RESUME_CKPT=""                           # resume this step: "$CKPT_DIR/epoch<N>_$UNSEEN_CLASSES.pth"
mkdir -p "$CKPT_DIR"
echo "Sweep run $RUN: lambda_evm $LAMBDA_EVM, lambda_ood $LAMBDA_OOD, max_exemplars $MAX_EXEMPLARS"

python src/class_incremental/eaml/$SCRIPT \
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
  --temperature 2.0 \
  --lambda_distill 1.0 \
  --lambda_ewc 5000.0 \
  --use_ewc \
  --use_exemplars \
  --max_exemplars "$MAX_EXEMPLARS" \
  --exemplar_selection herding \
  --training_mode last_layer \
  --full_model_acc "$EAML_BASE_11_ACC" \
  --patience 10 \
  --use_balanced_sampler \
  --use_bias_correction \
  "${METHOD_ARGS[@]}" \
  ${RESUME_CKPT:+--resume --resume_checkpoint "$RESUME_CKPT"}

echo "Finished. Checkpoints: $CKPT_DIR"
