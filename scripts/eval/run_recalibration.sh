#!/bin/bash -l

#SBATCH --job-name=5_recalibration
#SBATCH --output=logs/5_eval/%x_%j.out
#SBATCH --error=logs/5_eval/%x_%j.err
#SBATCH --partition=rtx3080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=12:00:00
#SBATCH --export=NONE

# Classifier recalibration after the last CIL step with the rebuilt replay memory (tools/eval/recalibration.py):
# NCM on the memory and weight aligning, for No EVM, EVM (three seeds) and fine-tuning. Needs the drift features of
# step 5 (scripts/eval/run_drift_features.sh). Output: $RUN_ROOT/analysis/recalibration/ (both backbones in one table)
# Submit from the repo root, one job per backbone (run them one after the other, they share the output table):
#   sbatch scripts/eval/run_recalibration.sh eaml
#   sbatch --export=LLMV3_MODEL=hf scripts/eval/run_recalibration.sh llmv3hf

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

BACKBONE=${1:?backbone: eaml or llmv3hf}
case "$BACKBONE" in
    eaml) OCR=$EAML_OCR_RVL ;;
    llmv3hf) [ "$LLMV3_MODEL" = hf ] || { echo "ERROR: submit with --export=LLMV3_MODEL=hf" >&2; exit 1; }; OCR=$LLMV3_OCR_RVL ;;
    *) echo "ERROR: backbone must be eaml or llmv3hf" >&2; exit 1 ;;
esac
python tools/eval/recalibration.py --backbone "$BACKBONE" --drift "$RUN_ROOT/features/drift" --data_dir "$RVL_DIR" \
  --ocr "$OCR" --out "$RUN_ROOT/analysis/recalibration" "${@:2}"
