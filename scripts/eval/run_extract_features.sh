#!/bin/bash -l

#SBATCH --job-name=5_features
#SBATCH --output=logs/5_eval/%x_%j.out
#SBATCH --error=logs/5_eval/%x_%j.err
#SBATCH --partition=rtx3080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=02:00:00
#SBATCH --export=NONE

# Per-modality features of one CIL checkpoint for the post-hoc EVT comparison (tools/eval/posthoc_evt.py, MEVM
# go / no-go): <which> = base (11-class base model, step 0) or step1 (best model of step 1 of the No-EVM / standard
# run, 12 classes). Output: $RUN_ROOT/features/<eaml|llmv3|llmv3hf>/<which>_cil.pt
# Submit from the repo root:
#   sbatch scripts/eval/run_extract_features.sh eaml base
#   sbatch scripts/eval/run_extract_features.sh eaml step1
#   sbatch --export=LLMV3_MODEL=hf scripts/eval/run_extract_features.sh llmv3 base     # pre-trained LayoutLMv3
# Then (CPU): python tools/eval/posthoc_evt.py $RUN_ROOT/features/eaml/*.pt $RUN_ROOT/features/llmv3hf/*.pt

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

BACKBONE=${1:?usage: sbatch scripts/eval/run_extract_features.sh eaml|llmv3 base|step1}
WHICH=${2:?usage: sbatch scripts/eval/run_extract_features.sh eaml|llmv3 base|step1}
NEW1=${CIL_ORDER[0]}                       # class added in step 1
case "$BACKBONE" in
    eaml)  TAG=eaml; OCR=$EAML_OCR_RVL; BASE=$EAML_BASE_11
           STEP1=$CIL_ROOT/eaml/noevm_std/seed$SEED/best_model_$NEW1.pth ;;
    llmv3) TAG=$LLMV3_TAG; OCR=$LLMV3_OCR_RVL; BASE=$LLMV3_BASE_11
           STEP1=$CIL_ROOT/$LLMV3_TAG/noevm_std/seed$SEED/layoutlmv3_cil_incremental_${NEW1}_best.pt ;;
    *) echo "ERROR: backbone must be eaml or llmv3" >&2; exit 1 ;;
esac
case "$WHICH" in
    base)  CKPT=$BASE;  KNOWN=$CIL_BASE_CLASSES ;;
    step1) CKPT=$STEP1; KNOWN=$CIL_BASE_CLASSES,$NEW1 ;;
    *) echo "ERROR: which must be base or step1" >&2; exit 1 ;;
esac
require_file "$CKPT" "train the model / run CIL step 1 (No EVM, standard) first"

python tools/eval/extract_step_features.py --backbone "$BACKBONE" --checkpoint "$CKPT" \
  --known_classes "$KNOWN" --all_classes "$ALL_CLASSES" --data_dir "$RVL_DIR" --ocr "$OCR" \
  --out "$RUN_ROOT/features/$TAG/${WHICH}_cil.pt"
