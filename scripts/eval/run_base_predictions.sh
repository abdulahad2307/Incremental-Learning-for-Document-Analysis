#!/bin/bash -l

#SBATCH --job-name=2_base_predictions
#SBATCH --output=logs/2_base/%x_%j.out
#SBATCH --error=logs/2_base/%x_%j.err
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100:1
#SBATCH --time=04:00:00
#SBATCH --export=NONE

# Step 2b - step-0 predictions of the base models (tools/eval/base_predictions.py): the CIL base model (11 classes)
# on the test documents of the base and the later classes, the DIL base model (16 classes) on the RVL-CDIP and
# Tobacco-3482 test sets. Run once per backbone after its base models are trained; the incremental scripts save the
# predictions of every later step themselves.
# Output: <base model dir>/predictions/base_{cil,dil}_seed<SEED>.pt
# The test splits are fixed (Tobacco-3482: utils/data_subset.split_documents), so one run serves all seeds.
# Submit from the repo root:
#   sbatch scripts/eval/run_base_predictions.sh eaml
#   sbatch scripts/eval/run_base_predictions.sh llmv3                            # Custom LayoutLMv3
#   sbatch --export=LLMV3_MODEL=hf scripts/eval/run_base_predictions.sh llmv3    # pre-trained LayoutLMv3

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

BACKBONE=${1:?usage: sbatch scripts/eval/run_base_predictions.sh eaml|llmv3}
case "$BACKBONE" in
    eaml)  CKPT_11=$EAML_BASE_11;  CKPT_16=$EAML_BASE_16;  OCR_RVL=$EAML_OCR_RVL;  OCR_TOB=$EAML_OCR_TOB ;;
    llmv3) CKPT_11=$LLMV3_BASE_11; CKPT_16=$LLMV3_BASE_16; OCR_RVL=$LLMV3_OCR_RVL; OCR_TOB=$LLMV3_OCR_TOB ;;
    *) echo "ERROR: backbone must be eaml or llmv3" >&2; exit 1 ;;
esac
require_file "$CKPT_11" "train the 11-class base model first"
require_file "$CKPT_16" "train the 16-class base model first"

python tools/eval/base_predictions.py --backbone "$BACKBONE" --setting CIL --checkpoint "$CKPT_11" \
  --base_classes "$CIL_BASE_CLASSES" --all_classes "$ALL_CLASSES" \
  --data_root "$DATA_ROOT" --ocr "$OCR_RVL" --seed "$SEED" || exit 1

python tools/eval/base_predictions.py --backbone "$BACKBONE" --setting DIL --checkpoint "$CKPT_16" \
  --base_classes "$ALL_CLASSES" --all_classes "$DIL_CLASSES" \
  --data_root "$DATA_ROOT" --ocr "$OCR_RVL" "$OCR_TOB" --seed "$SEED" || exit 1

echo "Finished: $(dirname "$CKPT_11")/predictions/ and $(dirname "$CKPT_16")/predictions/"
