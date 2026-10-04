#!/bin/bash -l

#SBATCH --job-name=1_ocr_llmv3_bert_bbox
#SBATCH --output=logs/1_data_prep/%x_%A_%a.out
#SBATCH --error=logs/1_data_prep/%x_%A_%a.err
#SBATCH --partition=broadwell512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=23:00:00
#SBATCH --array=0-7
#SBATCH --export=NONE

# Step 1 - OCR for the LayoutLMv3 backbone: Tesseract words + boxes, bert-base-uncased tokens (the model's text
# encoder); each word-piece gets its word's box. Writes one <image stem>.pt per image. CPU job on TinyFAT.
# Submit from the repo root (paths: scripts/config.sh):
#   sbatch.tinyfat scripts/data_prep/run_ocrextractor_bert_bbox.sh                    # RVL-CDIP: 8 shards -> $LLMV3_OCR_RVL
#   sbatch.tinyfat --array=0 scripts/data_prep/run_ocrextractor_bert_bbox.sh tobacco  # Tobacco-3482 -> $LLMV3_OCR_TOB

unset SLURM_EXPORT_ENV
module load python/3.12-conda
conda activate ocr_env
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

SHARD=${SLURM_ARRAY_TASK_ID:-0}
if [ "${1:-rvl}" = "tobacco" ]; then
    DATA_DIR=$DATA_ROOT/$TOB_DOMAIN
    OUTPUT_DIR=$LLMV3_OCR_TOB
else
    DATA_DIR=$RVL_DIR
    OUTPUT_DIR=$LLMV3_OCR_RVL
fi
OFFSET=$((SHARD * OCR_SHARD_SIZE))
mkdir -p "$OUTPUT_DIR"

echo "LayoutLMv3 OCR: $DATA_DIR (offset $OFFSET, $OCR_SHARD_SIZE images) -> $OUTPUT_DIR"
python tools/ocr/ocr_extraction_bbox_bert.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --ocr_engine tesseract \
  --max_size 1024 \
  --offset "$OFFSET" \
  --max_images "$OCR_SHARD_SIZE" \
  --skip_existing

echo "Finished. Output: $OUTPUT_DIR"
