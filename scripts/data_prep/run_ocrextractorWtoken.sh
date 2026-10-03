#!/bin/bash -l

#SBATCH --job-name=1_ocr_eaml_tokens
#SBATCH --output=logs/1_data_prep/%x_%A_%a.out
#SBATCH --error=logs/1_data_prep/%x_%A_%a.err
#SBATCH --partition=broadwell512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=0-31
#SBATCH --export=NONE

# Step 1 - OCR for the EAML backbone (Tesseract text tokens). CPU job on TinyFAT.
# Task k covers sorted images [k * EAML_OCR_PART_SIZE, (k + 1) * EAML_OCR_PART_SIZE) and writes
# $EAML_OCR_SHARDS/part_<offset>.pt; the file is only written when the task finishes.
# Submit from the repo root (paths: scripts/config.sh):
#   sbatch.tinyfat scripts/data_prep/run_ocrextractorWtoken.sh                    # RVL-CDIP: tasks 0-31
#   sbatch.tinyfat --array=16-31 scripts/data_prep/run_ocrextractorWtoken.sh      # only some tasks (e.g. reruns)
#   sbatch.tinyfat --array=0 scripts/data_prep/run_ocrextractorWtoken.sh tobacco  # Tobacco-3482 -> $EAML_OCR_TOB
# Afterwards merge the RVL-CDIP parts: sbatch.tinyfat scripts/data_prep/run_combinetensors.sh

unset SLURM_EXPORT_ENV
module load python/3.12-conda
conda activate ocr_env
source /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad-Extension/scripts/config.sh

TASK=${SLURM_ARRAY_TASK_ID:-0}
if [ "${1:-rvl}" = "tobacco" ]; then
    DATA_DIR=$DATA_ROOT/$TOB_DOMAIN
    OUTPUT_PT=$EAML_OCR_TOB
    OFFSET=0; MAX_IMAGES=""
else
    DATA_DIR=$RVL_DIR
    OFFSET=$((TASK * EAML_OCR_PART_SIZE)); MAX_IMAGES=$EAML_OCR_PART_SIZE
    if (( OFFSET >= RVL_NUM_IMAGES )); then echo "Task $TASK starts after the last image; nothing to do."; exit 0; fi
    OUTPUT_PT=$EAML_OCR_SHARDS/part_$(printf '%06d' "$OFFSET").pt
fi
mkdir -p "$(dirname "$OUTPUT_PT")"

echo "EAML OCR: $DATA_DIR (offset $OFFSET${MAX_IMAGES:+, $MAX_IMAGES images}) -> $OUTPUT_PT"
python tools/ocr/ocr_extraction_token.py \
  --data_dir "$DATA_DIR" \
  --output_pt "$OUTPUT_PT" \
  --ocr_engine tesseract \
  --max_size 1024 \
  --offset "$OFFSET" \
  ${MAX_IMAGES:+--max_images "$MAX_IMAGES"}

echo "Finished. Output: $OUTPUT_PT"
