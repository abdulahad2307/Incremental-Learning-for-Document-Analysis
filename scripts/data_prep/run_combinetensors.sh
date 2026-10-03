#!/bin/bash -l

#SBATCH --job-name=1_merge_tensors
#SBATCH --output=logs/1_data_prep/%x_%j.out
#SBATCH --error=logs/1_data_prep/%x_%j.err
#SBATCH --partition=broadwell512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --export=NONE

# Step 1 - merge the EAML OCR parts of RVL-CDIP ($EAML_OCR_SHARDS/part_*.pt) into one file ($EAML_OCR_RVL).
# Run after all OCR tasks finished; fails if the merged file does not cover all $RVL_NUM_IMAGES images.
# Submit from the repo root (paths: scripts/config.sh):
#   sbatch.tinyfat scripts/data_prep/run_combinetensors.sh

unset SLURM_EXPORT_ENV
module load python/3.12-conda
conda activate ocr_env
source /home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad-Extension/scripts/config.sh

shopt -s nullglob
PARTS=("$EAML_OCR_SHARDS"/part_*.pt)
if (( ${#PARTS[@]} == 0 )); then echo "ERROR: no parts in $EAML_OCR_SHARDS" >&2; exit 1; fi
echo "Merging ${#PARTS[@]} parts"

python tools/data/combine_tensorTokens.py \
  --inputs "${PARTS[@]}" \
  --output "$EAML_OCR_RVL" | tee /dev/stderr | grep -q "merged dict with $RVL_NUM_IMAGES entries" || {
    echo "ERROR: merged file does not have $RVL_NUM_IMAGES entries (missing or duplicate parts?)" >&2
    exit 1
}

echo "Finished. Output: $EAML_OCR_RVL"
