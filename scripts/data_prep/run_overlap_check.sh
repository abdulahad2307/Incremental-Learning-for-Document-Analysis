#!/bin/bash -l

#SBATCH --job-name=1_overlap_check
#SBATCH --output=logs/1_data_prep/%x_%j.out
#SBATCH --error=logs/1_data_prep/%x_%j.err
#SBATCH --partition=broadwell512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=08:00:00
#SBATCH --export=NONE

# Near-duplicate check: Tobacco-3482 and RVL-CDIP-N against RVL-CDIP (tools/data/overlap_check.py). CPU job on
# TinyFAT; hashes all ~400,000 RVL-CDIP images once (cached in $RUN_ROOT/overlap/hashes_rvl.npz).
# Submit from the repo root:
#   sbatch.tinyfat scripts/data_prep/run_overlap_check.sh

unset SLURM_EXPORT_ENV
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

python tools/data/overlap_check.py --rvl "$RVL_DIR" --tobacco "$DATA_ROOT/$TOB_DOMAIN" \
  --rvl_n "$DATA_ROOT/RVL-CDIP-N" --out "$RUN_ROOT/overlap" --workers "${SLURM_CPUS_PER_TASK:-8}"
