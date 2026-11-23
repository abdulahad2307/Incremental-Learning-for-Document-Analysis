#!/bin/bash -l

#SBATCH --job-name=5_closed_form
#SBATCH --output=logs/5_eval/%x_%j.out
#SBATCH --error=logs/5_eval/%x_%j.err
#SBATCH --partition=rtx3080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=06:00:00
#SBATCH --export=NONE

# Closed-form incremental classifier on frozen base-model features (tools/eval/closed_form_classifier.py; CPU only,
# the GPU is the cluster's requirement for a job). <which> = the base feature file of scripts/eval/run_extract_features.sh
# (base_full: all training documents per class, base_all: 1000 per class), compared with the learners' results table.
# Output: $RUN_ROOT/analysis/closed_form_<which>/
# Submit from the repo root, after run_extract_features.sh <backbone> <which> for both backbones:
#   sbatch scripts/eval/run_closed_form.sh base_full

unset SLURM_EXPORT_ENV
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

WHICH=${1:-base_full}
python tools/eval/closed_form_classifier.py --root "$RUN_ROOT/features/drift" --runs \
  --base "$RUN_ROOT/features/{backbone}/${WHICH}_cil.pt" --table "$RUN_ROOT/il_runs.csv" \
  --train_per_class 1000 2500 5000 12500 --out "$RUN_ROOT/analysis/closed_form_$WHICH"
