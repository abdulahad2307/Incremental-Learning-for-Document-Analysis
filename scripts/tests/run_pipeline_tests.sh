#!/bin/bash -l

#SBATCH --job-name=il_pipeline_tests        # Job name
#SBATCH --output=logs/%x_%j.out            # Standard output log
#SBATCH --error=logs/%x_%j.err             # Error log
#SBATCH --partition=v100                   # GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:00:00
#SBATCH --export=NONE

# End-to-end tests of every pipeline (baseline, EAML-CIL, EAML-DIL, LayoutLMv3-CIL, LayoutLMv3-DIL) on a tiny
# synthetic dataset: all methods x both strategies (standard, distillation). Submit from the repository root:
#   sbatch scripts/tests/run_pipeline_tests.sh                        # everything (55 tests)
#   sbatch scripts/tests/run_pipeline_tests.sh tests.test_layout_dil  # one pipeline
# Narrow down with PIPELINE_TEST_METHODS=evm,evm_ood and/or PIPELINE_TEST_STRATEGIES=standard.
# Each run writes its command and full output to $PIPELINE_TEST_DIR/il_pipeline_tests/runs/<pipeline>/<test>/run.log.

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

REPO=/home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad-Extension
export PYTHONPATH=$REPO:$PYTHONPATH
export PIPELINE_TEST_DIR=${PIPELINE_TEST_DIR:-$TMPDIR}   # node-local disk: synthetic data + ~3 GB of test checkpoints
cd $REPO

if [ $# -gt 0 ]; then
    python -m unittest -v "$@"
else
    python -m unittest discover -v -s tests -t .
fi
