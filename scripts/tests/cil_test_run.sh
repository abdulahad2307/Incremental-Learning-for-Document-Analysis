#!/bin/bash -l

#SBATCH --job-name=9_test_cil
#SBATCH --output=logs/tests/%x_%j.out
#SBATCH --error=logs/tests/%x_%j.err
#SBATCH --partition=v100                 # GPU partition name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=4                # CPU cores
#SBATCH --gres=gpu:v100:1                # Number of GPUs
#SBATCH --time=23:00:00                  # Shorter time for test run
#SBATCH --export=NONE                    # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

# Load required modules
module load cuda/12.6
module load python/3.12-conda
conda activate mtil

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# Move to the repository folder
export PYTHONPATH=${SLURM_SUBMIT_DIR:-$PWD}:$PYTHONPATH

echo "Starting CIL Test Run (Temporary Test Script Mode)..."

# Create test_cil.py if it doesn't exist
TEST_SCRIPT="src/tests/test_cil.py"
if [ ! -f "$TEST_SCRIPT" ]; then
    cat << 'EOF' > $TEST_SCRIPT
from src.class_incremental.eaml.class_incremental import run_incremental_learning

CLASS_ORDER = ["news article","specification","memo"]

run_incremental_learning(
    data_root="/home/woody/iwi5/iwi5280h/dataset/small_dataset",
    class_order=CLASS_ORDER,
    base_model_path="/home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad/outputs/funsd/docformer_best_model.pt",
    model_name="docformer",
    checkpoint_dir="checkpoints/cil_test",
    start_step=0,
    batch_size=8,
    lr=2e-5,
    num_epochs=5
)
EOF
    echo "Created temporary test script at $TEST_SCRIPT"
fi

# Run the test script
export CUDA_LAUNCH_BLOCKING=1
python $TEST_SCRIPT

echo "CIL Test Run Completed (Temporary Script Mode)."

# To run:
# sbatch cil_test_run.sh