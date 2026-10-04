#!/bin/bash -l

#SBATCH --job-name=2_base_cnn
#SBATCH --output=logs/2_base/other/%x_%j.out
#SBATCH --error=logs/2_base/other/%x_%j.err
#SBATCH --partition=v100                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --gres=gpu:v100:1                 # Number of GPUs
#SBATCH --time=23:30:00                    # Time limit hrs:min:sec
#SBATCH --export=NONE                     # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

# Load required modules
module load cuda/12.6
module load python/3.12-conda
conda activate mtil

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# Move to the repository folder

export PYTHONPATH=${SLURM_SUBMIT_DIR:-$PWD}:$PYTHONPATH

CUSTOM_CLASSES=("letter" "form" "email" "handwritten" "advertisement" "scientific report" "invoice" "presentation" "questionnaire" "resume" "memo" )


echo "Starting Baseline Model Training..."

# Run Model Training
echo "Training ResNet50..."
export CUDA_LAUNCH_BLOCKING=1
python3 src/base_models/baseline_model.py \
    --data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
    --model_name resnet50 \
    --classes "${CUSTOM_CLASSES[@]}" \
    --batch_size 64 \
    --epochs 10 \
    --learning_rate 0.001 \
    --optimizer adamw \
    --device cuda

echo "Training DenseNet121..."
python3 src/base_models/baseline_model.py \
    --data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
    --model_name densenet121 \
    --classes "${CUSTOM_CLASSES[@]}" \
    --batch_size 64 \
    --epochs 10 \
    --learning_rate 0.001 \
    --optimizer adamw \
    --device cuda

echo "Training Completed. Check logs/training_log.csv for results."

#sbatch.tinygpu run_basemodel.sh