#!/bin/bash -l

#SBATCH --job-name=2_base_eaml_multigpu
#SBATCH --output=logs/2_base/eaml/%x_%j.out
#SBATCH --error=logs/2_base/eaml/%x_%j.err
#SBATCH --partition=v100                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # Number of CPU cores per task
#SBATCH --gres=gpu:v100:4                 # Number of GPUs
#SBATCH --time=23:55:00                   # Time limit hrs:min:sec
#SBATCH --export=NONE                     # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

# Load required modules
module load cuda/12.6
module load python/3.12-conda
conda activate mtil

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

export PYTHONPATH=${SLURM_SUBMIT_DIR:-$PWD}:$PYTHONPATH

echo "Starting EAML Training..."

# Define classes as space-separated list like DocFormer
CLASSES="letter form email handwritten advertisement scientific_report invoice presentation questionnaire resume memo"

# Create output directory
OUTPUT_DIR="outputs/small_eaml_trocr_$(date +%Y%m%d_%H%M%S)_MultiPros"
OCR_JSON_PATH="/home/woody/iwi5/iwi5280h/dataset/small_dataset_ocr_texts_trocr.json"
OCR_DATA_PATH="/home/woody/iwi5/iwi5280h/dataset/small_dataset_ocr_texts_trocr.pt"
mkdir -p $OUTPUT_DIR
mkdir -p logs

$DATA_DIR="/home/woody/iwi5/iwi5280h/dataset/small_dataset"

# Multi-GPU configuration
WORLD_SIZE=4  # Number of GPUs
MASTER_PORT="12355"

# Training parameters
BATCH_SIZE=16  # Per GPU batch size
NUM_EPOCHS=3
LEARNING_RATE=5e-5
WEIGHT_DECAY=0.05

# OCR handling
HANDLE_EMPTY_TEXT="fallback"  # Options: exclude, fallback, include
FALLBACK_TEXT="Empty Text"

ulimit -n 4096

# Resume from checkpoint
CHECKPOINT_PATH=""  # Set path if resuming
RESUME_ARG=""
if [ -n "$CHECKPOINT_PATH" ] && [ -f "$CHECKPOINT_PATH" ]; then
    RESUME_ARG="--resume $CHECKPOINT_PATH"
    echo "Resuming training from checkpoint: $CHECKPOINT_PATH"
fi

echo "======== Configuration ======= "
echo "  Data directory: $DATA_DIR"
echo "  OCR data path: $OCR_DATA_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  World size (GPUs): $WORLD_SIZE"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Empty text handling: $HANDLE_EMPTY_TEXT"
echo "  Classes: $CLASSES"
echo " ============================== "

python src/base_models/sota_eaml_model_multipros.py \
    --data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset\
    --ocr_data_path $OCR_DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --classes $CLASSES \
    --world_size $WORLD_SIZE \
    --master_port $MASTER_PORT \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --handle_empty_text $HANDLE_EMPTY_TEXT \
    --fallback_text "$FALLBACK_TEXT" \
    --num_workers 2 \
    --patience 15 \
    --keep_checkpoints 3 \
    --cls_weight 1.0 \
    --kld_weight 0.5 \
    --kld_threshold 0.1 \
    --embed_dim 512 \
    --dropout_rate 0.3 \
    $RESUME_ARG

echo "Training Completed. Output: $OUTPUT_DIR"


#sbatch run_eamlmodel_multipros.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
#--data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \

# json_small_trocr -- "/home/woody/iwi5/iwi5280h/dataset/small_dataset_ocr_texts_trocr.json"
# json_all_trocr -- "/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr.json"