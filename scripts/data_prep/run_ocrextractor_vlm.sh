#!/bin/bash -l

#SBATCH --job-name=1_vlm_ocr
#SBATCH --output=logs/1_data_prep/%x_%j.out
#SBATCH --error=logs/1_data_prep/%x_%j.err
#SBATCH --partition=v100                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=1                 # Number of CPU cores per task
#SBATCH --gres=gpu:v100:1                 # Number of GPUs
#SBATCH --time=23:55:00                   # Time limit hrs:min:sec
#SBATCH --export=NONE                     # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

module load cuda/12.6
module load python/3.12-conda
conda activate ocr_env

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HUB_DISABLE_XET=1
export HF_HOME=$WORK/.cache/huggingface

export PYTHONPATH=/home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad-Extension:$PYTHONPATH

# Default values
PREPROCESSED_DIR="/home/woody/iwi5/iwi5280h/dataset/preprocessed/small_dataset2_qwen2-vl"
OUTPUT_DIR="/home/woody/iwi5/iwi5280h/dataset/vlm_outputs"
IMAGES_PER_JOB=1
MICROBATCH_SIZE=25
VLM_MODEL="qwen2-vl"
MAX_LENGTH=200
EXTRACT_MODE="description"

# Calculate offset for this job
OFFSET=$((SLURM_ARRAY_TASK_ID * IMAGES_PER_JOB))

# Output file for this job
OUTPUT_PT="${OUTPUT_DIR}/vlm_extraction_${SLURM_ARRAY_TASK_ID}.pt"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Print job information
echo "=== Job Information ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $HOSTNAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "Offset: $OFFSET"
echo "Max Images: $IMAGES_PER_JOB"
echo "Microbatch Size: $MICROBATCH_SIZE"
echo "Output: $OUTPUT_PT"
echo "======================="

# Run extraction with memory monitoring
python tools/vlm/extraction_vlm_ocr.py \
    --preprocessed_dir $PREPROCESSED_DIR \
    --output_pt $OUTPUT_PT \
    --vlm_model $VLM_MODEL \
    --offset $OFFSET \
    --max_images $IMAGES_PER_JOB \
    --max_length $MAX_LENGTH \
    --extract_mode $EXTRACT_MODE \
    --microbatch_size $MICROBATCH_SIZE \
    --use_filename_key \
    --save_intermediate \
    --intermediate_save_freq 15

# Check if job completed successfully
if [[ $? -eq 0 ]]; then
    echo "Job completed successfully!"
    echo "Output saved to: $OUTPUT_PT"
else
    echo "Job failed!"
    exit 1
fi

#sbatch run_ocrextractor_vlm.sh

#python tools/ocr/pre_processing.py \
#    --data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset2 \
#    --output_dir /home/woody/iwi5/iwi5280h/dataset/preprocessed/small_dataset2_qwen2-vl \
#    --vlm_model qwen2-vl


module load cuda/12.6
module load python/3.12-conda
conda activate ocr_env

# === PROXY & HF FIX ===
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HUB_DISABLE_XET=1
export HF_HOME=$WORK/.cache/huggingface
export PYTHONPATH=/home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad-Extension:$PYTHONPATH

# === CONFIGURE HERE ===
IMAGES_DIR="/home/woody/iwi5/iwi5280h/dataset/all_prepdataset/train/memo"    # Raw images folder
OUTPUT_DIR="/home/woody/iwi5/iwi5280h/dataset/vlm_outputs/raw_offset"
MODEL_PATH=""  # ""=online download, or "/work/models/Qwen2-VL-2B-Instruct"

# === OFFSET CONTROL ===
GLOBAL_OFFSET=0          # Skip first N images (0=first image)
IMAGES_PER_JOB=2        # Images per job
TOTAL_JOBS=2           # --array=0-$(($TOTAL_JOBS-1))

mkdir -p "$OUTPUT_DIR"

# Job info & calculations
JOB_OFFSET=$((SLURM_ARRAY_TASK_ID * IMAGES_PER_JOB))
THIS_OFFSET=$((GLOBAL_OFFSET + JOB_OFFSET))
THIS_MAX_IMAGES=$IMAGES_PER_JOB

echo "=== Raw VLM Offset Job ==="
echo "Job ID: $SLURM_JOB_ID | Array ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $HOSTNAME | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "Global Offset: $GLOBAL_OFFSET"
echo "This Offset: $THIS_OFFSET | Max Images: $THIS_MAX_IMAGES"
echo "Model: $MODEL_PATH"
echo "============================="

# Output filename
OUTPUT_PT="${OUTPUT_DIR}/vlm_raw_${GLOBAL_OFFSET}_${THIS_OFFSET}_${THIS_MAX_IMAGES}.pt"

# RUN EXTRACTION (Option 2: Raw images, no preprocessing)
python tools/vlm/extraction_vlm_q_single.py \
    --images_dir "$IMAGES_DIR" \
    --output_pt "$OUTPUT_PT" \
    --offset "$THIS_OFFSET" \
    --max_images "$THIS_MAX_IMAGES" \
    --max_tokens 200

echo "✓ Completed: $THIS_MAX_IMAGES images -> $OUTPUT_PT"

===========================================

module load cuda/12.6
module load python/3.12-conda
conda activate ocr_env

# === PROXY & HF FIX ===
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HUB_DISABLE_XET=1
export HF_HOME=$WORK/.cache/huggingface
export PYTHONPATH=/home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad-Extension:$PYTHONPATH

# === CONFIGURE HERE ===
DATA_DIR="/home/woody/iwi5/iwi5280h/dataset/all_prepdataset/train/memo"  # Dataset root OR specific dir
OUTPUT_DIR="/home/woody/iwi5/iwi5280h/dataset/vlm_outputs/bbox_offset"  # Changed for bbox version
MODEL_PATH=""  # ""=online download, or "/work/models/Qwen2-VL-2B-Instruct"

# === OFFSET CONTROL ===
GLOBAL_OFFSET=0          # Skip first N images (0=first image)
IMAGES_PER_JOB=2        # Images per job
TOTAL_JOBS=2           # --array=0-$(($TOTAL_JOBS-1))

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Job info & calculations
JOB_OFFSET=$((SLURM_ARRAY_TASK_ID * IMAGES_PER_JOB))
THIS_OFFSET=$((GLOBAL_OFFSET + JOB_OFFSET))
THIS_MAX_IMAGES=$IMAGES_PER_JOB

echo "=== VLM+BBox Offset Job ==="
echo "Job ID: $SLURM_JOB_ID | Array ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $HOSTNAME | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "Global Offset: $GLOBAL_OFFSET"
echo "This Offset: $THIS_OFFSET | Max Images: $THIS_MAX_IMAGES"
echo "Data Dir: $DATA_DIR"
echo "Model: $MODEL_PATH"
echo "============================="

# Output filename
OUTPUT_PT="${OUTPUT_DIR}/vlm_bbox_${GLOBAL_OFFSET}_${THIS_OFFSET}_${THIS_MAX_IMAGES}.pt"

# NEW: Use --data_dir for dataset root (walks subdirs) OR --images_dir for specific
# Detect if dataset root or single dir (flexible)
if [ -d "$DATA_DIR" ] && ls "$DATA_DIR"/*/* >/dev/null 2>&1; then
    # Has subdirs -> use --data_dir
    DATA_ARG="--data_dir $DATA_DIR"
else
    # Single dir -> use --images_dir
    DATA_ARG="--images_dir $DATA_DIR"
fi

echo "Using: $DATA_ARG"

# RUN EXTRACTION (Updated script with bbox support)
python tools/vlm/extraction_vlm_bbox.py \
    $DATA_ARG \
    --output_pt "$OUTPUT_PT" \
    --offset "$THIS_OFFSET" \
    --max_images "$THIS_MAX_IMAGES" \
    --max_tokens 400 \
    ${MODEL_PATH:+"--model $MODEL_PATH"} \
    --device cuda

echo "✓ Completed: $THIS_MAX_IMAGES images -> $OUTPUT_PT (with texts/bboxes_rect)"