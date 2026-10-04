#!/bin/bash -l

#SBATCH --job-name=1_vlm_ocr_bbox
#SBATCH --output=logs/1_data_prep/%x_%j.out
#SBATCH --error=logs/1_data_prep/%x_%j.err
#SBATCH --partition=v100                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # DataLoader workers + async post-processing threads + main
#SBATCH --gres=gpu:v100:1                 # Number of GPUs
#SBATCH --time=23:55:00                   # Time limit hrs:min:sec
#SBATCH --export=NONE                     # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

module load cuda/12.6
module load python/3.12-conda
conda activate ocr_env

# === PROXY & HF FIX ===
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HF_HUB_DISABLE_XET=1
export HF_HOME=$WORK/.cache/huggingface
export PYTHONPATH=${SLURM_SUBMIT_DIR:-$PWD}:$PYTHONPATH

# === CONFIGURE HERE ===
DATA_DIR="/home/woody/iwi5/iwi5280h/dataset/small_dataset2"   # dataset root
OUTPUT_ROOT="/home/woody/iwi5/iwi5280h/dataset/vlm_outputs"   # will contain /content and /layout
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"                        # or local path

# === HOW MANY IMAGES TO EXTRACT ===
IMAGES_PER_CLASS="5"     # empty = full dataset; set a number (e.g. 200) to cap images PER CLASS
BATCH_SIZE=0             # VLM batch size; 0 = auto-detect from free GPU memory
NUM_WORKERS=4            # DataLoader workers for parallel image decode/resize (must fit --cpus-per-task)
VIZ_PER_CLASS=3          # number of annotated QA samples to save per class

# === OFFSET CONTROL (array-job chunking of the — possibly per-class-capped — image list) ===
GLOBAL_OFFSET=0
IMAGES_PER_JOB=5000
TOTAL_JOBS=1   # → use --array=0-$((TOTAL_JOBS-1)) if needed.
               # NOTE: when running as a real array, submit one extra job afterwards with
               #   python tools/vlm/extraction_vlm_bbox.py --merge_only --output_root "$OUTPUT_ROOT"
               # once ALL array tasks have finished, to guarantee a consistent final merge
               # (auto-merge-per-job is safe for sequential runs but not for tasks racing in parallel).

mkdir -p "$OUTPUT_ROOT"
mkdir -p logs

JOB_OFFSET=$((SLURM_ARRAY_TASK_ID * IMAGES_PER_JOB))
THIS_OFFSET=$((GLOBAL_OFFSET + JOB_OFFSET))
THIS_MAX_IMAGES=$IMAGES_PER_JOB

echo "=== VLM Extraction Job ==="
echo "Job ID: $SLURM_JOB_ID | Array ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $HOSTNAME | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "Global Offset: $GLOBAL_OFFSET"
echo "This Offset: $THIS_OFFSET | Max Images: $THIS_MAX_IMAGES"
echo "Data Dir: $DATA_DIR"
echo "Model: $MODEL_NAME"
echo "Output Root: $OUTPUT_ROOT"
echo "Images per class: ${IMAGES_PER_CLASS:-full dataset}"
echo "============================="

CMD="tools/vlm/extraction_vlm_bbox.py \
    --data_dir $DATA_DIR \
    --output_root $OUTPUT_ROOT \
    --max_size 1024 \
    --max_len_layout 512 \
    --max_length_content 128 \
    --offset $THIS_OFFSET \
    --max_images $THIS_MAX_IMAGES \
    --model_name $MODEL_NAME \
    --max_new_tokens 400 \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --viz_per_class $VIZ_PER_CLASS \
    --device cuda"

[ -n "${IMAGES_PER_CLASS}" ] && CMD="${CMD} --images_per_class ${IMAGES_PER_CLASS}"

python ${CMD}

echo "Completed: $THIS_MAX_IMAGES images -> $OUTPUT_ROOT/content & $OUTPUT_ROOT/layout"
echo "QA visualizations (bbox overlays, ${VIZ_PER_CLASS}/class) -> $OUTPUT_ROOT/viz"

#sbatch run_ocr_extractor_vlm_bbox.sh