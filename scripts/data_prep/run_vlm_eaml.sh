#!/bin/bash -l
###############################################################################
#  VLM Content Extraction (EAML) — FAU TinyGPU V100 partition
#  Same as run_vlm_content.sh but targeting V100 GPUs (32 GB each).
#
#  Submit:  sbatch.tinygpu run_vlm_eaml.sh
###############################################################################

#SBATCH --job-name=1_vlm_eaml_content
#SBATCH --output=logs/1_data_prep/%x_%j.out
#SBATCH --error=logs/1_data_prep/%x_%j.err
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:4
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --mail-type=END,FAIL

unset SLURM_EXPORT_ENV
 
# ---------- Configuration — EDIT THESE PATHS --------------------------------
DATA_DIR="/home/woody/iwi5/iwi5280h/dataset/small_dataset2"
OUTPUT_DIR="/home/woody/iwi5/iwi5280h/dataset/vlm_content/small2"
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MAX_LENGTH=128
MAX_NEW_TOKENS=512
MAX_SIZE=1024
NUM_WORKERS=4
SAVE_EVERY=200
BATCH_SIZE=0
USE_COMPILE=""
OFFSET=0
MAX_IMAGES=""
 
# ---------- Environment -----------------------------------------------------
module load python
conda activate ocr_env
 
#Compute nodes have NO internet access !!
MODEL_CACHE="/home/woody/iwi5/iwi5280h/model_cache"
export HF_HOME="${MODEL_CACHE}"
export TRANSFORMERS_CACHE="${MODEL_CACHE}/transformers"
export HF_HUB_CACHE="${MODEL_CACHE}/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
 
mkdir -p logs "${OUTPUT_DIR}"
export TMPDIR="${TMPDIR:-/tmp}"
echo "TMPDIR=${TMPDIR}  |  Node: $(hostname)  |  GPUs: ${CUDA_VISIBLE_DEVICES}"
 
NGPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
echo "Launching ${NGPUS} GPU workers (V100)"
 
# ---------- Launch ----------------------------------------------------------
CMD="tools/vlm/vlm_extractor_eaml.py \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --output_filename vlm_content_all.pt \
    --model_name ${MODEL_NAME} \
    --max_length ${MAX_LENGTH} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --max_size ${MAX_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --save_every ${SAVE_EVERY} \
    --batch_size ${BATCH_SIZE} \
    --offset ${OFFSET} \
    ${USE_COMPILE}"
 
[ -n "${MAX_IMAGES}" ] && CMD="${CMD} --max_images ${MAX_IMAGES}"
 
torchrun --standalone --nproc_per_node=${NGPUS} ${CMD}
exit $?


#DATA_DIR="/home/woody/iwi5/iwi5280h/dataset/small_dataset2"
#OUTPUT_DIR="/home/woody/iwi5/iwi5280h/dataset/vlm_content/small"