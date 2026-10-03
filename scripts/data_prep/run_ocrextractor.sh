#!/bin/bash -l

#SBATCH --job-name=1_ocr_text
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
export PYTHONPATH=/home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad-Extension:$PYTHONPATH

echo "Starting OCR Extraction"

DATA_DIR="/home/woody/iwi5/iwi5280h/dataset/all_prepdataset"
OUTPUT_TENSOR="/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr1.pt"
OUTPUT_JSON="/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_tesseract.json"
OCR_ENGINE="tesseract"
MAX_SIZE=1024

python tools/ocr/ocr_extraction.py \
  --data_dir $DATA_DIR \
  --output_json $OUTPUT_JSON\
  --ocr_engine $OCR_ENGINE\
  --offset 0\
  --max_images 100000 \

echo "OCR Extraction Completed. Output: $OUTPUT_TENSOR |&| $OUTPUT_JSON "

#sbatch run_ocrextractor.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/all_prepdataset \
#--data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \