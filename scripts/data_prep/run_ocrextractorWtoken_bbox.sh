#!/bin/bash -l

#SBATCH --job-name=1_ocr_llmv3_tokenizer_bbox
#SBATCH --output=logs/1_data_prep/%x_%j.out
#SBATCH --error=logs/1_data_prep/%x_%j.err
#SBATCH --partition=broadwell512                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=16                 # Number of CPU cores per task
#SBATCH --time=23:55:00                   # Time limit hrs:min:sec
#SBATCH --export=NONE                    # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

module load cuda/12.6
module load python/3.12-conda
conda activate ocr_env

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export PYTHONPATH=/home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad-Extension:$PYTHONPATH

echo "Starting OCR Extraction"

DATA_DIR="/home/woody/iwi5/iwi5280h/dataset/all_prepdataset"
OUTPUT_TENSOR="/home/woody/iwi5/iwi5280h/dataset/all_prepdata_combined_ocr_texts_rectabbox_tesseract_single_image"
#OUTPUT_JSON="/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr1.json"
#OCR_ENGINE="trocr"
OCR_ENGINE="tesseract"
MAX_SIZE=1024
BBOX_STYLE="rect"

#tools/ocr/ocr_extraction_token.py \

python tools/ocr/ocr_extraction_bbox_single.py \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUTPUT_TENSOR" \
  --ocr_engine "$OCR_ENGINE" \
  --max_size "$MAX_SIZE" \
  --offset  360000\
  --max_images 40000 \
  #--bbox_style "$BBOX_STYLE"


echo "OCR Extraction Completed. Output: $OUTPUT_TENSOR "

#python tools/ocr/ocr_extraction_token.py \
#  --data_dir $DATA_DIR \
#  --output_pt $OUTPUT_TENSOR \
#  --ocr_engine $OCR_ENGINE \
#  --max_size 1024\
#  --offset 350000 \
#  --max_images 49999 \

#echo "OCR Extraction Completed. Output: $OUTPUT_TENSOR "

#sbatch.tinyfat run_ocrextractorWtoken_bbox.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/all_prepdataset \
#--data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \