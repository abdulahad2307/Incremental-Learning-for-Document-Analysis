#!/bin/bash -l

#SBATCH --job-name=1_merge_json
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

echo "Combining Jsons"

python tools/data/combine_jsons.py \
  --input_files /home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_pero1.json /home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_pero2.json /home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_pero3.json /home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_pero4.json \
  --output_file /home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_trocr.json

echo "JSON combination Completed. Output: $OUTPUT_JSON"

#sbatch run_combinejson.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/prepdata \
#--data_dir /home/woody/iwi5/iwi5280h/dataset/small_dataset \