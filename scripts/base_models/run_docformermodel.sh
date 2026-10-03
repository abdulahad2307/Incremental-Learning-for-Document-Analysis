#!/bin/bash -l

#SBATCH --job-name=2_base_docformer
#SBATCH --output=logs/2_base/other/%x_%j.out
#SBATCH --error=logs/2_base/other/%x_%j.err
#SBATCH --partition=v100                  # GPU partition name
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=4                 # Increased CPU cores for OCR processing
#SBATCH --gres=gpu:v100:1                 # Number of GPUs
#SBATCH --time=23:30:00                   # Time limit hrs:min:sec
#SBATCH --export=NONE                     # Avoid inheriting unwanted environment variables

unset SLURM_EXPORT_ENV

# Load required modules
module load cuda/12.6
module load python/3.12-conda
conda activate mtil

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# Move to the repository folder
export PYTHONPATH=/home/hpc/iwi5/iwi5280h/projects/FAU-Masters_Thesis-Ahad-Extension:$PYTHONPATH

echo "Starting Progressive DocFormer Training..."

# Configuration with defaults, override via script args if needed
DATA_DIR=${1:-"/home/woody/iwi5/iwi5280h/dataset/all_prepdataset"}
OCR_TOKEN_DIR=${2:-"/home/woody/iwi5/iwi5280h/dataset/all_dataset_ocr_texts_bbox_tesseract.pt"}
OUTPUT_DIR="outputs/all_class_docformer_outputs_$(date +%Y%m%d_%H%M%S)"
# Full set of classes available
ALL_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,scientific_publication,specification,file_folder,news_article,budget,invoice,presentation,questionnaire,resume,memo"
# Common subset for training/evaluation
CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo"
EVAL_ONLY="false"
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Check if EVAL_ONLY environment variable is set for evaluation
if [ "$EVAL_ONLY" = "true" ]; then
    echo "=== Running Evaluation Only ==="
    if [ ! -f "$OUTPUT_DIR/best_model.pt" ]; then
        echo "Error: No trained model found at $OUTPUT_DIR/best_model.pt"
        echo "Please train a model first or provide the correct output directory"
        exit 1
    fi

    python src/base_models/sota_docformer_model.py \
        --data_dir "$DATA_DIR" \
        --ocr_token_dir "$OCR_TOKEN_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --classes "$ALL_CLASSES" \
        --batch_size 8 \
        --eval_batch_size 16 \
        --num_epochs 1 \
        --classes "$CLASSES" \
        --evaluate_only

else
    echo "=== Starting Training ==="
    python src/base_models/sota_docformer_model.py \
        --data_dir "$DATA_DIR" \
        --ocr_token_dir "$OCR_TOKEN_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --classes "$CLASSES" \
        --batch_size 4 \
        --eval_batch_size 8 \
        --num_epochs 100 \
        --learning_rate 5e-5 \
        --finetune_lr 2.5e-5 \
        --num_workers 4 \
        --max_seq_len 512 \
        --weight_decay 0.01 \
        --early_stop_patience 15 \
        --training_stage finetune \
        --use_amp

    echo "Training completed!"

    echo "=== Starting Final Evaluation ==="
    python src/base_models/sota_docformer_model.py \
        --data_dir "$DATA_DIR" \
        --ocr_token_dir "$OCR_TOKEN_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --batch_size 8 \
        --eval_batch_size 16 \
        --num_epochs 1 \
        --classes "$CLASSES" \
        --evaluate_only
fi

echo "=== Process Completed ==="
echo "Results saved to: $OUTPUT_DIR"
echo "Training and Evaluation Completed."


# sbatch run_docformermodel.sh
#--data_dir /home/woody/iwi5/iwi5280h/dataset/all_prepdataset \
#"/home/woody/iwi5/iwi5280h/dataset/small_dataset"