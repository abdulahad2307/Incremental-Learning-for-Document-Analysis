# Shared configuration for the SLURM job scripts (sourced by scripts/**/*.sh).
# Paths, class lists and base-model accuracies live here once; the job scripts only hold hyperparameters.

# Repo root = the directory the job was submitted from, so every checkout (e.g. a git worktree) runs its own code.
# Submit from the repo root: cd <repo> && sbatch scripts/...   (outside SLURM: the current directory)
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "$REPO/scripts/config.sh" ]; then
    echo "ERROR: $REPO is not the repo root; submit jobs from the repo root (cd <repo> && sbatch scripts/...)" >&2
    exit 1
fi
cd "$REPO" || exit 1
export PYTHONPATH=$REPO:$PYTHONPATH
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Pretrained models (bert-base-uncased, timm backbones, microsoft/layoutlmv3-base) are cached on woody, not in the
# small $HOME quota. hf_xet is not installed and xet downloads fail here, so use plain HTTP downloads.
export HF_HOME=/home/woody/iwi5/iwi5280h/model_cache/hf_home
export HF_HUB_DISABLE_XET=1


# ---------------- Data ----------------
DATA_ROOT=/home/woody/iwi5/iwi5280h/dataset
RVL_DOMAIN=all_prepdataset                  # RVL-CDIP as <split>/<class>/ folders under DATA_ROOT
TOB_DOMAIN=Tobacco3482-jpg                  # Tobacco-3482 as <class>/ folders under DATA_ROOT
RVL_DIR=$DATA_ROOT/$RVL_DOMAIN

# OCR (step 1), all Tesseract. EAML: text tokens per image in one .pt dict (merged from shards);
# LayoutLMv3: bert-base-uncased tokens + boxes, one .pt per image.
# Earlier OCR outputs directly under $DATA_ROOT (all_dataset_ocr_texts_tesseract.pt, ...) are not used.
OCR_ROOT=$DATA_ROOT/ocr
EAML_OCR_SHARDS=$OCR_ROOT/eaml_rvl_shards   # part_<offset>.pt written by run_ocrextractorWtoken.sh, merged by run_combinetensors.sh
EAML_OCR_RVL=$OCR_ROOT/eaml_rvl_tesseract.pt
EAML_OCR_TOB=$OCR_ROOT/eaml_tobacco_tesseract.pt
LLMV3_OCR_RVL=$OCR_ROOT/llmv3_rvl_bert_tesseract
LLMV3_OCR_TOB=$OCR_ROOT/llmv3_tobacco_bert_tesseract
# Pre-trained LayoutLMv3 (LLMV3_MODEL=hf): LayoutLMv3 tokens + boxes (+ OCR words), one .pt per image
LLMV3HF_OCR_RVL=$OCR_ROOT/llmv3hf_rvl_tesseract
LLMV3HF_OCR_TOB=$OCR_ROOT/llmv3hf_tobacco_tesseract
OCR_SHARD_SIZE=50000                        # LayoutLMv3 OCR: images per array task; RVL-CDIP (399,829) = tasks 0-7
EAML_OCR_PART_SIZE=12500                    # EAML OCR: images per array task (~6-7 h); RVL-CDIP = tasks 0-31
RVL_NUM_IMAGES=399829                       # images in $RVL_DIR; the merged EAML OCR file must have this many entries

# ---------------- Outputs of this pipeline ----------------
RUN_ROOT=/home/woody/iwi5/iwi5280h/il_runs
BASE_ROOT=$RUN_ROOT/base                    # step 2: <backbone>_<n>cls/
CIL_ROOT=$RUN_ROOT/cil                      # step 3: <backbone>/<method>_<strategy>/seed<SEED>/
DIL_ROOT=$RUN_ROOT/dil                      # step 4: <backbone>/<method>_<strategy>/seed<SEED>/
# Seed of the incremental runs (steps 3-4); every seed has its own checkpoint folder. Data splits do not depend on it.
# Repeat a run with another seed: sbatch --export=SEED=1 <script> ...   (with LLMV3_MODEL: --export=SEED=1,LLMV3_MODEL=hf)
SEED=${SEED:-42}
export IL_RESULTS_TABLE=$RUN_ROOT/il_runs.csv   # one table with every IL run (utils/run_log.py)

# Base checkpoints written by step 2 and read by steps 3-4
EAML_BASE_11=$BASE_ROOT/eaml_11cls/eaml_best_model.pt
EAML_BASE_16=$BASE_ROOT/eaml_16cls/eaml_best_model.pt
LLMV3_BASE_11=$BASE_ROOT/llmv3_11cls/layoutlmv3_rvl_cdip_best.pt
LLMV3_BASE_16=$BASE_ROOT/llmv3_16cls/layoutlmv3_rvl_cdip_best.pt

# Base-model test accuracies (fractions) that G_IL is measured against.
# Defaults are the paper's values; replace them with your own step-2 results before running steps 3-4.
EAML_BASE_11_ACC=0.9302
EAML_BASE_16_ACC=0.9120
LLMV3_BASE_11_ACC=0.8928
LLMV3_BASE_16_ACC=0.8892
LLMV3HF_BASE_11_ACC=                        # pre-trained LayoutLMv3: fill in after its step-2 runs
LLMV3HF_BASE_16_ACC=

# ---------------- LayoutLMv3 variant ----------------
# custom: Custom LayoutLMv3 (bert-base-uncased + ViT + fusion transformer; the thesis / workshop model)
# hf:     pre-trained LayoutLMv3 (microsoft/layoutlmv3-base)
# Choose it when submitting: sbatch --export=LLMV3_MODEL=hf <llmv3 script> ...   (default: custom)
# It selects the OCR tensors, the base checkpoints and accuracies, the output folders ($CIL_ROOT/$LLMV3_TAG,
# $DIL_ROOT/$LLMV3_TAG) and the backbone name in the results table (llmv3 | llmv3hf).
export LLMV3_MODEL=${LLMV3_MODEL:-custom}
case "$LLMV3_MODEL" in
    custom) LLMV3_TAG=llmv3 ;;
    hf)
        LLMV3_TAG=llmv3hf
        LLMV3_OCR_RVL=$LLMV3HF_OCR_RVL
        LLMV3_OCR_TOB=$LLMV3HF_OCR_TOB
        LLMV3_BASE_11=$BASE_ROOT/llmv3hf_11cls/layoutlmv3_rvl_cdip_best.pt
        LLMV3_BASE_16=$BASE_ROOT/llmv3hf_16cls/layoutlmv3_rvl_cdip_best.pt
        # until filled in, steps 3-4 stop with "invalid float value: 'set_LLMV3HF_BASE_.._ACC_in_config.sh'"
        LLMV3_BASE_11_ACC=${LLMV3HF_BASE_11_ACC:-set_LLMV3HF_BASE_11_ACC_in_config.sh}
        LLMV3_BASE_16_ACC=${LLMV3HF_BASE_16_ACC:-set_LLMV3HF_BASE_16_ACC_in_config.sh}
        ;;
    *) echo "ERROR: LLMV3_MODEL must be custom or hf, got '$LLMV3_MODEL'" >&2; exit 1 ;;
esac

# ---------------- Classes ----------------
ALL_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,scientific_publication,specification,file_folder,news_article,budget,invoice,presentation,questionnaire,resume,memo"
CIL_BASE_CLASSES="letter,form,email,handwritten,advertisement,scientific_report,invoice,presentation,questionnaire,resume,memo"
CIL_ORDER=(scientific_publication specification file_folder news_article budget)   # class added in step 1, 2, ...
DIL_CLASSES="$ALL_CLASSES,Note,Report"      # RVL-CDIP classes + the two Tobacco-only classes

# ---------------- Helpers ----------------
require_file() {
    if [ ! -e "$1" ]; then echo "ERROR: $1 not found${2:+ ($2)}" >&2; exit 1; fi
}

# cil_step <step> <ckpt_dir> <step-1 base checkpoint> <best-checkpoint name, {class} = class of that step>
# Sets BASE_MODEL, BASE_CLASSES and UNSEEN_CLASSES for class-incremental step <step> (1..${#CIL_ORDER[@]}).
# Step 1 starts from the base model; step k starts from the best checkpoint of step k-1 in <ckpt_dir>.
cil_step() {
    local step=$1 ckpt_dir=$2 base_ckpt=$3 pattern=$4 i
    if ! [[ "$step" =~ ^[0-9]+$ ]] || (( step < 1 || step > ${#CIL_ORDER[@]} )); then
        echo "ERROR: CIL step must be 1..${#CIL_ORDER[@]}; submit with sbatch --array=<step> <script>" >&2
        exit 1
    fi
    UNSEEN_CLASSES=${CIL_ORDER[step - 1]}
    BASE_CLASSES=$CIL_BASE_CLASSES
    for ((i = 0; i < step - 1; i++)); do BASE_CLASSES+=",${CIL_ORDER[i]}"; done
    if (( step == 1 )); then
        BASE_MODEL=$base_ckpt
        require_file "$BASE_MODEL" "train the base model first"
    else
        BASE_MODEL=$ckpt_dir/${pattern//\{class\}/${CIL_ORDER[step - 2]}}
        require_file "$BASE_MODEL" "finish CIL step $((step - 1)) first"
    fi
    export IL_STEP=$step                 # recorded in the saved predictions (utils/eval/predictions.py)
    echo "CIL step $step: adding '$UNSEEN_CLASSES' to $BASE_MODEL"
}
