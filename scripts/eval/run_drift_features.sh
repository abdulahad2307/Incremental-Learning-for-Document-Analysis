#!/bin/bash -l

#SBATCH --job-name=5_drift_features
#SBATCH --output=logs/5_eval/%x_%j.out
#SBATCH --error=logs/5_eval/%x_%j.err
#SBATCH --partition=rtx3080
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --time=04:00:00
#SBATCH --export=NONE

# Per-modality features of every CIL step of one run for the drift analysis (tools/eval/drift_analysis.py, plan
# item 13), on the same documents at every step: 200 base training documents and 300 test documents of each of the
# 16 classes. Step 0 is the base model (shared by all runs); steps whose checkpoint does not exist yet and files that
# exist already are skipped, so the job can be resubmitted as the CIL run progresses.
# Output: $RUN_ROOT/features/drift/<eaml|llmv3|llmv3hf>/step0.pt and .../<run>/step<k>.pt
# Submit from the repo root (pre-trained LayoutLMv3: --export=LLMV3_MODEL=hf; any free GPU type via -p / --gres):
#   sbatch scripts/eval/run_drift_features.sh eaml noevm_std
#   sbatch --export=LLMV3_MODEL=hf scripts/eval/run_drift_features.sh llmv3 evm_std

unset SLURM_EXPORT_ENV
module load cuda/12.6
module load python/3.12-conda
conda activate mtil
source "${SLURM_SUBMIT_DIR:-$PWD}/scripts/config.sh" || exit 1

BACKBONE=${1:?usage: sbatch scripts/eval/run_drift_features.sh eaml|llmv3 <run, e.g. noevm_std>}
RUN=${2:?usage: sbatch scripts/eval/run_drift_features.sh eaml|llmv3 <run, e.g. noevm_std>}
METHOD=${RUN%_*}                                   # noevm | evm | evmood | ievm | regevm | finetune | joint
case "$BACKBONE" in
    eaml)  TAG=eaml; OCR=$EAML_OCR_RVL; BASE=$EAML_BASE_11; PATTERN="best_model_{class}.pth" ;;
    llmv3) TAG=$LLMV3_TAG; OCR=$LLMV3_OCR_RVL; BASE=$LLMV3_BASE_11
           case "$METHOD" in                       # checkpoint names of the LayoutLMv3 CIL scripts
               evm) INFIX=evm_ ;; evmood) INFIX=evm_ood_ ;; ievm) INFIX=ievm_ ;; regevm) INFIX=reg_evm_ ;; *) INFIX="" ;;
           esac
           PATTERN="layoutlmv3_cil_incremental_${INFIX}{class}_best.pt" ;;
    *) echo "ERROR: backbone must be eaml or llmv3" >&2; exit 1 ;;
esac
OUT=$RUN_ROOT/features/drift/$TAG
mkdir -p "$OUT/$RUN"

extract() {   # <checkpoint> <known classes> <output>
    [ -f "$3" ] && { echo "exists: $3"; return; }
    python tools/eval/extract_step_features.py --backbone "$BACKBONE" --checkpoint "$1" --known_classes "$2" \
      --all_classes "$ALL_CLASSES" --data_dir "$RVL_DIR" --ocr "$OCR" --train_classes all --train_per_class 200 \
      --test_per_class 300 --out "$3"
}

require_file "$BASE" "train the base model first"
extract "$BASE" "$CIL_BASE_CLASSES" "$OUT/step0.pt"
KNOWN=$CIL_BASE_CLASSES
for ((k = 1; k <= ${#CIL_ORDER[@]}; k++)); do
    NEW=${CIL_ORDER[k - 1]}
    KNOWN=$KNOWN,$NEW
    CKPT=$CIL_ROOT/$TAG/$RUN/seed$SEED/${PATTERN//\{class\}/$NEW}
    if [ ! -f "$CKPT" ]; then echo "step $k: no checkpoint yet ($CKPT)"; continue; fi
    extract "$CKPT" "$KNOWN" "$OUT/$RUN/step$k.pt"
done
echo "Finished: $OUT"
