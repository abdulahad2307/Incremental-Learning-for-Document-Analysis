# Run Guide

How to run the document-classification incremental-learning pipeline (EAML and LayoutLMv3 backbones, class- and domain-incremental learning, EVM / EVM+OOD / iEVM / RegEVM) on the FAU HPC cluster, from the base models to the final results table.

## 1. Repository layout

```
FAU-Masters_Thesis-Ahad-Extension/
├── README.md                 Project overview
├── RUN_README.md             This file
├── configs/
│   └── class_mapping.json    RVL-CDIP class id -> name
├── requirements/
│   ├── il-env-requirement.txt    Training env  (conda: mtil)
│   └── ocr-env-requirement.txt   OCR env       (conda: ocr_env)
├── scripts/                  SLURM job scripts (submit from repo root, see §3)
│   ├── config.sh             Shared paths, class lists, base-model accuracies (sourced by the IL/base scripts)
│   ├── data_prep/            Step 1: OCR extraction, merging tensors
│   ├── base_models/          Step 2: base classifiers (EAML, LayoutLMv3, DocFormer, CNN baseline)
│   ├── class_incremental/    Step 3: eaml/run_classIL*.sh, llmv3/run_llmv3_classIL*.sh
│   ├── domain_incremental/   Step 4: eaml/run_domainIL*.sh, llmv3/run_llmv3_domainIL*.sh
│   └── tests/                run_pipeline_tests.sh (end-to-end tests, see §8)
├── logs/                     SLURM logs, one folder per step (created by you, see §3.2)
├── src/                      Python entry points (called by scripts/)
│   ├── base_models/
│   ├── class_incremental/{eaml,llmv3}/
│   ├── domain_incremental/{eaml,llmv3}/
│   ├── evaluation/
│   └── tests/
├── tools/                    Standalone utilities (called by scripts/data_prep or by hand)
│   ├── data/  ocr/
│   └── eval/                 Evaluation of trained models; il_results_table.py summarises all IL runs
├── utils/                    Library code imported as `utils.*` (models, dataloaders, EVM, IL strategies, run_log)
└── archive/                  Superseded files kept for reference (not used by any script)
```

Rule of thumb: **`scripts/*.sh` → `src/**` or `tools/**` → `utils/**`**. Nothing imports from `src/` or `tools/`, except the test in `src/tests/`.

## 2. One-time setup

### 2.1 Conda environments

```bash
module load python/3.12-conda
conda create -n mtil python=3.12 -y && conda activate mtil
pip install -r requirements/il-env-requirement.txt

conda create -n ocr_env python=3.12 -y && conda activate ocr_env
pip install -r requirements/ocr-env-requirement.txt
```

| Env       | Used by                                                   |
|-----------|-----------------------------------------------------------|
| `mtil`    | `scripts/base_models`, `class_incremental`, `domain_incremental`, `tests`, `tools/eval/il_results_table.py` |
| `ocr_env` | `scripts/data_prep`                                       |

### 2.2 Datasets

1. Download **RVL-CDIP** (images + `labels/{train,val,test}.txt`) and **Tobacco-3482**.
2. Arrange RVL-CDIP into `split/class/` folders:
   ```bash
   # Edit RAW_DATA_PATH, LABEL_PATHS, CSV_PATHS, OUTPUT_DIR at the top first
   python tools/data/dataprep.py
   ```
   Result: `<OUTPUT_DIR>/{train,val,test}/<class_name>/*.tif`.
   Tobacco-3482 is used as-is (`Tobacco3482-jpg/<class_name>/`); the domain-IL loader makes a stratified train/val/test split when those folders are missing.

The data root and folder names are set in `scripts/config.sh` (`DATA_ROOT`, `RVL_DOMAIN`, `TOB_DOMAIN`).

## 3. How the jobs are organised

### 3.1 `scripts/config.sh`

All base-model, CIL and DIL scripts `source scripts/config.sh`. It holds everything that is shared, so the job scripts only contain hyperparameters:

| Variable | Meaning |
|---|---|
| `DATA_ROOT`, `RVL_DIR`, `RVL_DOMAIN`, `TOB_DOMAIN` | Datasets |
| `OCR_ROOT`, `EAML_OCR_RVL/_TOB`, `LLMV3_OCR_RVL/_TOB`, `OCR_SHARD_SIZE` | OCR outputs of step 1 |
| `RUN_ROOT` (`/home/woody/iwi5/iwi5280h/il_runs`) | Everything this pipeline writes: `base/`, `cil/`, `dil/`, `il_runs.csv` |
| `EAML_BASE_11/_16`, `LLMV3_BASE_11/_16` | Base checkpoints written by step 2 and read by steps 3–4 |
| `*_BASE_*_ACC` | Base-model test accuracies that G_IL is measured against (**update after step 2**) |
| `ALL_CLASSES`, `CIL_BASE_CLASSES`, `CIL_ORDER`, `DIL_CLASSES` | Class lists; `CIL_ORDER` is the order in which CIL adds classes |
| `IL_RESULTS_TABLE` | The shared results table (`utils/run_log.py`) |

To start a fresh set of runs without overwriting the previous one, change `RUN_ROOT`.

### 3.2 Job names and log folders

Job names start with the pipeline step, then setting, backbone, method and strategy (`std` = standard IL, `kd` = distillation-based IL). Logs go to one folder per step and backbone:

| Step | Job names | Log folder |
|---|---|---|
| 1 Data prep | `1_ocr_*`, `1_merge_*` | `logs/1_data_prep/<job>_<arrayid>_<task>.out` |
| 2 Base models | `2_base_eaml_11cls`, `2_base_eaml_16cls`, `2_base_llmv3_11cls`, `2_base_llmv3_16cls` (+ `2_base_eaml_multigpu`, `2_base_docformer`, `2_base_cnn`) | `logs/2_base/{eaml,llmv3,other}/` |
| 3 CIL | `3_cil_<backbone>_<method>_<std\|kd>`, e.g. `3_cil_eaml_evmood_kd` | `logs/3_cil/{eaml,llmv3}/<job>_step<N>_<jobid>.out` |
| 4 DIL | `4_dil_<backbone>_<method>_<std\|kd>`, e.g. `4_dil_llmv3_evm_std` | `logs/4_dil/{eaml,llmv3}/<job>_<jobid>.out` |

Methods: `noevm` (ER + EWC + BC), `evm`, `evmood`, `ievm`, `regevm`; extras: `evmposthoc`, `ood`.

**SLURM does not create log folders**; a job whose log folder is missing dies without any output. Create them once:

```bash
cd <repo root>
mkdir -p logs/{1_data_prep,2_base/{eaml,llmv3,other},3_cil/{eaml,llmv3},4_dil/{eaml,llmv3},tests}
```

### 3.3 Submitting

Always submit **from the repository root**. The log paths are relative to it, and the scripts find the code through it: they source `$SLURM_SUBMIT_DIR/scripts/config.sh`, which sets `REPO` and `PYTHONPATH` to that directory. A job therefore runs the code of the checkout it was submitted from (for example a git worktree used for development), and a job submitted from any other directory stops at once with an error.

- **CIL steps** are job-array indices: `sbatch --array=3 <script>` runs step 3 (adds `CIL_ORDER[2]` = `file_folder`). Step *k* starts from the best checkpoint of step *k−1* in the same checkpoint folder, so submit the steps of one method **in order, each after the previous one finished**. A step whose previous checkpoint is missing stops at once with an error.
- **LayoutLMv3 variant**: every LayoutLMv3 script (OCR excepted) runs either the **Custom LayoutLMv3** (default; `bert-base-uncased` + ViT + a 2-layer fusion transformer, the thesis / workshop model) or the **pre-trained LayoutLMv3** (`microsoft/layoutlmv3-base`). Select the pre-trained one with `--export=LLMV3_MODEL=hf`, e.g. `sbatch --export=LLMV3_MODEL=hf --array=1 scripts/class_incremental/llmv3/run_llmv3_classIL.sh`. The variant selects the OCR tensors, base checkpoints, base accuracies and output folders (`$CIL_ROOT/llmv3hf/`, `$DIL_ROOT/llmv3hf/`), and the results table names it `llmv3hf`. A script stops with an error if the checkpoint it loads belongs to the other variant. Add `-J <name>_hf` so the job and log names say which one ran.
- **LayoutLMv3 strategy** is the first script argument: nothing = standard IL, `distillation` = distillation-based IL. Pass `-J <name>_kd` so the job and log name say `kd` (the commands below do). EAML has separate `*_KD.sh` scripts instead.
- **Resuming** an interrupted job: set `RESUME_CKPT` (and for the LayoutLMv3 base models `RESUME_EPOCH`) near the top of the script to the last checkpoint, then resubmit the same command.
- To chain all steps of one CIL method so each starts when the previous one succeeded:
  ```bash
  S=scripts/class_incremental/eaml/run_classIL_evm_ood.sh
  j=$(sbatch --parsable --array=1 $S); for k in 2 3 4 5; do j=$(sbatch --parsable --dependency=afterok:$j --array=$k $S); done
  ```

Monitor with `squeue -u $USER`; cancel with `scancel <jobid>`.

## 4. Full run, one command at a time

### Step 1 — OCR extraction (`scripts/data_prep/`, env `ocr_env`, CPU jobs on TinyFAT)

All OCR is Tesseract and goes to `$OCR_ROOT` (`/home/woody/iwi5/iwi5280h/dataset/ocr/`). The OCR jobs run on the TinyFAT partition `broadwell512`, so they are submitted with **`sbatch.tinyfat`**; SLURM dependencies cannot link them to the TinyGPU training jobs, so wait until they have finished before step 2. RVL-CDIP (399,829 images) is split into 8 array tasks of 50,000 images (`OCR_SHARD_SIZE`).

```bash
# EAML: text tokens
sbatch.tinyfat scripts/data_prep/run_ocrextractorWtoken.sh                      # RVL-CDIP, tasks 0-7 -> $EAML_OCR_SHARDS/shard<k>.pt
sbatch.tinyfat --array=0 scripts/data_prep/run_ocrextractorWtoken.sh tobacco    # Tobacco-3482       -> $EAML_OCR_TOB
# LayoutLMv3: bert-base-uncased tokens + boxes, one file per image
sbatch.tinyfat scripts/data_prep/run_ocrextractor_bert_bbox.sh                   # RVL-CDIP, tasks 0-7 -> $LLMV3_OCR_RVL/
sbatch.tinyfat --array=0 scripts/data_prep/run_ocrextractor_bert_bbox.sh tobacco # Tobacco-3482       -> $LLMV3_OCR_TOB/
# pre-trained LayoutLMv3 (LLMV3_MODEL=hf): LayoutLMv3 tokens + boxes + OCR words, one file per image
sbatch.tinyfat scripts/data_prep/run_ocrextractor_layoutlmv3_bbox.sh                   # RVL-CDIP, tasks 0-7 -> $LLMV3HF_OCR_RVL/
sbatch.tinyfat --array=0 scripts/data_prep/run_ocrextractor_layoutlmv3_bbox.sh tobacco # Tobacco-3482       -> $LLMV3HF_OCR_TOB/

# after all 8 EAML RVL-CDIP tasks have finished: merge the shards into one file
sbatch.tinyfat scripts/data_prep/run_combinetensors.sh                           # -> $EAML_OCR_RVL
```

Check: `sacct -j <jobid>` shows every task `COMPLETED`; `ls $LLMV3_OCR_RVL | wc -l` ≈ 399,829; inspect a file with `python tools/data/read_tensor.py` / `read_tensor_bbox.py`. A failed shard can be rerun alone with `--array=<k>` (the LayoutLMv3 job skips images that already have a file).

Other extraction variants (not used by the pipeline): `run_ocrextractor.sh`, `run_ocrextractor_multi.sh` (plain text), `run_ocrextractorWtoken_bbox.sh` (older LayoutLMv3-tokenizer output; use `run_ocrextractor_layoutlmv3_bbox.sh` for the pre-trained LayoutLMv3). These still hold their own paths.

### Step 2 — Base models (4 jobs, independent, can run in parallel; need the OCR of step 1)

```bash
sbatch scripts/base_models/run_eamlmodel.sh        # 2_base_eaml_11cls  -> $EAML_BASE_11  (CIL base)
sbatch scripts/base_models/run_eamlmodel_all.sh    # 2_base_eaml_16cls  -> $EAML_BASE_16  (DIL base)
sbatch scripts/base_models/run_llmv3_11class.sh    # 2_base_llmv3_11cls -> $LLMV3_BASE_11 (CIL base)
sbatch scripts/base_models/run_llmv3.sh            # 2_base_llmv3_16cls -> $LLMV3_BASE_16 (DIL base)
# pre-trained LayoutLMv3 (needs the hf OCR of step 1) -> $BASE_ROOT/llmv3hf_{11,16}cls/
sbatch --export=LLMV3_MODEL=hf -J 2_base_llmv3hf_11cls scripts/base_models/run_llmv3_11class.sh
sbatch --export=LLMV3_MODEL=hf -J 2_base_llmv3hf_16cls scripts/base_models/run_llmv3.sh
```

When they are done, put their **test accuracies** (last lines of each log) into `EAML_BASE_11_ACC`, `EAML_BASE_16_ACC`, `LLMV3_BASE_11_ACC`, `LLMV3_BASE_16_ACC` (and `LLMV3HF_BASE_11_ACC`, `LLMV3HF_BASE_16_ACC` for the pre-trained LayoutLMv3) in `scripts/config.sh`. Steps 3–4 pass them as `--full_model_acc` / `--base_model_acc`, so G_IL is measured against the base model.

### Step 3 — Class-incremental learning (needs the two 11-class base models)

Five steps per method, adding `scientific_publication`, `specification`, `file_folder`, `news_article`, `budget` (the paper reports steps 1–4). Different methods are independent and can run in parallel; the steps of one method run one after another.

#### 3a. EAML — standard IL

```bash
# No EVM  (job 3_cil_eaml_noevm_std)
sbatch --array=1 scripts/class_incremental/eaml/run_classIL.sh
sbatch --array=2 scripts/class_incremental/eaml/run_classIL.sh
sbatch --array=3 scripts/class_incremental/eaml/run_classIL.sh
sbatch --array=4 scripts/class_incremental/eaml/run_classIL.sh
sbatch --array=5 scripts/class_incremental/eaml/run_classIL.sh
```
```bash
# EVM  (job 3_cil_eaml_evm_std)
sbatch --array=1 scripts/class_incremental/eaml/run_classIL_evm_training.sh
sbatch --array=2 scripts/class_incremental/eaml/run_classIL_evm_training.sh
sbatch --array=3 scripts/class_incremental/eaml/run_classIL_evm_training.sh
sbatch --array=4 scripts/class_incremental/eaml/run_classIL_evm_training.sh
sbatch --array=5 scripts/class_incremental/eaml/run_classIL_evm_training.sh
```
```bash
# EVM+OOD  (job 3_cil_eaml_evmood_std)
sbatch --array=1 scripts/class_incremental/eaml/run_classIL_evm_ood.sh
sbatch --array=2 scripts/class_incremental/eaml/run_classIL_evm_ood.sh
sbatch --array=3 scripts/class_incremental/eaml/run_classIL_evm_ood.sh
sbatch --array=4 scripts/class_incremental/eaml/run_classIL_evm_ood.sh
sbatch --array=5 scripts/class_incremental/eaml/run_classIL_evm_ood.sh
```
```bash
# iEVM  (job 3_cil_eaml_ievm_std)
sbatch --array=1 scripts/class_incremental/eaml/run_classIL_ievm_training.sh
sbatch --array=2 scripts/class_incremental/eaml/run_classIL_ievm_training.sh
sbatch --array=3 scripts/class_incremental/eaml/run_classIL_ievm_training.sh
sbatch --array=4 scripts/class_incremental/eaml/run_classIL_ievm_training.sh
sbatch --array=5 scripts/class_incremental/eaml/run_classIL_ievm_training.sh
```
```bash
# RegEVM  (job 3_cil_eaml_regevm_std)
sbatch --array=1 scripts/class_incremental/eaml/run_classIL_reg_evm_training.sh
sbatch --array=2 scripts/class_incremental/eaml/run_classIL_reg_evm_training.sh
sbatch --array=3 scripts/class_incremental/eaml/run_classIL_reg_evm_training.sh
sbatch --array=4 scripts/class_incremental/eaml/run_classIL_reg_evm_training.sh
sbatch --array=5 scripts/class_incremental/eaml/run_classIL_reg_evm_training.sh
```

#### 3b. EAML — distillation-based IL

```bash
# No EVM  (job 3_cil_eaml_noevm_kd)
sbatch --array=1 scripts/class_incremental/eaml/run_classIL_KD.sh
sbatch --array=2 scripts/class_incremental/eaml/run_classIL_KD.sh
sbatch --array=3 scripts/class_incremental/eaml/run_classIL_KD.sh
sbatch --array=4 scripts/class_incremental/eaml/run_classIL_KD.sh
sbatch --array=5 scripts/class_incremental/eaml/run_classIL_KD.sh
```
```bash
# EVM  (job 3_cil_eaml_evm_kd)
sbatch --array=1 scripts/class_incremental/eaml/run_classIL_evm_training_KD.sh
sbatch --array=2 scripts/class_incremental/eaml/run_classIL_evm_training_KD.sh
sbatch --array=3 scripts/class_incremental/eaml/run_classIL_evm_training_KD.sh
sbatch --array=4 scripts/class_incremental/eaml/run_classIL_evm_training_KD.sh
sbatch --array=5 scripts/class_incremental/eaml/run_classIL_evm_training_KD.sh
```
```bash
# EVM+OOD  (job 3_cil_eaml_evmood_kd)
sbatch --array=1 scripts/class_incremental/eaml/run_classIL_evm_ood_KD.sh
sbatch --array=2 scripts/class_incremental/eaml/run_classIL_evm_ood_KD.sh
sbatch --array=3 scripts/class_incremental/eaml/run_classIL_evm_ood_KD.sh
sbatch --array=4 scripts/class_incremental/eaml/run_classIL_evm_ood_KD.sh
sbatch --array=5 scripts/class_incremental/eaml/run_classIL_evm_ood_KD.sh
```
```bash
# iEVM  (job 3_cil_eaml_ievm_kd)
sbatch --array=1 scripts/class_incremental/eaml/run_classIL_ievm_training_KD.sh
sbatch --array=2 scripts/class_incremental/eaml/run_classIL_ievm_training_KD.sh
sbatch --array=3 scripts/class_incremental/eaml/run_classIL_ievm_training_KD.sh
sbatch --array=4 scripts/class_incremental/eaml/run_classIL_ievm_training_KD.sh
sbatch --array=5 scripts/class_incremental/eaml/run_classIL_ievm_training_KD.sh
```
```bash
# RegEVM  (job 3_cil_eaml_regevm_kd)
sbatch --array=1 scripts/class_incremental/eaml/run_classIL_reg_evm_training_KD.sh
sbatch --array=2 scripts/class_incremental/eaml/run_classIL_reg_evm_training_KD.sh
sbatch --array=3 scripts/class_incremental/eaml/run_classIL_reg_evm_training_KD.sh
sbatch --array=4 scripts/class_incremental/eaml/run_classIL_reg_evm_training_KD.sh
sbatch --array=5 scripts/class_incremental/eaml/run_classIL_reg_evm_training_KD.sh
```

#### 3c. LayoutLMv3 — standard IL

```bash
# No EVM  (job 3_cil_llmv3_noevm_std)
sbatch --array=1 scripts/class_incremental/llmv3/run_llmv3_classIL.sh
sbatch --array=2 scripts/class_incremental/llmv3/run_llmv3_classIL.sh
sbatch --array=3 scripts/class_incremental/llmv3/run_llmv3_classIL.sh
sbatch --array=4 scripts/class_incremental/llmv3/run_llmv3_classIL.sh
sbatch --array=5 scripts/class_incremental/llmv3/run_llmv3_classIL.sh
```
```bash
# EVM  (job 3_cil_llmv3_evm_std)
sbatch --array=1 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_training.sh
sbatch --array=2 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_training.sh
sbatch --array=3 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_training.sh
sbatch --array=4 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_training.sh
sbatch --array=5 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_training.sh
```
```bash
# EVM+OOD  (job 3_cil_llmv3_evmood_std)
sbatch --array=1 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_ood.sh
sbatch --array=2 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_ood.sh
sbatch --array=3 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_ood.sh
sbatch --array=4 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_ood.sh
sbatch --array=5 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_ood.sh
```
```bash
# iEVM  (job 3_cil_llmv3_ievm_std)
sbatch --array=1 scripts/class_incremental/llmv3/run_llmv3_classIL_ievm_training.sh
sbatch --array=2 scripts/class_incremental/llmv3/run_llmv3_classIL_ievm_training.sh
sbatch --array=3 scripts/class_incremental/llmv3/run_llmv3_classIL_ievm_training.sh
sbatch --array=4 scripts/class_incremental/llmv3/run_llmv3_classIL_ievm_training.sh
sbatch --array=5 scripts/class_incremental/llmv3/run_llmv3_classIL_ievm_training.sh
```
```bash
# RegEVM  (job 3_cil_llmv3_regevm_std)
sbatch --array=1 scripts/class_incremental/llmv3/run_llmv3_classIL_reg_evm_training.sh
sbatch --array=2 scripts/class_incremental/llmv3/run_llmv3_classIL_reg_evm_training.sh
sbatch --array=3 scripts/class_incremental/llmv3/run_llmv3_classIL_reg_evm_training.sh
sbatch --array=4 scripts/class_incremental/llmv3/run_llmv3_classIL_reg_evm_training.sh
sbatch --array=5 scripts/class_incremental/llmv3/run_llmv3_classIL_reg_evm_training.sh
```

#### 3d. LayoutLMv3 — distillation-based IL

Not run in the paper: the teacher copy did not fit next to the student on a 32 GB V100 (see the note below the step list).

```bash
# No EVM  (job 3_cil_llmv3_noevm_kd)
sbatch -J 3_cil_llmv3_noevm_kd --array=1 scripts/class_incremental/llmv3/run_llmv3_classIL.sh distillation
sbatch -J 3_cil_llmv3_noevm_kd --array=2 scripts/class_incremental/llmv3/run_llmv3_classIL.sh distillation
sbatch -J 3_cil_llmv3_noevm_kd --array=3 scripts/class_incremental/llmv3/run_llmv3_classIL.sh distillation
sbatch -J 3_cil_llmv3_noevm_kd --array=4 scripts/class_incremental/llmv3/run_llmv3_classIL.sh distillation
sbatch -J 3_cil_llmv3_noevm_kd --array=5 scripts/class_incremental/llmv3/run_llmv3_classIL.sh distillation
```
```bash
# EVM  (job 3_cil_llmv3_evm_kd)
sbatch -J 3_cil_llmv3_evm_kd --array=1 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_training.sh distillation
sbatch -J 3_cil_llmv3_evm_kd --array=2 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_training.sh distillation
sbatch -J 3_cil_llmv3_evm_kd --array=3 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_training.sh distillation
sbatch -J 3_cil_llmv3_evm_kd --array=4 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_training.sh distillation
sbatch -J 3_cil_llmv3_evm_kd --array=5 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_training.sh distillation
```
```bash
# EVM+OOD  (job 3_cil_llmv3_evmood_kd)
sbatch -J 3_cil_llmv3_evmood_kd --array=1 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_ood.sh distillation
sbatch -J 3_cil_llmv3_evmood_kd --array=2 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_ood.sh distillation
sbatch -J 3_cil_llmv3_evmood_kd --array=3 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_ood.sh distillation
sbatch -J 3_cil_llmv3_evmood_kd --array=4 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_ood.sh distillation
sbatch -J 3_cil_llmv3_evmood_kd --array=5 scripts/class_incremental/llmv3/run_llmv3_classIL_evm_ood.sh distillation
```
```bash
# iEVM  (job 3_cil_llmv3_ievm_kd)
sbatch -J 3_cil_llmv3_ievm_kd --array=1 scripts/class_incremental/llmv3/run_llmv3_classIL_ievm_training.sh distillation
sbatch -J 3_cil_llmv3_ievm_kd --array=2 scripts/class_incremental/llmv3/run_llmv3_classIL_ievm_training.sh distillation
sbatch -J 3_cil_llmv3_ievm_kd --array=3 scripts/class_incremental/llmv3/run_llmv3_classIL_ievm_training.sh distillation
sbatch -J 3_cil_llmv3_ievm_kd --array=4 scripts/class_incremental/llmv3/run_llmv3_classIL_ievm_training.sh distillation
sbatch -J 3_cil_llmv3_ievm_kd --array=5 scripts/class_incremental/llmv3/run_llmv3_classIL_ievm_training.sh distillation
```
```bash
# RegEVM  (job 3_cil_llmv3_regevm_kd)
sbatch -J 3_cil_llmv3_regevm_kd --array=1 scripts/class_incremental/llmv3/run_llmv3_classIL_reg_evm_training.sh distillation
sbatch -J 3_cil_llmv3_regevm_kd --array=2 scripts/class_incremental/llmv3/run_llmv3_classIL_reg_evm_training.sh distillation
sbatch -J 3_cil_llmv3_regevm_kd --array=3 scripts/class_incremental/llmv3/run_llmv3_classIL_reg_evm_training.sh distillation
sbatch -J 3_cil_llmv3_regevm_kd --array=4 scripts/class_incremental/llmv3/run_llmv3_classIL_reg_evm_training.sh distillation
sbatch -J 3_cil_llmv3_regevm_kd --array=5 scripts/class_incremental/llmv3/run_llmv3_classIL_reg_evm_training.sh distillation
```

Extras (not in the paper's comparison), same step-by-step usage:

```bash
sbatch --array=1 scripts/class_incremental/eaml/run_classIL_evm.sh   # 3_cil_eaml_evmposthoc_kd: EVM fitted after each step, no L_EVM
sbatch --array=1 scripts/class_incremental/eaml/run_classIL_ood.sh   # 3_cil_eaml_ood_kd: ViM OOD detector only
```

**Note on LayoutLMv3 with distillation:** the teacher is a frozen copy of the model, so a job needs roughly twice the GPU memory of standard IL. In the paper this did not fit on a 32 GB V100. If it runs out of memory, add a larger GPU on the command line, e.g. `sbatch -p a100 --gres=gpu:a100:1 …` (check which partitions your account can use).

### Step 4 — Domain-incremental learning RVL-CDIP → Tobacco-3482 (needs the two 16-class base models)

One job per method; all are independent and can run in parallel.

#### 4a. EAML — standard IL, then distillation-based IL

```bash
sbatch scripts/domain_incremental/eaml/run_domainIL.sh                    # No EVM, standard IL
sbatch scripts/domain_incremental/eaml/run_domainIL_evm_training.sh       # EVM, standard IL
sbatch scripts/domain_incremental/eaml/run_domainIL_evm_ood.sh            # EVM+OOD, standard IL
sbatch scripts/domain_incremental/eaml/run_domainIL_ievm_training.sh      # iEVM, standard IL
sbatch scripts/domain_incremental/eaml/run_domainIL_reg_evm_training.sh   # RegEVM, standard IL
sbatch scripts/domain_incremental/eaml/run_domainIL_KD.sh                 # No EVM, distillation-based IL
sbatch scripts/domain_incremental/eaml/run_domainIL_evm_training_KD.sh    # EVM, distillation-based IL
sbatch scripts/domain_incremental/eaml/run_domainIL_evm_ood_KD.sh         # EVM+OOD, distillation-based IL
sbatch scripts/domain_incremental/eaml/run_domainIL_ievm_training_KD.sh   # iEVM, distillation-based IL
sbatch scripts/domain_incremental/eaml/run_domainIL_reg_evm_training_KD.sh  # RegEVM, distillation-based IL
```

#### 4b. LayoutLMv3 — standard IL, then distillation-based IL

```bash
sbatch scripts/domain_incremental/llmv3/run_llmv3_domainIL.sh            # No EVM, standard IL
sbatch scripts/domain_incremental/llmv3/run_llmv3_domainIL_evm.sh        # EVM, standard IL
sbatch scripts/domain_incremental/llmv3/run_llmv3_domainIL_evm_ood.sh    # EVM+OOD, standard IL
sbatch scripts/domain_incremental/llmv3/run_llmv3_domainIL_ievm.sh       # iEVM, standard IL
sbatch scripts/domain_incremental/llmv3/run_llmv3_domainIL_reg_evm.sh    # RegEVM, standard IL
sbatch -J 4_dil_llmv3_noevm_kd scripts/domain_incremental/llmv3/run_llmv3_domainIL.sh distillation
sbatch -J 4_dil_llmv3_evm_kd scripts/domain_incremental/llmv3/run_llmv3_domainIL_evm.sh distillation
sbatch -J 4_dil_llmv3_evmood_kd scripts/domain_incremental/llmv3/run_llmv3_domainIL_evm_ood.sh distillation
sbatch -J 4_dil_llmv3_ievm_kd scripts/domain_incremental/llmv3/run_llmv3_domainIL_ievm.sh distillation
sbatch -J 4_dil_llmv3_regevm_kd scripts/domain_incremental/llmv3/run_llmv3_domainIL_reg_evm.sh distillation
```

Extras (not in the paper's comparison):

```bash
sbatch scripts/domain_incremental/eaml/run_domainIL_evm.sh   # 4_dil_eaml_evmposthoc_kd: EVM on adapted features, no L_EVM
sbatch scripts/domain_incremental/eaml/run_domainIL_ood.sh   # 4_dil_eaml_ood_kd: domain-shift detection with ViM
```

### Step 5 — Results

Every IL job appends its metrics (per epoch, per step, final, open-set and OOD) to `$RUN_ROOT/il_runs.csv`, and its full arguments to `$RUN_ROOT/il_runs_config.jsonl`. Each row also records:
- the run's **seed**: every script takes `--seed`, default 42. For repeated runs, add `--seed <n>` to the python command in a copy of the launcher.
- the **git commit** of the code. A `-dirty` suffix means tracked files had uncommitted changes, so commit before running anything that goes into the paper.

Runs that differ only in their seed are kept as separate results. If the table was written by an older version with other columns, jobs stop with an error instead of appending misaligned rows; move the old file aside.

Summarise with the `mtil` env:

```bash
module load python/3.12-conda && conda activate mtil
export IL_RESULTS_TABLE=/home/woody/iwi5/iwi5280h/il_runs/il_runs.csv
python tools/eval/il_results_table.py                                  # final results, latest run per configuration
python tools/eval/il_results_table.py --backbone eaml --setting CIL --out results_eaml_cil.md
python tools/eval/il_results_table.py --phase epoch --run-id <run_id>  # training curve of one run
```

Stand-alone evaluation of a finished model: `tools/eval/eaml_eval.py`, `tools/eval/sota_llmv3_testing.py`, `tools/eval/verify_model.py`, `src/evaluation/dil_eval.py`, `tools/eval/eaml_eval_dil.py`.

## 5. What the IL scripts do

**Scripts and entry points**

| Method | EAML CIL (`scripts/class_incremental/eaml/`) | LayoutLMv3 CIL (`…/llmv3/`) | EAML DIL (`scripts/domain_incremental/eaml/`) | LayoutLMv3 DIL (`…/llmv3/`) |
|---|---|---|---|---|
| No EVM | `run_classIL[_KD].sh` | `run_llmv3_classIL.sh` | `run_domainIL[_KD].sh` | `run_llmv3_domainIL.sh` |
| EVM | `run_classIL_evm_training[_KD].sh` | `run_llmv3_classIL_evm_training.sh` | `run_domainIL_evm_training[_KD].sh` | `run_llmv3_domainIL_evm.sh` |
| EVM+OOD | `run_classIL_evm_ood[_KD].sh` | `run_llmv3_classIL_evm_ood.sh` | `run_domainIL_evm_ood[_KD].sh` | `run_llmv3_domainIL_evm_ood.sh` |
| iEVM | `run_classIL_ievm_training[_KD].sh` | `run_llmv3_classIL_ievm_training.sh` | `run_domainIL_ievm_training[_KD].sh` | `run_llmv3_domainIL_ievm.sh` |
| RegEVM | `run_classIL_reg_evm_training[_KD].sh` | `run_llmv3_classIL_reg_evm_training.sh` | `run_domainIL_reg_evm_training[_KD].sh` | `run_llmv3_domainIL_reg_evm.sh` |

Each script calls the matching entry point in `src/class_incremental/<backbone>/` or `src/domain_incremental/<backbone>/`; run `python <entry point> --help` for all options.

**CIL training data per step (EAML):** the new class's training images plus, with `--use_exemplars`, an exemplar memory of every previous class (`--max_exemplars` in total, `--exemplar_selection herding|random`, herding on a 1000-image pool per class). The memory is rebuilt from the previous step's model when a job starts, so one-class-per-job runs work. `--joint_training` instead trains on all images of all classes seen so far — an upper bound, not incremental learning. New-class images (~20k) far outnumber exemplars (~20 per class): use `--use_balanced_sampler`.

**EVM state between steps:** the EVM-training CIL scripts (`evm`, `evmood`, `ievm`, `regevm`, both backbones) pass `--evm_persist`: after each step the EVM (and for EVM+OOD the ViM behind L_OOD) is saved as `evm_state_<class>.pt` next to that step's best checkpoint, and the next step's job loads it, so L_EVM is active from step 2 on. Without `--evm_persist` the EVM lives in memory only, and with one class per job L_EVM would never be active. iEVM adds each new class with its partial update (Eq. 8) instead of refitting from scratch.

**L_OOD** (`*_evm_ood`, `--lambda_ood`, default 0.1): ViM virtual-logit penalty −log(1 − p_virtual) on training samples of classes the ViM was fitted on. CIL: fitted at the end of each step on that step's training data and used in the next step. DIL: fitted once before adaptation on the pretrained domain (`--ood_max_per_class` images per class) with the base model. `--lambda_ood 0` disables it.

**DIL:** adapts the 16-class RVL-CDIP model (domain 0) to Tobacco-3482 (domain 1; classes `Note` and `Report` are new) and evaluates on both test sets. With `--use_exemplars`, a replay memory of RVL-CDIP exemplars is selected with the base model before training and mixed into every Tobacco batch. The distillation teacher is the unmodified 16-class base model.

**Open-set / OOD evaluation** (`--ood_method` `msp` | `vim` | `gradnorm`; all scores "higher = known", shared code in `utils/ood/ood_eval.py`):

| Setting | Known (in-distribution) | Unknown (OOD) |
|---|---|---|
| CIL, after every step | test images of the classes learned so far | test images of the classes not learned yet (skipped after the last step) |
| DIL, before and after adaptation | pretrained-domain test (RVL-CDIP) | incremental-domain test (Tobacco-3482) |

The detector is fitted on known training data and its threshold calibrated on known validation data so that `--ood_tpr` (default 0.95) of known samples are accepted. Reported: AUROC, AUPR-In/Out, FPR@95%TPR, and at the calibrated threshold known acceptance, unknown rejection and open-set accuracy. `--ood_max_per_class` (default 500) caps images per class. The EVM scripts report EVM known accuracy and unknown rejection on the same split and take `--evm_tailsize`.

## 6. Outputs

| What | Where |
|---|---|
| SLURM logs | `logs/<step>/<backbone>/<job-name>[_step<N>]_<jobid>.{out,err}` |
| Base models | `$RUN_ROOT/base/{eaml,llmv3}_{11,16}cls/` (`eaml_best_model.pt`, `layoutlmv3_rvl_cdip_best.pt`) |
| CIL checkpoints | `$RUN_ROOT/cil/<backbone>/<method>_<std\|kd>/`: EAML `best_model_<class>.pth`, `epoch<N>_<class>.pth`, `final_global_best_model.pth`; LayoutLMv3 `layoutlmv3_cil_incremental_[<method>_]<class>_best.pt`; `evm_state_<class>.pt` |
| DIL checkpoints | `$RUN_ROOT/dil/<backbone>/<method>_<std\|kd>/`: `best_model.pth` + last epochs (LayoutLMv3: `layoutlmv3_domain_incremental_best.pt`) |
| Metrics of all IL runs | `$RUN_ROOT/il_runs.csv`, arguments in `$RUN_ROOT/il_runs_config.jsonl` |
| EVM / OOD plots | `evm_hist*.png`, `ood_*.png` in the checkpoint folder |

Checkpoints are dicts with `model_state_dict`, `optimizer_state_dict`, `epoch` and step metadata; load with
`model.load_state_dict(torch.load(path, weights_only=False)["model_state_dict"])`.

## 7. Notes

- **Tests:** end-to-end tests for every pipeline are described in §8. The old smoke test (`scripts/tests/cil_test_run.sh`, `src/tests/test_cil.py`) uses an outdated function signature.
- **LayoutLMv3 OCR:** the model's text encoder is `bert-base-uncased`, so its OCR must be extracted with `run_ocrextractor_bert_bbox.sh` (each word-piece gets its word's box). The older LayoutLMv3-tokenizer folders fail the model's vocabulary check at the first batch.
- **EAML domain-IL reads OCR text.** `--ocr_tensor_dirs` accepts a directory of per-image `.pt` files or a single `.pt` dict file. Earlier DIL results (before this fix) are image-only.
- **Training scope** (`--training_mode` for CIL, `--finetune_mode` for EAML DIL; `training_mode` for LayoutLMv3 DIL): `last_layer` trains the model's last feature layer + the classifier heads (EAML: the output projections of the image and text encoders, `image_encoder.model.classif` + `text_encoder.fc`, 1.2M params incl. heads — EAML fuses by element-wise sum as in the paper, so the fusion itself has no parameters; Custom LayoutLMv3: last fusion-transformer layer, 5.5M params; HF LayoutLMv3: last encoder layer). `classifier_only` / `head_only` trains only the heads — then EWC (which excludes the heads) and L_EVM (a loss on the features) have no effect and the scripts print a warning. `full` / `full_model` / `full_finetune` trains everything; EAML DIL also has `partial_finetune` (`--unfreeze_depth`). Until Sep 2026 `last_layer` meant `classifier_only`.
- **EAML class-incremental training changed.** Earlier CIL runs trained on all images of all seen classes (joint retraining), and cross-entropy was counted twice; reproduce them with `--joint_training`.
- **Domain-IL EVM-training scripts apply `--strategy distillation` and `--use_ewc`** (they used to be ignored), and domain-IL `--use_exemplars` replays pretrained-domain exemplars.
- **RegEVM** fits each extreme vector's Weibull with Eq. 10: tail negative log-likelihood + α·λ² (α = `lambda_reg`, default 0.1); α = 0 is the standard EVM. α acts in the units of the feature distances.
- **EVM training loss** is differentiable and matched to classes by name (`utils/evm/evm_loss.py`). With steep Weibull fits it saturates quickly, so tune `--lambda_evm`.
- **LayoutLMv3 data fixes:** val/test come from their own split folders, labels use one label space for all loaders, the distillation teacher is a frozen copy of the base model (it used to be the student itself), and images use the same preprocessing as the base model. LayoutLMv3 IL scripts default to `--strategy standard` and pass `--use_bias_correction`.
- **EVM scoring** (Sep 2026): scores are the maximum inclusion probability over extreme vectors with half-distance Weibull fits, as in Rudd et al. Earlier results are not comparable.
- **Bias correction in EAML CIL:** the EVM, iEVM and RegEVM CIL scripts do not pass `--use_bias_correction` (as in the runs reported in the paper); the No-EVM, EVM+OOD and OOD scripts do.
- `archive/` holds old versions and scratch files. `utils/eaml/dataloader_old.py` stays in place because `src/base_models/sota_eaml_model_multipros.py` imports it.

## 8. Pipeline tests

`tests/` runs every pipeline end to end on a tiny synthetic dataset (real class names and directory layouts, both OCR formats, randomly initialised base checkpoints), one epoch per run:

| File | Covers |
|---|---|
| `tests/test_baseline.py` | CNN baseline, EAML base, LayoutLMv3 base |
| `tests/test_eaml_cil.py` | No EVM, EVM, EVM+OOD, iEVM, RegEVM (+ OOD-only, post-hoc EVM) × standard / distillation; EVM persistence over two jobs; joint training |
| `tests/test_eaml_dil.py` | the same methods × both strategies, RVL-CDIP → Tobacco-3482 with replay |
| `tests/test_layout_cil.py` | No EVM, EVM, EVM+OOD, iEVM, RegEVM × both strategies; EVM persistence; joint training |
| `tests/test_layout_dil.py` | No EVM, EVM, EVM+OOD, iEVM, RegEVM × both strategies |

A test passes when the script exits cleanly, logs no traceback and writes its checkpoint (plus, per method, its EVM / open-set output). The tests check that the pipelines run; they do not measure accuracy. Run them on a GPU node:

```bash
sbatch scripts/tests/run_pipeline_tests.sh                          # all 55 tests
sbatch scripts/tests/run_pipeline_tests.sh tests.test_eaml_dil      # one pipeline
PIPELINE_TEST_METHODS=evm_ood PIPELINE_TEST_STRATEGIES=distillation sbatch scripts/tests/run_pipeline_tests.sh
```

Each run's command and full output are in `$PIPELINE_TEST_DIR/il_pipeline_tests/runs/<pipeline>/<test>/run.log` (default: the node's `$TMPDIR`; the synthetic data and test checkpoints take about 3 GB).
