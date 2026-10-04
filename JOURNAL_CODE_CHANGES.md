# Code Change Plan for the Journal Extension

Companion to [JOURNAL_EXTENSION_PLAN.md](JOURNAL_EXTENSION_PLAN.md). This file lists **what has to change in the existing code** so the new pieces fit in: MEVM, seeds, the shared protocol, new metrics, and the new baselines. It is a plan only; nothing here is implemented.

Sections are in dependency order: each one only needs the sections above it.

---

## 0. Decision needed first: which "LayoutLMv3"?

`utils/llmv3/llmv3_model_loader.py::LayoutLMv3` is **not** Microsoft's pretrained LayoutLMv3. It is a custom model made of:
- `bert-base-uncased` as the text encoder;
- a timm `vit_base_patch16_224`;
- a 2-layer `nn.TransformerEncoder` for fusion;
- a layout embedding that is only `Linear(2 → 768)` on box centres.

The paper describes the real LayoutLMv3 (unified pre-training, 2-D position embeddings of full boxes, ViT patches). A journal reviewer who reads the code will flag this. Choose one:

| Option | Changes | Cost |
|---|---|---|
| **A. Switch to the real model** (recommended if compute allows) | Wrap `transformers.LayoutLMv3ForSequenceClassification` (`microsoft/layoutlmv3-base`) behind the same interface: `forward`, `forward_features`, `extract_features`, `classifier`. Inputs: LayoutLMv3 tokenizer ids (RoBERTa vocabulary), 4-coordinate boxes in 0–1000, `pixel_values`. `_check_token_ids` hints that tensors tokenized with the LayoutLMv3 tokenizer may already exist; check `tools/ocr/` outputs before re-running OCR. | Base model retrained; all LayoutLMv3 IL runs repeated (they are repeated anyway for seeds) |
| **B. Keep the custom model and rename it** | In the paper, call it a "BERT+ViT layout-fusion model". Optionally make the layout embedding use all 4 box coordinates. No library change. | Cheap, but weakens the "layout-aware backbone" claim |

Everything below works with either option. Only the feature pooling in §2.2 differs.

**Decision (issue #3): option A, keeping the custom model as well.**
- **`utils/llmv3/llmv3_model_loader.py`** now holds two models:
  - `CustomLayoutLMv3`: the old model. `LayoutLMv3` stays as an alias, so old imports and checkpoints keep working.
  - `HFLayoutLMv3`: `microsoft/layoutlmv3-base` with a plain `nn.Linear` classifier on [CLS].
- **Loading:** `load_llmv3_checkpoint` picks the right model from the checkpoint. It uses the stored `model_type`, or otherwise infers it from the parameter names.
- **Selecting the variant for jobs:** `sbatch --export=LLMV3_MODEL=hf ...`. In `scripts/config.sh`, this switches the OCR tensors, base checkpoints, base accuracies, output folders (`llmv3hf`) and the results-table name.
- **OCR for the HF model:** `tools/ocr/ocr_extraction_bbox_layoutlmv3.py` / `scripts/data_prep/run_ocrextractor_layoutlmv3_bbox.sh`.
- **Feature pooling for HF:** in §2.2, pool `backbone` `last_hidden_state`; text tokens come first, then the visual [CLS] and patches.

---

## 1. Shared foundations

### 1.1 Seeds
- **New `utils/seed.py`:** `set_seed(seed, deterministic=True)` sets Python, NumPy and torch seeds, cuDNN deterministic mode, and returns a `torch.Generator` for DataLoader shuffling.
- **Add `--seed`** to every EAML CIL/DIL script. Today only the LayoutLMv3 scripts have it; for example, `src/class_incremental/eaml/class_incremental.py` has none. Call `set_seed(args.seed)` right after argument parsing.
- **Pass the seed through** to everything that samples:
  - `ExemplarManager` (random selection);
  - `subsample_loader(..., seed=)` in `utils/ood/ood_eval.py`, which currently defaults to `seed=0`;
  - the DataLoader `generator=` in `build_cil_train_loader` and the EAML loaders;
  - `DILDataLoader` / `stratified_split(random_state=)`.

### 1.2 Shared protocol config
- **New `configs/journal_protocol.json`:** split manifests, class orders, seeds, `training_mode`, `num_epochs`, `patience`, `lr`, exemplar budget and selection, EVM/MEVM settings, λ values.
- **New `utils/protocol.py`:** `apply_protocol(parser, path)` reads the JSON and calls `parser.set_defaults(...)`, so explicit command-line flags still override it.
- **In every CIL/DIL script:** add `--protocol` and a single `apply_protocol(...)` call before `parse_args()`. No other logic changes.

### 1.3 Run logging (`utils/run_log.py`, `tools/eval/il_results_table.py`)
- **`COLUMNS`:** add `seed`, `protocol`, `class_order`, `evm_type`, `ood_auroc_evm`, `oscr`, `avg_inc_acc`, `forgetting`, `bwt`.
- **`run_log.init`:** store `seed` and `protocol` from the arguments, the same way `strategy` is stored now.
- **New phase `"step_eval"`:** written once at the end of every step, with per-class test accuracy for all classes seen so far, stored as JSON in `extra`. This is the accuracy matrix that forgetting and BWT are computed from.
- **`il_results_table.py`:**
  - add the new columns to `METRICS`;
  - add `seed` to the run identity, so re-runs with different seeds are no longer collapsed by `groupby(CONFIG)["run_id"].transform("last")`.

### 1.4 Saving predictions (needed for bootstrap confidence intervals and McNemar)
- **New helper `save_predictions(path, y_true, y_pred, scores, class_names)`**, which writes an `.npz`. Call it at the end of every step for val and test in every script. Write the files next to the step checkpoint, e.g. `preds_step{t}_{split}.npz`.

---

## 2. Model feature interface (needed by MEVM)

### 2.1 EAML (`utils/eaml/eaml_model.py`)
- **Add `forward_features(images, texts=None, input_ids=None, attention_mask=None, modalities=False)`.** It should return:
  - `logits` (fusion);
  - `fused`;
  - when `modalities=True`, also `image` and `text`, the L2-normalised branch features (the same ones the fusion module consumes).
- **Get logits and features from one forward pass.** Today `train_one_epoch_evm_ood` (EAML CIL/DIL) calls `model(...)` and then `model.extract_features(...)`. That runs both encoders twice per batch, and the two passes see different dropout:
  - `forward` applies dropout before normalisation;
  - `extract_features` does not.

  Switching to one pass roughly halves the cost of every EVM/OOD run.
- **Keep `extract_features`** (it is used by herding in `ExemplarManager`) as a thin wrapper around `forward_features(...)["fused"]`.

### 2.2 LayoutLMv3 (`utils/llmv3/llmv3_model_loader.py`, or the new wrapper from option A)
- **Extend `forward_features(..., modalities=False)`.** With `modalities=True`, return a dict:
  - `cls`: `fused_out[:, 0]` (current behaviour);
  - `text`: attention-masked mean of `fused_out[:, :L_text]`;
  - `visual`: mean of `fused_out[:, L_text:]`.

  `L_text = input_ids.size(1)`. With option A, pool `hidden_states[-1]` the same way.
- **Give `forward` the same `return_features` flag**, so logits and features come from one pass.

### 2.3 One feature-collection helper
- **New `utils/features/modality_features.py`:**
  - `batch_features(model, batch, device, modalities=False)` handles all three batch formats that are currently duplicated in:
    - `train_one_epoch_evm_ood` (`src/class_incremental/eaml/class_incremental_evm_ood.py`);
    - `extract_features` (`utils/class_IL/cil_utils.py`);
    - `collect_features_logits` (`utils/ood/ood_eval.py`);
    - `features_by_class` (`utils/llmv3/llmv3_il_common.py`).
  - `features_by_class(model, loader, device, modalities=False, max_per_class=None)` returns `{class: array}` or `{class: {modality: array}}`.
- **Adopt it in the new code paths first.** Move the old functions over only when they are touched anyway.

---

## 3. Extreme value theory

### 3.1 Changes to `utils/evm/evm_classifier.py`
- **Factor out** `_tail_half_distances(points, negatives)`, which both `fit` and `fit_cuda` repeat, so MEVM can reuse it.
- **`fit_cuda` ignores `distance_metric`**: it always uses `torch.cdist(p=2)`. Respect the metric (Euclidean or cosine), because the sweep needs it.
- **Expose the hidden settings on the command line** in every script:
  - `max_fit_samples` (default 50, never set from the command line);
  - `cover_threshold`: EAML constructs `EVMClassifier(tailsize=..., cover_threshold=0.7)` with a hard-coded value, e.g. `class_incremental_evm_ood.py:219`.

  Add `--evm_cover_threshold`, `--evm_max_fit_samples` and `--evm_distance`. The LayoutLMv3 `add_il_args` already has `--evm_threshold`; use one consistent name for both backbones.

### 3.2 New `utils/evm/copula.py`
- `kendall_tau(a, b)` and `mean_pairwise_tau(margins: (N, m))`.
- `theta_from_tau(tau) = max(1, 1/(1 − tau))`.
- `log_l_theta(log_t: (..., m), theta) = logsumexp(theta · log_t, -1) / theta`. This is the numerically stable `log ‖t‖_θ`, in both NumPy and torch versions.

### 3.3 New `utils/evm/mevm_classifier.py` — `MultivariateEVM`
- **Constructor:** `modalities`, `tailsize`, `cover_threshold`, `max_fit_samples`, `distance_metric`, `theta` (`"kendall"` or a float), and optional modality `weights`.
- **`fit(features: {class: {modality: (N, D_m)}})`:**
  - For each class, prune the points with the **same indices in every modality**. Extreme vectors must stay aligned across modalities, so pruning by index is required.
  - For each extreme vector and each modality, fit a Weibull (κ_ij, λ_ij) on the half-distances to the nearest negatives *in that modality*, reusing §3.1.
  - For θ per class: take each sample's per-modality nearest-negative margin, compute the mean pairwise Kendall τ, and convert it with `theta_from_tau`.
- **Storage:** `weibull_models[class] = [(points: {mod: vec}, scales: (m,), shapes: (m,))]`, plus `theta[class]`, plus `class_features` and `class_means`. `evm_openset_metrics` checks membership with `yt in evm.class_features`, so MEVM must provide that attribute.
- **`predict_proba(features: {mod: (N, D_m)})`:** Ψ = exp(−‖t‖_θ), maximum over extreme vectors.
- **`predict_proba_tensor`, `predict(threshold)`:** the same return contract as `EVMClassifier`.
- **`state_dict()` / `load_state_dict()`**, so `utils/evm/evm_state.py` can persist it with `--evm_persist`.
- **Optional modality `weights`:** the weighted-logistic variant, item 14 in the plan.

### 3.4 MEVM loss (`utils/evm/evm_loss.py`)
- **Add `mevm_nll_loss(mevm, features: {mod: tensor}, labels, class_names=None)`:**
  - per modality: `log_u_j = κ_j · log(d_j/λ_j + 1e-12)`;
  - combine: `log ‖u‖_θ = log_l_theta(log_u, θ_class)`;
  - then the same clamp and `min` over extreme vectors as `evm_nll_loss`.
- **Same contract as `evm_nll_loss`:** it returns `None` when the EVM knows none of the classes in the batch.
- **Generalise `_extreme_vectors`** to read the MEVM storage format.

### 3.5 Open-set evaluation (`utils/evm/evm_eval.py`)
- **Accept nested `{class: {mod: array}}` input**, concatenating per modality.
- **Add threshold-free metrics** computed from the returned `scores` (maximum class Ψ): AUROC and FPR95 for known vs unknown, and OSCR. Put them in the new `utils/metrics/openset_metrics.py` and call them from here.
- **Log them** via `run_log.log("open_set", ...)` in every script that already logs open-set results.

### 3.6 Optional MEVM variants (only after the core works)
- **Incremental MEVM:** in `utils/ievm/ievm.py`, either add a `MultivariateIncrementalEVM` or keep one `IncrementalEVM` per modality.
  - **Design constraint:** the partial update and the K-set-cover reduction must keep the **same extreme-vector set in every modality**. Coverage therefore has to be computed jointly, e.g. on the joint Ψ, not per modality.
  - Re-estimate θ only for classes that changed.
- **ViM residual as an extra dimension:** `utils/ood/vim.py::VIM_OOD` gets `residual_norm(features)`. MEVM then accepts one "scalar modality" whose marginal Weibull is fitted on class residual norms rather than pairwise distances.
- **GPD tail ablation:** add `tail_model={"weibull","gpd"}` to the Weibull-fitting helper (`scipy.stats.genpareto`, peaks over threshold).

---

## 4. Wiring MEVM into the training scripts

Recommendation: add an **`--evm_type {evm, mevm}`** flag to the existing EVM scripts instead of creating more near-duplicate scripts. The repo already has about 7 near-copies per backbone and setting. The new `*_mevm.py` files listed in plan section H are then unnecessary; only the launchers are new.

| File | Change |
|---|---|
| `src/class_incremental/eaml/class_incremental_evm_ood.py` | `--evm_type`, `--mevm_modalities` (default `image,text,fused`), `--mevm_theta`. Build `EVMClassifier` or `MultivariateEVM` (line ~219). In `train_one_epoch_evm_ood`: one forward with `modalities=True`; `mevm_nll_loss` on the modality dict, `vim_ood_loss` on `fused`. At end of step (line ~525): fit on modality features. Open-set evaluation on modality features. `--lambda_ood 0` already turns it into MEVM without OOD. |
| `src/class_incremental/eaml/class_incremental_evm_training.py`, `class_incremental_ievm_training.py` | Same `--evm_type` switch (and iEVM only if §3.6 is done). |
| `src/class_incremental/llmv3/llmv3_class_incremental_evm_ood.py`, `llmv3_class_incremental_evm_training.py` | Same. In `utils/llmv3/llmv3_il_common.py`: `add_il_args(evm=True)` gains `--evm_type` / `--mevm_*`; `fit_evm` and `evm_open_set_eval` call `features_by_class(..., modalities=(evm_type=="mevm"))`. |
| `src/domain_incremental/eaml/domain_incremental_evm_ood.py`, `domain_incremental_evm_training.py` | Same switch; `utils/domain_IL/dil_train_utils.py::train_one_epoch_dil_evm_ood` gets the single-forward modality features. |
| `src/domain_incremental/llmv3/llmv3_domain_incremental_evm_ood.py`, `..._evm_training.py` | Same, through `llmv3_il_common`. |
| `utils/il_checks.py::warn_inactive_terms` | Also warn for `L_MEVM` when only the classifier is trainable. |
| `run_log.init(...)` calls | Method name becomes `"MEVM"` / `"MEVM+OOD"` when `--evm_type mevm`. |

---

## 5. Distillation (KD) for LayoutLMv3

- **The code path already exists:** `add_il_args(strategy=...)`, `make_teacher`, `distill_term` in `llmv3_il_common.py`.
- **Change `make_teacher`:** `teacher.half()`, and run the teacher forward under `torch.no_grad()` + `torch.autocast("cuda", dtype=torch.float16)`. V100 has no bf16, so fp16 is the option. Run one step and record peak memory with `torch.cuda.max_memory_allocated()` to check that it fits.
- **Fallback if it still does not fit:** new `tools/eval/teacher_logits.py` caches the teacher logits per sample id for each step. This is valid for LayoutLMv3 because its inputs are pre-tokenised tensors with no random augmentation. It is **not** valid for EAML, which uses flip/shear augmentation; EAML keeps its online teacher.
- **New launchers:** `scripts/class_incremental/llmv3/run_llmv3_classIL*_KD.sh` and the DIL equivalents, i.e. existing launchers with `--strategy distillation`.

---

## 6. Baselines

### 6.1 Lower and upper bounds (mostly launcher work)
- **EAML lower bound:** `class_incremental.py` run without `--use_exemplars`, `--use_ewc` or `--use_bias_correction`, since these are all opt-in. No code change.
- **LayoutLMv3 lower bound:** check whether exemplars can be switched off. `llmv3_class_incremental_*.py` always builds `ExemplarHandler`; add `--no_exemplars` if it cannot.
- **Upper bound:** `--joint_training` already exists for EAML and LayoutLMv3 CIL. Add it to DIL (train on RVL-CDIP subset + Tobacco jointly) only if the DIL upper bound is wanted.

### 6.2 DER++
- **New `utils/class_IL/der_buffer.py`:**
  - The buffer stores `(sample, label, logits)`. Logits are recorded with the end-of-step model when exemplars are selected, which fits the current herding at the end of each step.
  - `sample(batch_size)` returns a replay batch.
  - `loss(model) = α · MSE(z[:, :k], z_stored) + β · CE(z, y_stored)`, where only the first `k` columns are compared because the classifier grows.
- **Training-loop change:** DER++ draws a **separate replay batch every iteration**. The current loops concatenate exemplars into the training dataset (`ConcatDataset`), so DER++ needs its own `train_one_epoch_derpp`. Put it in a new script per backbone and setting (`class_incremental_derpp.py`, `llmv3_class_incremental_derpp.py`, and DIL versions if wanted), built from `class_incremental.py` / `llmv3_class_incremental.py`.
- **Arguments:** `--der_alpha`, `--der_beta`, `--buffer_size` (same memory budget as ER for fairness).

### 6.3 Frozen-backbone NCM
- **New `src/evaluation/ncm_frozen.py`:** load the base checkpoint, cache features for each step's training data (§9), keep class means, and classify test samples by nearest mean. No training.
- **Optional:** report NCM-on-exemplars for every IL run as well, computed from cached step features.

---

## 7. Metrics and per-step evaluation

- **New `utils/metrics/il_metrics.py`:**
  - `accuracy_matrix(step_eval_rows)` → `A[t, k]`;
  - `avg_incremental_acc(A)`, `forgetting(A)`, `bwt(A)`, `gil(acc, ref)`;
  - `grouped_class_acc(per_class, groups)` for the base/new and overlapping/non-overlapping/new breakdowns.
- **New `utils/metrics/stats.py`:** `bootstrap_ci(y_true, y_pred, n=1000)`, `mcnemar(pred_a, pred_b, y_true)`, `mean_std(values)`.
- **End of each step, in every script:** evaluate per-class accuracy on the test set of **all classes seen so far**, then call `run_log.log("step_eval", ...)` and `save_predictions(...)`. Most scripts only report a final test result plus `step_test` at epoch level, which is not enough to compute forgetting.

---

## 8. Data and split manifests

- **`tools/data/data_subset.py`:** today `create_subset` copies a fixed `subset_size_per_class`. Add:
  - a `--fraction` mode (e.g. 0.5);
  - `--manifest_out`, which writes a JSON of file ids per class and split **instead of copying images**.
- **Loaders read the manifest:**
  - EAML `get_class_il_loader`;
  - LayoutLMv3 `get_incremental_dataloader`, which currently selects via `images_per_class` + `seed`;
  - `DILDataLoader`.

  Each gets `--split_manifest` and keeps only the listed samples, so both backbones see **exactly** the same documents.
- **New `scripts/base_models/run_eamlmodel_50pct.sh`:** retrains the EAML base model with the manifest.
- **Class orders** stay as `--unseen_classes` strings. The protocol file lists the orders and the launcher picks one, so no loader change is needed.

---

## 9. Post-hoc pipeline (needed for the go/no-go check and the sweeps)

- **New `tools/eval/extract_step_features.py`:** for a run directory, load each step's best checkpoint and write `features_step{t}.npz` containing, per split, `labels`, `logits` and features for every modality (via §2.3).
- **New `tools/eval/posthoc_evt.py`:**
  - fit EVM and MEVM (and variants) on cached training features;
  - evaluate closed-set accuracy, AUROC/FPR95/OSCR with future classes as unknowns;
  - grid over `tailsize × cover_threshold × max_fit_samples × distance × θ-mode`;
  - write one CSV.
- **New `tools/eval/evt_drift_analysis.py`:**
  - θ per class over steps;
  - per-modality λ_ij distributions over steps;
  - linear CKA between step-t and base features per modality;
  - plots.
- **New `tools/eval/aggregate_seeds.py`:** read `il_results.csv` and the prediction files; group by configuration without the seed; compute mean ± std, bootstrap confidence intervals and McNemar tests; produce paper-ready Markdown/LaTeX tables.
- **New `tools/eval/plot_sensitivity.py`.**
- **Launchers:** `scripts/journal/run_feature_extraction.sh`, `run_posthoc_evt.sh`, and `submit_matrix.sh`, a Slurm array over backbone × method × seed × strategy that reads `journal_protocol.json`.

---

## 10. Tests

- **New `tests/test_mevm.py`:**
  - with one modality, MEVM gives the same result as `EVMClassifier`;
  - θ = 1 gives the product of the marginal Ψ;
  - θ = 1e3 gives approximately the minimum;
  - `mevm_nll_loss` gradients are finite;
  - the estimated θ is ≥ 1;
  - extreme vectors stay aligned across modalities after pruning.
- **New `tests/test_il_metrics.py`:** forgetting, BWT and the open-set metrics on hand-made matrices and scores.
- **Extend `tests/fixtures.py` and `tests/runner.py`** with MEVM, DER++ and KD-LayoutLMv3 smoke runs (1 epoch, tiny data), following the existing `tests/test_eaml_cil.py` / `test_layout_cil.py` pattern.

---

## 11. Small fixes found while planning

These affect correctness or fairness, so fix them alongside everything else:

- **Unused duplicate EVM:** `utils/domain_IL/evm_classifier.py` (`EVMDomainClassifier`) is not imported anywhere. Move it to `archive/` so nobody fits the wrong EVM.
- **`EVMClassifier.fit_cuda` ignores `distance_metric`** (§3.1).
- **EVM settings hard-coded or hidden:** `cover_threshold=0.7` and `max_fit_samples=50` (§3.1).
- **Double forward pass with different dropout** in the EVM/OOD training loops (§2.1).
- **Table 3 anomaly:** confirm the training scope and the per-step checkpoint loading in the LayoutLMv3 CIL scripts before reusing old results (plan item 1).

---

## Implementation order (dependencies only)

1. §0 decision → §1 seeds, protocol, logging, prediction saving
2. §2 feature interface
3. §3.1–3.5 EVT code + §10 MEVM tests
4. §9 feature extraction + `posthoc_evt.py` → **go/no-go on MEVM**
5. §4 script wiring, §5 KD for LayoutLMv3, §7 per-step evaluation
6. §6 baselines, §8 manifests and EAML 50 % retrain
7. §9 aggregation, drift analysis, sensitivity plots
8. §3.6 optional variants
