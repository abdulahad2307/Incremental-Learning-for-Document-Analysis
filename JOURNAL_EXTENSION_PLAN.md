# Journal Extension Plan

Plan for extending the ICDAR workshop paper *"Incremental Document Image Classification: A Comparative Study of Content and Layout Representations"* into a journal paper. This is a plan only: nothing here has been run yet.

**Aims**
1. Address the technical points raised in the workshop reviews.
2. Add one real methodological contribution: a **Multivariate Extreme Value Machine (MEVM)**, which follows the "Advanced extreme value theory" item in the thesis future scope.

---

## 0. How the reviewer points are covered

| Review point | Covered by |
|---|---|
| R1: no per-method hyperparameter sweeps | Items 18–19 |
| R1: limited to classical IL and EVM-family methods | Items 15–16 (DER++, frozen-backbone NCM), 6 (upper/lower bounds) |
| R2: single runs, no statistical analysis | Items 4, 7 |
| R2: backbones set up differently | Items 2, 3 |

---

## A. Protocol and credibility

1. **Check the LayoutLMv3 CIL anomaly.** In Table 3 (step 2), every method reports exactly 60.87 % test accuracy with identical G_B and G_P scores. Things to check:
   - which `--training_mode` the runs used;
   - whether `L_EVM` / `L_OOD` could reach any trainable parameter (`utils/il_checks.py` warns when it cannot);
   - whether the step-2 evaluation loaded the step-2 checkpoint.

   Fix this before any LayoutLMv3 result is reused.
   - **Decide which "LayoutLMv3" the paper uses.** The model in `utils/llmv3/llmv3_model_loader.py` is a custom BERT + ViT + 2-layer fusion model with a `Linear(2→768)` box-centre embedding, not the pretrained `microsoft/layoutlmv3-base`. Either switch to the real model or rename it in the paper. See section 0 of [JOURNAL_CODE_CHANGES.md](JOURNAL_CODE_CHANGES.md).
2. **One shared protocol for both backbones.** Use the same:
   - data subset (50 % RVL-CDIP split, saved as file lists) and class order;
   - `training_mode`, epoch cap, patience, optimizer and learning-rate schedule;
   - exemplar budget and selection method;
   - EVM settings (`tailsize`, `cover_threshold`, `max_fit_samples`) and λ values.

   Keep all of these in one config file that every CIL/DIL script reads.
3. **Remove the backbone asymmetry.**
   - Retrain the EAML base model on the same 50 % split as LayoutLMv3. Keep the full-data EAML only as a supplementary row.
   - Enable distillation-based IL (KD) for LayoutLMv3. Run the frozen teacher in fp16 under `torch.no_grad()`; if that still does not fit in memory, precompute the teacher logits offline for each step.
4. **Seeds.** Add `--seed` to every EAML script (only the LayoutLMv3 scripts have it now), set deterministic flags, and record the seed in `run_log`. Target: 3 seeds for the main methods.
5. **Metrics.** Keep accuracy and G_IL, and add:
   - average incremental accuracy, average forgetting, and backward transfer (BWT);
   - per-class accuracy, reported separately for base, new, overlapping and non-overlapping classes;
   - **open-set metrics** (AUROC, FPR95, open-set classification rate / OSCR), using classes not yet seen as the unknowns at each step. The paper motivates EVM with open-set recognition but has never measured it.
6. **Bounds.**
   - Lower bound: fine-tuning with no exemplars, EWC or bias correction.
   - Upper bound: joint training on all classes seen so far.
7. **Statistics.**
   - Report mean ± std over seeds.
   - Give bootstrap 95 % confidence intervals on the test sets.
   - Use McNemar tests for key pairwise comparisons, e.g. MEVM+OOD vs EVM+OOD.

---

## B. Main contribution: Multivariate EVM (MEVM)

**Idea.** Classic EVM fits a single Weibull to one scalar margin in the fused feature space. MEVM does this instead:

- Fit one Weibull per **modality**, per extreme vector.
- Join the per-modality Weibulls with an extreme-value (Gumbel / logistic) copula.

```
modalities j = 1..m
  EAML       : image, text, fused
  LayoutLMv3 : pooled text tokens, pooled visual patches, [CLS]

per extreme vector i and modality j:
  Weibull (κ_ij, λ_ij) fitted on half-distances to the nearest negatives in modality j
  t_ij(x) = ( d_j(x_i, x) / λ_ij ) ^ κ_ij          # per-modality EVM energy

joint inclusion (Gumbel copula, θ ≥ 1):
  Ψ_i(x)      = exp( −( Σ_j t_ij^θ )^(1/θ) )
  −log Ψ_i(x) = ‖ t_i ‖_θ                          # ℓθ-norm of the energies

class score:  P(c | x) = max_{i ∈ c} Ψ_i(x)
```

Properties (each one becomes a unit test):
- with m = 1, MEVM is exactly classic EVM;
- with θ = 1, the joint inclusion is the product of the modality inclusions (every modality must agree);
- as θ → ∞, it becomes the minimum over modalities (modalities fully dependent);
- θ is estimated per class from data, not tuned: θ = 1/(1 − τ), where τ is Kendall's τ between the modality margins (mean pairwise τ when m > 2), clipped to θ ≥ 1;
- it stays differentiable, so it can replace `L_EVM` directly.

Work items:

8. **Per-modality features.** Give both models a single `forward_features(...)` that returns `{image, text, fused}` (EAML) or `{text, visual, cls}` (LayoutLMv3).
9. **`MultivariateEVM` class.**
   - Per-modality Weibull fitting, reusing the tail logic of `EVMClassifier`.
   - θ estimation.
   - Joint inclusion, prediction and open-set scoring.
   - The same `fit` / `predict` / `weibull_models`-style interface, so the existing evaluation code works unchanged.
10. **MEVM loss.** Add `mevm_nll_loss` next to `evm_nll_loss`: compute the per-modality energies in log space, combine them with the ℓθ-norm, then take the minimum over extreme vectors.
11. **Post-hoc go/no-go check.** Using cached features from the **existing checkpoints**, compare EVM against MEVM on closed-set accuracy and open-set AUROC. This costs no GPU training and decides how much weight MEVM gets in the paper.
12. **Training integration.** Add MEVM and MEVM+OOD as CIL and DIL methods for both backbones, under standard IL and KD.
13. **Drift analysis.** Track θ, the per-modality scales λ_ij, and CKA to the base features across incremental steps. The goal is to show *which modality's margin collapses* when forgetting happens, which would explain the content-vs-layout gap rather than only reporting it.
14. **Optional variants**, only if the core MEVM works:
   - **Weighted logistic:** `(Σ_j (w_j t_ij)^θ)^(1/θ)`, e.g. to down-weight noisy OCR text.
   - **ViM residual as an extra dimension:** merges EVM+OOD into one EVT model instead of two summed losses.
   - **Incremental MEVM:** iEVM's partial-update rule applied to each marginal, with θ re-estimated only for classes that changed.
   - **GPD tail ablation:** a peaks-over-threshold / GPD tail in place of the Weibull.

---

## C. Baselines

15. **DER++.** Store logits together with the exemplars and add the replay terms for logit matching and labels.
16. **Frozen-backbone NCM / prototype classifier.** Freeze the base model and update only the class means. This is cheap on cached features and is a strong reference point.
17. **Method set for the journal version.**
   - Keep: fine-tuning, ER+EWC+BC, EVM, iEVM, EVM+OOD, MEVM, MEVM+OOD, DER++, NCM, joint.
   - Drop RegEVM as a configuration. Mention its negative result from the workshop paper in one sentence.

---

## D. Hyperparameter sensitivity

18. **Post-hoc EVT sweep** on cached features (cheap):
    - `tailsize`
    - `cover_threshold`
    - `max_fit_samples` (currently 50 per class, which is a confounder)
    - distance metric (Euclidean or cosine)
    - for MEVM: fixed θ against estimated θ
19. **Small training sweep** on EAML, 1 seed, steps 1–2:
    - λ_EVM ∈ {0.01, 0.1, 1}
    - λ_OOD ∈ {0.01, 0.1, 1}
    - exemplars per class ∈ {16, 64}

    Report the results as sensitivity plots.

---

## E. Cross-collection DIL (RVL-CDIP → Tobacco-3482)

20. Rerun DIL under the shared protocol with all main methods, including MEVM, for both backbones. Report the breakdown into overlapping, non-overlapping and new classes as the main view, not only the aggregate accuracy.

---

## F. Stretch items (only if time remains)

21. A second class order for CIL.
22. RVL-CDIP-O as real out-of-distribution documents for open-set evaluation.
23. A LoRA-adapter or prompt-based baseline on LayoutLMv3.

---

## G. Go / no-go decision for the framing

- **MEVM improves accuracy or open-set AUROC over EVM+OOD:** present the paper as a method paper (MEVM for multimodal incremental document classification) with the benchmark as supporting material.
- **MEVM does not improve on those, but the drift analysis (item 13) is informative:** present it as an analysis paper explaining *why* layout-aware models forget, with MEVM as the diagnostic tool.

---

## H. New and changed code (planned, not implemented)

The detailed, file-by-file change plan is in [JOURNAL_CODE_CHANGES.md](JOURNAL_CODE_CHANGES.md). It recommends an `--evm_type {evm, mevm}` flag on the existing EVM scripts instead of the separate `*_mevm.py` scripts listed below; with that flag only the launchers are new.

### New files

| Path | Purpose |
|---|---|
| `configs/journal_protocol.json` | Shared protocol: split files, class orders, seeds, training mode, epochs, exemplars, EVM/MEVM settings, λ values |
| `utils/seed.py` | `set_seed(seed)` plus deterministic flags |
| `utils/features/modality_features.py` | Per-modality feature extraction for EAML and LayoutLMv3; `features_by_class_and_modality(...)` |
| `utils/evm/copula.py` | Gumbel copula, Kendall-τ → θ estimation, ℓθ aggregation |
| `utils/evm/mevm_classifier.py` | `MultivariateEVM`: per-modality fitting, θ, joint Ψ, predict, open-set scores |
| `utils/evm/mevm_loss.py` | `mevm_nll_loss` (differentiable, computed in log space) |
| `utils/class_IL/der_buffer.py` | DER++ buffer (samples, labels, stored logits) and replay loss |
| `utils/metrics/il_metrics.py` | Average incremental accuracy, forgetting, BWT, G_IL, grouped per-class accuracy |
| `utils/metrics/openset_metrics.py` | AUROC, FPR95, OSCR with not-yet-seen classes as unknowns |
| `utils/metrics/stats.py` | Seed aggregation, bootstrap confidence intervals, McNemar test |
| `src/class_incremental/eaml/class_incremental_mevm.py` | EAML CIL with MEVM (`--use_ood` switches to MEVM+OOD; `--strategy` standard or KD) |
| `src/class_incremental/llmv3/llmv3_class_incremental_mevm.py` | The same for LayoutLMv3 |
| `src/class_incremental/eaml/class_incremental_derpp.py` | DER++ baseline for EAML |
| `src/class_incremental/llmv3/llmv3_class_incremental_derpp.py` | DER++ baseline for LayoutLMv3 |
| `src/domain_incremental/eaml/domain_incremental_mevm.py` | EAML DIL with MEVM / MEVM+OOD |
| `src/domain_incremental/llmv3/llmv3_domain_incremental_mevm.py` | LayoutLMv3 DIL with MEVM / MEVM+OOD |
| `src/evaluation/ncm_frozen.py` | Frozen-backbone NCM baseline on cached features (CIL and DIL) |
| `tools/eval/extract_step_features.py` | Cache per-modality features and logits for every step checkpoint |
| `tools/eval/posthoc_evt.py` | Fit EVM / MEVM / variants on cached features; go/no-go check (item 11) and EVT sweep (item 18) |
| `tools/eval/evt_drift_analysis.py` | θ, λ_ij and CKA across steps; plots |
| `tools/eval/aggregate_seeds.py` | Combine runs into mean ± std tables with confidence intervals and significance tests |
| `tools/eval/plot_sensitivity.py` | Hyperparameter sensitivity plots |
| `tools/eval/teacher_logits.py` | Optional offline teacher logits for LayoutLMv3 KD, if the fp16 teacher does not fit |
| `scripts/base_models/run_eamlmodel_50pct.sh` | Retrain the EAML base model on the shared 50 % split |
| `scripts/class_incremental/{eaml,llmv3}/run_*_mevm.sh`, `run_*_mevm_ood.sh`, `run_*_derpp.sh` | Slurm launchers (standard IL and KD) |
| `scripts/class_incremental/llmv3/run_llmv3_classIL_*_KD.sh` | LayoutLMv3 KD launchers, which do not exist yet |
| `scripts/domain_incremental/{eaml,llmv3}/run_*_mevm*.sh` | DIL launchers for MEVM |
| `scripts/journal/submit_matrix.sh` | Slurm array job over backbone × method × seed × strategy, reading `journal_protocol.json` |
| `scripts/journal/run_feature_extraction.sh`, `run_posthoc_evt.sh` | Launchers for the post-hoc pipeline |
| `tests/test_mevm.py` | m = 1 gives EVM; θ = 1 gives the product; large θ gives the minimum; gradients are finite; θ ≥ 1 |
| `tests/test_il_metrics.py` | Forgetting, BWT and open-set metrics on toy inputs |

### Files to change

| Path | Change |
|---|---|
| `utils/eaml/eaml_model.py` | `forward_features` returns `{image, text, fused}` |
| `utils/llmv3/llmv3_model_loader.py` | `forward_features` returns `{text, visual, cls}` |
| Existing CIL/DIL scripts in `src/` | `--seed`, load `journal_protocol.json`, log the new metrics; `--no_exemplars` / `--joint_training` flags for the bounds |
| `utils/llmv3/llmv3_il_common.py` | Hook up KD for LayoutLMv3 (fp16 teacher or cached logits); add MEVM to `fit_evm` / `evm_open_set_eval` |
| `tools/data/data_subset.py` | Write the fixed 50 % split as file lists for both backbones |
| `tools/eval/il_results_table.py` | New columns (forgetting, BWT, AUROC, mean ± std) |
