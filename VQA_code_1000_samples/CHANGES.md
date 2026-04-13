# Antahkarana — IEEE Bug Fixes & Changes

**Version:** v3-fixed (April 2026)  
**Reviewer:** IEEE Senior Evaluator  
**Status:** All 10 identified issues resolved

---

## Critical Fixes (would cause desk rejection)

### Bug 1 — Synthetic dataset rigged to produce desired accuracy ordering
**File:** `dataset.py` · `_CONDITION_MIX`  
**Problem:** Each experimental condition received a *different* proportion of easy/medium/hard questions. "Full Antahkarana" got 65% easy questions; baselines got 50–60%. This made the dataset — not the pipeline — responsible for the observed accuracy ordering.  
**Fix:** All conditions now receive **identical** difficulty distributions `(0.60, 0.25, 0.15)`. Accuracy differences arise solely from pipeline behaviour, not question selection.

```python
# BEFORE (rigged)
"full":         (0.65, 0.22, 0.13),   # easier than every baseline
"cot_baseline": (0.50, 0.30, 0.20),   # harder than full

# AFTER (fair)
"full":         (0.60, 0.25, 0.15),   # same as every other condition
"cot_baseline": (0.60, 0.25, 0.15),
```

### Bug 2 — Self-Consistency outperformed Full Antahkarana (unaddressed anomaly)
**File:** `main.py` · `run_ablation_study()`  
**Problem:** The ablation table showed Self-Consistency (0.760) > Full Antahkarana (0.743). This directly undermines the paper's core claim, yet it was published in the results table without any discussion.  
**Fix:** After running the ablation study, the code now **automatically detects this anomaly**, prints a detailed explanation of possible causes, and writes `anomaly_self_consistency.json` to the output directory. The paper must discuss this finding — it is not silently ignored.

---

## Major Fixes (would cause rejection at review stage)

### Bug 3 — CoT hallucination_pct always 0% (stated fix didn't work)
**File:** `utils.py` · `detect_hallucination()`  
**Problem:** The hallucination detection threshold was `conf_threshold=0.67`. Beam scores from BLIP-2 single-pass inference typically return values in the 0.50–0.65 range, so even confidently-wrong answers were never flagged.  
**Fix:** Lowered default threshold to `0.50`. Any confident wrong prediction (beam confidence > 50%) is now correctly flagged. The 3-pass full pipeline uses `answer_confidence_from_agreement()` which returns 0.67 or 1.00, so multi-pass hallucinations are still caught.

### Bug 4 — `seed_everything()` inside `load_vqav2()` corrupted RNG state
**File:** `dataset.py` · `load_vqav2()`  
**Two sub-issues fixed:**  
1. `seed_everything(cfg.seed)` was called at the start of every `load_vqav2()` invocation, resetting the global random state to the *same* seed for every condition. This created artificial correlation between condition samples.  
2. `DataLoader` with `num_workers > 1` uses multiprocessing — seeds were not propagated to worker processes, so multi-worker runs produced non-reproducible results.  
**Fix:** Removed the internal `seed_everything()` call. Added `worker_init_fn` to `DataLoader` that propagates `cfg.seed + worker_id` to each worker process.

### Bug 5 — Batch inference returned hardcoded confidence = 0.5 for all samples
**File:** `model.py` · `_batch_forward()`  
**Problem:** The batch inference path did not request generation scores and returned `(answer, 0.5)` for every sample. Since the hallucination threshold is 0.50 (after Bug 3 fix), this would still mean batch-processed samples sit exactly at the boundary. More importantly, the code did not use the correct per-sample confidence at all.  
**Fix:** Added `output_scores=True, return_dict_in_generate=True` to the batch `generate()` call. Added `_beam_confidence_for_sample(outputs, sample_idx)` to extract per-sample confidence from `outputs.scores` tensor.

### Bug 6 — `exact_match_accuracy()` compared against first GT answer only
**File:** `evaluator.py` · `exact_match_accuracy()`  
**Problem:** VQA 2.0 provides up to 10 annotator answers per sample. The function compared only against `ground_truth[0]`, penalising synonymous correct answers (e.g. "grey" vs "gray") and producing results inconsistent with the VQA 2.0 evaluation protocol.  
**Fix:** Prediction is correct if it matches **any** of the normalised ground-truth annotations.

```python
# BEFORE
normalize_answer(r.ground_truth[0] if r.ground_truth else "")

# AFTER
any(pred_norm == normalize_answer(gt) for gt in gt_list if gt)
```

---

## Minor Fixes (would be flagged during review)

### Bug 7 — `macro_f1_by_type()` double-counted FP and FN per error
**File:** `ieee_stats.py` · `macro_f1_by_type()`  
**Problem:** For a wrong prediction, both `fp[atype] += 1` and `fn[atype] += 1` were incremented simultaneously. This artificially suppressed both precision and recall by ~50%, causing macro-F1 to be severely underestimated.  
**Fix:** Separated TP/FP/FN accounting. Predictions are matched against **any** GT annotation (consistent with VQA 2.0). FP is incremented when the prediction matches none of the GT; FN is incremented when the correct answer was not predicted.

### Bug 8 — Wilcoxon test silent fallback returned fake `p_value=1.0`
**File:** `ieee_stats.py` · `wilcoxon_signed_rank()`  
**Problem:** When fewer than 10 paired differences were available, the function returned `{"w": 0, "p_value": 1.0}` with no indication the test was skipped. The significance report showed `p=1.0` as if the test ran and found no significant difference.  
**Fix:** Now returns `{"test_skipped": True, "reason": "insufficient non-tied paired samples (n=N < 10)"}` alongside the placeholder values.

---

## Design Fixes (architecture / engineering quality)

### Bug 9 — 6 separate BLIP-2 model loads in ablation study (OOM risk)
**File:** `main.py` · `run_ablation_study()`  
**Problem:** The ablation study created a new `AntahkaranaVLM` object (i.e. loaded BLIP-2 from disk) for each of the 6 conditions. On a 16GB GPU this caused OOM errors. The `del pipe; gc.collect()` pattern does not guarantee immediate VRAM release, so measured latency for later conditions was contaminated by cached state from earlier ones.  
**Fix:** BLIP-2 is loaded **once** into a `shared_vlm` instance before the ablation loop. All 6 pipeline constructors now accept a `shared_vlm` parameter and reuse it. This is both VRAM-efficient and ensures all conditions run on the same warmed-up model state, making latency comparisons fair.

### Bug 10 — Image-noise self-consistency is non-standard and semantically risky
**File:** `dataset.py` · `make_self_consistency_sample()`, `main.py` · `SelfConsistencyPipeline`  
**Problem:** Gaussian pixel noise (±8–17 intensity) was added to images to break BLIP-2's beam-search determinism. This is not the standard self-consistency method (Wang et al., 2022). For low-contrast colour boundaries (e.g. teal vs cyan), the noise could genuinely alter the correct answer.  
**Fix:** `make_self_consistency_sample()` now returns the original sample unchanged. Diversity across passes is obtained via `do_sample=True, temperature=0.7` in the `generate()` call, which is both the established approach and is deterministic given a fixed seed.

---

## Additional IEEE-Required Improvements

### Wilson 95% Confidence Intervals on all accuracy numbers
Added to `evaluate_experiment()`. All accuracy metrics are now reported as `0.743 ± 0.044 (95% CI)`. The `ieee_stats.wilson_ci()` function was already correctly implemented; it is now surfaced in all metric outputs.

### `requirements.txt` with pinned dependency versions
Added `requirements.txt` with exact package versions. BLIP-2 results are sensitive to `transformers` version; without pinning, results cannot be reproduced.

---

## Summary Table

| # | Severity | File | Fix |
|---|----------|------|-----|
| 1 | **Critical** | `dataset.py` | Uniform difficulty mix across all conditions |
| 2 | **Critical** | `main.py` | Anomaly detection + mandatory discussion flag |
| 3 | **Major** | `utils.py` | Hallucination threshold lowered 0.67 → 0.50 |
| 4 | **Major** | `dataset.py` | Removed internal seed; added `worker_init_fn` |
| 5 | **Major** | `model.py` | Per-sample beam confidence in batch mode |
| 6 | **Major** | `evaluator.py` | Exact match checks all GT annotations |
| 7 | **Minor** | `ieee_stats.py` | Fixed FP/FN double-counting in macro-F1 |
| 8 | **Minor** | `ieee_stats.py` | Wilcoxon skipped flag instead of fake p=1.0 |
| 9 | **Design** | `main.py` | Shared VLM across ablation (single model load) |
| 10 | **Design** | `dataset.py`, `model.py`, `main.py` | Temperature sampling replaces image noise |
