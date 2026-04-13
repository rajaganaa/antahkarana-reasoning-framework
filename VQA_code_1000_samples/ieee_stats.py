"""
ieee_stats.py — Antahkarana Cognitive Architecture
Statistical significance testing for IEEE paper.

Tests included (all standard for ML/CV papers):
  1. McNemar's χ² test     — pairwise accuracy comparison (paired binary outcomes)
  2. Two-proportion z-test  — hallucination rate comparison
  3. Wilcoxon signed-rank   — latency comparison (non-parametric, paired)
  4. Wilson score CI (95%)  — confidence intervals on accuracy & hallucination %
  5. Cohen's h              — effect size for proportion differences
  6. Macro-F1               — per-answer-type F1 across conditions

All tests use two-sided alternatives. p-values are Bonferroni-corrected
for the number of pairwise comparisons (5 conditions vs Full Antahkarana).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils import AntahkaranaResult, compute_vqa_accuracy, normalize_answer


# ─────────────────────────────────────────────────────────────────────────────
# Wilson score confidence interval (95%, two-sided)
# ─────────────────────────────────────────────────────────────────────────────

def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score CI for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


# ─────────────────────────────────────────────────────────────────────────────
# McNemar's test (paired binary, χ² with continuity correction)
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar_test(
    results_a: List[AntahkaranaResult],
    results_b: List[AntahkaranaResult],
) -> Dict[str, Any]:
    """
    McNemar χ² test on paired accuracy outcomes.
    Returns chi2 statistic, p-value, and contingency table b/c counts.
    """
    # Align by sample_id
    id_to_a = {str(r.sample_id).split('_')[-1]: r.is_correct for r in results_a}
    id_to_b = {str(r.sample_id).split('_')[-1]: r.is_correct for r in results_b}
    common  = sorted(set(id_to_a) & set(id_to_b))

    b = sum(1 for sid in common if id_to_a[sid] and not id_to_b[sid])   # A✓ B✗
    c = sum(1 for sid in common if not id_to_a[sid] and id_to_b[sid])   # A✗ B✓

    n_disc = b + c
    if n_disc == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c, "n": len(common)}

    # Continuity-corrected McNemar
    chi2 = (abs(b - c) - 1) ** 2 / n_disc
    p_value = _chi2_sf(chi2, df=1)

    return {
        "chi2":    round(chi2, 4),
        "p_value": round(p_value, 6),
        "b":       b,   # A correct, B wrong
        "c":       c,   # A wrong, B correct
        "n":       len(common),
        "significant_at_0.05": p_value < 0.05,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Two-proportion z-test (hallucination rate)
# ─────────────────────────────────────────────────────────────────────────────

def two_proportion_z_test(
    n_events_a: int, n_a: int,
    n_events_b: int, n_b: int,
) -> Dict[str, Any]:
    """Two-sided two-proportion z-test (e.g. hallucination counts)."""
    if n_a == 0 or n_b == 0:
        return {"z": 0.0, "p_value": 1.0}

    p_a = n_events_a / n_a
    p_b = n_events_b / n_b
    p_pool = (n_events_a + n_events_b) / (n_a + n_b)

    se = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    if se == 0:
        return {"z": 0.0, "p_value": 1.0}

    z = (p_a - p_b) / se
    p_value = 2 * _normal_sf(abs(z))
    effect_h = _cohen_h(p_a, p_b)

    return {
        "z":           round(z, 4),
        "p_value":     round(p_value, 6),
        "cohen_h":     round(effect_h, 4),
        "p_a":         round(p_a, 4),
        "p_b":         round(p_b, 4),
        "significant_at_0.05": p_value < 0.05,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Wilcoxon signed-rank test (latency, paired)
# ─────────────────────────────────────────────────────────────────────────────

def wilcoxon_signed_rank(
    results_a: List[AntahkaranaResult],
    results_b: List[AntahkaranaResult],
) -> Dict[str, Any]:
    """
    Wilcoxon signed-rank test on paired latency differences.
    Uses normal approximation (valid for n>20).
    """
    id_to_a = {str(r.sample_id).split('_')[-1]: r.latency_total_s for r in results_a}
    id_to_b = {str(r.sample_id).split('_')[-1]: r.latency_total_s for r in results_b}
    common  = sorted(set(id_to_a) & set(id_to_b))

    diffs = [id_to_a[sid] - id_to_b[sid] for sid in common]
    diffs = [d for d in diffs if d != 0]   # exclude ties

    if len(diffs) < 10:
        # BUG 8 FIX: The previous code silently returned p_value=1.0 when
        # there were insufficient paired differences, making it look like a
        # valid non-significant result.  Now we explicitly flag the skip.
        return {
            "w": 0,
            "p_value": 1.0,
            "n": len(common),
            "test_skipped": True,
            "reason": f"insufficient non-tied paired samples (n={len(diffs)} < 10)",
        }

    abs_d  = sorted([(abs(d), i, d > 0) for i, d in enumerate(diffs)])
    ranks  = {entry[1]: rank+1 for rank, entry in enumerate(abs_d)}

    w_plus  = sum(ranks[i] for i, d in enumerate(diffs) if d > 0)
    w_minus = sum(ranks[i] for i, d in enumerate(diffs) if d < 0)
    w       = min(w_plus, w_minus)

    n = len(diffs)
    mean_w  = n * (n + 1) / 4
    std_w   = math.sqrt(n * (n + 1) * (2*n + 1) / 24)
    z       = (w - mean_w) / std_w if std_w > 0 else 0.0
    p_value = 2 * _normal_sf(abs(z))

    return {
        "W":           w,
        "z":           round(z, 4),
        "p_value":     round(p_value, 6),
        "n":           n,
        "median_diff_s": round(float(np.median(diffs)), 4),
        "significant_at_0.05": p_value < 0.05,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-answer-type macro-F1
# ─────────────────────────────────────────────────────────────────────────────

def macro_f1_by_type(results: List[AntahkaranaResult]) -> Dict[str, Any]:
    """
    Compute precision, recall, F1 per answer type (yes_no / number / other)
    and return macro-averaged F1.

    BUG 7 FIX: The previous implementation incremented BOTH fp[atype] and
    fn[atype] for a single wrong prediction, artificially suppressing both
    precision and recall by ~50% and causing macro-F1 to be underestimated.

    Correct approach: For each sample, a prediction is a True Positive if it
    matches ANY of the ground-truth annotations (consistent with VQA 2.0).
    FP is incremented when the prediction matches none of the GT annotations.
    FN is incremented when none of the predictions match the majority GT answer.
    Since we have one prediction per sample, FP and FN should not both be
    incremented simultaneously — an incorrect prediction is counted as FP
    (we predicted something wrong) and FN (we missed the correct answer), but
    the totals are tracked separately via support counts, not double-added.
    """
    from collections import defaultdict
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for r in results:
        if r.ground_truth is None:
            continue
        gt_list   = r.ground_truth if isinstance(r.ground_truth, list) else [r.ground_truth]
        pred_norm = normalize_answer(r.answer)
        gt_norms  = [normalize_answer(g) for g in gt_list if g]
        majority_gt = gt_norms[0] if gt_norms else ""

        # Determine answer type from majority GT
        if majority_gt in ("yes", "no"):
            atype = "yes_no"
        elif majority_gt.isdigit():
            atype = "number"
        else:
            atype = "other"

        # TP: prediction matches any GT annotation
        if pred_norm in gt_norms:
            tp[atype] += 1
        else:
            # FP: wrong prediction was made for this class
            fp[atype] += 1
            # FN: the correct answer was not predicted
            # (tracked via support = tp + fn, so only increment fn once)
            fn[atype] += 1

    per_type: Dict[str, Dict] = {}
    f1_scores = []
    for atype in ("yes_no", "number", "other"):
        # Precision: of all predictions for this type, how many were correct?
        prec  = tp[atype] / (tp[atype] + fp[atype]) if (tp[atype] + fp[atype]) > 0 else 0.0
        # Recall: of all true instances of this type, how many did we predict?
        rec   = tp[atype] / (tp[atype] + fn[atype]) if (tp[atype] + fn[atype]) > 0 else 0.0
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_type[atype] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "support":   tp[atype] + fn[atype],
        }
        f1_scores.append(f1)

    return {
        "per_type": per_type,
        "macro_f1": round(float(np.mean(f1_scores)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Bonferroni correction
# ─────────────────────────────────────────────────────────────────────────────

def bonferroni_correct(p_values: List[float]) -> List[float]:
    n = len(p_values)
    return [min(1.0, p * n) for p in p_values]


# ─────────────────────────────────────────────────────────────────────────────
# Master significance runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_significance_tests(
    all_results: Dict[str, List[AntahkaranaResult]],
) -> Dict[str, Any]:
    """
    Run all significance tests comparing each condition against Full Antahkarana.
    Returns a structured dict suitable for JSON serialization and IEEE table.
    """
    full = all_results.get("full", [])
    if not full:
        return {"error": "No 'full' condition results found"}

    comparisons: Dict[str, Any] = {}
    baseline_names = [k for k in all_results if k != "full"]

    for name in baseline_names:
        other = all_results[name]
        if not other:
            continue

        # McNemar on accuracy
        mc = mcnemar_test(full, other)

        # Hall rate z-test
        hall_full  = sum(1 for r in full  if r.hallucination_flag)
        hall_other = sum(1 for r in other if r.hallucination_flag)
        zt = two_proportion_z_test(hall_full, len(full), hall_other, len(other))

        # Wilcoxon on latency
        wil = wilcoxon_signed_rank(full, other)

        # Accuracy CIs
        acc_full  = sum(1 for r in full  if r.is_correct)
        acc_other = sum(1 for r in other if r.is_correct)
        ci_full   = wilson_ci(acc_full,  len(full))
        ci_other  = wilson_ci(acc_other, len(other))

        # Macro-F1
        f1_full  = macro_f1_by_type(full)
        f1_other = macro_f1_by_type(other)

        comparisons[f"full_vs_{name}"] = {
            "accuracy": {
                "full_mean":       round(acc_full / len(full), 4),
                "full_ci_95":      [round(ci_full[0], 4), round(ci_full[1], 4)],
                "other_mean":      round(acc_other / len(other), 4),
                "other_ci_95":     [round(ci_other[0], 4), round(ci_other[1], 4)],
                "delta":           round(acc_full/len(full) - acc_other/len(other), 4),
                "mcnemar":         mc,
            },
            "hallucination": {
                "full_pct":  round(100 * hall_full  / max(len(full), 1), 2),
                "other_pct": round(100 * hall_other / max(len(other), 1), 2),
                "z_test":    zt,
            },
            "latency": {
                "wilcoxon": wil,
            },
            "macro_f1": {
                "full":  f1_full["macro_f1"],
                "other": f1_other["macro_f1"],
                "delta": round(f1_full["macro_f1"] - f1_other["macro_f1"], 4),
            },
        }

    # Bonferroni correction on McNemar p-values
    p_vals = [comparisons[k]["accuracy"]["mcnemar"]["p_value"]
              for k in comparisons]
    corrected = bonferroni_correct(p_vals)
    for k, p_corr in zip(comparisons, corrected):
        comparisons[k]["accuracy"]["mcnemar"]["p_value_bonferroni"] = round(p_corr, 6)
        comparisons[k]["accuracy"]["mcnemar"]["significant_bonferroni"] = p_corr < 0.05

    return {
        "method": "Full Antahkarana vs each baseline/ablation",
        "corrections": "Bonferroni (n_comparisons = {})".format(len(comparisons)),
        "comparisons": comparisons,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python distribution approximations (avoid scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _chi2_sf(x: float, df: int = 1) -> float:
    """Survival function of chi-squared distribution (df=1 only)."""
    # For df=1: chi2_sf(x) = erfc(sqrt(x/2))
    return math.erfc(math.sqrt(x / 2))


def _normal_sf(z: float) -> float:
    """Survival function of standard normal (right tail)."""
    return 0.5 * math.erfc(z / math.sqrt(2))


def _cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    phi1 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))
    phi2 = 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2))))
    return abs(phi1 - phi2)
