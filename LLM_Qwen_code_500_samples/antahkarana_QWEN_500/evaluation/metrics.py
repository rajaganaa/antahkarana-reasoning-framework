"""
evaluation/metrics.py — IEEE-standard metrics for Antahkarana evaluation.

Computes:
  - Exact Match (EM)
  - F1 Score (token-level)
  - Supporting Fact EM / F1 (HotpotQA)
  - Latency stats
  - Bootstrap CI
  - Paired t-test significance
"""

import re
import string
import math
import time
import logging
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Text normalization
# ─────────────────────────────────────────────────────────────────────────────

_ARTICLES = {"a", "an", "the"}
_PUNCT    = set(string.punctuation)


def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    # Remove articles
    s = " ".join(w for w in s.split() if w not in _ARTICLES)
    # Remove punctuation
    s = "".join(c if c not in _PUNCT else " " for c in s)
    # Collapse whitespace
    return " ".join(s.split())


# ─────────────────────────────────────────────────────────────────────────────
# Core EM / F1
# ─────────────────────────────────────────────────────────────────────────────

def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    common    = Counter(pred_toks) & Counter(gold_toks)
    num_same  = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall    = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def em_f1_pair(pred: str, gold: str) -> Tuple[float, float]:
    return exact_match(pred, gold), token_f1(pred, gold)


# ─────────────────────────────────────────────────────────────────────────────
# MMLU accuracy
# ─────────────────────────────────────────────────────────────────────────────

def mmlu_accuracy(pred: str, sample: Dict) -> float:
    """
    pred can be a letter (A/B/C/D) or the full choice text.
    Returns 1.0 if correct.
    """
    answer_idx  = sample.get("answer", -1)
    answer_text = sample.get("answer_text", "")
    choices     = sample.get("choices", [])
    labels      = ["A", "B", "C", "D"]

    pred_clean = pred.strip().upper()

    # Try letter match
    if pred_clean and pred_clean[0] in labels:
        pred_idx = labels.index(pred_clean[0])
        return float(pred_idx == answer_idx)

    # Try text match
    em = exact_match(pred, answer_text)
    if em:
        return 1.0

    # Try text match against all choices
    for idx, choice in enumerate(choices):
        if normalize_answer(pred) == normalize_answer(choice):
            return float(idx == answer_idx)

    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FEVER accuracy
# ─────────────────────────────────────────────────────────────────────────────

_FEVER_MAP = {
    "supports":        "SUPPORTS",
    "support":         "SUPPORTS",
    "true":            "SUPPORTS",
    "refutes":         "REFUTES",
    "refute":          "REFUTES",
    "false":           "REFUTES",
    "not enough info": "NOT ENOUGH INFO",
    "not enough information": "NOT ENOUGH INFO",
    "nei":             "NOT ENOUGH INFO",
    "uncertain":       "NOT ENOUGH INFO",
}


def fever_accuracy(pred: str, gold: str) -> float:
    pred_norm = _FEVER_MAP.get(pred.lower().strip(), pred.upper().strip())
    gold_norm = gold.upper().strip()
    return float(pred_norm == gold_norm)


# ─────────────────────────────────────────────────────────────────────────────
# SVAMP / math accuracy
# ─────────────────────────────────────────────────────────────────────────────

def math_accuracy(pred: str, gold: str, tolerance: float = 1e-3) -> float:
    def _parse(s: str) -> Optional[float]:
        nums = re.findall(r'-?\d+(?:\.\d+)?', s)
        if nums:
            try:
                return float(nums[-1])
            except ValueError:
                pass
        return None

    p = _parse(pred)
    g = _parse(gold)
    if p is None or g is None:
        return exact_match(pred, gold)
    if g == 0:
        return float(abs(p - g) < tolerance)
    return float(abs(p - g) / max(abs(g), 1e-9) < tolerance)


# ─────────────────────────────────────────────────────────────────────────────
# Supporting Fact metrics (HotpotQA)
# ─────────────────────────────────────────────────────────────────────────────

def supporting_fact_em_f1(
    pred_spans: List[Dict],
    gold_sf:    Dict,
) -> Tuple[float, float]:
    """
    pred_spans: list of {"title": ..., "sent_id": ...}
    gold_sf:    {"title": [...], "sent_id": [...]}
    """
    gold_pairs = set(
        zip(gold_sf.get("title", []), gold_sf.get("sent_id", []))
    )
    pred_pairs = set(
        (s.get("title", ""), s.get("sent_id", -1)) for s in pred_spans
    )

    if not gold_pairs:
        return 1.0, 1.0

    em = float(pred_pairs == gold_pairs)

    common = len(pred_pairs & gold_pairs)
    prec   = common / len(pred_pairs) if pred_pairs else 0.0
    rec    = common / len(gold_pairs) if gold_pairs else 0.0
    f1     = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    return em, f1


# ─────────────────────────────────────────────────────────────────────────────
# Score a single result dict
# ─────────────────────────────────────────────────────────────────────────────

def score_result(result: Dict, sample: Dict, dataset: str) -> Dict:
    pred = result.get("predicted", "")
    gold = result.get("gold", sample.get("answer", ""))

    scores: Dict[str, float] = {}

    if dataset == "mmlu":
        acc = mmlu_accuracy(pred, sample)
        scores["em"]   = acc
        scores["f1"]   = acc

    elif dataset == "fever":
        acc = fever_accuracy(pred, gold)
        scores["em"]   = acc
        scores["f1"]   = acc

    elif dataset == "svamp":
        acc = math_accuracy(pred, gold)
        scores["em"]   = acc
        scores["f1"]   = acc

    else:  # hotpotqa, truthfulqa
        scores["em"], scores["f1"] = em_f1_pair(pred, gold)

    # Supporting facts for HotpotQA — use predicted supporting_facts field
    if dataset == "hotpotqa":
        gold_sf = sample.get("supporting_facts", {})
        # Use predicted supporting_facts from result if available
        pred_spans = result.get("supporting_facts", result.get("evidence_spans", []))
        sf_em, sf_f1 = supporting_fact_em_f1(pred_spans, gold_sf)
        scores["sf_em"]     = sf_em
        scores["sf_f1"]     = sf_f1
        scores["joint_em"]  = scores["em"]  * sf_em
        scores["joint_f1"]  = scores["f1"]  * sf_f1

    scores["latency"]    = result.get("latency", 0.0)
    scores["confidence"] = result.get("confidence", 0.5)

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate over result lists
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_scores(score_list: List[Dict]) -> Dict[str, Dict]:
    """
    Returns dict of metric_name → {mean, std, ci_low, ci_high}
    """
    if not score_list:
        return {}

    all_keys = set()
    for s in score_list:
        all_keys |= set(s.keys())

    agg = {}
    for key in all_keys:
        vals = [s[key] for s in score_list if key in s]
        if not vals:
            continue
        arr  = np.array(vals, dtype=float)
        mean = float(arr.mean())
        std  = float(arr.std())
        ci_lo, ci_hi = bootstrap_ci(arr)
        agg[key] = {
            "mean":    round(mean, 4),
            "std":     round(std, 4),
            "ci_low":  round(ci_lo, 4),
            "ci_high": round(ci_hi, 4),
            "n":       len(vals),
        }
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence interval
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    scores: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float]:
    if len(scores) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(42)
    means = [
        rng.choice(scores, size=len(scores), replace=True).mean()
        for _ in range(n_boot)
    ]
    alpha = (1 - ci) / 2
    return (
        float(np.percentile(means, alpha * 100)),
        float(np.percentile(means, (1 - alpha) * 100)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Statistical significance (paired t-test)
# ─────────────────────────────────────────────────────────────────────────────

def paired_ttest(
    scores_a: List[float],
    scores_b: List[float],
) -> Tuple[float, float, str]:
    """
    Returns (t_stat, p_value, significance_marker).
    significance_marker: '***' p<0.001, '**' p<0.01, '*' p<0.05, 'ns' otherwise.
    """
    n = min(len(scores_a), len(scores_b))
    a = np.array(scores_a[:n], dtype=float)
    b = np.array(scores_b[:n], dtype=float)

    if n < 2 or np.allclose(a, b):
        return 0.0, 1.0, "ns"

    try:
        t_stat, p_value = scipy_stats.ttest_rel(a, b)
    except Exception:
        return 0.0, 1.0, "ns"

    if p_value < 0.001:
        marker = "***"
    elif p_value < 0.01:
        marker = "**"
    elif p_value < 0.05:
        marker = "*"
    else:
        marker = "ns"

    return float(t_stat), float(p_value), marker


def run_significance_tests(
    antahkarana_scores: List[float],
    baseline_scores:    Dict[str, List[float]],
    metric:             str = "f1",
) -> Dict[str, Dict]:
    """Compare Antahkarana vs each baseline. Returns dict of stats."""
    results = {}
    for method, scores in baseline_scores.items():
        t, p, sig = paired_ttest(antahkarana_scores, scores)
        ant_mean  = np.mean(antahkarana_scores) if antahkarana_scores else 0
        base_mean = np.mean(scores) if scores else 0
        improvement = (
            100 * (ant_mean - base_mean) / base_mean
            if base_mean > 1e-9 else 0.0
        )
        results[method] = {
            "antahkarana_mean": round(float(ant_mean), 4),
            "baseline_mean":    round(float(base_mean), 4),
            "improvement_pct":  round(improvement, 2),
            "t_stat":           round(t, 4),
            "p_value":          round(p, 6),
            "significance":     sig,
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Throughput measurement
# ─────────────────────────────────────────────────────────────────────────────

def compute_throughput_stats(
    results: List[Dict],
) -> Dict[str, float]:
    latencies = [r.get("latency", 0.0) for r in results]
    if not latencies:
        return {}
    arr = np.array(latencies)
    return {
        "mean_latency_s":   round(float(arr.mean()), 4),
        "std_latency_s":    round(float(arr.std()), 4),
        "p50_latency_s":    round(float(np.percentile(arr, 50)), 4),
        "p95_latency_s":    round(float(np.percentile(arr, 95)), 4),
        "throughput_sps":   round(1.0 / float(arr.mean()) if arr.mean() > 0 else 0, 2),
        "n_samples":        len(results),
    }
