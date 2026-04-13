"""
evaluator.py — Antahkarana Cognitive Architecture
VQA-style accuracy, consistency, latency, throughput; ablation reporting.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import (
    AntahkaranaResult,
    ExperimentConfig,
    LOG,
    answer_consistency,
    compute_vqa_accuracy,
    normalize_answer,
    print_metrics_panel,
    save_json,
)


# ─────────────────────────────────────────────────────────────────────────────
# Core metric functions
# ─────────────────────────────────────────────────────────────────────────────

def vqa_accuracy(results: List[AntahkaranaResult]) -> float:
    """Mean VQA 2.0 accuracy across all results."""
    if not results:
        return 0.0
    accs = []
    for r in results:
        if r.ground_truth is not None:
            gt_list = (
                r.ground_truth
                if isinstance(r.ground_truth, list)
                else [r.ground_truth] * 3
            )
            accs.append(compute_vqa_accuracy(r.answer, gt_list))
    return float(np.mean(accs)) if accs else 0.0


def exact_match_accuracy(results: List[AntahkaranaResult]) -> float:
    """
    Exact string match after normalisation.

    BUG 6 FIX: The previous implementation compared the prediction only against
    ground_truth[0].  VQA 2.0 provides up to 10 annotator answers per sample;
    a prediction is correct if it matches ANY of the normalised GT answers.
    Comparing against only the first annotation penalises synonymous-but-valid
    answers (e.g. "grey" vs "gray") and produces a metric inconsistent with the
    VQA 2.0 evaluation protocol.
    """
    if not results:
        return 0.0
    correct = 0
    for r in results:
        if r.ground_truth is None:
            continue
        gt_list = r.ground_truth if isinstance(r.ground_truth, list) else [r.ground_truth]
        pred_norm = normalize_answer(r.answer)
        # Match if prediction equals ANY normalised ground-truth annotation
        if any(pred_norm == normalize_answer(gt) for gt in gt_list if gt):
            correct += 1
    return correct / len(results)


def mean_consistency(results: List[AntahkaranaResult]) -> float:
    """Mean consistency score across multi-pass results."""
    scores = [r.consistency_score for r in results if r.consistency_score > 0]
    return float(np.mean(scores)) if scores else 0.0


def latency_stats(results: List[AntahkaranaResult]) -> Dict[str, float]:
    """Latency statistics across all results."""
    lats = [r.latency_total_s for r in results if r.latency_total_s > 0]
    if not lats:
        return {}
    return {
        "mean_s": float(np.mean(lats)),
        "std_s": float(np.std(lats)),
        "min_s": float(np.min(lats)),
        "max_s": float(np.max(lats)),
        "p50_s": float(np.percentile(lats, 50)),
        "p95_s": float(np.percentile(lats, 95)),
    }


def per_pass_latency(results: List[AntahkaranaResult]) -> Dict[str, float]:
    """Average latency per reasoning pass."""
    p1 = [r.latency_pass1_s for r in results if r.latency_pass1_s > 0]
    p2 = [r.latency_pass2_s for r in results if r.latency_pass2_s > 0]
    p3 = [r.latency_pass3_s for r in results if r.latency_pass3_s > 0]
    return {
        "pass1_mean_s": float(np.mean(p1)) if p1 else 0.0,
        "pass2_mean_s": float(np.mean(p2)) if p2 else 0.0,
        "pass3_mean_s": float(np.mean(p3)) if p3 else 0.0,
    }


def throughput(results: List[AntahkaranaResult], wall_time_s: float) -> float:
    """Samples per second based on wall-clock time."""
    if wall_time_s <= 0 or not results:
        return 0.0
    return len(results) / wall_time_s


def gpu_stats(results: List[AntahkaranaResult]) -> Dict[str, float]:
    utils = [r.gpu_util_avg_pct for r in results if r.gpu_util_avg_pct >= 0]
    mems = [r.peak_gpu_mem_gb for r in results if r.peak_gpu_mem_gb > 0]
    return {
        "mean_gpu_util_pct": float(np.mean(utils)) if utils else -1.0,
        "mean_peak_mem_gb": float(np.mean(mems)) if mems else 0.0,
        "max_peak_mem_gb": float(np.max(mems)) if mems else 0.0,
    }


def accuracy_by_answer_type(results: List[AntahkaranaResult]) -> Dict[str, float]:
    """Breakdown accuracy by answer type (yes/no, number, other)."""
    groups: Dict[str, List[float]] = defaultdict(list)
    for r in results:
        if r.ground_truth is None:
            continue
        gt_list = r.ground_truth if isinstance(r.ground_truth, list) else [r.ground_truth] * 3
        acc = compute_vqa_accuracy(r.answer, gt_list)
        gt_norm = normalize_answer(gt_list[0] if gt_list else "")
        if gt_norm in ("yes", "no"):
            groups["yes_no"].append(acc)
        elif gt_norm.isdigit():
            groups["number"].append(acc)
        else:
            groups["other"].append(acc)
    return {k: float(np.mean(v)) for k, v in groups.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Ablation comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_ablations(
    full_results: List[AntahkaranaResult],
    no_pass2_results: List[AntahkaranaResult],
    no_pass3_results: List[AntahkaranaResult],
    baseline_results: List[AntahkaranaResult],
    wall_times: Dict[str, float],
    extra_conditions: Optional[Dict[str, List[AntahkaranaResult]]] = None,
) -> pd.DataFrame:
    """
    Build an ablation comparison DataFrame.
    Columns: condition, vqa_acc, exact_match, consistency, mean_lat_s, throughput
    extra_conditions: optional dict of {display_name: results} for CoT, self-consistency, etc.
    """
    def _row(name: str, res: List[AntahkaranaResult], wt: float) -> Dict[str, Any]:
        return {
            "condition": name,
            "n_samples": len(res),
            "vqa_accuracy": round(vqa_accuracy(res), 4),
            "exact_match": round(exact_match_accuracy(res), 4),
            "consistency": round(mean_consistency(res), 4),
            "hallucination_pct": round(
                100.0 * sum(1 for r in res if r.hallucination_flag) / max(len(res), 1), 1
            ),
            "mean_latency_s": round(latency_stats(res).get("mean_s", 0.0), 4),
            "p95_latency_s": round(latency_stats(res).get("p95_s", 0.0), 4),
            "throughput_sps": round(throughput(res, wt), 4),
            "mean_gpu_util_pct": round(gpu_stats(res).get("mean_gpu_util_pct", -1.0), 1),
            "total_model_calls": sum(r.num_model_calls for r in res),
        }

    rows = [
        _row("Single-Pass Baseline",  baseline_results,  wall_times.get("baseline", 1.0)),
        _row("CoT Baseline",           extra_conditions.get("cot_baseline", []) if extra_conditions else [],
             wall_times.get("cot_baseline", 1.0)),
        _row("Self-Consistency (3x)",  extra_conditions.get("self_consistency", []) if extra_conditions else [],
             wall_times.get("self_consistency", 1.0)),
        _row("No Verification (–P2)",  no_pass2_results,  wall_times.get("no_pass2", 1.0)),
        _row("No Consistency (–P3)",   no_pass3_results,  wall_times.get("no_pass3", 1.0)),
        _row("Full Antahkarana ★",     full_results,      wall_times.get("full", 1.0)),
    ]
    # Drop rows with 0 samples (extra_conditions not provided)
    rows = [r for r in rows if r["n_samples"] > 0]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Save outputs
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(
    results: List[AntahkaranaResult],
    path: str | Path,
    condition_name: str = "full",
) -> None:
    rows = []
    for r in results:
        # Flatten ground_truth list for CSV readability
        gt_flat = (
            " | ".join(str(x) for x in r.ground_truth)
            if isinstance(r.ground_truth, list)
            else str(r.ground_truth or "")
        )
        rows.append({
            "sample_id": r.sample_id,
            "condition": condition_name,
            "question": r.passes.get("question", ""),
            "answer": r.answer,
            "ground_truth": gt_flat,
            "is_correct": r.is_correct,
            "vqa_accuracy_score": round(r.vqa_accuracy_score, 4),   # FIX: real per-sample VQA
            "confidence": round(r.confidence, 4),                    # FIX: agreement-based
            "consistency": round(r.consistency_score, 4),
            "hallucination_flag": r.hallucination_flag,              # BONUS
            "selected_pass": r.selected_pass,
            "pass1_answer": r.passes.get("pass1", {}).get("answer", ""),
            "pass2_answer": r.passes.get("pass2", {}).get("answer", ""),
            "pass3_answer": r.passes.get("pass3", {}).get("answer", ""),
            "latency_total_s": round(r.latency_total_s, 4),
            "latency_pass1_s": round(r.latency_pass1_s, 4),
            "latency_pass2_s": round(r.latency_pass2_s, 4),
            "latency_pass3_s": round(r.latency_pass3_s, 4),
            "gpu_util_avg_pct": round(r.gpu_util_avg_pct, 1),
            "peak_gpu_mem_gb": round(r.peak_gpu_mem_gb, 3),
            "num_model_calls": r.num_model_calls,
            "modality": r.modality,
        })
    df = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    LOG.info(f"[IO] Saved results CSV → {path}  ({len(df)} rows)")


def save_metrics_summary(
    metrics: Dict[str, Any],
    path: str | Path,
) -> None:
    save_json(metrics, path)
    print_metrics_panel(metrics)


def save_latency_report(
    results: List[AntahkaranaResult],
    path: str | Path,
) -> None:
    report = {
        "total_latency": latency_stats(results),
        "per_pass_latency": per_pass_latency(results),
        "gpu": gpu_stats(results),
        "n_samples": len(results),
    }
    save_json(report, path)


def save_ablation_table(
    df: pd.DataFrame,
    csv_path: str | Path,
    json_path: str | Path,
) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    save_json(df.to_dict(orient="records"), json_path)
    LOG.info(f"[IO] Ablation table → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Full experiment evaluation entry point
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_experiment(
    results: List[AntahkaranaResult],
    wall_time_s: float,
    cfg: ExperimentConfig,
    condition_name: str = "full",
) -> Dict[str, Any]:
    """Compute and save all metrics for a single experiment condition."""
    n = len(results)
    hall_count = sum(1 for r in results if r.hallucination_flag)
    n_correct  = sum(1 for r in results if r.is_correct)

    # Wilson 95% CI on accuracy (IEEE requirement: report as 0.743 ± 0.044)
    try:
        from ieee_stats import wilson_ci
        ci_lo, ci_hi = wilson_ci(n_correct, n)
        acc_ci = {"ci95_lo": round(ci_lo, 4), "ci95_hi": round(ci_hi, 4),
                  "margin":  round((ci_hi - ci_lo) / 2, 4)}
    except Exception:
        acc_ci = {}

    metrics = {
        "condition": condition_name,
        "n_samples": n,
        "vqa_accuracy": vqa_accuracy(results),
        "vqa_accuracy_ci95": acc_ci,
        "exact_match_accuracy": exact_match_accuracy(results),
        "mean_consistency": mean_consistency(results),
        "hallucination_count": hall_count,
        "hallucination_pct": round(100 * hall_count / max(n, 1), 1),
        "throughput_samples_per_sec": throughput(results, wall_time_s),
        "wall_time_s": wall_time_s,
        **latency_stats(results),
        **per_pass_latency(results),
        **gpu_stats(results),
        "accuracy_by_type": accuracy_by_answer_type(results),
        "model": results[0].model if results else cfg.model_name,
    }

    out_dir = Path(cfg.output_dir)
    save_results_csv(results, out_dir / f"results_{condition_name}.csv", condition_name)
    save_metrics_summary(metrics, out_dir / f"metrics_{condition_name}.json")
    save_latency_report(results, out_dir / f"latency_{condition_name}.json")

    return metrics
