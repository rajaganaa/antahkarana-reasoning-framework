"""
baselines/runner.py — Batch all 4 baselines through the single vLLM engine.
Methods: direct, cot, self_consistency (n=5), tot
"""

import re
import time
import logging
from collections import Counter
from typing import List, Dict, Any, Optional

from vllm_engine import batch_infer, batch_infer_multi, get_engine
from baselines.prompts import build_prompts_for_method

logger = logging.getLogger(__name__)

BATCH_SIZE = 16  # adjust if OOM; vLLM handles dynamic batching internally


# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_answer(raw: str, dataset: str) -> str:
    """
    Parse model output to a clean final answer string.
    """
    raw = raw.strip()

    # Look for explicit ANSWER: tag first
    m = re.search(r'ANSWER\s*:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # MMLU: letter choice
    if dataset == "mmlu":
        m = re.search(r'\b([A-D])\b', raw)
        if m:
            return m.group(1).upper()

    # FEVER: label
    if dataset == "fever":
        for label in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
            if label.lower() in raw.lower():
                return label

    # SVAMP: last number
    if dataset == "svamp":
        nums = re.findall(r'-?\d+(?:\.\d+)?', raw)
        if nums:
            return nums[-1]

    # Fallback: last non-empty line
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    return lines[-1] if lines else raw


def _sc_majority_vote(outputs: List[str], dataset: str) -> str:
    """Majority vote across n self-consistency samples."""
    answers = [_extract_answer(o, dataset) for o in outputs]
    if not answers:
        return ""
    # Normalise for counting
    normed = [a.lower().strip() for a in answers]
    most_common = Counter(normed).most_common(1)[0][0]
    # Return original-cased version
    for a, n in zip(answers, normed):
        if n == most_common:
            return a
    return answers[0]


# ─────────────────────────────────────────────────────────────────────────────
# Core per-method runner
# ─────────────────────────────────────────────────────────────────────────────

def run_direct(samples: List[Dict], dataset: str) -> List[Dict]:
    engine = get_engine()
    prompts = build_prompts_for_method(samples, dataset, "direct", engine)

    t0 = time.time()
    raw_outputs = batch_infer(prompts, method="direct")
    elapsed = time.time() - t0

    results = []
    for i, (s, raw) in enumerate(zip(samples, raw_outputs)):
        pred = _extract_answer(raw, dataset)
        results.append({
            "id":        s.get("id", i),
            "question":  s["question"],
            "gold":      s["answer"],
            "predicted": pred,
            "raw":       raw,
            "latency":   elapsed / len(samples),
            "method":    "direct",
            "dataset":   dataset,
        })
    return results


def run_cot(samples: List[Dict], dataset: str) -> List[Dict]:
    engine = get_engine()
    prompts = build_prompts_for_method(samples, dataset, "cot", engine)

    t0 = time.time()
    raw_outputs = batch_infer(prompts, method="cot")
    elapsed = time.time() - t0

    results = []
    for i, (s, raw) in enumerate(zip(samples, raw_outputs)):
        pred = _extract_answer(raw, dataset)
        results.append({
            "id":           s.get("id", i),
            "question":     s["question"],
            "gold":         s["answer"],
            "predicted":    pred,
            "raw":          raw,
            "reasoning":    raw,  # full CoT kept for analysis
            "latency":      elapsed / len(samples),
            "method":       "cot",
            "dataset":      dataset,
        })
    return results


def run_self_consistency(samples: List[Dict], dataset: str) -> List[Dict]:
    """
    Generate n=5 samples per prompt with temperature=0.7, then majority vote.
    vLLM handles all 5 completions per prompt in a single batched call.
    """
    engine = get_engine()
    prompts = build_prompts_for_method(samples, dataset, "self_consistency", engine)

    t0 = time.time()
    multi_outputs = batch_infer_multi(prompts, method="self_consistency")
    elapsed = time.time() - t0

    results = []
    for i, (s, outputs) in enumerate(zip(samples, multi_outputs)):
        pred = _sc_majority_vote(outputs, dataset)
        results.append({
            "id":           s.get("id", i),
            "question":     s["question"],
            "gold":         s["answer"],
            "predicted":    pred,
            "all_samples":  outputs,
            "latency":      elapsed / len(samples),
            "method":       "self_consistency",
            "dataset":      dataset,
        })
    return results


def run_tot(samples: List[Dict], dataset: str) -> List[Dict]:
    engine = get_engine()
    prompts = build_prompts_for_method(samples, dataset, "tot", engine)

    t0 = time.time()
    raw_outputs = batch_infer(prompts, method="tot")
    elapsed = time.time() - t0

    results = []
    for i, (s, raw) in enumerate(zip(samples, raw_outputs)):
        pred = _extract_answer(raw, dataset)
        results.append({
            "id":        s.get("id", i),
            "question":  s["question"],
            "gold":      s["answer"],
            "predicted": pred,
            "raw":       raw,
            "latency":   elapsed / len(samples),
            "method":    "tot",
            "dataset":   dataset,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_RUNNERS = {
    "direct":           run_direct,
    "cot":              run_cot,
    "self_consistency": run_self_consistency,
    "tot":              run_tot,
}


def run_all_baselines(
    samples: List[Dict],
    dataset: str,
    methods: Optional[List[str]] = None,
) -> Dict[str, List[Dict]]:
    """
    Run all (or selected) baselines on samples from a given dataset.
    Returns dict: method_name → list of result dicts.
    """
    if methods is None:
        methods = list(BASELINE_RUNNERS.keys())

    all_results: Dict[str, List[Dict]] = {}
    for method in methods:
        logger.info(f"  [{dataset}] Running baseline: {method} ({len(samples)} samples)…")
        runner = BASELINE_RUNNERS[method]
        t0 = time.time()
        results = runner(samples, dataset)
        elapsed = time.time() - t0
        all_results[method] = results
        logger.info(
            f"  [{dataset}][{method}] done in {elapsed:.1f}s "
            f"({len(samples)/elapsed:.1f} samp/s)"
        )

    return all_results
