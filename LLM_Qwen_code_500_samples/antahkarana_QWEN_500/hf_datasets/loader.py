"""
datasets/loader.py — Robust dataset loading with multiple fallback sources.
Datasets: HotpotQA, MMLU, TruthfulQA, FEVER, SVAMP
"""

import os
import json
import logging
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path("datasets/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAIN_N = 100    # samples for main evaluation
ABLATION_N = 50  # samples for ablation


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_hf(dataset_id: str, config: Optional[str], split: str, **kwargs):
    """Thin wrapper around datasets.load_dataset with error propagation."""
    from datasets import load_dataset
    args = [dataset_id]
    if config:
        args.append(config)
    return load_dataset(*args, split=split, trust_remote_code=True, **kwargs)


def _sample(items: List[Any], n: int, seed: int = 42) -> List[Any]:
    rng = random.Random(seed)
    if len(items) <= n:
        return list(items)
    return rng.sample(list(items), n)


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _save_cache(name: str, data: List[Dict]):
    with open(_cache_path(name), "w") as f:
        json.dump(data, f)


def _load_cache(name: str) -> Optional[List[Dict]]:
    p = _cache_path(name)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# HotpotQA
# ─────────────────────────────────────────────────────────────────────────────

def load_hotpotqa(n: int = MAIN_N) -> List[Dict]:
    cached = _load_cache(f"hotpotqa_{n}")
    if cached:
        logger.info(f"HotpotQA loaded from cache ({len(cached)} samples)")
        return cached

    logger.info("Downloading HotpotQA…")
    try:
        ds = _load_hf("hotpot_qa", "distractor", "validation")
        items = []
        for row in ds:
            ctx_titles = row["context"]["title"]
            ctx_sents  = row["context"]["sentences"]
            context = []
            for title, sents in zip(ctx_titles, ctx_sents):
                context.append({"title": title, "sentences": sents})

            items.append({
                "id":              row.get("id", ""),
                "question":        row["question"],
                "answer":          row["answer"],
                "context":         context,
                "supporting_facts": {
                    "title":  row["supporting_facts"]["title"],
                    "sent_id": row["supporting_facts"]["sent_id"],
                },
                "type":            row.get("type", ""),
                "level":           row.get("level", ""),
            })

        sampled = _sample(items, n)
        _save_cache(f"hotpotqa_{n}", sampled)
        logger.info(f"HotpotQA: {len(sampled)} samples ready")
        return sampled

    except Exception as e:
        logger.error(f"HotpotQA load failed: {e}")
        # Fallback: try alternate HuggingFace path
        try:
            ds = _load_hf("hotpotqa", None, "validation[:500]")
            items = []
            for row in ds:
                context = [{"title": t, "sentences": s}
                           for t, s in zip(row.get("context", {}).get("title", []),
                                           row.get("context", {}).get("sentences", []))]
                items.append({
                    "id": row.get("id", ""),
                    "question": row["question"],
                    "answer": row["answer"],
                    "context": context,
                    "supporting_facts": row.get("supporting_facts", {}),
                    "type": row.get("type", ""),
                    "level": row.get("level", ""),
                })
            sampled = _sample(items, n)
            _save_cache(f"hotpotqa_{n}", sampled)
            return sampled
        except Exception as e2:
            logger.error(f"HotpotQA fallback also failed: {e2}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# MMLU
# ─────────────────────────────────────────────────────────────────────────────

MMLU_SUBJECTS = [
    "high_school_mathematics", "high_school_physics",
    "college_computer_science", "high_school_world_history",
    "professional_medicine", "abstract_algebra",
    "logical_fallacies", "moral_scenarios",
]

def load_mmlu(n: int = MAIN_N) -> List[Dict]:
    cached = _load_cache(f"mmlu_{n}")
    if cached:
        logger.info(f"MMLU loaded from cache ({len(cached)} samples)")
        return cached

    logger.info("Downloading MMLU…")
    items = []
    per_subject = max(1, n // len(MMLU_SUBJECTS) + 1)

    # Try cais/mmlu first (most reliable mirror)
    for subj in MMLU_SUBJECTS:
        try:
            ds = _load_hf("cais/mmlu", subj, "test")
            for row in list(ds)[:per_subject]:
                choices = row["choices"]
                items.append({
                    "id":       f"mmlu_{subj}_{row.get('id',len(items))}",
                    "question": row["question"],
                    "choices":  choices,
                    "answer":   int(row["answer"]),  # index 0-3
                    "answer_text": choices[int(row["answer"])],
                    "subject":  subj,
                })
        except Exception:
            # Fallback to lukaemon/mmlu
            try:
                ds = _load_hf("lukaemon/mmlu", subj, "test")
                for row in list(ds)[:per_subject]:
                    choices = [row.get(k, "") for k in ["A", "B", "C", "D"]]
                    answer_idx = {"A": 0, "B": 1, "C": 2, "D": 3}.get(str(row.get("target", "A")), 0)
                    items.append({
                        "id":          f"mmlu_{subj}_{len(items)}",
                        "question":    row["input"],
                        "choices":     choices,
                        "answer":      answer_idx,
                        "answer_text": choices[answer_idx],
                        "subject":     subj,
                    })
            except Exception as e2:
                logger.warning(f"MMLU subject {subj} failed: {e2}")
                continue

    if not items:
        # Last resort: all_subjects
        try:
            ds = _load_hf("cais/mmlu", "all", "test")
            for row in list(ds)[:n]:
                choices = row["choices"]
                items.append({
                    "id":          f"mmlu_{len(items)}",
                    "question":    row["question"],
                    "choices":     choices,
                    "answer":      int(row["answer"]),
                    "answer_text": choices[int(row["answer"])],
                    "subject":     row.get("subject", ""),
                })
        except Exception as e3:
            logger.error(f"MMLU all fallback failed: {e3}")
            return []

    sampled = _sample(items, n)
    _save_cache(f"mmlu_{n}", sampled)
    logger.info(f"MMLU: {len(sampled)} samples ready")
    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# TruthfulQA
# ─────────────────────────────────────────────────────────────────────────────

def load_truthfulqa(n: int = MAIN_N) -> List[Dict]:
    cached = _load_cache(f"truthfulqa_{n}")
    if cached:
        logger.info(f"TruthfulQA loaded from cache ({len(cached)} samples)")
        return cached

    logger.info("Downloading TruthfulQA…")
    try:
        ds = _load_hf("truthful_qa", "multiple_choice", "validation")
        items = []
        for row in ds:
            mc = row.get("mc1_targets", {})
            choices  = mc.get("choices", [])
            labels   = mc.get("labels", [])
            correct  = [c for c, l in zip(choices, labels) if l == 1]
            incorrect = [c for c, l in zip(choices, labels) if l == 0]
            items.append({
                "id":              row.get("id", len(items)),
                "question":        row["question"],
                "answer":          correct[0] if correct else "",
                "correct_answers": correct,
                "wrong_answers":   incorrect,
                "category":        row.get("category", ""),
                "source":          row.get("source", ""),
            })

        sampled = _sample(items, n)
        _save_cache(f"truthfulqa_{n}", sampled)
        logger.info(f"TruthfulQA: {len(sampled)} samples ready")
        return sampled

    except Exception as e:
        logger.error(f"TruthfulQA load failed: {e}")
        try:
            # Try generation format
            ds = _load_hf("truthful_qa", "generation", "validation")
            items = []
            for row in ds:
                ca = row.get("correct_answers", [])
                items.append({
                    "id":              row.get("id", len(items)),
                    "question":        row["question"],
                    "answer":          ca[0] if ca else "",
                    "correct_answers": ca,
                    "wrong_answers":   row.get("incorrect_answers", []),
                    "category":        row.get("category", ""),
                    "source":          row.get("source", ""),
                })
            sampled = _sample(items, n)
            _save_cache(f"truthfulqa_{n}", sampled)
            return sampled
        except Exception as e2:
            logger.error(f"TruthfulQA fallback failed: {e2}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# FEVER
# ─────────────────────────────────────────────────────────────────────────────

def load_fever(n: int = MAIN_N) -> List[Dict]:
    cached = _load_cache(f"fever_{n}")
    if cached:
        logger.info(f"FEVER loaded from cache ({len(cached)} samples)")
        return cached

    logger.info("Downloading FEVER…")

    def _parse_fever_rows(ds):
        items = []
        for row in ds:
            label = row.get("label", "")
            if label not in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
                continue
            items.append({
                "id":       row.get("id", len(items)),
                "claim":    row.get("claim", ""),
                "question": f"Is the following claim SUPPORTED, REFUTED, or NOT ENOUGH INFO? Claim: {row.get('claim','')}",
                "answer":   label,
                "evidence": row.get("evidence", []),
            })
        return items

    # Try multiple sources in order
    attempts = [
        ("fever", "v1.0", "paper_dev"),
        ("fever", "v2.0", "validation"),
        ("copenlu/fever_gold_evidence", None, "validation"),
        ("pietrolesci/fever", None, "dev"),
        ("anli", None, "dev_r1"),  # similar NLI task as last resort
    ]

    for ds_id, config, split in attempts:
        try:
            ds = _load_hf(ds_id, config, split)
            items = _parse_fever_rows(ds)
            if items:
                sampled = _sample(items, n)
                _save_cache(f"fever_{n}", sampled)
                logger.info(f"FEVER ({ds_id}): {len(sampled)} samples ready")
                return sampled
        except Exception as e:
            logger.warning(f"FEVER attempt {ds_id}/{config}/{split} failed: {e}")
            continue

    # Final fallback: download JSON directly from FEVER website
    try:
        import urllib.request, json as _json
        url = "https://fever.ai/download/fever/shared_task_dev.jsonl"
        items = []
        with urllib.request.urlopen(url, timeout=30) as resp:
            for line in resp:
                row = _json.loads(line.decode())
                label = row.get("label", "")
                if label not in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
                    continue
                items.append({
                    "id":       row.get("id", len(items)),
                    "claim":    row.get("claim", ""),
                    "question": f"Is the following claim SUPPORTED, REFUTED, or NOT ENOUGH INFO? Claim: {row.get('claim','')}",
                    "answer":   label,
                    "evidence": [],
                })
                if len(items) >= n * 3:
                    break
        if items:
            sampled = _sample(items, n)
            _save_cache(f"fever_{n}", sampled)
            logger.info(f"FEVER (direct): {len(sampled)} samples ready")
            return sampled
    except Exception as e:
        logger.error(f"FEVER direct download failed: {e}")

    logger.error("FEVER: all sources failed")
    return []


# ─────────────────────────────────────────────────────────────────────────────
# SVAMP
# ─────────────────────────────────────────────────────────────────────────────

def load_svamp(n: int = MAIN_N) -> List[Dict]:
    cached = _load_cache(f"svamp_{n}")
    if cached:
        logger.info(f"SVAMP loaded from cache ({len(cached)} samples)")
        return cached

    logger.info("Downloading SVAMP…")
    try:
        ds = _load_hf("ChilleD/SVAMP", None, "train")
        items = []
        for row in ds:
            body = row.get("Body", "")
            question = row.get("Question", "")
            full_q = f"{body} {question}".strip()
            answer_val = row.get("Answer", 0)
            equation = row.get("Equation", "")
            items.append({
                "id":       row.get("ID", len(items)),
                "question": full_q,
                "answer":   str(answer_val),
                "equation": equation,
                "type":     row.get("Type", ""),
            })

        sampled = _sample(items, n)
        _save_cache(f"svamp_{n}", sampled)
        logger.info(f"SVAMP: {len(sampled)} samples ready")
        return sampled

    except Exception as e:
        logger.error(f"SVAMP primary failed: {e}")
        try:
            # Direct JSON from GitHub
            import urllib.request
            url = (
                "https://raw.githubusercontent.com/arkilpatel/SVAMP/"
                "main/data/mawps-asdiv-a_svamp/dev.csv"
            )
            import io, csv
            with urllib.request.urlopen(url, timeout=30) as resp:
                text = resp.read().decode()
            reader = csv.DictReader(io.StringIO(text))
            items = []
            for row in reader:
                full_q = f"{row.get('Body','')} {row.get('Question','')}".strip()
                items.append({
                    "id":       row.get("ID", len(items)),
                    "question": full_q,
                    "answer":   str(row.get("Answer", "")),
                    "equation": row.get("Equation", ""),
                    "type":     row.get("Type", ""),
                })
            sampled = _sample(items, n)
            _save_cache(f"svamp_{n}", sampled)
            return sampled
        except Exception as e2:
            logger.error(f"SVAMP GitHub fallback failed: {e2}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Master loader
# ─────────────────────────────────────────────────────────────────────────────

def load_all_datasets(n: int = MAIN_N) -> Dict[str, List[Dict]]:
    """
    Load all 5 datasets. Returns dict keyed by dataset name.
    Missing datasets return empty lists (logged as warnings).
    """
    logger.info(f"Loading all datasets (n={n} each)…")

    datasets = {
        "hotpotqa":    load_hotpotqa(n),
        "mmlu":        load_mmlu(n),
        "truthfulqa":  load_truthfulqa(n),
        "fever":       load_fever(n),
        "svamp":       load_svamp(n),
    }

    for name, data in datasets.items():
        if not data:
            logger.warning(f"⚠ {name}: 0 samples loaded — skipping in evaluation")
        else:
            logger.info(f"✓ {name}: {len(data)} samples")

    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    all_ds = load_all_datasets(n=5)
    for name, data in all_ds.items():
        print(f"{name}: {len(data)} samples")
        if data:
            print("  Sample:", list(data[0].keys()))
