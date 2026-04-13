"""
evaluation/ablation.py — Ablation study for Antahkarana components.

Ablates:
  - routing (use simple path for all)
  - pramana (skip grounding verification)
  - samsaya (skip self-consistency repair)
  - evidence_module (skip Chitta context scoring)

Runs on up to 3 datasets simultaneously:
  - HotpotQA  : tests Routing + Evidence (multi-hop reasoning path)
  - SVAMP     : tests Routing (math path) + Samsaya (numeric repair)
  - FEVER     : tests Pramana (only dataset where it fires)

Each dataset uses ABLATION_N samples (default 50 / 100 for 500-run).
"""

import logging
import time
import re
from typing import List, Dict, Optional

from vllm_engine import batch_infer, batch_infer_multi, get_engine
from antahkarana.system import (
    Manas, Chitta, Buddhi, QType,
    _extract_answer, _extract_pramana_answer, _is_bad_answer,
)
from collections import Counter

logger = logging.getLogger(__name__)


class AblationConfig:
    def __init__(
        self,
        skip_routing:  bool = False,
        skip_pramana:  bool = False,
        skip_samsaya:  bool = False,
        skip_evidence: bool = False,
        name:          str  = "full",
    ):
        self.skip_routing  = skip_routing
        self.skip_pramana  = skip_pramana
        self.skip_samsaya  = skip_samsaya
        self.skip_evidence = skip_evidence
        self.name          = name


ABLATION_CONFIGS = [
    AblationConfig(name="Full Antahkarana"),
    AblationConfig(skip_routing=True,  name="w/o Routing"),
    AblationConfig(skip_pramana=True,  name="w/o Pramana"),
    AblationConfig(skip_samsaya=True,  name="w/o Samsaya"),
    AblationConfig(skip_evidence=True, name="w/o Evidence"),
]

# Datasets used for ablation and the rationale for each
ABLATION_DATASETS = ["hotpotqa", "svamp", "fever"]
# hotpotqa: tests Routing (multihop) + Evidence (dense retrieval)
# svamp   : tests Routing (math path) + Samsaya (numeric repair)
# fever   : tests Pramana (only dataset where fact-verification fires)


class AblationRunner:
    def __init__(self):
        self.manas  = Manas()
        self.chitta = Chitta()
        self.buddhi = Buddhi(self.manas, self.chitta)

    def run(
        self,
        samples:  List[Dict],
        dataset:  str,
        config:   AblationConfig,
    ) -> List[Dict]:
        t0 = time.time()
        engine = get_engine()

        # ── Routing ──────────────────────────────────────────────────────────
        q_types = []
        for s in samples:
            if config.skip_routing:
                q_types.append(QType.SIMPLE)
            else:
                q_types.append(self.manas.classify(s["question"], dataset))

        # ── Context scoring ───────────────────────────────────────────────────
        context_strs = []
        for s, q_type in zip(samples, q_types):
            ctx = s.get("context", [])
            if ctx and not config.skip_evidence:
                entities = self.manas.extract_entities(s["question"])
                ctx_str  = self.chitta.top_k_context(s["question"], ctx, entities, k=5)
            elif ctx:
                # Naive concat without scoring (w/o Evidence condition)
                parts = []
                for para in ctx[:4]:
                    title = para.get("title", "")
                    sents = para.get("sentences", [])
                    text  = " ".join(sents) if isinstance(sents, list) else str(sents)
                    parts.append(f"[{title}]: {text}")
                ctx_str = "\n".join(parts)
            else:
                ctx_str = ""
            context_strs.append(ctx_str)

        # ── Pass 1: Tarka ─────────────────────────────────────────────────────
        tarka_prompts = [
            self.buddhi.build_tarka_prompt(
                s["question"], ctx_str, q_type, choices=s.get("choices")
            )
            for s, ctx_str, q_type in zip(samples, context_strs, q_types)
        ]
        tarka_raw     = batch_infer(tarka_prompts, method="antahkarana")
        tarka_answers = [_extract_answer(r, dataset) for r in tarka_raw]

        # ── Pass 2: Pramana (FEVER only) ──────────────────────────────────────
        grounded_answers = list(tarka_answers)
        if not config.skip_pramana and dataset in ("fever",):
            pramana_prompts = [
                self.buddhi.build_pramana_prompt(s["question"], draft, ctx_str)
                for s, draft, ctx_str in zip(samples, tarka_answers, context_strs)
            ]
            pramana_raw = batch_infer(pramana_prompts, method="antahkarana")
            for i, (draft, p_raw) in enumerate(zip(tarka_answers, pramana_raw)):
                grounded, _ = _extract_pramana_answer(p_raw, draft)
                if not _is_bad_answer(grounded):
                    grounded_answers[i] = grounded

        # ── Pass 3: Samsaya ───────────────────────────────────────────────────
        final_answers = list(grounded_answers)
        if not config.skip_samsaya:
            uncertain_idx = [i for i, a in enumerate(grounded_answers) if _is_bad_answer(a)]
            if uncertain_idx:
                sc_prompts = [tarka_prompts[i] for i in uncertain_idx]
                sc_multi   = batch_infer_multi(sc_prompts, method="self_consistency")
                for idx, multi_out in zip(uncertain_idx, sc_multi):
                    candidates = [_extract_answer(o, dataset) for o in multi_out]
                    counter    = Counter([c.lower() for c in candidates])
                    best_lower = counter.most_common(1)[0][0]
                    best_orig  = next(
                        (c for c in candidates if c.lower() == best_lower), candidates[0]
                    )
                    final_answers[idx] = best_orig

        elapsed = time.time() - t0

        results = []
        for i, s in enumerate(samples):
            results.append({
                "id":        s.get("id", i),
                "question":  s["question"],
                "gold":      s["answer"],
                "predicted": final_answers[i],
                "latency":   elapsed / len(samples),
                "method":    config.name,
                "dataset":   dataset,
            })
        return results


def run_ablation_study(
    samples:   List[Dict],
    dataset:   str,
    configs:   Optional[List[AblationConfig]] = None,
) -> Dict[str, List[Dict]]:
    """
    Run all ablation configs on the given samples for one dataset.
    Returns dict: config_name -> list of result dicts.
    """
    if configs is None:
        configs = ABLATION_CONFIGS

    runner  = AblationRunner()
    results = {}

    for cfg in configs:
        logger.info(f"  Ablation [{dataset}] config: {cfg.name} ({len(samples)} samples)...")
        t0  = time.time()
        res = runner.run(samples, dataset, cfg)
        elapsed = time.time() - t0
        results[cfg.name] = res
        logger.info(f"  Ablation [{cfg.name}] done in {elapsed:.1f}s")

    return results
