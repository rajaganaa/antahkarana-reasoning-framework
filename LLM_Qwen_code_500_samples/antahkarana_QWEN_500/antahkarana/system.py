"""
antahkarana/system.py — Antahkarana v16
Model: Qwen/Qwen2.5-7B-Instruct

Qwen2.5 notes:
- Native ChatML format with dedicated system role — no prompt merging needed.
- Stronger instruction following than Mistral-7B — all structured format
  prompts (PRAMANA, VERIFY, MULTIHOP) work correctly without workarounds.
- Mistral-specific comments kept for traceability but fixes remain in place
  as they represent good prompt hygiene for any 7B-class model.
"""

import re
import time
import logging
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

from vllm_engine import get_engine, batch_infer, batch_infer_multi

logger = logging.getLogger(__name__)

# ── Sentence-transformer for dense retrieval ────────────────────────────────
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("[CHITTA] Dense embedder loaded: all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"[CHITTA] Embedder load failed ({e}), falling back to lexical")
            _embedder = "lexical"
    return _embedder


# ─────────────────────────────────────────────────────────────────────────────
# QType
# ─────────────────────────────────────────────────────────────────────────────

class QType:
    SIMPLE       = "simple"
    MULTIHOP     = "multihop"
    COMPARISON   = "comparison"
    MATH         = "math"
    VERIFICATION = "verification"
    MCHOICE      = "mchoice"


# ─────────────────────────────────────────────────────────────────────────────
# Manas — routing
# HotpotQA always → MULTIHOP (ablation: w/o routing = +13% F1)
# ─────────────────────────────────────────────────────────────────────────────

MATH_KW = {
    "how many", "how much", "calculate", "total", "sum", "average",
    "percent", "ratio", "difference", "multiply", "divide",
}
COMPARE_KW = {
    "older", "younger", "longer", "shorter", "more", "less",
    "earlier", "later", "higher", "lower", "bigger", "smaller",
    "first", "last", "both", "same",
}
VERIFY_KW = {"supports", "refutes", "true or false", "is it true", "verify", "claim"}


class Manas:
    def classify(self, question: str, dataset: str, sample_id: Any = None) -> str:
        # Hard dataset overrides
        if dataset == "mmlu":    return QType.MCHOICE
        if dataset == "fever":   return QType.VERIFICATION
        if dataset == "svamp":   return QType.MATH
        # HotpotQA: always multihop — ablation proved routing hurts
        if dataset == "hotpotqa": return QType.MULTIHOP

        q = question.lower()
        if any(k in q for k in VERIFY_KW):  return QType.VERIFICATION
        if any(k in q for k in MATH_KW):    return QType.MATH
        if any(k in q for k in COMPARE_KW): return QType.COMPARISON
        return QType.SIMPLE

    def extract_entities(self, question: str) -> List[str]:
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', question)
        return list(dict.fromkeys(entities))


# ─────────────────────────────────────────────────────────────────────────────
# Chitta — dense retrieval + lexical fallback + proper SF extraction
# ─────────────────────────────────────────────────────────────────────────────

class Chitta:

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9: return 0.0
        return float(np.dot(a, b) / (na * nb))

    def score_context(
        self, question: str, context: List[Dict], entities: List[str]
    ) -> List[Tuple[float, Dict]]:
        """Dense semantic scoring with lexical fallback."""
        embedder = _get_embedder()
        scored   = []

        if embedder != "lexical":
            try:
                para_texts = []
                for para in context:
                    sents = para.get("sentences", [])
                    text  = " ".join(sents) if isinstance(sents, list) else str(sents)
                    para_texts.append(text)

                if para_texts:
                    q_emb    = embedder.encode(question, convert_to_numpy=True, show_progress_bar=False)
                    p_embs   = embedder.encode(para_texts, convert_to_numpy=True, show_progress_bar=False)
                    for i, (para, text) in enumerate(zip(context, para_texts)):
                        sem_score = self._cosine(q_emb, p_embs[i])
                        # Add entity bonus on top of semantic score
                        ent_bonus = 0.1 * sum(1 for e in entities if e.lower() in text.lower())
                        scored.append((sem_score + ent_bonus, para))
                    return sorted(scored, key=lambda x: x[0], reverse=True)
            except Exception as e:
                logger.debug(f"[CHITTA] Dense scoring failed ({e}), using lexical")

        # Lexical fallback
        q_words = set(question.lower().split())
        for para in context:
            sents = para.get("sentences", [])
            text  = " ".join(sents) if isinstance(sents, list) else str(sents)
            words = set(text.lower().split())
            overlap   = len(q_words & words) / (len(q_words) + 1)
            ent_bonus = 0.1 * sum(1 for e in entities if e.lower() in text.lower())
            scored.append((overlap + ent_bonus, para))
        return sorted(scored, key=lambda x: x[0], reverse=True)

    def top_k_context(
        self, question: str, context: List[Dict], entities: List[str], k: int = 5
    ) -> str:
        scored = self.score_context(question, context, entities)
        parts  = []
        for _, para in scored[:k]:
            title = para.get("title", "")
            sents = para.get("sentences", [])
            text  = " ".join(sents) if isinstance(sents, list) else str(sents)
            parts.append(f"[{title}]: {text}")
        return "\n".join(parts)

    def extract_evidence_spans(
        self, answer: str, context: List[Dict]
    ) -> List[Dict]:
        """Token-overlap span extraction (unchanged — works well)."""
        evidence  = []
        ans_lower = answer.lower().strip()
        if not ans_lower or len(ans_lower) < 2:
            return evidence
        ans_tokens = set(ans_lower.split())
        for para in context:
            sents = para.get("sentences", [])
            title = para.get("title", "")
            for idx, sent in enumerate(sents):
                sent_lower  = sent.lower()
                sent_tokens = set(sent_lower.split())
                overlap = len(ans_tokens & sent_tokens) / (len(ans_tokens) + 1e-9)
                if ans_lower in sent_lower or overlap > 0.5:
                    evidence.append({"title": title, "sent_id": idx, "sentence": sent})
        return evidence

    def extract_supporting_facts(
        self, answer: str, tarka_raw: str, context: List[Dict]
    ) -> List[Dict]:
        """
        Extract supporting facts for HotpotQA SF-EM/F1 evaluation.
        Parses SUPPORTING: field from Tarka output, with answer-span fallback.
        """
        sf = []

        # Method 1: Parse SUPPORTING: field from model output
        m = re.search(r'SUPPORTING\s*:\s*(.+?)(?:\n|$)', tarka_raw, re.IGNORECASE)
        if m:
            titles_raw = m.group(1).strip()
            for title_part in titles_raw.split(","):
                title_part = title_part.strip().strip('"\'')
                for para in context:
                    if title_part.lower() in para.get("title", "").lower():
                        sf.append({"title": para["title"], "sent_id": 0})
                        break

        # Method 2: Fallback — paragraphs with highest answer overlap
        if not sf:
            ans_lower  = answer.lower().strip()
            ans_tokens = set(ans_lower.split())
            for para in context:
                sents = para.get("sentences", [])
                title = para.get("title", "")
                for idx, sent in enumerate(sents):
                    sent_toks = set(sent.lower().split())
                    overlap = len(ans_tokens & sent_toks) / (len(ans_tokens) + 1e-9)
                    if ans_lower in sent.lower() or overlap > 0.5:
                        sf.append({"title": title, "sent_id": idx})

        return sf


# ─────────────────────────────────────────────────────────────────────────────
# Prompt systems
# ─────────────────────────────────────────────────────────────────────────────

MULTIHOP_SYSTEM = (
    # Good practice for all 7B models: explicit instruction not to echo the
    # question in the ANSWER line. Qwen2.5 follows structured formats reliably.
    "You are a multi-hop reasoning expert. "
    "Use ONLY the provided context to answer. Read carefully.\n\n"
    "Step 1: Find the first key fact from context.\n"
    "Step 2: Use it to find the final answer.\n\n"
    "CRITICAL: The ANSWER line must be SHORT — 1 to 5 words only. "
    "Do NOT repeat the question. Do NOT explain. Write only the answer entity.\n\n"
    "Format:\n"
    "Step 1: ...\n"
    "Step 2: ...\n"
    "ANSWER: <short answer only, 1-5 words>"
)

TARKA_SYSTEM = (
    "You are a careful reasoning assistant. "
    "Use the context to answer.\n\n"
    "Reasoning: <step-by-step>\n"
    "ANSWER: <final answer>"
)

MATH_SYSTEM = (
    "You are a precise math solver.\n"
    "Step 1: ...\nStep 2: ...\n"
    "ANSWER: <number only>"
)

VERIFY_SYSTEM = (
    # Qwen2.5 follows structured formats well. Keeping the few-shot example
    # from the Mistral version as it reliably anchors label output for all
    # 7B-class models and prevents paraphrased / freeform label variants.
    "You are a fact-checking assistant. "
    "You must decide if the claim is supported or refuted by evidence.\n\n"
    "You MUST respond with exactly one of these three labels on the ANSWER line:\n"
    "  SUPPORTS\n"
    "  REFUTES\n"
    "  NOT ENOUGH INFO\n\n"
    "Example:\n"
    "Claim: The Eiffel Tower is in Paris.\n"
    "Evidence assessment: The claim states Paris. This is correct.\n"
    "ANSWER: SUPPORTS\n\n"
    "Now answer:\n"
    "Evidence assessment: <your reasoning>\n"
    "ANSWER: SUPPORTS or REFUTES or NOT ENOUGH INFO"
)

MCHOICE_SYSTEM = (
    "You are a knowledge expert.\n\n"
    "Analysis: ...\n"
    "ANSWER: <letter A/B/C/D only>"
)

COMPARE_SYSTEM = (
    "You are a comparative reasoning expert.\n\n"
    "Entity 1: ...\nEntity 2: ...\nComparison: ...\nANSWER: <answer>"
)

PRAMANA_SYSTEM = (
    # Clean two-block format works reliably across Mistral and Qwen2.5.
    # Qwen2.5's stronger instruction following means parse rate is higher,
    # but the explicit format is kept for robustness.
    "You are a strict fact verifier. "
    "Check if the draft answer is supported by the context.\n\n"
    "You MUST follow this exact format (no other text):\n"
    "Supported: yes\n"
    "Evidence: <quote from context, max 15 words>\n"
    "Revised answer: <repeat the draft answer>\n\n"
    "OR if NOT supported:\n"
    "Supported: no\n"
    "Evidence: <quote from context, max 15 words>\n"
    "Revised answer: <corrected answer from context>"
)

SAMSAYA_SYSTEM = (
    "You are a consistency checker. Pick the best answer from candidates.\n\n"
    "You MUST follow this exact format:\n"
    "Most consistent: <best answer>\n"
    "Confidence: <0.0 to 1.0>\n"
    "Reason: <one sentence>"
)


# ─────────────────────────────────────────────────────────────────────────────
# Buddhi
# ─────────────────────────────────────────────────────────────────────────────

class Buddhi:
    def __init__(self, manas: Manas, chitta: Chitta):
        self.manas  = manas
        self.chitta = chitta

    def build_tarka_prompt(self, question: str, context_str: str, q_type: str,
                           choices: Optional[List[str]] = None) -> str:
        engine = get_engine()
        if q_type == QType.MULTIHOP:
            return engine.apply_chat_template(
                MULTIHOP_SYSTEM, f"Context:\n{context_str}\n\nQuestion: {question}"
            )
        if q_type == QType.MATH:
            return engine.apply_chat_template(MATH_SYSTEM, f"Problem: {question}")
        if q_type == QType.VERIFICATION:
            return engine.apply_chat_template(VERIFY_SYSTEM, question)
        if q_type == QType.MCHOICE:
            labels = ["A", "B", "C", "D"]
            cs     = "\n".join(f"{labels[i]}. {c}" for i, c in enumerate(choices or []))
            return engine.apply_chat_template(MCHOICE_SYSTEM, f"Question: {question}\n{cs}")
        if q_type == QType.COMPARISON:
            return engine.apply_chat_template(
                COMPARE_SYSTEM, f"Context:\n{context_str}\n\nQuestion: {question}"
            )
        return engine.apply_chat_template(
            TARKA_SYSTEM, f"Context:\n{context_str}\n\nQuestion: {question}"
        )

    def build_pramana_prompt(self, question: str, draft: str, context_str: str) -> str:
        engine = get_engine()
        return engine.apply_chat_template(
            PRAMANA_SYSTEM,
            f"Question: {question}\n\nDraft answer: {draft}\n\nContext:\n{context_str[:2000]}"
        )

    def build_samsaya_prompt(self, question: str, candidates: List[str]) -> str:
        engine   = get_engine()
        cand_str = "\n".join(f"Candidate {i+1}: {c}" for i, c in enumerate(candidates))
        return engine.apply_chat_template(SAMSAYA_SYSTEM, f"Question: {question}\n\nCandidates:\n{cand_str}")


# ─────────────────────────────────────────────────────────────────────────────
# Answer extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_answer(raw: str, dataset: str) -> str:
    raw = raw.strip()
    m   = re.search(r'ANSWER\s*:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        # Strip trailing punctuation
        ans = re.sub(r'[.,;:!?]+$', '', ans)
        # Strip chain prefixes like "Chain: Fact1 → " or "Fact1 + Fact2 → "
        ans = re.sub(r'^.*?→\s*', '', ans)
        # Strip "Chain:" prefix
        ans = re.sub(r'^Chain\s*:\s*', '', ans, flags=re.IGNORECASE)
        # Strip "Fact N:" prefixes
        ans = re.sub(r'^Fact\s*\d+\s*:\s*', '', ans, flags=re.IGNORECASE)
        ans = ans.strip()
        return ans

    if dataset == "mmlu":
        m = re.search(r'\b([A-D])\b', raw)
        if m: return m.group(1).upper()
    if dataset == "fever":
        for label in ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"):
            if label.lower() in raw.lower(): return label
    if dataset == "svamp":
        nums = re.findall(r'-?\d+(?:\.\d+)?', raw)
        if nums: return nums[-1]
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    return lines[-1] if lines else raw


def _extract_pramana_answer(raw: str, draft: str) -> Tuple[str, bool]:
    supported_m = re.search(r'Supported\s*:\s*(yes|no)', raw, re.IGNORECASE)
    revised_m   = re.search(r'Revised answer\s*:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
    if revised_m and supported_m and supported_m.group(1).lower() == "no":
        revised = revised_m.group(1).strip()
        if revised and not _is_bad_answer(revised):
            return revised, True
    return draft, False


def _extract_samsaya_answer(raw: str) -> Optional[str]:
    m = re.search(r'Most consistent\s*:\s*(.+?)(?:\n|$)', raw, re.IGNORECASE)
    if m:
        ans = m.group(1).strip()
        if ans: return ans
    return None


def _extract_confidence(raw: str) -> float:
    m = re.search(r'Confidence\s*:\s*([\d.]+)', raw, re.IGNORECASE)
    if m:
        try: return min(1.0, max(0.0, float(m.group(1))))
        except ValueError: pass
    return 0.6


APOLOGY_PATTERNS = [
    "i don't know", "i cannot", "i am not sure", "no information",
    "sorry", "cannot determine", "unclear", "i'm not certain",
]

def _is_bad_answer(answer: str) -> bool:
    if not answer or len(answer.strip()) < 1: return True
    return any(p in answer.lower() for p in APOLOGY_PATTERNS)


# ─────────────────────────────────────────────────────────────────────────────
# AntahkaranaSystem
# ─────────────────────────────────────────────────────────────────────────────

class AntahkaranaSystem:
    def __init__(self):
        self.manas  = Manas()
        self.chitta = Chitta()
        self.buddhi = Buddhi(self.manas, self.chitta)
        # Pre-load embedder at init time (not per-batch)
        _get_embedder()

    def run_batch(self, samples: List[Dict], dataset: str) -> List[Dict]:
        t0 = time.time()

        # ── Classify & prepare ───────────────────────────────────────────────
        q_types      = []
        context_strs = []
        entities_list = []

        for s in samples:
            q_type   = self.manas.classify(s["question"], dataset, sample_id=s.get("id"))
            entities = self.manas.extract_entities(s["question"])
            ctx      = s.get("context", [])
            # Dense retrieval via Chitta
            ctx_str  = self.chitta.top_k_context(s["question"], ctx, entities, k=5) if ctx else ""
            q_types.append(q_type)
            context_strs.append(ctx_str)
            entities_list.append(entities)

        route_dist = Counter(q_types)
        logger.info(f"[ROUTE_DIST] dataset={dataset} n={len(samples)} dist={dict(route_dist)}")

        # ── Pass 1: Tarka ────────────────────────────────────────────────────
        tarka_prompts = [
            self.buddhi.build_tarka_prompt(s["question"], ctx_str, q_type, choices=s.get("choices"))
            for s, ctx_str, q_type in zip(samples, context_strs, q_types)
        ]
        tarka_raw     = batch_infer(tarka_prompts, method="antahkarana")
        tarka_answers = [_extract_answer(r, dataset) for r in tarka_raw]

        # ── Pass 2: Pramana — FEVER only ─────────────────────────────────────
        grounded_answers   = list(tarka_answers)
        pramana_revised    = [False] * len(samples)
        pramana_fire_count = 0

        if dataset in ("fever",):
            pramana_prompts = [
                self.buddhi.build_pramana_prompt(s["question"], draft, ctx_str)
                for s, draft, ctx_str in zip(samples, tarka_answers, context_strs)
            ]
            pramana_raw = batch_infer(pramana_prompts, method="antahkarana")

            for i, (draft, p_raw) in enumerate(zip(tarka_answers, pramana_raw)):
                grounded, was_revised = _extract_pramana_answer(p_raw, draft)
                grounded_answers[i]   = grounded
                pramana_revised[i]    = was_revised
                if was_revised:
                    pramana_fire_count += 1
                    logger.debug(f"[PRAMANA_FIRED] i={i} {draft!r}→{grounded!r}")

            # Hard assertion if ALL outputs empty
            all_empty = all(
                not re.search(r'Supported\s*:', r, re.IGNORECASE) for r in pramana_raw
            )
            if all_empty and pramana_raw:
                raise AssertionError(
                    f"[PRAMANA BUG] All outputs empty. Sample: {pramana_raw[0][:300]!r}"
                )
            logger.info(
                f"[PRAMANA] dataset={dataset} fired={pramana_fire_count}/{len(samples)} "
                f"({pramana_fire_count/len(samples):.1%})"
            )

        # ── Pass 3: Samsaya — uncertain answers ──────────────────────────────
        final_answers      = list(grounded_answers)
        # Real confidence: 0.9 if direct clean answer, 0.75 if pramana-revised
        confidences        = [0.75 if pramana_revised[i] else 0.9 for i in range(len(samples))]
        samsaya_fire_count = 0
        samsaya_indices    = [i for i, a in enumerate(grounded_answers) if _is_bad_answer(a)]

        if samsaya_indices:
            sc_prompts   = [tarka_prompts[i] for i in samsaya_indices]
            sc_multi     = batch_infer_multi(sc_prompts, method="self_consistency")
            candidates_p = [[_extract_answer(o, dataset) for o in mo] for mo in sc_multi]

            samsaya_prompts = [
                self.buddhi.build_samsaya_prompt(samples[idx]["question"], cands)
                for idx, cands in zip(samsaya_indices, candidates_p)
            ]
            samsaya_raw = batch_infer(samsaya_prompts, method="antahkarana")

            for j, (idx, sam_raw, cands) in enumerate(
                zip(samsaya_indices, samsaya_raw, candidates_p)
            ):
                samsaya_ans = _extract_samsaya_answer(sam_raw)
                if samsaya_ans and not _is_bad_answer(samsaya_ans):
                    final_answers[idx]  = samsaya_ans
                    confidences[idx]    = _extract_confidence(sam_raw)
                    samsaya_fire_count += 1
                else:
                    counter    = Counter(c.lower() for c in cands)
                    best_lower = counter.most_common(1)[0][0]
                    best_orig  = next((c for c in cands if c.lower() == best_lower), cands[0])
                    final_answers[idx] = best_orig
                    # Real confidence = vote fraction
                    confidences[idx]   = round(counter[best_lower] / len(cands), 3)

            all_empty = all(
                not re.search(r'Most consistent\s*:', r, re.IGNORECASE) for r in samsaya_raw
            )
            if all_empty and samsaya_raw:
                raise AssertionError(
                    f"[SAMSAYA BUG] All outputs empty. Sample: {samsaya_raw[0][:300]!r}"
                )
            logger.info(
                f"[SAMSAYA] dataset={dataset} triggered={len(samsaya_indices)} "
                f"structured_used={samsaya_fire_count}"
            )

        # ── Fallback repair ──────────────────────────────────────────────────
        repair_indices = [i for i, a in enumerate(final_answers) if _is_bad_answer(a)]
        if repair_indices:
            engine = get_engine()
            repair_prompts = [
                engine.apply_chat_template(
                    "Answer in 1-5 words only.",
                    f"Question: {samples[i]['question']}"
                )
                for i in repair_indices
            ]
            repair_raw = batch_infer(repair_prompts, method="direct")
            for i, raw in zip(repair_indices, repair_raw):
                repaired = _extract_answer(raw, dataset)
                if not _is_bad_answer(repaired):
                    final_answers[i] = repaired
                    confidences[i]   = 0.4

        elapsed = time.time() - t0

        # ── Assemble results ─────────────────────────────────────────────────
        results = []
        for i, s in enumerate(samples):
            ctx = s.get("context", [])

            # Evidence spans (token-overlap based)
            evidence_spans = self.chitta.extract_evidence_spans(final_answers[i], ctx)

            # Supporting facts (for HotpotQA SF-EM/F1 — proper gold matching)
            supporting_facts = self.chitta.extract_supporting_facts(
                final_answers[i], tarka_raw[i], ctx
            ) if dataset == "hotpotqa" else []

            results.append({
                "id":               s.get("id", i),
                "question":         s["question"],
                "gold":             s["answer"],
                "predicted":        final_answers[i],
                "tarka_raw":        tarka_raw[i],
                "tarka_answer":     tarka_answers[i],
                "grounded_answer":  grounded_answers[i],
                "confidence":       confidences[i],
                "q_type":           q_types[i],
                "evidence_spans":   evidence_spans,
                "supporting_facts": supporting_facts,   # ← new: for SF metrics
                "latency":          elapsed / len(samples),
                "method":           "antahkarana",
                "dataset":          dataset,
            })
        return results
