"""
ANTAHKARANA v7 — Manas Module (Router) + Chitta (Dual-Stream Retrieval)
Exact logic preserved from original notebook Cell 6.
"""

import numpy as np
from typing import List, Tuple

from PIL import Image

from .utils import extract_entities, free_gpu
from . import preprocessing  # import module — embedder is set after init_models()
from .data_loader import VQASample
from config import (
    VISUAL_KW, TEXT_KW, MATH_KW, COMPARE_KW, VERIFY_KW, MCHOICE_KW,
    DATASET_OVERRIDES, VIS_BETA, ENTITY_LAMBDA, TOP_K_RETRIEVAL
)
from collections import Counter


def manas_route(question: str, has_image: bool, dataset_name: str) -> Tuple[str, set]:
    """
    Priority: text_reading > math > mchoice > comparison > verification
             > visual (only if ≥2 keywords) > simple (default)
    Defaulting to 'simple' (neutral prompt) avoids over-routing penalty.
    """
    if dataset_name in DATASET_OVERRIDES:
        return DATASET_OVERRIDES[dataset_name], extract_entities(question)
    q_lower  = question.lower()
    entities = extract_entities(question)
    if any(kw in q_lower for kw in TEXT_KW):    return 'text_reading', entities
    if any(kw in q_lower for kw in MATH_KW):    return 'math', entities
    if any(kw in q_lower for kw in MCHOICE_KW): return 'mchoice', entities
    if any(kw in q_lower for kw in COMPARE_KW): return 'comparison', entities
    if any(kw in q_lower for kw in VERIFY_KW):  return 'verification', entities
    vis_hits = sum(1 for kw in VISUAL_KW if kw in q_lower)
    if has_image and vis_hits >= 2:             return 'visual', entities
    return 'simple', entities


def clean_context(context: str) -> str:
    """Strip list markers and roman numerals from retrieved context
    to prevent BLIP-2 from echoing them as answers.
    Fixes 39 known garbage predictions: (4)., iii., [ii], iv., (iv)., etc."""
    import re
    # Remove standalone roman numeral tokens
    context = re.sub(r'\b(i{1,4}|vi{0,3}|ix|iv|x{0,3})(\.|\)|\])', '', context, flags=re.IGNORECASE)
    # Remove numbered list markers like (4). [3] 4.
    context = re.sub(r'[\(\[]\d+[\)\]]\.?', '', context)
    context = re.sub(r'\b\d+\.\s', ' ', context)
    # Remove "not enough information" phrases from context
    context = re.sub(r'not enough information', '', context, flags=re.IGNORECASE)
    context = re.sub(r'\s+', ' ', context).strip()
    return context


def build_prompt(question: str, q_type: str, context: str = '', choices_str: str = '') -> str:
    ctx = f'Context: {clean_context(context)}\n' if context else ''
    if q_type == 'text_reading':
        return f'{ctx}Question: {question}\nRead the text in the image and answer:'
    elif q_type == 'math':
        # V7-C: explicit counting instruction — this is CoT's edge on math questions
        return f'{ctx}Question: {question}\nCount every relevant object carefully. Answer with a number:'
    elif q_type == 'comparison':
        return f'{ctx}Question: {question}\nAnswer concisely:'
    elif q_type == 'verification':
        return f'{ctx}Question: {question}\nAnswer:'
    elif q_type == 'mchoice':
        if choices_str:
            return (f'{ctx}Question: {question}\n'
                    f'Options: {choices_str}\n'
                    f'Answer with ONLY a single letter (A, B, C, or D):')
        return f'{ctx}Question: {question}\nAnswer with one word or letter:'
    elif q_type == 'visual':
        return f'{ctx}Question: {question}\nAnswer briefly:'
    else:  # simple
        opts = f'\nOptions: {choices_str}' if choices_str else ''
        # V7-C: if context exists, make the model use it explicitly
        if context:
            return f'Context: {context}\nUsing the context above, answer: {question}{opts}\nAnswer:'
        return f'Question: {question}{opts}\nAnswer:'


def build_pass2_prompt(q: str, ans_p1: str, q_type: str, choices_str: str = '') -> str:
    """FIX H: fresh prompts for mchoice/text_reading — no failed P1 echo (anchoring bias)."""
    if q_type == 'text_reading':
        # FIX H: no failed attempt echo — BLIP-2 anchors to wrong text
        return f'Look carefully at all text visible in the image. Question: {q}\nShort answer:'
    elif q_type == 'mchoice':
        opts = f'\nOptions: {choices_str}' if choices_str else ''
        # FIX H: no failed attempt; clean re-prompt with options only
        return f'Question: {q}{opts}\nAnswer ONLY with the exact text of the correct option:'
    elif q_type == 'math':
        return f'Question: {q}\nCount carefully, answer with a number only:'
    elif q_type == 'comparison':
        return f'Look at the image. Question: {q}\nCompare carefully and answer briefly:'
    else:
        # For other types, P1 hint is helpful rather than anchoring
        return f'Question: {q}\nFirst attempt: {ans_p1}\nImproved answer in 1-5 words:'


def smart_truncate_context(context: str, max_chars: int = 280) -> str:
    if len(context) <= max_chars: return context
    pairs = context.split(' | ')
    result, used = [], 0
    for pair in pairs:
        if used + len(pair) + 3 > max_chars: break
        result.append(pair); used += len(pair) + 3
    return ' | '.join(result) if result else context[:max_chars]


class ChittaRetriever:
    def __init__(self, samples: List[VQASample], beta=VIS_BETA, lam=ENTITY_LAMBDA, top_k=TOP_K_RETRIEVAL):
        self.beta, self.lam, self.top_k = beta, lam, top_k
        self.questions = [s.question for s in samples]
        self.answers   = [Counter(s.answers).most_common(1)[0][0] if s.answers else '' for s in samples]
        self.images    = [s.image for s in samples]
        self._build_text_index()
        self._build_visual_index()

    def _build_text_index(self):
        print('    Chitta: text index...', end=' ', flush=True)
        self.txt_embs = preprocessing.embedder.encode(
            self.questions, batch_size=128, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True
        )
        print(f'{len(self.questions)} passages ✓')

    def _build_visual_index(self):
        print('    Chitta: visual index...', end=' ', flush=True)
        vis_embs = []
        # FIX #10: batch visual embeddings for index building (not per-image)
        for i in range(0, len(self.images), 32):
            vis_embs.append(preprocessing.get_visual_embedding(self.images[i:i+32]))
            free_gpu()
        self.vis_embs = np.vstack(vis_embs)
        norms = np.linalg.norm(self.vis_embs, axis=1, keepdims=True) + 1e-8
        self.vis_embs /= norms
        print('done ✓')

    def retrieve(self, query_question: str, query_img: Image.Image,
                 query_entities: set, has_image: bool = True, query_idx: int = -1) -> str:
        try:
            q_txt      = preprocessing.embedder.encode([query_question], normalize_embeddings=True, convert_to_numpy=True)[0]
            txt_scores = self.txt_embs @ q_txt
            if has_image:
                q_vis = preprocessing.get_visual_embedding([query_img])[0]
                q_vis /= (np.linalg.norm(q_vis) + 1e-8)
                vis_scores = self.vis_embs @ q_vis
            else:
                vis_scores = np.zeros(len(self.questions))
            entity_scores = np.array([
                len(query_entities & extract_entities(q)) / max(len(query_entities | extract_entities(q)), 1)
                for q in self.questions
            ])
            combined   = txt_scores + self.beta * vis_scores + self.lam * entity_scores
            sorted_idx = np.argsort(combined)[::-1]
            top_idx    = [i for i in sorted_idx if i != query_idx][:self.top_k]
            return ' | '.join(f'Q: {self.questions[i]} A: {self.answers[i]}' for i in top_idx)
        except Exception as e:
            print(f'  Chitta error: {e}'); return ''


def build_chitta_index(all_samples: List[VQASample]) -> ChittaRetriever:
    """Build the Chitta retrieval index from loaded samples."""
    print('Building Chitta retrieval index...')
    chitta = ChittaRetriever(all_samples)
    print('✅ Chitta ready.')
    return chitta
