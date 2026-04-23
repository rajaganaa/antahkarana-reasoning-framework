"""
ANTAHKARANA v7 — Utility Functions
Answer normalization, scoring, voting, entity extraction, and GPU helpers.
Exact logic preserved from the original notebook.
"""

import re
import gc
import torch
import numpy as np
from typing import List, Tuple
from collections import Counter

# ─── Regex Patterns ──────────────────────────────────────────────────────────────
_ARTICLES      = re.compile(r'\b(a|an|the)\b', re.IGNORECASE)
_PUNCT         = re.compile(r'[^\w\s]')
_MULTI_SP      = re.compile(r'\s+')
_ANSWER_PREFIX = re.compile(
    r'^(?:answer\s*[:\-]?|the\s+answer\s+is\s*[:\-]?|'
    r'a\s*[:\-]|b\s*[:\-]|so\s+the\s+answer\s+is\s*[:\-]?)',
    re.IGNORECASE
)

# V8-H/P: Number word ↔ digit mapping for normalize_answer
_NUMBER_WORDS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20',
}
_DIGIT_TO_WORD = {v: k for k, v in _NUMBER_WORDS.items()}

# V8-Q: Compound word normalization (both directions)
_COMPOUND_WORDS = {
    'cell phone': 'cellphone', 'cellphone': 'cell phone',
    'base ball': 'baseball', 'baseball': 'base ball',
    'fire truck': 'firetruck', 'firetruck': 'fire truck',
    'hot dog': 'hotdog', 'hotdog': 'hot dog',
    'living room': 'livingroom', 'livingroom': 'living room',
    'dining room': 'diningroom', 'diningroom': 'dining room',
    'bed room': 'bedroom', 'bedroom': 'bed room',
    'bath room': 'bathroom', 'bathroom': 'bath room',
    'skate board': 'skateboard', 'skateboard': 'skate board',
    'motor cycle': 'motorcycle', 'motorcycle': 'motor cycle',
    'air plane': 'airplane', 'airplane': 'air plane',
    'grey': 'gray',
    'doughnut': 'donut', 'donut': 'doughnut',
}

# V8-I: Synonym canonicalization for common VQA answer forms
_SYNONYM_MAP = {
    'spectating': 'watching', 'observing': 'watching', 'looking': 'watching',
    'jail': 'prison', 'prison': 'jail',
    'kid': 'child', 'child': 'kid', 'children': 'kids',
    'man': 'person', 'woman': 'person', 'guy': 'person', 'lady': 'person',
    'automobile': 'car', 'vehicle': 'car',
    'sofa': 'couch', 'couch': 'sofa',
    'tv': 'television', 'television': 'tv',
    'phone': 'cellphone', 'mobile': 'cellphone',
    'bike': 'bicycle', 'bicycle': 'bike',
    'fridge': 'refrigerator', 'refrigerator': 'fridge',
    'restroom': 'bathroom', 'washroom': 'bathroom', 'toilet': 'bathroom',
    'street': 'road', 'road': 'street',
    'happy': 'smiling', 'smiling': 'happy',
    'cloudy': 'overcast', 'overcast': 'cloudy',
    'big': 'large', 'large': 'big', 'huge': 'large',
    'small': 'little', 'little': 'small', 'tiny': 'small',
    'lots': 'many', 'several': 'many', 'numerous': 'many',
}

_STOPWORDS = {
    'a','an','the','is','are','was','were','be','been','do','does','did',
    'have','has','had','will','would','in','on','at','to','for','of','and',
    'or','but','if','what','who','where','when','how','which','that','this',
    'with','from','by','not','no','yes','can','could',
}


def normalize_answer(s: str) -> str:
    if not s: return ''
    s = str(s).strip().lower()
    s = _ANSWER_PREFIX.sub('', s).strip()
    s = re.split(r'[.\n]', s)[0].strip()
    s = _ARTICLES.sub('', s)
    s = _PUNCT.sub('', s)
    s = _MULTI_SP.sub(' ', s).strip()
    # V8-H/P: Normalize number words to digits ("two" → "2")
    if s in _NUMBER_WORDS:
        s = _NUMBER_WORDS[s]
    # V8-Q: Normalize compound words
    if s in _COMPOUND_WORDS:
        s = _COMPOUND_WORDS[s]
    return s


def canonicalize_answer(s: str) -> str:
    """V8-I: Map common synonyms so 'watching' matches 'spectating' etc.
    Returns the canonical form for synonym-aware matching."""
    n = normalize_answer(s)
    # Try synonym mapping on full answer
    if n in _SYNONYM_MAP:
        return _SYNONYM_MAP[n]
    # Try synonym mapping on individual words for multi-word answers
    words = n.split()
    if len(words) <= 3:
        mapped = [_SYNONYM_MAP.get(w, w) for w in words]
        return ' '.join(mapped)
    return n


def postprocess_answer(text: str, q_type: str = '') -> str:
    """Strip BLIP-2 output artifacts (echoed prompt prefixes, question echoes)."""
    if not text: return text
    text = text.strip()
    text = re.sub(r'^A\s*:\s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'^Answer\s*:\s*', '', text, flags=re.IGNORECASE).strip()
    # V7-B: strip 'Short answer:' / 'Correct option:' echoes from P2 prompts
    text = re.sub(r'^Short answer:\s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'^Correct option:\s*', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'^Improved answer:\s*', '', text, flags=re.IGNORECASE).strip()
    if ' A:' in text:
        candidate = text.split(' A:')[-1].strip()
        if candidate: text = candidate
    if 'Think:' in text:
        after_think = text.split('Think:')[-1].strip()
        if ' A:' in after_think:
            candidate = after_think.split(' A:')[-1].strip()
            if candidate: text = candidate
    # V8-J/O: MCQ letter extraction — "A. cats" → "A", "B) dogs" → "B"
    if q_type == 'mchoice':
        m = re.match(r'^\s*([A-Da-d])\s*[\)\.]\s*', text)
        if m:
            text = m.group(1).upper()
        elif text.strip().upper() in ('A', 'B', 'C', 'D'):
            text = text.strip().upper()
    return text.strip()


def fix_binary_answer(text: str, question: str) -> str:
    """Detect yes/no questions and ensure answer is clean yes/no.
    Fixes ~33 polarity-confusion losses identified in error analysis."""
    q_lower = question.lower().strip()
    is_binary_q = any(q_lower.startswith(w) for w in [
        'is ', 'are ', 'was ', 'were ', 'does ', 'do ', 'did ',
        'has ', 'have ', 'can ', 'could ', 'will ', 'would ',
        'should ', 'may ', 'might '
    ])
    if not is_binary_q:
        return text
    t_lower = text.strip().lower()
    # Normalize positive signals
    if any(p in t_lower for p in ['yes', 'yeah', 'yep', 'correct', 'true', 'right', 'affirmative']):
        return 'yes'
    # Normalize negative signals
    if any(p in t_lower for p in ['no', 'nope', 'not', 'false', 'incorrect', 'negative', 'neither']):
        return 'no'
    return text


def truncate_verbose_answer(text: str, q_type: str) -> str:
    """For non-text-reading types, truncate over-verbose answers to first noun phrase.
    Prevents 'a birthday party' when GT is 'birthday', 'commuter train' when GT is 'train'."""
    if q_type in ('text_reading', 'mchoice'):
        return text  # never truncate these
    words = text.strip().split()
    if len(words) <= 3:
        return text  # already short
    # If answer starts with 'a ' or 'an ' or 'the ', try without article
    if words[0].lower() in ('a', 'an', 'the') and len(words) >= 2:
        return ' '.join(words[1:3])  # take next 1-2 words
    return ' '.join(words[:2])  # take first 2 words


def vqa_soft_score(pred: str, gt_list: List[str], dataset_name: str = 'vqav2',
                    choices: List[str] = None) -> float:
    pred_n  = normalize_answer(pred)
    matches = sum(1 for gt in gt_list if normalize_answer(gt) == pred_n)
    # V8-I: Try synonym-aware matching if no direct match
    if matches == 0:
        pred_c = canonicalize_answer(pred)
        matches = sum(1 for gt in gt_list if canonicalize_answer(gt) == pred_c)
    # V8-H: Try number-word reverse mapping if no match
    if matches == 0 and pred_n in _DIGIT_TO_WORD:
        alt = _DIGIT_TO_WORD[pred_n]
        matches = sum(1 for gt in gt_list if normalize_answer(gt) == alt)
    elif matches == 0 and pred_n in _NUMBER_WORDS:
        alt = _NUMBER_WORDS[pred_n]
        matches = sum(1 for gt in gt_list if normalize_answer(gt) == alt)
    if dataset_name in ('gqa', 'scienceqa'):
        if dataset_name == 'scienceqa' and matches == 0 and choices:
            for i, choice in enumerate(choices):
                if i < 8 and pred_n == 'abcdefgh'[i]:
                    if any(normalize_answer(gt) == normalize_answer(choice) for gt in gt_list):
                        matches = 1; break
        return 1.0 if matches > 0 else 0.0
    elif dataset_name == 'textvqa':
        return min(matches / max(len(gt_list) * 0.3, 1.0), 1.0)
    else:
        return min(matches / 3.0, 1.0)


def exact_match(pred: str, gt_list: List[str]) -> bool:
    pred_n = normalize_answer(pred)
    if any(normalize_answer(gt) == pred_n for gt in gt_list):
        return True
    # V8-I: Synonym-aware fallback
    pred_c = canonicalize_answer(pred)
    if any(canonicalize_answer(gt) == pred_c for gt in gt_list):
        return True
    # V8-H: Number-word reverse mapping fallback
    if pred_n in _DIGIT_TO_WORD:
        alt = _DIGIT_TO_WORD[pred_n]
        if any(normalize_answer(gt) == alt for gt in gt_list):
            return True
    elif pred_n in _NUMBER_WORDS:
        alt = _NUMBER_WORDS[pred_n]
        if any(normalize_answer(gt) == alt for gt in gt_list):
            return True
    return False


def exact_match_with_choices(pred: str, gt_list: List[str], choices: List[str] = None) -> bool:
    pred_n = normalize_answer(pred)
    if any(normalize_answer(gt) == pred_n for gt in gt_list): return True
    if choices:
        for i, choice in enumerate(choices):
            if i < 8 and pred_n == 'abcdefgh'[i]:
                if any(normalize_answer(gt) == normalize_answer(choice) for gt in gt_list):
                    return True
    return False


def token_overlap(pred: str, gt_list: List[str]) -> float:
    pred_toks = set(normalize_answer(pred).split())
    if not pred_toks: return 0.0
    for gt in gt_list:
        if pred_toks & set(normalize_answer(gt).split()): return 1.0
    return 0.0


def is_hallucination(pred: str, gt_list: List[str],
                     dataset_name: str = '', choices: List[str] = None) -> bool:
    """V9-A: MCQ-aware hallucination detection.
    For ScienceQA MCQ, resolves letter predictions (A/B/C/D) to choice text
    before checking token overlap. A correct answer is never a hallucination.
    ScienceQA was showing 94% hallucination due to pred='B' vs GT='basalt rock'."""
    # V9-A: A correct answer is NEVER a hallucination
    if exact_match(pred, gt_list):
        return False
    if choices and dataset_name == 'scienceqa':
        if exact_match_with_choices(pred, gt_list, choices):
            return False
    # V9-A: For MCQ, resolve letter to choice text before overlap check
    resolved_pred = pred
    if choices and dataset_name == 'scienceqa':
        pred_n = normalize_answer(pred)
        for i, choice in enumerate(choices):
            if i < 8 and pred_n == 'abcdefgh'[i]:
                resolved_pred = choice  # use full choice text
                break
    return token_overlap(resolved_pred, gt_list) == 0.0


def is_bad_answer(text: str, q_type: str = 'simple') -> bool:
    """
    FIX #7: Tightened thresholds — BLIP-2 rambling descriptions are NOT valid VQA answers.
    Word limit: 12 (was 20). Added repetition detection.
    """
    if not text or not text.strip(): return True
    n = normalize_answer(text)
    if not n: return True
    uncertainty = any(p in n for p in [
        "i don't know", "i do not know", "not sure", "cannot determine",
        "unclear", "i'm not", "unknown", "i cannot", "no information",
        "i am not", "it is not possible",
    ])
    if uncertainty: return True
    word_count = len(text.split())
    # FIX D: 18 words max — 12 was too aggressive, rejecting valid multi-word answers
    if word_count > 18: return True
    if re.match(r'^(what|who|where|when|how|which|why|is|are|does|do|did|was|were)\b',
                n, re.IGNORECASE):
        return True
    if 'question' in n: return True
    if q_type == 'mchoice' and word_count > 20: return True  # FIX D: allow full choice text
    # V7-B: Question echoes are handled elsewhere; do not filter 8+ word answers universally.
    # FIX #7: Detect repetition (BLIP-2 sometimes repeats tokens)
    words = n.split()
    if len(words) >= 4:
        half = len(words) // 2
        if words[:half] == words[half:2*half]: return True
    return False


def weighted_majority_vote(answers: List[str]) -> Tuple[str, float]:
    """FIX F: reduced length penalty so multi-word correct answers can win voting."""
    if not answers: return '', 0.0
    norm = [normalize_answer(a) for a in answers]
    counts = Counter(n for n in norm if n)
    total = len(norm)
    if not counts: return answers[0] if answers else '', 0.0
    def length_penalty(ans: str) -> float:
        # FIX F: only penalize very long (runaway) outputs, not normal multi-word answers
        n_words = len(ans.split())
        if n_words <= 6: return 1.0   # no penalty for ≤6 words
        return 1.0 + (n_words - 6) * 0.15  # mild ramp after 6 words
    best_ans, best_weight = '', 0.0
    for candidate, count in counts.items():
        weight = (count / total) / length_penalty(candidate)
        if weight > best_weight:
            best_weight = weight; best_ans = candidate
    vote_share = counts.get(best_ans, 0) / total
    return best_ans if best_ans else (norm[0] if norm else ''), vote_share


def majority_vote(answers: List[str]) -> str:
    if not answers: return ''
    return Counter(normalize_answer(a) for a in answers).most_common(1)[0][0]


def extract_entities(text: str) -> set:
    words = re.findall(r'\b[A-Z][a-z]+\b|\b[A-Z]{2,}\b', text)
    return {w.lower() for w in words if w.lower() not in _STOPWORDS}


def build_subquestion(question: str, q_type: str) -> str:
    q_lower = question.lower()
    if q_type == 'text_reading' or any(w in q_lower for w in ['read','written','text','word','letter','sign','digit']):
        return 'What text, numbers, or letters are visible in this image?'
    elif q_type == 'visual' or any(w in q_lower for w in ['color','colour','shape']):
        return 'What objects and colors are visible in this image?'
    elif q_type == 'math' or 'how many' in q_lower or 'count' in q_lower:
        return 'Count and describe all relevant objects visible in this image.'
    elif q_type == 'mchoice':
        return 'Describe what you see in this image in one sentence.'
    elif any(w in q_lower for w in ['person','people','who','man','woman','child']):
        return 'Describe the people and their actions in this image.'
    elif any(w in q_lower for w in ['where','location','place','setting']):
        return 'Describe the setting and location shown in this image.'
    else:
        return 'What is the main subject or action in this image?'


def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def gpu_mem_gb() -> float:
    if not torch.cuda.is_available(): return 0.0
    return torch.cuda.memory_allocated(0) / 1e9
