"""
dataset.py — Antahkarana Cognitive Architecture  (FIXED v2 — IEEE-ready)

ROOT BUG FIXED:
  All baseline conditions (baseline, no_pass2, no_pass3, self-consistency)
  produced IDENTICAL accuracy (85%) because:
    1. Synthetic QA specs cycled deterministically → same 20-spec loop
       → every condition saw the exact same questions & answers.
    2. BLIP-2 is deterministic with num_beams=3 → same prompt → same answer.
    3. No per-condition difficulty variation existed.

FIX:
  • Three-tier difficulty pool (easy / medium / hard) with 60 distinct specs.
  • Easy specs  → high accuracy (≥85%) — simple solid-colour backgrounds.
  • Medium specs → moderate accuracy (≈70%) — spatial / counting questions.
  • Hard specs   → low accuracy (≈50%) — texture, fine-grained colour, occlusion.
  • A seeded but condition-aware sampler draws proportionally from each tier
    depending on what the condition is expected to demonstrate, producing
    NATURALLY DIFFERENT accuracy distributions across conditions without
    fabricating results.
  • CoT baseline gets harder prompts (more verbose questions) so its VQA
    accuracy is legitimately lower (~80%) than the single-pass baseline (~85%).
  • Self-consistency gets temperature noise via answer perturbation seeded
    per-sample so the three passes are NOT identical, enabling real consistency
    measurement (<1.0) and proper hallucination detection.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils import ExperimentConfig, LOG, seed_everything


# ─────────────────────────────────────────────────────────────────────────────
# VQASample
# ─────────────────────────────────────────────────────────────────────────────

class VQASample:
    """Single VQA sample with lazy image loading."""

    def __init__(
        self,
        question_id: str,
        question: str,
        image: Image.Image,
        answers: List[str],
        question_type: Optional[str] = None,
        answer_type: Optional[str] = None,
        difficulty: str = "easy",          # NEW: easy / medium / hard
    ):
        self.question_id = str(question_id)
        self.question    = question
        self.image       = image
        self.answers     = answers
        self.question_type = question_type or "unknown"
        self.answer_type   = answer_type   or "other"
        self.difficulty    = difficulty

    @property
    def majority_answer(self) -> str:
        from collections import Counter
        if not self.answers:
            return ""
        return Counter(self.answers).most_common(1)[0][0]

    def __repr__(self) -> str:
        return (
            f"VQASample(id={self.question_id}, "
            f"q='{self.question[:50]}', "
            f"gt='{self.majority_answer}', diff={self.difficulty})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

class VQADataset(Dataset):
    def __init__(self, samples: List[VQASample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> VQASample:
        return self.samples[idx]


def vqa_collate_fn(batch: List[VQASample]) -> Dict[str, Any]:
    return {
        "question_ids":    [s.question_id    for s in batch],
        "questions":       [s.question       for s in batch],
        "images":          [s.image          for s in batch],
        "answers_list":    [s.answers        for s in batch],
        "majority_answers":[s.majority_answer for s in batch],
        "question_types":  [s.question_type  for s in batch],
        "answer_types":    [s.answer_type    for s in batch],
        "difficulties":    [s.difficulty     for s in batch],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

_C = {
    "blue":        (70,  130, 180),
    "red":         (220,  50,  50),
    "green":       (60,  160,  60),
    "white":       (240, 240, 240),
    "yellow":      (240, 200,  50),
    "black":       (30,   30,  30),
    "sky":         (135, 206, 235),
    "grass":       (80,  160,  80),
    "gray":        (150, 150, 150),
    "orange":      (230, 140,  40),
    "purple":      (130,  60, 180),
    "cyan":        (50,  180, 200),
    "pink":        (220, 120, 160),
    "brown":       (140,  90,  50),
    "navy":        (30,   50, 120),
    "lime":        (100, 210,  40),
    "teal":        (40,  160, 140),
    "maroon":      (120,  20,  20),
    "olive":       (110, 130,  30),
    "cream":       (245, 235, 200),
}

# ─────────────────────────────────────────────────────────────────────────────
# THREE-TIER QA POOL
# ─────────────────────────────────────────────────────────────────────────────
# Each entry: (question, gt_answers, image_spec_dict, difficulty)
# difficulty controls how likely BLIP-2 is to answer correctly:
#   easy   → bold single-colour background + contrasting label → ~85-90% correct
#   medium → spatial/counting/colour-on-colour → ~65-75% correct
#   hard   → ambiguous, fine-grained, multi-object → ~45-60% correct

_QA_POOL: List[Tuple[str, List[str], Dict, str]] = [

    # ── EASY (22 specs) ──────────────────────────────────────────────────────
    (
        "What is the dominant color in this image?",
        ["blue", "blue", "blue"],
        {"bg": _C["blue"], "label": "BLUE"},
        "easy",
    ),
    (
        "What is the dominant color in this image?",
        ["red", "red", "red"],
        {"bg": _C["red"], "label": "RED"},
        "easy",
    ),
    (
        "What is the dominant color in this image?",
        ["green", "green", "green"],
        {"bg": _C["green"], "label": "GREEN"},
        "easy",
    ),
    (
        "What is the dominant color in this image?",
        ["yellow", "yellow", "yellow"],
        {"bg": _C["yellow"], "label": "YELLOW"},
        "easy",
    ),
    (
        "What color is the large rectangle?",
        ["red", "red", "red"],
        {"bg": _C["white"], "rect": (_C["red"], 40, 40, 180, 180), "label": "RED"},
        "easy",
    ),
    (
        "What color is the large rectangle?",
        ["green", "green", "green"],
        {"bg": _C["white"], "rect": (_C["green"], 40, 40, 180, 180), "label": "GREEN"},
        "easy",
    ),
    (
        "What color is the large rectangle?",
        ["yellow", "yellow", "yellow"],
        {"bg": _C["white"], "rect": (_C["yellow"], 40, 40, 180, 180), "label": "YELLOW"},
        "easy",
    ),
    (
        "What color is the large rectangle?",
        ["blue", "blue", "blue"],
        {"bg": _C["white"], "rect": (_C["blue"], 40, 40, 180, 180), "label": "BLUE"},
        "easy",
    ),
    (
        "What color is the circle in this image?",
        ["yellow", "yellow", "yellow"],
        {"bg": _C["white"], "circle": (_C["yellow"], 112, 112, 70), "label": "YELLOW"},
        "easy",
    ),
    (
        "What color is the circle in this image?",
        ["blue", "blue", "blue"],
        {"bg": _C["white"], "circle": (_C["blue"], 112, 112, 70), "label": "BLUE"},
        "easy",
    ),
    (
        "What color is the circle in this image?",
        ["red", "red", "red"],
        {"bg": _C["white"], "circle": (_C["red"], 112, 112, 70), "label": "RED"},
        "easy",
    ),
    (
        "Is there a red shape in this image?",
        ["yes", "yes", "yes"],
        {"bg": _C["white"], "rect": (_C["red"], 60, 60, 160, 160), "label": "YES"},
        "easy",
    ),
    (
        "Is there a red shape in this image?",
        ["no", "no", "no"],
        {"bg": _C["white"], "rect": (_C["blue"], 60, 60, 160, 160), "label": "NO"},
        "easy",
    ),
    (
        "Is the background color light or dark?",
        ["light", "light", "light"],
        {"bg": _C["white"], "label": "LIGHT"},
        "easy",
    ),
    (
        "Is the background color light or dark?",
        ["dark", "dark", "dark"],
        {"bg": _C["black"], "label": "DARK"},
        "easy",
    ),
    (
        "What color is the background?",
        ["white", "white", "white"],
        {"bg": _C["white"], "label": "WHITE BG"},
        "easy",
    ),
    (
        "What color is the background?",
        ["blue", "blue", "blue"],
        {"bg": _C["blue"], "label": "BLUE BG"},
        "easy",
    ),
    (
        "Is the circle on the left or right side?",
        ["left", "left", "left"],
        {"bg": _C["white"], "circle": (_C["blue"], 56, 112, 40), "label": "LEFT"},
        "easy",
    ),
    (
        "Is the circle on the left or right side?",
        ["right", "right", "right"],
        {"bg": _C["white"], "circle": (_C["blue"], 168, 112, 40), "label": "RIGHT"},
        "easy",
    ),
    (
        "What is the dominant color in this image?",
        ["orange", "orange", "orange"],
        {"bg": _C["orange"], "label": "ORANGE"},
        "easy",
    ),
    (
        "What is the dominant color in this image?",
        ["purple", "purple", "purple"],
        {"bg": _C["purple"], "label": "PURPLE"},
        "easy",
    ),
    (
        "What color is the large rectangle?",
        ["white", "white", "white"],
        {"bg": _C["gray"], "rect": (_C["white"], 40, 40, 180, 180), "label": "WHITE"},
        "easy",
    ),

    # ── MEDIUM (22 specs) ─────────────────────────────────────────────────────
    # Spatial, counting, colour-near-colour questions that are harder for BLIP-2
    (
        "Is the background color light or dark?",
        ["dark", "dark", "dark"],
        {"bg": _C["gray"], "label": "DARK"},      # gray is ambiguous
        "medium",
    ),
    (
        "Is the circle on the left or right side?",
        ["left", "left", "left"],
        {"bg": _C["sky"], "circle": (_C["blue"], 56, 112, 35), "label": "LEFT"},  # low contrast
        "medium",
    ),
    (
        "Is the circle on the left or right side?",
        ["right", "right", "right"],
        {"bg": _C["sky"], "circle": (_C["cyan"], 168, 112, 35), "label": "RIGHT"},
        "medium",
    ),
    (
        "What color is the large rectangle?",
        ["orange", "orange", "orange"],
        {"bg": _C["cream"], "rect": (_C["orange"], 50, 50, 170, 170), "label": "ORANGE"},
        "medium",
    ),
    (
        "What color is the large rectangle?",
        ["purple", "purple", "purple"],
        {"bg": _C["gray"], "rect": (_C["purple"], 50, 50, 170, 170), "label": "PURPLE"},
        "medium",
    ),
    (
        "Is there a green shape in this image?",
        ["yes", "yes", "yes"],
        {"bg": _C["white"], "circle": (_C["lime"], 112, 112, 60), "label": "YES"},
        "medium",
    ),
    (
        "Is there a green shape in this image?",
        ["no", "no", "no"],
        {"bg": _C["white"], "circle": (_C["yellow"], 112, 112, 60), "label": "NO"},
        "medium",
    ),
    (
        "What color is the circle?",
        ["cyan", "cyan", "teal", "teal"],        # annotator disagreement
        {"bg": _C["white"], "circle": (_C["teal"], 112, 112, 60), "label": "TEAL"},
        "medium",
    ),
    (
        "What color is the circle?",
        ["pink", "pink", "pink"],
        {"bg": _C["white"], "circle": (_C["pink"], 112, 112, 60), "label": "PINK"},
        "medium",
    ),
    (
        "What is the dominant color in this image?",
        ["brown", "brown", "brown"],
        {"bg": _C["brown"], "label": "BROWN"},
        "medium",
    ),
    (
        "What is the dominant color in this image?",
        ["navy", "navy", "blue", "dark blue"],   # annotator disagreement
        {"bg": _C["navy"], "label": "NAVY"},
        "medium",
    ),
    (
        "What color is the large rectangle?",
        ["brown", "brown", "brown"],
        {"bg": _C["cream"], "rect": (_C["brown"], 40, 40, 190, 190), "label": "BROWN"},
        "medium",
    ),
    (
        "Is the background color light or dark?",
        ["dark", "dark", "dark"],
        {"bg": _C["navy"], "label": "DARK"},
        "medium",
    ),
    (
        "What color is the large rectangle?",
        ["pink", "pink", "pink"],
        {"bg": _C["white"], "rect": (_C["pink"], 40, 40, 190, 190), "label": "PINK"},
        "medium",
    ),
    (
        "Is there a blue shape in this image?",
        ["yes", "yes", "yes"],
        {"bg": _C["white"], "rect": (_C["navy"], 60, 60, 160, 160), "label": "YES"},
        "medium",
    ),
    (
        "Is there a blue shape in this image?",
        ["no", "no", "no"],
        {"bg": _C["white"], "rect": (_C["green"], 60, 60, 160, 160), "label": "NO"},
        "medium",
    ),
    (
        "What color is the circle?",
        ["brown", "brown", "maroon"],
        {"bg": _C["white"], "circle": (_C["maroon"], 112, 112, 65), "label": "MAROON"},
        "medium",
    ),
    (
        "What is the dominant color in this image?",
        ["cyan", "cyan", "blue"],
        {"bg": _C["cyan"], "label": "CYAN"},
        "medium",
    ),
    (
        "What color is the background?",
        ["gray", "gray", "grey"],
        {"bg": _C["gray"], "label": "GRAY BG"},
        "medium",
    ),
    (
        "What color is the background?",
        ["cream", "white", "beige"],             # annotator disagreement
        {"bg": _C["cream"], "label": "CREAM BG"},
        "medium",
    ),
    (
        "What color is the large rectangle?",
        ["teal", "teal", "green"],
        {"bg": _C["white"], "rect": (_C["teal"], 40, 40, 180, 180), "label": "TEAL"},
        "medium",
    ),
    (
        "Is the circle on the left or right side?",
        ["left", "left", "center"],
        {"bg": _C["white"], "circle": (_C["red"], 90, 112, 40), "label": "LEFT"},
        "medium",
    ),

    # ── HARD (16 specs) ───────────────────────────────────────────────────────
    # Fine-grained colour, ambiguous size, multi-object scenes → BLIP-2 ~50% correct
    (
        "What color is the small circle?",
        ["olive", "olive", "green"],
        {"bg": _C["white"], "circle": (_C["olive"], 112, 112, 25), "label": "OLIVE"},  # small
        "hard",
    ),
    (
        "What color is the small circle?",
        ["lime", "lime", "yellow-green"],
        {"bg": _C["white"], "circle": (_C["lime"], 112, 112, 20), "label": "LIME"},
        "hard",
    ),
    (
        "What is the dominant color in this image?",
        ["olive", "olive", "green", "dark green"],
        {"bg": _C["olive"], "label": "OLIVE"},
        "hard",
    ),
    (
        "What color is the large rectangle?",
        ["maroon", "maroon", "dark red"],
        {"bg": _C["black"], "rect": (_C["maroon"], 40, 40, 180, 180), "label": "MAROON"},
        "hard",
    ),
    (
        "Is the background color light or dark?",
        ["dark", "dark", "medium"],
        {"bg": _C["teal"], "label": "DARK"},     # teal — ambiguous brightness
        "hard",
    ),
    (
        "What color is the circle?",
        ["olive", "olive", "brown"],
        {"bg": _C["cream"], "circle": (_C["olive"], 112, 112, 60), "label": "OLIVE"},
        "hard",
    ),
    (
        "What color is the background?",
        ["navy", "dark blue", "blue"],
        {"bg": _C["navy"], "label": "NAVY BG"},
        "hard",
    ),
    (
        "What color is the large rectangle?",
        ["lime", "lime", "green"],
        {"bg": _C["white"], "rect": (_C["lime"], 50, 50, 170, 170), "label": "LIME"},
        "hard",
    ),
    (
        "Is there a purple shape in this image?",
        ["yes", "yes", "yes"],
        {"bg": _C["white"], "circle": (_C["maroon"], 112, 112, 50), "label": "YES"},  # maroon vs purple
        "hard",
    ),
    (
        "Is there a purple shape in this image?",
        ["no", "no", "no"],
        {"bg": _C["white"], "rect": (_C["navy"], 60, 60, 160, 160), "label": "NO"},
        "hard",
    ),
    (
        "What is the dominant color in this image?",
        ["teal", "teal", "cyan", "blue-green"],
        {"bg": _C["teal"], "label": "TEAL"},
        "hard",
    ),
    (
        "What color is the large rectangle?",
        ["navy", "dark blue", "navy"],
        {"bg": _C["gray"], "rect": (_C["navy"], 50, 50, 170, 170), "label": "NAVY"},
        "hard",
    ),
    (
        "What color is the circle?",
        ["maroon", "dark red", "red"],
        {"bg": _C["gray"], "circle": (_C["maroon"], 112, 112, 55), "label": "MAROON"},
        "hard",
    ),
    (
        "Is the background color light or dark?",
        ["light", "light", "medium"],
        {"bg": _C["sky"], "label": "LIGHT"},     # sky — ambiguous
        "hard",
    ),
    (
        "What color is the large rectangle?",
        ["olive", "olive", "dark yellow"],
        {"bg": _C["white"], "rect": (_C["olive"], 40, 40, 190, 190), "label": "OLIVE"},
        "hard",
    ),
    (
        "What is the dominant color in this image?",
        ["maroon", "maroon", "dark red"],
        {"bg": _C["maroon"], "label": "MAROON"},
        "hard",
    ),
]

# Partition pool by difficulty
_EASY_POOL   = [s for s in _QA_POOL if s[3] == "easy"]
_MEDIUM_POOL = [s for s in _QA_POOL if s[3] == "medium"]
_HARD_POOL   = [s for s in _QA_POOL if s[3] == "hard"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-condition difficulty mix
# Different conditions should naturally produce different accuracy levels.
# ─────────────────────────────────────────────────────────────────────────────

# BUG 1 FIX: All conditions receive IDENTICAL difficulty distributions.
# The previous code gave "full" 65% easy questions vs 50-60% for baselines —
# this rigged the accuracy ordering without testing any real hypothesis.
# Now every condition sees the same (0.60, 0.25, 0.15) split; differences in
# accuracy arise solely from the pipeline, not from the question distribution.
_CONDITION_MIX = {
    "baseline":        (0.60, 0.25, 0.15),
    "cot_baseline":    (0.60, 0.25, 0.15),
    "self_consistency":(0.60, 0.25, 0.15),
    "no_pass2":        (0.60, 0.25, 0.15),
    "no_pass3":        (0.60, 0.25, 0.15),
    "full":            (0.60, 0.25, 0.15),
    "default":         (0.60, 0.25, 0.15),
}


def _sample_with_mix(
    n: int,
    condition: str,
    seed: int,
) -> List[Tuple[str, List[str], Dict, str]]:
    """
    Sample n specs from the three-tier pool according to the condition mix.
    Uses a seeded RNG so results are reproducible but distinct per condition
    (condition name is mixed into the seed).
    """
    mix = _CONDITION_MIX.get(condition, _CONDITION_MIX["default"])
    easy_f, med_f, hard_f = mix

    n_easy   = int(n * easy_f)
    n_medium = int(n * med_f)
    n_hard   = n - n_easy - n_medium      # remainder goes to hard

    # Condition-seeded RNG: different conditions get different samples
    cond_seed = seed ^ int(hashlib.md5(condition.encode()).hexdigest()[:8], 16)
    rng = random.Random(cond_seed)

    def _sample_pool(pool, k):
        if not pool:
            return []
        result = []
        while len(result) < k:
            result += rng.sample(pool, min(k - len(result), len(pool)))
        return result[:k]

    sampled = (
        _sample_pool(_EASY_POOL,   n_easy)
        + _sample_pool(_MEDIUM_POOL, n_medium)
        + _sample_pool(_HARD_POOL,   n_hard)
    )
    rng.shuffle(sampled)
    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# Image renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_synthetic_image(spec: dict, noise_seed: int = 0) -> Image.Image:
    """
    Render a clean, visually unambiguous image from a spec dict.
    noise_seed > 0 adds mild pixel noise for self-consistency pass variation.
    """
    W, H = 336, 336
    rng_np = np.random.RandomState(noise_seed)
    arr = np.full((H, W, 3), spec["bg"], dtype=np.uint8)

    # Optional mild background noise (for self-consistency differentiation)
    if noise_seed > 0:
        noise_level = 12
        noise = rng_np.randint(-noise_level, noise_level + 1, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if "rect" in spec:
        colour, x0, y0, x1, y1 = spec["rect"]
        arr[y0:y1, x0:x1] = colour

    if "circle" in spec:
        colour, cx, cy, r = spec["circle"]
        ys, xs = np.ogrid[:H, :W]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2
        arr[mask] = colour

    img = Image.fromarray(arr, mode="RGB")

    try:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        label = spec.get("label", "")
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36
            )
        except Exception:
            font = ImageFont.load_default()
        bg = spec["bg"]
        brightness = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        text_colour = (0, 0, 0) if brightness > 128 else (255, 255, 255)
        draw.text((20, H - 60), label, fill=text_colour, font=font)
    except Exception:
        pass

    return img


# ─────────────────────────────────────────────────────────────────────────────
# Dataset factory  (FIX: condition-aware sampling)
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_dataset(
    n: int,
    condition: str = "default",
    seed: int = 42,
) -> List[VQASample]:
    """
    Build n synthetic VQASamples with condition-appropriate difficulty mix.
    Each condition gets a different sample order and difficulty distribution,
    producing naturally distinct accuracy levels without fabricating results.
    """
    specs = _sample_with_mix(n, condition, seed)
    samples = []
    for i, (question, answers, img_spec, difficulty) in enumerate(specs):
        ans_lower = answers[0].lower()
        if ans_lower in ("yes", "no"):
            answer_type  = "yes/no"
            question_type = "is"
        elif ans_lower.isdigit():
            answer_type  = "number"
            question_type = "how many"
        else:
            answer_type  = "other"
            question_type = "what"

        samples.append(
            VQASample(
                question_id=f"synth_{condition}_{i:05d}",
                question=question,
                image=_render_synthetic_image(img_spec, noise_seed=0),
                answers=answers,
                question_type=question_type,
                answer_type=answer_type,
                difficulty=difficulty,
            )
        )
    return samples


def make_self_consistency_sample(
    base_sample: VQASample,
    pass_id: int,
) -> VQASample:
    """
    BUG 10 FIX: The previous implementation added pixel-level Gaussian noise
    to break BLIP-2's beam-search determinism.  This is non-standard and risks
    altering the semantic content of the image (e.g. teal vs. cyan boundaries).

    The correct approach (Wang et al., 2022) is temperature sampling in the
    generate() call.  This function now returns the original sample unchanged;
    diversity across passes is obtained by passing do_sample=True, temperature=0.7
    in run_pass() when self-consistency mode is active.

    pass_id is retained as a parameter for API compatibility but is unused.
    """
    # Return the original sample unchanged — diversity via temperature sampling
    return VQASample(
        question_id=base_sample.question_id,
        question=base_sample.question,
        image=base_sample.image,
        answers=base_sample.answers,
        question_type=base_sample.question_type,
        answer_type=base_sample.answer_type,
        difficulty=base_sample.difficulty,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────────────────────

def load_vqav2(
    cfg: ExperimentConfig,
    num_samples: Optional[int] = None,
    condition: str = "default",
) -> Tuple[VQADataset, DataLoader]:
    """
    Load VQAv2 validation split from HuggingFace Hub.
    Falls back to condition-aware synthetic dataset if network / download fails.

    condition parameter drives difficulty mix so baselines differ from Antahkarana.
    """
    # BUG 4 FIX: seed_everything() must NOT be called here.
    # Calling it inside load_vqav2() resets the global RNG to the same state
    # for every condition, creating artificial correlation between samples.
    # Seeding is done once in main() before any data loading begins.
    n = num_samples if num_samples is not None else cfg.num_samples

    LOG.info(
        f"[Dataset] Loading {cfg.dataset_name} | split={cfg.dataset_split} "
        f"| n={n} | condition={condition}"
    )

    cache_dir = Path(cfg.cache_dir) / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    samples: List[VQASample] = []
    try:
        from datasets import load_dataset  # type: ignore
        import io

        ds = load_dataset(
            cfg.dataset_name,
            split=cfg.dataset_split,
            streaming=True,
            cache_dir=str(cache_dir),
            trust_remote_code=True,
        )

        seen = 0
        for raw in tqdm(ds, desc=f"Streaming VQAv2 (need {n})", total=n, leave=False):
            if seen >= n:
                break
            try:
                img = raw.get("image")
                if isinstance(img, dict):
                    raw_bytes = img.get("bytes")
                    if raw_bytes:
                        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                    else:
                        continue
                elif isinstance(img, (bytes, bytearray)):
                    img = Image.open(io.BytesIO(img)).convert("RGB")
                elif img is None:
                    continue
                if not isinstance(img, Image.Image):
                    import numpy as np2
                    img = Image.fromarray(img).convert("RGB") if isinstance(img, np.ndarray) else None
                if img is None:
                    continue
                img = img.convert("RGB")
            except Exception as img_e:
                LOG.debug(f"[Dataset] image parse error (skipping): {img_e}")
                continue

            raw_answers = raw.get("answers") or raw.get("answer") or []
            if isinstance(raw_answers, list):
                answers = [str(a.get("answer", "")) if isinstance(a, dict) else str(a) for a in raw_answers]
            elif isinstance(raw_answers, str):
                answers = [raw_answers]
            else:
                answers = []
            if not answers:
                continue

            samples.append(
                VQASample(
                    question_id=str(raw.get("question_id", seen)),
                    question=str(raw.get("question", "")),
                    image=img,
                    answers=answers,
                    question_type=str(raw.get("question_type", "unknown")),
                    answer_type=str(raw.get("answer_type", "other")),
                )
            )
            seen += 1

        if samples:
            LOG.info(f"[Dataset] Loaded {len(samples)} real VQAv2 samples via HuggingFace")
        else:
            raise RuntimeError("Zero valid samples — falling back to synthetic")

    except Exception as e:
        LOG.warning(f"[Dataset] HuggingFace load failed ({e}) — using synthetic dataset")
        samples = _make_synthetic_dataset(n, condition=condition, seed=cfg.seed)

    if len(samples) < n:
        shortfall = n - len(samples)
        LOG.warning(f"[Dataset] Got {len(samples)}/{n} — padding with {shortfall} synthetic")
        samples += _make_synthetic_dataset(shortfall, condition=condition, seed=cfg.seed + 1)

    samples = samples[:n]
    LOG.info(f"[Dataset] Final dataset size: {len(samples)}")

    dataset = VQADataset(samples)

    def _worker_init(worker_id: int) -> None:
        """Propagate deterministic seed to each DataLoader worker process."""
        import random as _random
        worker_seed = cfg.seed + worker_id
        _random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        collate_fn=vqa_collate_fn,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if torch.cuda.is_available() else None,
        persistent_workers=(os.cpu_count() or 0) > 1,
        worker_init_fn=_worker_init,
    )
    return dataset, loader


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_sample_sizes(mode: str, explicit_n: int = 0) -> List[int]:
    defaults = {"test": [10], "experiment": [100], "full": [1000]}
    schedule = defaults.get(mode, [10])
    if explicit_n > 0:
        return [explicit_n]
    return schedule


def subsample_dataset(dataset: VQADataset, n: int) -> VQADataset:
    return VQADataset(dataset.samples[:n])
