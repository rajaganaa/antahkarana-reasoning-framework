"""
ANTAHKARANA v7 — Core Model (Buddhi 3-Pass Pipeline + Ahamkara + Sakshi)
Exact logic preserved from original notebook Cell 7.
"""

import re
import time
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path

from .utils import (
    postprocess_answer, is_bad_answer, vqa_soft_score, exact_match,
    exact_match_with_choices, is_hallucination, weighted_majority_vote,
    free_gpu, build_subquestion, fix_binary_answer, truncate_verbose_answer
)
from .preprocessing import blip2_generate
from .manas_module import (
    manas_route, build_prompt, build_pass2_prompt, smart_truncate_context
)
from config import SC_N_PASSES, SC_TEMPERATURE, DEPTH_CAP
from .utils import normalize_answer


@dataclass
class SampleResult:
    qid: str; dataset: str; question: str; predicted: str; ground_truth: List[str]
    soft_score: float = 0.0; is_exact: bool = False; is_hallucination: bool = False
    latency_s: float = 0.0; model_calls: int = 1; q_type: str = 'simple'
    pass2_fired: bool = False; pass3_fired: bool = False
    consistency_score: float = 1.0; evidence_span: str = ''; condition: str = 'full'


class Sakshi:
    def __init__(self): self.log = []
    def record(self, r: SampleResult):
        self.log.append({'qid':r.qid,'dataset':r.dataset,'q_type':r.q_type,
                         'lat':r.latency_s,'pass2':r.pass2_fired,
                         'pass3':r.pass3_fired,'calls':r.model_calls})
    def save(self, path: Path):
        with open(path,'w') as f:
            for entry in self.log: f.write(json.dumps(entry)+'\n')


# Global Sakshi logger instance
sakshi_logger = Sakshi()


def run_buddhi_full(images, questions, qids, gt_answers, dataset_names,
                    sample_indices=None, all_samples=None, chitta=None) -> List[SampleResult]:
    """
    Full Antahkarana v7: Manas → Chitta → Buddhi(P1[+P1.5]+P2+P3) → Ahamkara → Sakshi

    v7 fixes:
      V7-D: P2 blocked for ScienceQA comparison/simple/mchoice (was giving 0%)
      V7-E: Math gets visual sub-question (P1.5) before main answer — closes CoT gap
      V7-F: postprocess_answer applied to all BLIP-2 outputs
      V7-C: Math prompt & simple context-injection already in build_prompt
    """
    t0 = time.time()
    n  = len(images)
    if sample_indices is None: sample_indices = [-1] * n

    # ── Step 1: Chitta retrieval + Manas routing ──────────────────────────────
    contexts, q_types, choices_strs, choices_lists = [], [], [], []
    for i in range(n):
        s_idx = sample_indices[i]
        q_type, seed_ents = manas_route(questions[i], images[i] is not None, dataset_names[i])
        # V8-B: OKVQA requires world knowledge — retrieved visual Q&A pairs
        # inject wrong facts. Confirmed: no_retrieval beats antahkarana 29.6% vs 26.4%.
        # V10-A: VQAv2 retrieval PROVEN harmful — 30/31 regressions caused by retrieval.
        #   no_retrieval VQAv2=67.2% vs antahkarana=61.6% (+5.6pp without retrieval).
        #   yes/no: -7.6pp from retrieval, other: -7.1pp from retrieval.
        #   Keep retrieval ONLY for text_reading where OCR context genuinely helps.
        SKIP_RETRIEVAL_DATASETS = {'okvqa'}
        skip_retrieval = (dataset_names[i] in SKIP_RETRIEVAL_DATASETS
                         or (dataset_names[i] == 'vqav2' and q_type != 'text_reading'))
        if skip_retrieval:
            ctx_str = ''
        else:
            ctx     = chitta.retrieve(questions[i], images[i], seed_ents,
                                      has_image=(images[i] is not None), query_idx=s_idx)
            max_ctx = 280
            ctx_str = smart_truncate_context(ctx, max_ctx)

        # V8-C: Math/counting needs pure visual focus — context adds wrong object counts
        if q_type == 'math':
            ctx_str = ''  # force visual-only for counting questions
        cs  = getattr(all_samples[s_idx], 'choices_str',  '') if 0 <= s_idx < len(all_samples) else ''
        cl  = getattr(all_samples[s_idx], 'choices_list', []) if 0 <= s_idx < len(all_samples) else []
        contexts.append(ctx_str); q_types.append(q_type)
        choices_strs.append(cs); choices_lists.append(cl)

    # ── Step 1.5: Visual sub-question (V7-E + V8-K) ───────────────────────────
    # V7-E: Math gets visual sub-Q (counting). CoT fires sub-Q for ALL types.
    # V8-K: Extend to GQA simple/comparison — closes 8pp gap vs CoT.
    # GQA questions are scene-understanding; visual grounding helps enormously.
    subq_idx = [i for i in range(n) if q_types[i] == 'math'
                or (dataset_names[i] == 'gqa' and q_types[i] in ('simple', 'comparison', 'verification'))]
    math_ctx_extra = [''] * n
    if subq_idx:
        subq_imgs   = [images[i] for i in subq_idx]
        subq_prompts = [f'{build_subquestion(questions[i], q_types[i])}\nAnswer:' for i in subq_idx]
        subq_ans = blip2_generate(subq_imgs, subq_prompts)
        free_gpu()
        for j, i in enumerate(subq_idx):
            sub = postprocess_answer(subq_ans[j])
            if sub and not is_bad_answer(sub, q_types[i]):
                math_ctx_extra[i] = f'Visual: {sub}'

    # ── Step 2: BATCH Pass 1 ─────────────────────────────────────────────────
    p1_prompts = []
    for i in range(n):
        ctx_combined = contexts[i]
        if math_ctx_extra[i]:
            ctx_combined = f'{math_ctx_extra[i]} | {ctx_combined}' if ctx_combined else math_ctx_extra[i]
        p1_prompts.append(build_prompt(questions[i], q_types[i], ctx_combined, choices_strs[i]))
    ans_p1_raw = blip2_generate(images, p1_prompts)
    ans_p1 = [postprocess_answer(a, q_types[i]) for i, a in enumerate(ans_p1_raw)]  # V7-F + V8-J
    ans_p1 = [fix_binary_answer(ans_p1[i], questions[i]) for i in range(n)]  # V8-E
    ans_p1 = [truncate_verbose_answer(ans_p1[i], q_types[i]) for i in range(n)]  # V8-G
    free_gpu()

    # ── Step 3: BATCH Pass 2 — conservative + ScienceQA guard (V7-D) ─────────
    # V7-D: ScienceQA comparison/simple/mchoice → P2 was giving 0%, block it
    # Only fire P2 when: (is_bad_answer) OR (text_reading type) but NOT for scienceqa non-text
    JUNK_PATTERNS = {
        'unanswerable', 'not enough', 'cannot determine', 'i do not know',
        'not possible', 'unclear', 'unknown', 'no information',
    }
    JUNK_REGEX = re.compile(
        r'^[\(\[]?(?:i{1,4}|vi{0,3}|ix|iv|x{0,3})[\)\].]$|'   # roman numerals
        r'^[\(\[]\d+[\)\]]\.?$|'                                  # (4). [3].
        r'^\d+\.$',                                               # 4.
        re.IGNORECASE
    )

    def should_fire_p2(i):
        pred = ans_p1[i].strip().lower()
        ds = dataset_names[i]
        qt = q_types[i]
        # Never fire P2 for ScienceQA non-text (V7-D preserved)
        if ds == 'scienceqa' and qt not in ('text_reading',):
            return False
        # V8-L: Never fire P2 for GQA — 0% success rate in V7 (5 fired, 0 correct)
        if ds == 'gqa':
            return False
        # Check for junk patterns
        is_junk_phrase = any(j in pred for j in JUNK_PATTERNS)
        is_junk_token  = bool(JUNK_REGEX.match(pred))
        is_empty       = not pred or not normalize_answer(pred)
        # V9-C: Only fire if genuinely bad — removed unconditional text_reading trigger
        # (text_reading P2 was producing roman numeral junk, hurting 3 correct answers)
        return is_junk_phrase or is_junk_token or is_empty

    p2_mask  = [should_fire_p2(i) for i in range(n)]
    ans_p2   = list(ans_p1)
    p2_fired = list(p2_mask)

    p2_idx = [i for i in range(n) if p2_mask[i]]
    if p2_idx:
        p2_imgs    = [images[i] for i in p2_idx]
        p2_prompts = [build_pass2_prompt(questions[i], ans_p1[i], q_types[i], choices_strs[i])
                      for i in p2_idx]
        p2_res_raw = blip2_generate(p2_imgs, p2_prompts)
        p2_res = [postprocess_answer(a, q_types[p2_idx[j]]) for j, a in enumerate(p2_res_raw)]  # V7-F + V8-J
        free_gpu()
        for j, i in enumerate(p2_idx):
            cand       = p2_res[j]
            # V8-N: Apply binary/verbose post-processing to P2 outputs too
            cand = fix_binary_answer(cand, questions[i])
            cand = truncate_verbose_answer(cand, q_types[i])
            # V9-B: NEVER accept P2 output if it's junk (roman numerals, empty, etc.)
            cand_lower = cand.strip().lower()
            is_cand_junk = (bool(JUNK_REGEX.match(cand_lower))
                           or any(jp in cand_lower for jp in JUNK_PATTERNS)
                           or not cand.strip() or not normalize_answer(cand))
            if is_cand_junk:
                continue  # keep P1 answer, P2 produced garbage
            p1_was_bad = is_bad_answer(ans_p1[i], q_types[i])
            if p1_was_bad:
                ans_p2[i] = cand  # P1 was bad, P2 is clean → accept P2
            # else: P1 was good → keep P1

    # ── Step 4: BATCH Pass 3 — SC fallback ───────────────────────────────────
    final_ans     = list(ans_p2)
    consistencies = [1.0] * n
    p3_fired      = [False] * n

    p3_idx = [i for i in range(n) if is_bad_answer(ans_p2[i], q_types[i])]
    if p3_idx:
        for _ in range(DEPTH_CAP):
            still_bad = [i for i in p3_idx if is_bad_answer(final_ans[i], q_types[i])]
            if not still_bad: break
            sc_pool    = {i: [] for i in still_bad}
            sb_imgs    = [images[i] for i in still_bad]
            sb_prompts = [build_prompt(questions[i], q_types[i], contexts[i], choices_strs[i])
                          for i in still_bad]
            for _ in range(SC_N_PASSES):
                sc_raw = blip2_generate(sb_imgs, sb_prompts, do_sample=True, temperature=SC_TEMPERATURE)
                for j, i in enumerate(still_bad):
                    sc_pool[i].append(postprocess_answer(sc_raw[j], q_types[i]))  # V7-F + V8-J
            free_gpu()
            for i in still_bad:
                p3_fired[i] = True
                voted, vs = weighted_majority_vote(sc_pool[i])
                # V8-N: Apply binary/verbose post-processing to P3 voted output
                voted = fix_binary_answer(voted, questions[i])
                voted = truncate_verbose_answer(voted, q_types[i])
                consistencies[i] = vs
                if not is_bad_answer(voted, q_types[i]): final_ans[i] = voted
            # Deterministic beam fallback
            still_bad2 = [i for i in p3_idx if is_bad_answer(final_ans[i], q_types[i])]
            if still_bad2:
                fb_imgs    = [images[i] for i in still_bad2]
                fb_prompts = [f'Answer in 1-4 words: {questions[i]}' for i in still_bad2]
                fb_res     = [postprocess_answer(a) for a in blip2_generate(fb_imgs, fb_prompts)]
                for j, i in enumerate(still_bad2):
                    if not is_bad_answer(fb_res[j], q_types[i]): final_ans[i] = fb_res[j]
                free_gpu()

    # ── Step 5: Score ─────────────────────────────────────────────────────────
    per_s   = (time.time() - t0) / n
    results = []
    for i in range(n):
        soft = vqa_soft_score(final_ans[i], gt_answers[i], dataset_names[i],
                               choices=choices_lists[i] if dataset_names[i] == 'scienceqa' else None)
        em   = (exact_match_with_choices(final_ans[i], gt_answers[i], choices_lists[i])
                if dataset_names[i] == 'scienceqa' else exact_match(final_ans[i], gt_answers[i]))
        hall = is_hallucination(final_ans[i], gt_answers[i],
                               dataset_name=dataset_names[i],
                               choices=choices_lists[i] if dataset_names[i] == 'scienceqa' else None)
        calls = 1 + int(bool(math_ctx_extra[i])) + int(p2_fired[i]) + int(p3_fired[i]) * SC_N_PASSES
        r = SampleResult(
            qid=qids[i], dataset=dataset_names[i], question=questions[i],
            predicted=final_ans[i], ground_truth=gt_answers[i],
            soft_score=soft, is_exact=em, is_hallucination=hall,
            latency_s=per_s, model_calls=calls, q_type=q_types[i],
            pass2_fired=p2_fired[i], pass3_fired=p3_fired[i],
            consistency_score=consistencies[i], evidence_span=contexts[i][:100],
            condition='antahkarana',
        )
        sakshi_logger.record(r); results.append(r)
    return results
