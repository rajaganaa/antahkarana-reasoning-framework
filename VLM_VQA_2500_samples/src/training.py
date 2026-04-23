"""
ANTAHKARANA v7 — Training / Baselines & Ablation Pipelines
All 11 pipeline functions preserved from original notebook Cell 8.
"""

import time
from typing import List, Tuple
from collections import Counter

from .utils import (
    normalize_answer, is_bad_answer, vqa_soft_score, exact_match,
    exact_match_with_choices, is_hallucination, weighted_majority_vote,
    postprocess_answer, free_gpu
)
from .preprocessing import blip2_generate
from .manas_module import (
    manas_route, build_prompt, build_pass2_prompt, smart_truncate_context
)
from .antahkarana_model import SampleResult, sakshi_logger
from config import SC_N_PASSES, SC_TEMPERATURE, DEPTH_CAP


def _get_ctx(si, idx, img, q, entities, has_img, all_samples, chitta):
    s_idx = si[idx] if si else -1
    ctx   = chitta.retrieve(q, img, entities, has_image=has_img, query_idx=s_idx)
    cs    = getattr(all_samples[s_idx], 'choices_str',  '') if 0 <= s_idx < len(all_samples) else ''
    cl    = getattr(all_samples[s_idx], 'choices_list', []) if 0 <= s_idx < len(all_samples) else []
    return s_idx, ctx, cs, cl


def _score(pred, gts, ds, choices_list=None):
    soft = vqa_soft_score(pred, gts, ds, choices=choices_list if ds=='scienceqa' else None)
    em   = (exact_match_with_choices(pred, gts, choices_list)
            if ds=='scienceqa' else exact_match(pred, gts))
    # V9-A: Pass dataset and choices for MCQ-aware hallucination detection
    hall = is_hallucination(pred, gts, dataset_name=ds,
                           choices=choices_list if ds=='scienceqa' else None)
    return soft, em, hall


# ── Direct Prompting ─────────────────────────────────────────────────────────
# FIX #6: include choices in prompt for ScienceQA
# FIX #4: per-sample latency = batch_time / n (consistent with other conditions)
# FIX #5: single blip2_generate call for entire batch
def run_direct_prompting(images, questions, qids, gt_answers, dataset_names,
                         sample_indices=None, all_samples=None, chitta=None):
    si = sample_indices or [-1] * len(images)
    t0 = time.time()
    prompts = []
    all_cl  = []
    for i, (q, ds) in enumerate(zip(questions, dataset_names)):
        s_idx = si[i]
        cs = getattr(all_samples[s_idx], 'choices_str', '') if 0 <= s_idx < len(all_samples) else ''
        cl = getattr(all_samples[s_idx], 'choices_list', []) if 0 <= s_idx < len(all_samples) else []
        # FIX #6: ScienceQA must have choices even in direct baseline
        if ds == 'scienceqa' and cs:
            prompts.append(f'Question: {q}\nOptions: {cs}\nAnswer:')
        else:
            prompts.append(f'Question: {q}\nAnswer:')
        all_cl.append(cl)
    all_ans = blip2_generate(images, prompts)
    free_gpu()
    per_s = (time.time() - t0) / max(len(images), 1)
    results = []
    for i, (q, qid, gts, ds, ans) in enumerate(zip(questions, qids, gt_answers, dataset_names, all_ans)):
        soft, em, hall = _score(ans, gts, ds, all_cl[i])
        results.append(SampleResult(qid=qid, dataset=ds, question=q, predicted=ans,
            ground_truth=gts, soft_score=soft, is_exact=em, is_hallucination=hall,
            latency_s=per_s, model_calls=1, condition='direct'))
    return results


# ── CoT Baseline — BATCHED (FIX #5) ─────────────────────────────────────────
def run_cot_baseline(images, questions, qids, gt_answers, dataset_names,
                     sample_indices=None, all_samples=None, chitta=None):
    from .utils import build_subquestion
    si = sample_indices or list(range(len(images)))
    n  = len(images)

    # Batch route + get context
    q_types_l, ctx_l, cs_l, cl_l = [], [], [], []
    for i in range(n):
        q_type, ents = manas_route(questions[i], images[i] is not None, dataset_names[i])
        _, ctx, cs, cl = _get_ctx(si, i, images[i], questions[i], ents, images[i] is not None, all_samples, chitta)
        q_types_l.append(q_type); ctx_l.append(smart_truncate_context(ctx, 200))
        cs_l.append(cs); cl_l.append(cl)

    t0 = time.time()
    # Stage 1: batch sub-questions
    sub_prompts = [f'Question: {build_subquestion(questions[i], q_types_l[i])}\nAnswer:'
                   for i in range(n)]
    sub_ans = blip2_generate(images, sub_prompts)
    free_gpu()

    # Stage 2: batch main questions with visual context
    main_prompts = []
    for i in range(n):
        cot_ctx = f'{ctx_l[i]} | Visual: {sub_ans[i]}' if ctx_l[i] else f'Visual: {sub_ans[i]}'
        main_prompts.append(build_prompt(questions[i], q_types_l[i],
                                          smart_truncate_context(cot_ctx, 280), cs_l[i]))
    ans = blip2_generate(images, main_prompts)
    free_gpu()

    per_s = (time.time() - t0) / n
    results = []
    for i in range(n):
        soft, em, hall = _score(ans[i], gt_answers[i], dataset_names[i], cl_l[i])
        results.append(SampleResult(qid=qids[i], dataset=dataset_names[i], question=questions[i],
            predicted=ans[i], ground_truth=gt_answers[i], soft_score=soft, is_exact=em,
            is_hallucination=hall, latency_s=per_s, model_calls=2,
            q_type=q_types_l[i], condition='cot'))
    return results


# ── Self-Consistency — BATCHED (FIX #5) ─────────────────────────────────────
def run_self_consistency(images, questions, qids, gt_answers, dataset_names,
                          n_passes=SC_N_PASSES, sample_indices=None, all_samples=None, chitta=None):
    si = sample_indices or list(range(len(images)))
    n  = len(images)

    q_types_l, prompts_l, cl_l = [], [], []
    for i in range(n):
        q_type, ents = manas_route(questions[i], images[i] is not None, dataset_names[i])
        _, ctx, cs, cl = _get_ctx(si, i, images[i], questions[i], ents, images[i] is not None, all_samples, chitta)
        q_types_l.append(q_type); cl_l.append(cl)
        prompts_l.append(build_prompt(questions[i], q_type, smart_truncate_context(ctx, 280), cs))

    t0 = time.time()
    # n_passes stochastic runs, all batched
    answer_pool = [[] for _ in range(n)]
    for _ in range(n_passes):
        sc_res = blip2_generate(images, prompts_l, do_sample=True, temperature=SC_TEMPERATURE)
        for i in range(n): answer_pool[i].append(sc_res[i])
    free_gpu()

    final_ans = []
    for i in range(n):
        voted, vs = weighted_majority_vote(answer_pool[i])
        if vs < 0.35 or is_bad_answer(voted, q_types_l[i]):
            voted = blip2_generate([images[i]], [prompts_l[i]], do_sample=False)[0]
            free_gpu()
        final_ans.append(voted)

    per_s = (time.time() - t0) / n
    results = []
    for i in range(n):
        top_count = Counter(normalize_answer(a) for a in answer_pool[i]).most_common(1)[0][1]
        soft, em, hall = _score(final_ans[i], gt_answers[i], dataset_names[i], cl_l[i])
        results.append(SampleResult(qid=qids[i], dataset=dataset_names[i], question=questions[i],
            predicted=final_ans[i], ground_truth=gt_answers[i], soft_score=soft, is_exact=em,
            is_hallucination=hall, latency_s=per_s, model_calls=n_passes,
            consistency_score=top_count/n_passes, q_type=q_types_l[i], condition='self_consistency'))
    return results


# ── Single Pass — BATCHED (FIX #5) ──────────────────────────────────────────
def run_single_pass_blip2(images, questions, qids, gt_answers, dataset_names,
                          sample_indices=None, all_samples=None, chitta=None):
    si = sample_indices or list(range(len(images)))
    n  = len(images)
    q_types_l, prompts_l, cl_l = [], [], []
    for i in range(n):
        q_type, ents = manas_route(questions[i], images[i] is not None, dataset_names[i])
        _, ctx, cs, cl = _get_ctx(si, i, images[i], questions[i], ents, images[i] is not None, all_samples, chitta)
        q_types_l.append(q_type); cl_l.append(cl)
        prompts_l.append(build_prompt(questions[i], q_type, smart_truncate_context(ctx, 280), cs))
    t0 = time.time()
    ans = blip2_generate(images, prompts_l)
    free_gpu()
    per_s = (time.time() - t0) / n
    results = []
    for i in range(n):
        soft, em, hall = _score(ans[i], gt_answers[i], dataset_names[i], cl_l[i])
        results.append(SampleResult(qid=qids[i], dataset=dataset_names[i], question=questions[i],
            predicted=ans[i], ground_truth=gt_answers[i], soft_score=soft, is_exact=em,
            is_hallucination=hall, latency_s=per_s, model_calls=1,
            q_type=q_types_l[i], condition='single_pass'))
    return results


# ── Ablation: -Routing — BATCHED ─────────────────────────────────────────────
def run_no_routing(images, questions, qids, gt_answers, dataset_names,
                   sample_indices=None, all_samples=None, chitta=None):
    si = sample_indices or list(range(len(images)))
    n  = len(images)
    prompts_l, cl_l = [], []
    for i in range(n):
        _, ctx, _, cl = _get_ctx(si, i, images[i], questions[i], set(), images[i] is not None, all_samples, chitta)
        cl_l.append(cl)
        prompts_l.append(f'Context: {smart_truncate_context(ctx, 280)}\nQuestion: {questions[i]}\nAnswer:')
    t0 = time.time()
    ans = blip2_generate(images, prompts_l)
    free_gpu()
    per_s = (time.time() - t0) / n
    results = []
    for i in range(n):
        soft, em, hall = _score(ans[i], gt_answers[i], dataset_names[i], cl_l[i])
        results.append(SampleResult(qid=qids[i], dataset=dataset_names[i], question=questions[i],
            predicted=ans[i], ground_truth=gt_answers[i], soft_score=soft, is_exact=em,
            is_hallucination=hall, latency_s=per_s, model_calls=1,
            q_type='simple', condition='no_routing'))
    return results


# ── Ablation: -Verification (no Pass2) — BATCHED ────────────────────────────
def run_no_verification(images, questions, qids, gt_answers, dataset_names,
                        sample_indices=None, all_samples=None, chitta=None):
    si = sample_indices or list(range(len(images)))
    n  = len(images)
    q_types_l, prompts_l, cl_l, cs_l = [], [], [], []
    for i in range(n):
        q_type, ents = manas_route(questions[i], images[i] is not None, dataset_names[i])
        _, ctx, cs, cl = _get_ctx(si, i, images[i], questions[i], ents, images[i] is not None, all_samples, chitta)
        q_types_l.append(q_type); cl_l.append(cl); cs_l.append(cs)
        prompts_l.append(build_prompt(questions[i], q_type, smart_truncate_context(ctx, 280), cs))
    t0 = time.time()
    p1_ans = blip2_generate(images, prompts_l)
    free_gpu()
    # SC fallback only (no P2)
    final_ans = list(p1_ans)
    p3_fired  = [False] * n
    bad_idx   = [i for i in range(n) if is_bad_answer(p1_ans[i], q_types_l[i])]
    if bad_idx:
        sc_pool = {i: [] for i in bad_idx}
        bi_imgs = [images[i] for i in bad_idx]
        bi_prom = [build_prompt(questions[i], q_types_l[i], '', cs_l[i]) for i in bad_idx]
        for _ in range(SC_N_PASSES):
            sc_res = blip2_generate(bi_imgs, bi_prom, do_sample=True, temperature=SC_TEMPERATURE)
            for j, i in enumerate(bad_idx): sc_pool[i].append(sc_res[j])
        free_gpu()
        for i in bad_idx:
            p3_fired[i] = True
            voted, _ = weighted_majority_vote(sc_pool[i])
            final_ans[i] = voted
    per_s = (time.time() - t0) / n
    results = []
    for i in range(n):
        soft, em, hall = _score(final_ans[i], gt_answers[i], dataset_names[i], cl_l[i])
        results.append(SampleResult(qid=qids[i], dataset=dataset_names[i], question=questions[i],
            predicted=final_ans[i], ground_truth=gt_answers[i], soft_score=soft, is_exact=em,
            is_hallucination=hall, latency_s=per_s,
            model_calls=1 + int(p3_fired[i]) * SC_N_PASSES,
            q_type=q_types_l[i], pass3_fired=p3_fired[i], condition='no_verification'))
    return results


# ── Ablation: -Consistency (no Pass3) — BATCHED ──────────────────────────────
def run_no_consistency(images, questions, qids, gt_answers, dataset_names,
                       sample_indices=None, all_samples=None, chitta=None):
    si = sample_indices or list(range(len(images)))
    n  = len(images)
    q_types_l, prompts_l, cl_l, cs_l = [], [], [], []
    for i in range(n):
        q_type, ents = manas_route(questions[i], images[i] is not None, dataset_names[i])
        _, ctx, cs, cl = _get_ctx(si, i, images[i], questions[i], ents, images[i] is not None, all_samples, chitta)
        q_types_l.append(q_type); cl_l.append(cl); cs_l.append(cs)
        prompts_l.append(build_prompt(questions[i], q_type, smart_truncate_context(ctx, 280), cs))
    t0 = time.time()
    p1_ans = blip2_generate(images, prompts_l)
    free_gpu()
    # V9-D: Use junk-only P2 trigger (consistent with run_buddhi_full V8-D)
    import re as _re
    _JUNK_PATTERNS_ABL = {
        'unanswerable', 'not enough', 'cannot determine', 'i do not know',
        'not possible', 'unclear', 'unknown', 'no information',
    }
    _JUNK_REGEX_ABL = _re.compile(
        r'^[\(\[]?(?:i{1,4}|vi{0,3}|ix|iv|x{0,3})[\)\].]$|'
        r'^[\(\[]\d+[\)\]]\.?$|'
        r'^\d+\.$',
        _re.IGNORECASE
    )
    def _is_junk_p2(pred_str):
        p = pred_str.strip().lower()
        return (any(j in p for j in _JUNK_PATTERNS_ABL) or
                bool(_JUNK_REGEX_ABL.match(p)) or not p or not normalize_answer(p))
    p2_mask = [_is_junk_p2(p1_ans[i]) for i in range(n)]
    final_ans = list(p1_ans)
    p2_idx = [i for i in range(n) if p2_mask[i]]
    if p2_idx:
        p2_imgs = [images[i] for i in p2_idx]
        p2_prom = [build_pass2_prompt(questions[i], p1_ans[i], q_types_l[i], cs_l[i])
                   for i in p2_idx]
        p2_res = blip2_generate(p2_imgs, p2_prom)
        free_gpu()
        for j, i in enumerate(p2_idx):
            cand = p2_res[j]
            # V9-B mirror: reject junk P2 output
            if _is_junk_p2(cand):
                continue
            if is_bad_answer(p1_ans[i], q_types_l[i]):
                final_ans[i] = cand
    per_s = (time.time() - t0) / n
    results = []
    for i in range(n):
        soft, em, hall = _score(final_ans[i], gt_answers[i], dataset_names[i], cl_l[i])
        results.append(SampleResult(qid=qids[i], dataset=dataset_names[i], question=questions[i],
            predicted=final_ans[i], ground_truth=gt_answers[i], soft_score=soft, is_exact=em,
            is_hallucination=hall, latency_s=per_s,
            model_calls=1 + int(p2_mask[i]),
            q_type=q_types_l[i], pass2_fired=p2_mask[i], condition='no_consistency'))
    return results


# ── Ablation: -Retrieval — BATCHED ───────────────────────────────────────────
def run_no_retrieval(images, questions, qids, gt_answers, dataset_names,
                     sample_indices=None, all_samples=None, chitta=None):
    si = sample_indices or list(range(len(images)))
    n  = len(images)
    q_types_l, prompts_l, cl_l, cs_l = [], [], [], []
    for i in range(n):
        q_type, _ = manas_route(questions[i], images[i] is not None, dataset_names[i])
        s_idx = si[i]
        cs = getattr(all_samples[s_idx], 'choices_str',  '') if 0 <= s_idx < len(all_samples) else ''
        cl = getattr(all_samples[s_idx], 'choices_list', []) if 0 <= s_idx < len(all_samples) else []
        q_types_l.append(q_type); cl_l.append(cl); cs_l.append(cs)
        prompts_l.append(build_prompt(questions[i], q_type, '', cs))  # no context
    t0 = time.time()
    p1_ans = blip2_generate(images, prompts_l)
    free_gpu()
    # V9-D: Use junk-only P2 trigger (consistent with run_buddhi_full V8-D)
    import re as _re
    _JUNK_PATTERNS_ABL = {
        'unanswerable', 'not enough', 'cannot determine', 'i do not know',
        'not possible', 'unclear', 'unknown', 'no information',
    }
    _JUNK_REGEX_ABL = _re.compile(
        r'^[\(\[]?(?:i{1,4}|vi{0,3}|ix|iv|x{0,3})[\)\].]$|'
        r'^[\(\[]\d+[\)\]]\.?$|'
        r'^\d+\.$',
        _re.IGNORECASE
    )
    def _is_junk_p2(pred_str):
        p = pred_str.strip().lower()
        return (any(j in p for j in _JUNK_PATTERNS_ABL) or
                bool(_JUNK_REGEX_ABL.match(p)) or not p or not normalize_answer(p))
    p2_mask = [_is_junk_p2(p1_ans[i]) for i in range(n)]
    ans_p2  = list(p1_ans)
    p2_idx  = [i for i in range(n) if p2_mask[i]]
    if p2_idx:
        p2_imgs = [images[i] for i in p2_idx]
        p2_prom = [build_pass2_prompt(questions[i], p1_ans[i], q_types_l[i], cs_l[i])
                   for i in p2_idx]
        p2_res = blip2_generate(p2_imgs, p2_prom)
        free_gpu()
        for j, i in enumerate(p2_idx):
            cand = p2_res[j]
            # V9-B mirror: reject junk P2 output
            if _is_junk_p2(cand):
                continue
            if is_bad_answer(p1_ans[i], q_types_l[i]):
                ans_p2[i] = cand
    # SC fallback
    final_ans = list(ans_p2)
    p3_fired  = [False] * n
    bad_idx   = [i for i in range(n) if is_bad_answer(ans_p2[i], q_types_l[i])]
    if bad_idx:
        sc_pool = {i: [] for i in bad_idx}
        bi_imgs = [images[i] for i in bad_idx]
        bi_prom = [build_prompt(questions[i], q_types_l[i], '', cs_l[i]) for i in bad_idx]
        for _ in range(SC_N_PASSES):
            sc_res = blip2_generate(bi_imgs, bi_prom, do_sample=True, temperature=SC_TEMPERATURE)
            for j, i in enumerate(bad_idx): sc_pool[i].append(sc_res[j])
        free_gpu()
        for i in bad_idx:
            p3_fired[i] = True
            voted, _ = weighted_majority_vote(sc_pool[i])
            final_ans[i] = voted
    per_s = (time.time() - t0) / n
    results = []
    for i in range(n):
        soft, em, hall = _score(final_ans[i], gt_answers[i], dataset_names[i], cl_l[i])
        results.append(SampleResult(qid=qids[i], dataset=dataset_names[i], question=questions[i],
            predicted=final_ans[i], ground_truth=gt_answers[i], soft_score=soft, is_exact=em,
            is_hallucination=hall, latency_s=per_s,
            model_calls=1 + int(p2_mask[i]) + int(p3_fired[i]) * SC_N_PASSES,
            q_type=q_types_l[i], pass2_fired=p2_mask[i], pass3_fired=p3_fired[i],
            condition='no_retrieval'))
    return results


# ── Ablation: -Output (P1 only) — BATCHED ────────────────────────────────────
def run_no_output(images, questions, qids, gt_answers, dataset_names,
                  sample_indices=None, all_samples=None, chitta=None):
    si = sample_indices or list(range(len(images)))
    n  = len(images)
    q_types_l, prompts_l, cl_l = [], [], []
    for i in range(n):
        q_type, ents = manas_route(questions[i], images[i] is not None, dataset_names[i])
        _, ctx, cs, cl = _get_ctx(si, i, images[i], questions[i], ents, images[i] is not None, all_samples, chitta)
        q_types_l.append(q_type); cl_l.append(cl)
        prompts_l.append(build_prompt(questions[i], q_type, smart_truncate_context(ctx, 280), cs))
    t0 = time.time()
    ans = blip2_generate(images, prompts_l)
    free_gpu()
    per_s = (time.time() - t0) / n
    results = []
    for i in range(n):
        soft, em, hall = _score(ans[i], gt_answers[i], dataset_names[i], cl_l[i])
        r = SampleResult(qid=qids[i], dataset=dataset_names[i], question=questions[i],
            predicted=ans[i], ground_truth=gt_answers[i], soft_score=soft, is_exact=em,
            is_hallucination=hall, latency_s=per_s, model_calls=1,
            q_type=q_types_l[i], condition='no_output')
        sakshi_logger.record(r); results.append(r)
    return results


# ── Ablation: -Logging (full pipeline, Sakshi skipped) — BATCHED ─────────────
def run_no_logging(images, questions, qids, gt_answers, dataset_names,
                   sample_indices=None, all_samples=None, chitta=None):
    """V9-D: Same as run_buddhi_full but skips Sakshi logging.
    Mirrors run_buddhi_full exactly (V8-B/K/L/M/N/V9-B/C) for ablation consistency."""
    from .utils import fix_binary_answer, truncate_verbose_answer, build_subquestion
    import re as _re
    t0 = time.time()
    n  = len(images)
    si = sample_indices or [-1] * n

    # ── Step 1: Chitta retrieval + Manas routing (mirrors run_buddhi_full) ──
    contexts, q_types, cs_l, cl_l = [], [], [], []
    for i in range(n):
        s_idx = si[i]
        q_type, seed_ents = manas_route(questions[i], images[i] is not None, dataset_names[i])
        # V8-B + V10-A: Skip retrieval for OKVQA and VQAv2 non-text-reading
        SKIP_RETRIEVAL_DATASETS = {'okvqa'}
        skip_retrieval = (dataset_names[i] in SKIP_RETRIEVAL_DATASETS
                         or (dataset_names[i] == 'vqav2' and q_type != 'text_reading'))
        if skip_retrieval:
            ctx_str = ''
        else:
            ctx = chitta.retrieve(questions[i], images[i], seed_ents,
                                 has_image=(images[i] is not None), query_idx=s_idx)
            max_ctx = 280
            ctx_str = smart_truncate_context(ctx, max_ctx)
        # V8-C: Math context bypass
        if q_type == 'math':
            ctx_str = ''
        cs = getattr(all_samples[s_idx], 'choices_str',  '') if 0 <= s_idx < len(all_samples) else ''
        cl = getattr(all_samples[s_idx], 'choices_list', []) if 0 <= s_idx < len(all_samples) else []
        contexts.append(ctx_str); q_types.append(q_type); cs_l.append(cs); cl_l.append(cl)

    # ── Step 1.5: Visual sub-question (V8-K) ──
    subq_idx = [i for i in range(n) if q_types[i] == 'math'
                or (dataset_names[i] == 'gqa' and q_types[i] in ('simple', 'comparison', 'verification'))]
    math_ctx_extra = [''] * n
    if subq_idx:
        subq_imgs = [images[i] for i in subq_idx]
        subq_prompts = [f'{build_subquestion(questions[i], q_types[i])}\nAnswer:' for i in subq_idx]
        subq_ans = blip2_generate(subq_imgs, subq_prompts)
        free_gpu()
        for j, i in enumerate(subq_idx):
            sub = postprocess_answer(subq_ans[j])
            if sub and not is_bad_answer(sub, q_types[i]):
                math_ctx_extra[i] = f'Visual: {sub}'

    # ── Step 2: Pass 1 ──
    p1_prompts = []
    for i in range(n):
        ctx_combined = contexts[i]
        if math_ctx_extra[i]:
            ctx_combined = f'{math_ctx_extra[i]} | {ctx_combined}' if ctx_combined else math_ctx_extra[i]
        p1_prompts.append(build_prompt(questions[i], q_types[i], ctx_combined, cs_l[i]))
    ans_p1 = blip2_generate(images, p1_prompts)
    ans_p1 = [postprocess_answer(a, q_types[i]) for i, a in enumerate(ans_p1)]
    ans_p1 = [fix_binary_answer(ans_p1[i], questions[i]) for i in range(n)]
    ans_p1 = [truncate_verbose_answer(ans_p1[i], q_types[i]) for i in range(n)]
    free_gpu()

    # ── Step 3: Pass 2 (V8-D/V8-L/V9-B/V9-C junk-only trigger) ──
    JUNK_PATTERNS = {
        'unanswerable', 'not enough', 'cannot determine', 'i do not know',
        'not possible', 'unclear', 'unknown', 'no information',
    }
    JUNK_REGEX = _re.compile(
        r'^[\(\[]?(?:i{1,4}|vi{0,3}|ix|iv|x{0,3})[\)\].]$|'
        r'^[\(\[]\d+[\)\]]\.?$|'
        r'^\d+\.$',
        _re.IGNORECASE
    )
    def should_fire_p2(i):
        pred = ans_p1[i].strip().lower()
        ds = dataset_names[i]
        qt = q_types[i]
        if ds == 'scienceqa' and qt not in ('text_reading',): return False
        if ds == 'gqa': return False  # V8-L
        is_junk_phrase = any(j in pred for j in JUNK_PATTERNS)
        is_junk_token  = bool(JUNK_REGEX.match(pred))
        is_empty       = not pred or not normalize_answer(pred)
        return is_junk_phrase or is_junk_token or is_empty  # V9-C: no text_reading trigger

    p2_mask  = [should_fire_p2(i) for i in range(n)]
    ans_p2   = list(ans_p1)
    p2_idx   = [i for i in range(n) if p2_mask[i]]
    if p2_idx:
        p2_res_raw = blip2_generate([images[i] for i in p2_idx],
                                    [build_pass2_prompt(questions[i], ans_p1[i], q_types[i], cs_l[i]) for i in p2_idx])
        p2_res = [postprocess_answer(a, q_types[p2_idx[j]]) for j, a in enumerate(p2_res_raw)]
        free_gpu()
        for j, i in enumerate(p2_idx):
            cand = fix_binary_answer(p2_res[j], questions[i])
            cand = truncate_verbose_answer(cand, q_types[i])
            # V9-B: reject junk P2 output
            cand_lower = cand.strip().lower()
            is_cand_junk = (bool(JUNK_REGEX.match(cand_lower))
                           or any(jp in cand_lower for jp in JUNK_PATTERNS)
                           or not cand.strip() or not normalize_answer(cand))
            if is_cand_junk: continue
            if is_bad_answer(ans_p1[i], q_types[i]):
                ans_p2[i] = cand

    # ── Step 4: Pass 3 SC fallback ──
    final_ans = list(ans_p2)
    consistencies = [1.0] * n
    p3_fired = [False] * n
    p3_idx = [i for i in range(n) if is_bad_answer(ans_p2[i], q_types[i])]
    if p3_idx:
        for _ in range(DEPTH_CAP):
            still_bad = [i for i in p3_idx if is_bad_answer(final_ans[i], q_types[i])]
            if not still_bad: break
            sc_pool = {i: [] for i in still_bad}
            sb_imgs = [images[i] for i in still_bad]
            sb_prom = [build_prompt(questions[i], q_types[i], contexts[i], cs_l[i]) for i in still_bad]
            for _ in range(SC_N_PASSES):
                sc_res = blip2_generate(sb_imgs, sb_prom, do_sample=True, temperature=SC_TEMPERATURE)
                for j, i in enumerate(still_bad): sc_pool[i].append(postprocess_answer(sc_res[j], q_types[i]))
            free_gpu()
            for i in still_bad:
                p3_fired[i] = True
                voted, vs = weighted_majority_vote(sc_pool[i])
                voted = fix_binary_answer(voted, questions[i])
                voted = truncate_verbose_answer(voted, q_types[i])
                consistencies[i] = vs
                if not is_bad_answer(voted, q_types[i]): final_ans[i] = voted

    per_s = (time.time() - t0) / n
    results = []
    for i in range(n):
        soft, em, hall = _score(final_ans[i], gt_answers[i], dataset_names[i], cl_l[i])
        results.append(SampleResult(qid=qids[i], dataset=dataset_names[i], question=questions[i],
            predicted=final_ans[i], ground_truth=gt_answers[i], soft_score=soft, is_exact=em,
            is_hallucination=hall, latency_s=per_s,
            model_calls=1 + int(bool(math_ctx_extra[i])) + int(p2_mask[i]) + int(p3_fired[i]) * SC_N_PASSES,
            q_type=q_types[i], pass2_fired=p2_mask[i], pass3_fired=p3_fired[i],
            consistency_score=consistencies[i], condition='no_logging'))
    return results
