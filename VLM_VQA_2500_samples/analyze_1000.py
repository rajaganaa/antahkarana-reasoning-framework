"""Deep analysis of 1000-sample results — Task 1: VQAv2 failure diagnosis."""
import csv, re, sys, os
from collections import Counter, defaultdict

BASE = r'E:\Downloads\study1-20260418T045910Z-3-001\vlm_1000\home\jupyter\VLM_experiment\outputs\results'

def load_csv(name):
    with open(f'{BASE}\\{name}', encoding='utf-8') as f:
        return list(csv.DictReader(f))

anta = load_csv('results_antahkarana.csv')
direct = load_csv('results_direct.csv')
no_ret = load_csv('results_no_retrieval.csv')

print("="*70)
print("TASK 1: VQAv2 FAILURE DIAGNOSIS")
print("="*70)

# Get VQAv2 samples
v2_anta   = {r['qid']: r for r in anta   if r['dataset']=='vqav2'}
v2_direct = {r['qid']: r for r in direct if r['dataset']=='vqav2'}
v2_noret  = {r['qid']: r for r in no_ret if r['dataset']=='vqav2'}

print(f"VQAv2 samples: {len(v2_anta)}")
print(f"  Direct EM:   {sum(1 for r in v2_direct.values() if r['is_exact']=='True')}/{len(v2_direct)} = {sum(1 for r in v2_direct.values() if r['is_exact']=='True')/len(v2_direct)*100:.1f}%")
print(f"  Anta EM:     {sum(1 for r in v2_anta.values() if r['is_exact']=='True')}/{len(v2_anta)} = {sum(1 for r in v2_anta.values() if r['is_exact']=='True')/len(v2_anta)*100:.1f}%")
print(f"  no_ret EM:   {sum(1 for r in v2_noret.values() if r['is_exact']=='True')}/{len(v2_noret)} = {sum(1 for r in v2_noret.values() if r['is_exact']=='True')/len(v2_noret)*100:.1f}%")

# Find regressions: Direct correct, Anta wrong
losses = []
for qid in v2_anta:
    a = v2_anta[qid]
    d = v2_direct.get(qid)
    nr = v2_noret.get(qid)
    if d and d['is_exact']=='True' and a['is_exact']=='False':
        losses.append({
            'qid': qid,
            'question': a['question'],
            'q_type': a.get('q_type',''),
            'anta_pred': a['predicted'],
            'direct_pred': d['predicted'],
            'noret_pred': nr['predicted'] if nr else '',
            'noret_em': nr['is_exact'] if nr else '',
            'gt': a['ground_truth'][:60],
            'p2': a.get('pass2_fired',''),
            'p3': a.get('pass3_fired',''),
        })

# Find gains: Anta correct, Direct wrong
gains = []
for qid in v2_anta:
    a = v2_anta[qid]
    d = v2_direct.get(qid)
    if d and a['is_exact']=='True' and d['is_exact']=='False':
        gains.append(qid)

print(f"\nRegressions (Direct OK, Anta WRONG): {len(losses)}")
print(f"Gains (Anta OK, Direct WRONG): {len(gains)}")
print(f"Net delta: {len(gains) - len(losses)} samples ({(len(gains)-len(losses))/len(v2_anta)*100:.1f}pp)")

# TASK 5: Classify by answer type
def classify_vqa_type(question, gt):
    gt_lower = gt.split('|')[0].strip().lower() if gt else ''
    if gt_lower in ('yes','no'):
        return 'yes/no'
    try:
        float(gt_lower)
        return 'number'
    except ValueError:
        pass
    if gt_lower.isdigit():
        return 'number'
    return 'other'

# Per answer-type analysis
print("\n" + "="*70)
print("TASK 5: PER-QUESTION-TYPE BREAKDOWN (VQAv2)")
print("="*70)
type_stats = defaultdict(lambda: {'anta_correct':0, 'direct_correct':0, 'noret_correct':0, 'total':0})
for qid in v2_anta:
    a = v2_anta[qid]
    d = v2_direct.get(qid)
    nr = v2_noret.get(qid)
    qt = classify_vqa_type(a['question'], a['ground_truth'])
    type_stats[qt]['total'] += 1
    if a['is_exact']=='True': type_stats[qt]['anta_correct'] += 1
    if d and d['is_exact']=='True': type_stats[qt]['direct_correct'] += 1
    if nr and nr['is_exact']=='True': type_stats[qt]['noret_correct'] += 1

print(f"\n{'Type':>10} | {'N':>5} | {'Direct%':>8} | {'Anta%':>8} | {'NoRet%':>8} | {'Delta':>7}")
print("-"*60)
for qt in ['yes/no', 'number', 'other']:
    s = type_stats[qt]
    n = s['total']
    d_em = s['direct_correct']/n*100 if n else 0
    a_em = s['anta_correct']/n*100 if n else 0
    nr_em = s['noret_correct']/n*100 if n else 0
    delta = a_em - d_em
    print(f"{qt:>10} | {n:>5} | {d_em:>7.1f}% | {a_em:>7.1f}% | {nr_em:>7.1f}% | {delta:>+6.1f}pp")

# Analyze loss patterns by type
print("\n" + "="*70)
print("LOSS PATTERN ANALYSIS")
print("="*70)
loss_types = Counter(classify_vqa_type(l['question'], l['gt']) for l in losses)
print(f"\nLosses by type: {dict(loss_types)}")

# Was no_retrieval also correct? (proves it's a retrieval problem)
ret_caused = sum(1 for l in losses if l['noret_em']=='True')
print(f"Losses where no_retrieval was ALSO correct: {ret_caused}/{len(losses)}")
print(f"  -> {ret_caused} losses are CAUSED by retrieval injecting wrong context")

# Show top failure examples by type
for qt in ['yes/no', 'number', 'other']:
    qt_losses = [l for l in losses if classify_vqa_type(l['question'], l['gt'])==qt]
    if not qt_losses: continue
    print(f"\n--- {qt.upper()} failures ({len(qt_losses)}) ---")
    for l in qt_losses[:5]:
        noret_mark = " [noret=OK]" if l['noret_em']=='True' else ""
        print(f"  Q: {l['question'][:60]}")
        print(f"    Direct: '{l['direct_pred'][:30]}' | Anta: '{l['anta_pred'][:30]}' | GT: '{l['gt'][:25]}'{noret_mark}")

# Hallucination comparison
print("\n" + "="*70)
print("HALLUCINATION BY TYPE (VQAv2)")
print("="*70)
hall_stats = defaultdict(lambda: {'anta_hall':0, 'direct_hall':0, 'noret_hall':0, 'total':0})
for qid in v2_anta:
    a = v2_anta[qid]
    d = v2_direct.get(qid)
    nr = v2_noret.get(qid)
    qt = classify_vqa_type(a['question'], a['ground_truth'])
    hall_stats[qt]['total'] += 1
    if a.get('is_hallucination')=='True': hall_stats[qt]['anta_hall'] += 1
    if d and d.get('is_hallucination')=='True': hall_stats[qt]['direct_hall'] += 1
    if nr and nr.get('is_hallucination')=='True': hall_stats[qt]['noret_hall'] += 1

print(f"\n{'Type':>10} | {'N':>5} | {'Direct%':>8} | {'Anta%':>8} | {'NoRet%':>8} | {'Anta-NoRet':>10}")
print("-"*65)
for qt in ['yes/no', 'number', 'other']:
    s = hall_stats[qt]
    n = s['total']
    d_h = s['direct_hall']/n*100 if n else 0
    a_h = s['anta_hall']/n*100 if n else 0
    nr_h = s['noret_hall']/n*100 if n else 0
    delta = a_h - nr_h
    print(f"{qt:>10} | {n:>5} | {d_h:>7.1f}% | {a_h:>7.1f}% | {nr_h:>7.1f}% | {delta:>+9.1f}pp")

# P2/P3 analysis on VQAv2
print("\n" + "="*70)
print("P2/P3 IMPACT ON VQAv2")
print("="*70)
v2_p2 = [r for r in anta if r['dataset']=='vqav2' and r.get('pass2_fired')=='True']
v2_p3 = [r for r in anta if r['dataset']=='vqav2' and r.get('pass3_fired')=='True']
print(f"VQAv2 P2 fired: {len(v2_p2)}")
print(f"VQAv2 P3 fired: {len(v2_p3)}")

# Summary
print("\n" + "="*70)
print("CRITICAL FINDINGS")
print("="*70)
print("1. VQAv2: Anta=61.6% vs Direct=66.0% (-4.4pp)")
print("2. no_retrieval VQAv2=67.2% — HIGHER than both Anta and Direct!")
print("3. Retrieval is POISONING VQAv2 answers")
print("4. no_retrieval Hall=20.8% vs Anta Hall=31.6% (+10.8pp from retrieval)")
print("5. FIX: Skip retrieval for VQAv2 verification/simple/visual q-types")
print("6. Anta beats all baselines on ScienceQA (52.8% vs 48.4% SC)")
print("7. Anta beats Direct on GQA (45.2% vs 34.0% +11.2pp!)")
