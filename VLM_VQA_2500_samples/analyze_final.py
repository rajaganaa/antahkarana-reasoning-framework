"""Deep analysis of ANTAHKARANA final_paper_results (V8 with fixes applied)."""
import csv, re
from collections import Counter, defaultdict

BASE = r'E:\Downloads\study1-20260418T045910Z-3-001\final_results\outputs\results'

def load_csv(name):
    with open(f'{BASE}\\{name}', encoding='utf-8') as f:
        return list(csv.DictReader(f))

anta = load_csv('results_antahkarana.csv')
direct = load_csv('results_direct.csv')
cot = load_csv('results_cot.csv')
sc = load_csv('results_self_consistency.csv')
no_ret = load_csv('results_no_retrieval.csv')
no_ver = load_csv('results_no_verification.csv')
no_log = load_csv('results_no_logging.csv')

print(f"Total samples: {len(anta)}")

# === TASK 1: ScienceQA Hallucination Bug ===
print("\n" + "="*70)
print("TASK 1: SCIENCEQA HALLUCINATION BUG AUDIT")
print("="*70)
sci_direct = [r for r in direct if r['dataset']=='scienceqa']
sci_anta   = [r for r in anta   if r['dataset']=='scienceqa']

print(f"\nScienceQA Direct: {sum(1 for r in sci_direct if r['is_hallucination']=='True')}/50 hallucination = {sum(1 for r in sci_direct if r['is_hallucination']=='True')/50*100:.0f}%")
print(f"ScienceQA Direct EM: {sum(1 for r in sci_direct if r['is_exact']=='True')}/50 = {sum(1 for r in sci_direct if r['is_exact']=='True')/50*100:.0f}%")
print(f"ScienceQA Anta:  {sum(1 for r in sci_anta if r['is_hallucination']=='True')}/50 hallucination = {sum(1 for r in sci_anta if r['is_hallucination']=='True')/50*100:.0f}%")

print("\n--- 10 ScienceQA Direct samples with is_hallucination ---")
for r in sci_direct[:10]:
    gt = r['ground_truth']
    pred = r['predicted']
    hall = r['is_hallucination']
    em = r['is_exact']
    print(f"  Pred: '{pred[:40]}' | GT: '{gt[:40]}' | Hall={hall} | EM={em}")

print("\n--- ScienceQA hallucination WHERE EM is correct ---")
sci_hall_but_correct = [r for r in sci_direct if r['is_hallucination']=='True' and r['is_exact']=='True']
print(f"  Count: {len(sci_hall_but_correct)} (hall=True but EM=True)")
for r in sci_hall_but_correct[:5]:
    print(f"    Pred: '{r['predicted'][:40]}' | GT: '{r['ground_truth'][:40]}'")

print("\n--- ScienceQA hallucination WHERE EM is wrong ---")
sci_hall_wrong = [r for r in sci_direct if r['is_hallucination']=='True' and r['is_exact']=='False']
print(f"  Count: {len(sci_hall_wrong)}")
for r in sci_hall_wrong[:5]:
    print(f"    Pred: '{r['predicted'][:50]}' | GT: '{r['ground_truth'][:50]}'")

print("\n--- KEY ISSUE: is_hallucination checks token_overlap = 0 ---")
print("  For MCQ: pred='B' GT='basalt rock' -> token_overlap=0 -> hall=True")
print("  This is WRONG: the model answered correctly (option B=basalt rock)")
print("  ScienceQA GTs are full-text answers, preds are often letters A/B/C/D")

# === TASK 2: Pass 2 Hurting Performance ===
print("\n" + "="*70)
print("TASK 2: PASS 2 VERIFICATION AUDIT")
print("="*70)
print(f"\nAntahkarana EM: 42.4% (VQA=40.8%)")
print(f"No_verification EM: 43.2% (VQA=41.87%)  <- BEATING full Anta!")
print(f"\nP2 fired: {sum(1 for r in anta if r['pass2_fired']=='True')}")
print(f"P3 fired: {sum(1 for r in anta if r['pass3_fired']=='True')}")

p2_samples = [r for r in anta if r['pass2_fired']=='True']
p2_correct = [r for r in p2_samples if r['is_exact']=='True']
print(f"\nP2 fired on {len(p2_samples)} samples, {len(p2_correct)} correct ({len(p2_correct)/max(len(p2_samples),1)*100:.0f}%)")

print("\nP2 by dataset:")
for ds in ['vqav2','gqa','okvqa','textvqa','scienceqa']:
    ds_p2 = [r for r in p2_samples if r['dataset']==ds]
    ds_p2_ok = [r for r in ds_p2 if r['is_exact']=='True']
    if ds_p2:
        print(f"  {ds:12s}: {len(ds_p2)} fired, {len(ds_p2_ok)} correct ({len(ds_p2_ok)/len(ds_p2)*100:.0f}%)")

# Compare P1 answers (no_verification) vs final (antahkarana) on P2 samples
print("\n--- P2 HELP/HURT analysis ---")
p2_qids = {r['qid'] for r in p2_samples}
nv_map = {r['qid']: r for r in no_ver}
helped = hurt = neutral = 0
for r in p2_samples:
    nv = nv_map.get(r['qid'])
    if nv:
        nv_em = nv['is_exact'] == 'True'
        a_em = r['is_exact'] == 'True'
        if a_em and not nv_em: helped += 1
        elif not a_em and nv_em: hurt += 1
        else: neutral += 1
print(f"  P2 HELPED (got right after P2): {helped}")
print(f"  P2 HURT (was right, P2 broke it): {hurt}")
print(f"  P2 NEUTRAL (both same): {neutral}")

# Show hurt cases
print("\n--- Samples where P2 HURT ---")
for r in p2_samples:
    nv = nv_map.get(r['qid'])
    if nv and nv['is_exact']=='True' and r['is_exact']=='False':
        print(f"  [{r['dataset']}] Q: {r['question'][:60]}")
        print(f"    no_ver pred: '{nv['predicted'][:40]}' (correct)")
        print(f"    antahk pred: '{r['predicted'][:40]}' (WRONG)")
        print(f"    GT: '{r['ground_truth'][:40]}'")

# === TASK 3: no_logging P2 inconsistency ===
print("\n" + "="*70)
print("TASK 3: ABLATION ISOLATION BUG")
print("="*70)
print(f"no_logging P2 fired: {sum(1 for r in no_log if r['pass2_fired']=='True')}")
print(f"antahkarana P2 fired: {sum(1 for r in anta if r['pass2_fired']=='True')}")
print(f"no_verification P2 fired: {sum(1 for r in no_ver if r['pass2_fired']=='True')}")

# Check if no_logging uses the OLD is_bad_answer trigger (not our V8-D junk-only trigger)
print("\n--- no_logging vs antahkarana P2 comparison ---")
nl_p2 = {r['qid'] for r in no_log if r['pass2_fired']=='True'}
a_p2  = {r['qid'] for r in anta   if r['pass2_fired']=='True'}
print(f"  P2 in no_logging only (not in anta): {len(nl_p2 - a_p2)}")
print(f"  P2 in anta only (not in no_logging): {len(a_p2 - nl_p2)}")
print(f"  P2 in both: {len(nl_p2 & a_p2)}")

# === VQAv2 regression ===
print("\n" + "="*70)
print("VQAV2 REGRESSION: Anta=60% vs CoT=66%!")
print("="*70)
v2_anta = [r for r in anta if r['dataset']=='vqav2']
v2_cot = [r for r in cot if r['dataset']=='vqav2']
losses = []
for a, c in zip(v2_anta, v2_cot):
    if a['is_exact']=='False' and c['is_exact']=='True':
        losses.append((a, c))
print(f"VQAv2 regressions (Anta wrong, CoT right): {len(losses)}")
for a, c in losses[:8]:
    print(f"  Q: {a['question'][:55]}")
    print(f"    Anta: '{a['predicted'][:35]}' | CoT: '{c['predicted'][:35]}' | GT: '{a['ground_truth'].split('|')[0][:25]}'")

# === GQA improvement check ===
print("\n" + "="*70)
print("GQA: Anta=48% vs CoT=54% (still -6pp)")
print("="*70)
gqa_anta = [r for r in anta if r['dataset']=='gqa']
gqa_cot = [r for r in cot if r['dataset']=='gqa']
losses = []
for a, c in zip(gqa_anta, gqa_cot):
    if a['is_exact']=='False' and c['is_exact']=='True':
        losses.append((a, c))
print(f"GQA regressions (Anta wrong, CoT right): {len(losses)}")
for a, c in losses[:5]:
    print(f"  Q: {a['question'][:55]}")
    print(f"    Anta: '{a['predicted'][:35]}' | CoT: '{c['predicted'][:35]}' | GT: '{a['ground_truth'].split('|')[0][:25]}'")

# === Overall summary ===
print("\n" + "="*70)
print("CRITICAL ISSUES SUMMARY")
print("="*70)
print("1. Anta EM=42.4% TIES with CoT 42.0% but LOSES to no_verification 43.2%")
print("2. ScienceQA 94% hallucination is SCORING BUG (letter vs full-text)")
print("3. no_logging has 47 P2 fires vs anta 21 - different trigger logic!")
print("4. VQAv2 dropped from 62% (V7) to 60% (final) - regression")
print("5. All McNemar tests ns - need 500+ samples for significance")
