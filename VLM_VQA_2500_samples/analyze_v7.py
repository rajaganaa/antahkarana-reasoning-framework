"""Deep error analysis of Antahkarana V7 results."""
import csv, json, re
from collections import Counter, defaultdict

BASE = r'E:\Downloads\study1-20260418T045910Z-3-001\v7_results\antahkarana_project\outputs\results'

def load_csv(name):
    with open(f'{BASE}\\{name}', encoding='utf-8') as f:
        return list(csv.DictReader(f))

anta = load_csv('results_antahkarana.csv')
direct = load_csv('results_direct.csv')
cot = load_csv('results_cot.csv')
sc = load_csv('results_self_consistency.csv')
no_ret = load_csv('results_no_retrieval.csv')

print(f"Total samples: {len(anta)}")
print()

# === 1. Overall comparison ===
def em_rate(rows):
    return sum(1 for r in rows if r['is_exact']=='True') / len(rows) * 100

print("="*70)
print("OVERALL EM COMPARISON")
print("="*70)
print(f"  Antahkarana:  {em_rate(anta):.1f}%")
print(f"  Direct:       {em_rate(direct):.1f}%")
print(f"  CoT:          {em_rate(cot):.1f}%")
print(f"  SC(5x):       {em_rate(sc):.1f}%")
print(f"  No-retrieval: {em_rate(no_ret):.1f}%")

# === 2. Per-dataset EM ===
print()
print("="*70)
print("PER-DATASET EM%")
print("="*70)
datasets = ['vqav2', 'gqa', 'okvqa', 'textvqa', 'scienceqa']
for ds in datasets:
    a_em = sum(1 for r in anta if r['dataset']==ds and r['is_exact']=='True') / sum(1 for r in anta if r['dataset']==ds) * 100
    d_em = sum(1 for r in direct if r['dataset']==ds and r['is_exact']=='True') / sum(1 for r in direct if r['dataset']==ds) * 100
    c_em = sum(1 for r in cot if r['dataset']==ds and r['is_exact']=='True') / sum(1 for r in cot if r['dataset']==ds) * 100
    s_em = sum(1 for r in sc if r['dataset']==ds and r['is_exact']=='True') / sum(1 for r in sc if r['dataset']==ds) * 100
    nr_em = sum(1 for r in no_ret if r['dataset']==ds and r['is_exact']=='True') / sum(1 for r in no_ret if r['dataset']==ds) * 100
    best_base = max(d_em, c_em, s_em)
    gap = a_em - best_base
    marker = " >> WINNING" if gap > 0 else f" << LOSING by {abs(gap):.0f}pp"
    print(f"  {ds:12s}  Anta={a_em:5.1f}  Dir={d_em:5.1f}  CoT={c_em:5.1f}  SC={s_em:5.1f}  NoRet={nr_em:5.1f}  {marker}")

# === 3. Regression analysis: WHERE does Anta lose vs baselines? ===
print()
print("="*70)
print("REGRESSION ANALYSIS: Anta WRONG but baseline CORRECT")
print("="*70)

losses = []
for i in range(len(anta)):
    a = anta[i]
    if a['is_exact'] == 'False':
        d_ok = direct[i]['is_exact'] == 'True'
        c_ok = cot[i]['is_exact'] == 'True'
        s_ok = sc[i]['is_exact'] == 'True'
        if d_ok or c_ok or s_ok:
            gt = a['ground_truth'].split('|')[0]
            losses.append({
                'qid': a['qid'], 'ds': a['dataset'], 'qt': a['q_type'],
                'pred': a['predicted'], 'gt': gt, 'question': a['question'][:80],
                'd_ok': d_ok, 'c_ok': c_ok, 's_ok': s_ok,
                'p2': a['pass2_fired'], 'p3': a['pass3_fired'],
                'hall': a['is_hallucination']
            })

print(f"Total regression losses: {len(losses)}")
print()

# By dataset
print("By dataset:")
ds_counts = Counter(l['ds'] for l in losses)
for ds, cnt in ds_counts.most_common():
    ds_total = sum(1 for r in anta if r['dataset']==ds)
    print(f"  {ds:12s}: {cnt} losses (out of {ds_total} samples)")

# By q_type
print("\nBy q_type:")
qt_counts = Counter(l['qt'] for l in losses)
for qt, cnt in qt_counts.most_common():
    print(f"  {qt:15s}: {cnt}")

# === 4. Failure categorization ===
print()
print("="*70)
print("FAILURE CATEGORIZATION")
print("="*70)

categories = defaultdict(list)
for l in losses:
    pred = l['pred'].strip().lower()
    gt = l['gt'].strip().lower()
    
    # Junk / garbage output
    if re.match(r'^[\(\[]?(i{1,4}|vi{0,3}|ix|iv|x{0,3})[\)\].]$', pred, re.I):
        categories['JUNK: Roman numeral'].append(l)
    elif re.match(r'^[\(\[]\d+[\)\]]\.?$', pred) or re.match(r'^\d+\.$', pred):
        categories['JUNK: Numbered marker'].append(l)
    elif any(p in pred for p in ['not enough', 'cannot determine', 'unanswerable', 'i do not know', 'unclear', 'unknown']):
        categories['JUNK: Uncertainty phrase'].append(l)
    elif not pred or pred in ('', ' '):
        categories['JUNK: Empty'].append(l)
    # Yes/No confusion
    elif gt in ('yes', 'no') and pred not in ('yes', 'no'):
        categories['YES/NO: Wrong format'].append(l)
    elif gt in ('yes', 'no') and pred in ('yes', 'no') and gt != pred:
        categories['YES/NO: Polarity flip'].append(l)
    # Verbose mismatch
    elif gt in pred and len(pred.split()) > len(gt.split()) + 2:
        categories['VERBOSE: GT inside verbose pred'].append(l)
    elif len(pred.split()) > 5 and l['qt'] not in ('text_reading', 'mchoice'):
        categories['VERBOSE: Overly long answer'].append(l)
    # MCQ issues
    elif l['qt'] == 'mchoice':
        categories['MCQ: Wrong choice/format'].append(l)
    # Hallucination
    elif l['hall'] == 'True':
        categories['HALLUCINATION: No token overlap'].append(l)
    else:
        categories['NEAR-MISS: Close but not exact'].append(l)

for cat, items in sorted(categories.items(), key=lambda x: -len(x[1])):
    print(f"\n  {cat}: {len(items)} samples")
    for item in items[:3]:
        print(f"    [{item['ds']}] Q: {item['question'][:60]}...")
        print(f"      Pred: '{item['pred'][:50]}' → GT: '{item['gt'][:50]}'")

# === 5. Hallucination analysis ===
print()
print("="*70)
print("HALLUCINATION RATES")
print("="*70)
for ds in datasets:
    a_hall = sum(1 for r in anta if r['dataset']==ds and r['is_hallucination']=='True')
    d_hall = sum(1 for r in direct if r['dataset']==ds and r['is_hallucination']=='True')
    n = sum(1 for r in anta if r['dataset']==ds)
    print(f"  {ds:12s}: Anta={a_hall}/{n} ({a_hall/n*100:.0f}%)  Direct={d_hall}/{n} ({d_hall/n*100:.0f}%)")

# === 6. Pass2/Pass3 effectiveness ===
print()
print("="*70)
print("PASS2/PASS3 ANALYSIS")
print("="*70)
p2_fired = [r for r in anta if r['pass2_fired']=='True']
p2_correct = [r for r in p2_fired if r['is_exact']=='True']
p3_fired = [r for r in anta if r['pass3_fired']=='True']
p3_correct = [r for r in p3_fired if r['is_exact']=='True']
print(f"  P2 fired: {len(p2_fired)}, correct: {len(p2_correct)} ({len(p2_correct)/max(len(p2_fired),1)*100:.0f}%)")
print(f"  P3 fired: {len(p3_fired)}, correct: {len(p3_correct)} ({len(p3_correct)/max(len(p3_fired),1)*100:.0f}%)")

# By dataset for P2
print("\n  P2 by dataset:")
for ds in datasets:
    p2_ds = [r for r in p2_fired if r['dataset']==ds]
    p2_ds_ok = [r for r in p2_ds if r['is_exact']=='True']
    if p2_ds:
        print(f"    {ds:12s}: {len(p2_ds)} fired, {len(p2_ds_ok)} correct ({len(p2_ds_ok)/len(p2_ds)*100:.0f}%)")

# === 7. OKVQA specific: retrieval helps or hurts? ===
print()
print("="*70)
print("OKVQA: RETRIEVAL vs NO-RETRIEVAL (per-sample)")
print("="*70)
ok_anta = [r for r in anta if r['dataset']=='okvqa']
ok_noret = [r for r in no_ret if r['dataset']=='okvqa']
a_wins = sum(1 for i in range(len(ok_anta)) if ok_anta[i]['is_exact']=='True' and ok_noret[i]['is_exact']=='False')
nr_wins = sum(1 for i in range(len(ok_anta)) if ok_anta[i]['is_exact']=='False' and ok_noret[i]['is_exact']=='True')
both = sum(1 for i in range(len(ok_anta)) if ok_anta[i]['is_exact']=='True' and ok_noret[i]['is_exact']=='True')
print(f"  Both correct: {both}")
print(f"  Only Anta correct (retrieval helped): {a_wins}")
print(f"  Only NoRet correct (retrieval HURT): {nr_wins}")

# === 8. VQAv2 gap analysis (biggest baseline lead) ===
print()
print("="*70)
print("VQAv2 DEEP DIVE (Anta=62%, CoT=66%, SC=66%)")
print("="*70)
v2_losses = [l for l in losses if l['ds']=='vqav2']
print(f"  VQAv2 regressions vs baselines: {len(v2_losses)}")
v2_yes_no = [l for l in v2_losses if l['gt'] in ('yes','no')]
print(f"  Of which yes/no: {len(v2_yes_no)}")
for l in v2_yes_no[:5]:
    print(f"    Q: {l['question'][:60]}  Pred: '{l['pred']}'  GT: '{l['gt']}'")

# === 9. ScienceQA analysis ===
print()
print("="*70) 
print("SCIENCEQA ANALYSIS (Anta=54%, best)")
print("="*70)
sci = [r for r in anta if r['dataset']=='scienceqa']
sci_types = Counter(r['q_type'] for r in sci)
print(f"  Q-type distribution: {dict(sci_types)}")
for qt in sci_types:
    qt_samples = [r for r in sci if r['q_type']==qt]
    qt_em = sum(1 for r in qt_samples if r['is_exact']=='True') / len(qt_samples) * 100
    print(f"    {qt}: {qt_em:.0f}% EM ({len(qt_samples)} samples)")

print()
print("="*70)
print("ANTAHKARANA UNIQUE WINS (correct where ALL baselines fail)")
print("="*70)
unique_wins = []
for i in range(len(anta)):
    if (anta[i]['is_exact']=='True' and direct[i]['is_exact']=='False' 
        and cot[i]['is_exact']=='False' and sc[i]['is_exact']=='False'):
        unique_wins.append(anta[i])
print(f"  Unique wins: {len(unique_wins)}")
uw_ds = Counter(w['dataset'] for w in unique_wins)
print(f"  By dataset: {dict(uw_ds)}")
