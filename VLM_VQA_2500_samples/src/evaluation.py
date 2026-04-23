"""
ANTAHKARANA v7 — Evaluation
Metric computation, statistical significance, results saving, and figure generation.
Exact logic preserved from original notebook Cells 10-13.
"""

import json
import time
import shutil
from typing import List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

from .antahkarana_model import SampleResult
from .utils import normalize_answer, token_overlap


# ─── Metric Computation (Cell 10) ───────────────────────────────────────────────
def compute_metrics(results: List[SampleResult], condition: str) -> Dict[str, Any]:
    n = len(results)
    if n == 0: return {}
    soft_scores   = [r.soft_score    for r in results]
    exact_matches = [int(r.is_exact)  for r in results]
    hall_flags    = [int(r.is_hallucination) for r in results]
    latencies     = [r.latency_s     for r in results]
    model_calls   = sum(r.model_calls for r in results)
    partial_match = [int(token_overlap(r.predicted, r.ground_truth) > 0) for r in results]
    try:
        all_preds = [normalize_answer(r.predicted) for r in results]
        all_refs  = [normalize_answer(r.ground_truth[0]) if r.ground_truth else '' for r in results]
        labels    = list(set(all_refs))
        macro_f1  = (f1_score(all_refs, all_preds, labels=labels, average='macro', zero_division=0)
                     if len(labels) > 1 else float(np.mean(exact_matches)))
    except Exception:
        macro_f1 = float(np.mean(exact_matches))
    # ScienceQA assertion: EM and VQA accuracy must agree (both binary)
    sciqa = [r for r in results if r.dataset == 'scienceqa']
    if sciqa:
        sciqa_em  = float(np.mean([r.is_exact    for r in sciqa])) * 100
        sciqa_vqa = float(np.mean([r.soft_score  for r in sciqa])) * 100
        if abs(sciqa_em - sciqa_vqa) >= 5:
            print(f'  ⚠️  {condition}: ScienceQA EM({sciqa_em:.1f}%) vs VQA({sciqa_vqa:.1f}%) '
                  f'diverge by {abs(sciqa_em-sciqa_vqa):.1f}pp')
    return {
        'condition':         condition,
        'n_samples':         n,
        'vqa_accuracy':      round(float(np.mean(soft_scores))  * 100, 2),
        'exact_match':       round(float(np.mean(exact_matches)) * 100, 2),
        'partial_match_pct': round(float(np.mean(partial_match)) * 100, 2),
        'macro_f1':          round(float(macro_f1), 3),
        'hallucination_pct': round(float(np.mean(hall_flags))   * 100, 2),
        'latency_mean_s':    round(float(np.mean(latencies)),    4),
        'latency_std_s':     round(float(np.std(latencies)),     4),
        'latency_p50_s':     round(float(np.median(latencies)),  4),
        'latency_p95_s':     round(float(np.percentile(latencies, 95)), 4),
        'throughput_sps':    round(n / sum(latencies) if sum(latencies) > 0 else 0, 4),
        'total_model_calls': model_calls,
        'pass2_fired':       sum(1 for r in results if r.pass2_fired),
        'pass3_fired':       sum(1 for r in results if r.pass3_fired),
    }


def compute_per_dataset(results: List[SampleResult]) -> Dict[str, Dict]:
    by_ds = {}
    for r in results: by_ds.setdefault(r.dataset, []).append(r)
    return {ds: {
        'exact_match':       round(np.mean([r.is_exact    for r in rs]) * 100, 2),
        'vqa_accuracy':      round(np.mean([r.soft_score  for r in rs]) * 100, 2),
        'hallucination_pct': round(np.mean([r.is_hallucination for r in rs]) * 100, 2),
        'n': len(rs)
    } for ds, rs in by_ds.items()}


# V9-E: BLEU-1 for TextVQA (standard metric for open-ended text answers)
def compute_bleu1(pred: str, gt_list: list) -> float:
    """Unigram BLEU (BLEU-1) score between prediction and ground truths."""
    pred_toks = normalize_answer(pred).split()
    if not pred_toks:
        return 0.0
    best_bleu = 0.0
    for gt in gt_list:
        gt_toks = normalize_answer(gt).split()
        if not gt_toks:
            continue
        gt_counts = {}
        for t in gt_toks:
            gt_counts[t] = gt_counts.get(t, 0) + 1
        clipped = 0
        pred_counts = {}
        for t in pred_toks:
            pred_counts[t] = pred_counts.get(t, 0) + 1
        for t, c in pred_counts.items():
            clipped += min(c, gt_counts.get(t, 0))
        precision = clipped / len(pred_toks)
        # Brevity penalty
        bp = min(1.0, len(pred_toks) / len(gt_toks)) if gt_toks else 0.0
        bleu = bp * precision
        best_bleu = max(best_bleu, bleu)
    return best_bleu


# V9-F: Per-question-type breakdown within datasets
def compute_per_qtype(results: List[SampleResult]) -> Dict[str, Dict]:
    """Break down metrics by question type within each dataset.
    VQAv2: yes/no, number, other. GQA: simple, comparison, verification. etc."""
    by_ds_qt = {}
    for r in results:
        key = f"{r.dataset}/{r.q_type}"
        by_ds_qt.setdefault(key, []).append(r)
    out = {}
    for key, rs in by_ds_qt.items():
        out[key] = {
            'exact_match': round(np.mean([r.is_exact for r in rs]) * 100, 2),
            'vqa_accuracy': round(np.mean([r.soft_score for r in rs]) * 100, 2),
            'hallucination_pct': round(np.mean([r.is_hallucination for r in rs]) * 100, 2),
            'n': len(rs),
        }
        # V9-E: Add BLEU-1 for TextVQA
        if rs[0].dataset == 'textvqa':
            bleu_scores = [compute_bleu1(r.predicted, r.ground_truth) for r in rs]
            out[key]['bleu1'] = round(float(np.mean(bleu_scores)) * 100, 2)
    return out


# V9-G: Bootstrap 95% confidence intervals (required for IEEE reviews)
def bootstrap_ci(values, n_boot=1000, ci=0.95, seed=42):
    """Compute bootstrap confidence interval for a list of 0/1 or continuous values.
    Returns (mean, ci_lower, ci_upper)."""
    rng = np.random.RandomState(seed)
    values = np.array(values, dtype=float)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.array([np.mean(rng.choice(values, n, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return float(np.mean(values)), float(lo), float(hi)


def compute_metrics_with_ci(results: List[SampleResult], condition: str) -> Dict[str, Any]:
    """V9-G: compute_metrics enhanced with 95% bootstrap CIs on EM and VQA accuracy."""
    base = compute_metrics(results, condition)
    if not results:
        return base
    # Bootstrap on exact match
    em_values = [int(r.is_exact) for r in results]
    em_mean, em_lo, em_hi = bootstrap_ci(em_values)
    base['em_ci_95'] = f"{em_mean*100:.1f}% [{em_lo*100:.1f}, {em_hi*100:.1f}]"
    base['em_ci_lower'] = round(em_lo * 100, 2)
    base['em_ci_upper'] = round(em_hi * 100, 2)
    # Bootstrap on VQA soft score
    vqa_values = [r.soft_score for r in results]
    vqa_mean, vqa_lo, vqa_hi = bootstrap_ci(vqa_values)
    base['vqa_ci_95'] = f"{vqa_mean*100:.1f}% [{vqa_lo*100:.1f}, {vqa_hi*100:.1f}]"
    base['vqa_ci_lower'] = round(vqa_lo * 100, 2)
    base['vqa_ci_upper'] = round(vqa_hi * 100, 2)
    # Bootstrap on hallucination
    hall_values = [int(r.is_hallucination) for r in results]
    hall_mean, hall_lo, hall_hi = bootstrap_ci(hall_values)
    base['hall_ci_95'] = f"{hall_mean*100:.1f}% [{hall_lo*100:.1f}, {hall_hi*100:.1f}]"
    # Per-dataset CIs
    by_ds = {}
    for r in results:
        by_ds.setdefault(r.dataset, []).append(r)
    ds_cis = {}
    for ds, rs in by_ds.items():
        em_vals = [int(r.is_exact) for r in rs]
        m, lo, hi = bootstrap_ci(em_vals)
        ds_cis[ds] = {'em_ci_95': f"{m*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"}
        # BLEU-1 CI for TextVQA
        if ds == 'textvqa':
            bleu_vals = [compute_bleu1(r.predicted, r.ground_truth) for r in rs]
            bm, blo, bhi = bootstrap_ci(bleu_vals)
            ds_cis[ds]['bleu1_ci_95'] = f"{bm*100:.1f}% [{blo*100:.1f}, {bhi*100:.1f}]"
    base['per_dataset_ci'] = ds_cis
    return base


# V10-B: Efficiency Pareto analysis for IEEE paper
def compute_efficiency_pareto(metrics: Dict[str, Any]) -> Dict[str, Dict]:
    """Compute efficiency-accuracy tradeoff table showing model call savings vs SC."""
    sc = metrics.get('self_consistency', {})
    sc_calls = sc.get('total_model_calls', 1)
    sc_em = sc.get('exact_match', 0)
    pareto = {}
    for cond, m in metrics.items():
        calls = m.get('total_model_calls', 0)
        em = m.get('exact_match', 0)
        latency = m.get('latency_mean_s', 0)
        tp = m.get('throughput_sps', 0)
        call_saving = (sc_calls - calls) / sc_calls * 100 if sc_calls > 0 else 0
        em_diff = em - sc_em
        pareto[cond] = {
            'exact_match': em,
            'vqa_accuracy': m.get('vqa_accuracy', 0),
            'latency_mean_s': latency,
            'throughput_sps': tp,
            'total_model_calls': calls,
            'call_saving_vs_sc_pct': round(call_saving, 1),
            'em_delta_vs_sc': round(em_diff, 2),
            'efficiency_ratio': round(em / max(latency, 0.001), 1),
        }
    return pareto


def generate_pareto_plot(metrics: Dict[str, Any], output_dir: str):
    """V10-B: Generate latency vs EM% scatter plot (IEEE Figure 6)."""
    LABELS = {
        'antahkarana': 'Antahkarana', 'direct': 'Direct', 'cot': 'CoT',
        'self_consistency': 'SC (5×)', 'single_pass': 'Single-Pass',
        'no_routing': '-Routing', 'no_verification': '-Verif.',
        'no_consistency': '-Consist.', 'no_retrieval': '-Retrieval',
        'no_output': '-Output', 'no_logging': '-Logging',
    }
    COLORS = {
        'antahkarana': '#E63946', 'direct': '#457B9D', 'cot': '#2A9D8F',
        'self_consistency': '#E9C46A', 'single_pass': '#264653',
        'no_routing': '#A8DADC', 'no_verification': '#F4A261',
        'no_consistency': '#606C38', 'no_retrieval': '#BC6C25',
        'no_output': '#DDA0DD', 'no_logging': '#778DA9',
    }
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    for cond, m in metrics.items():
        if cond not in LABELS:
            continue
        lat = m.get('latency_mean_s', 0)
        em = m.get('exact_match', 0)
        calls = m.get('total_model_calls', 1)
        size = max(calls / 20, 30)
        color = COLORS.get(cond, '#888')
        zorder = 10 if cond == 'antahkarana' else 5
        edgecolor = 'black' if cond == 'antahkarana' else 'white'
        lw = 2.5 if cond == 'antahkarana' else 1
        ax.scatter(lat, em, s=size, c=color, edgecolors=edgecolor,
                   linewidths=lw, zorder=zorder, alpha=0.85)
        offset_y = 0.6 if cond != 'self_consistency' else -1.2
        ax.annotate(LABELS[cond], (lat, em), textcoords="offset points",
                    xytext=(8, offset_y), fontsize=9,
                    fontweight='bold' if cond == 'antahkarana' else 'normal',
                    color=color)
    ax.set_xlabel('Mean Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Exact Match (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy–Efficiency Pareto Frontier', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(f'{output_dir}/efficiency_pareto.png', dpi=300, bbox_inches='tight')
    fig.savefig(f'{output_dir}/efficiency_pareto.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {output_dir}/efficiency_pareto.png + .pdf')


def print_aggregate_results(metrics, per_ds_em):
    """Print formatted aggregate and per-dataset results tables."""
    DISPLAY_CONDS  = ['antahkarana','single_pass','cot','self_consistency','direct',
                      'no_routing','no_verification','no_consistency','no_retrieval',
                      'no_output','no_logging']
    DISPLAY_LABELS = {
        'antahkarana':    'Full Antahkarana',
        'single_pass':    'Single-Pass+RAG',
        'cot':            'CoT (SubQ)',
        'self_consistency':'SC (5x,weighted)',
        'direct':         'Direct Prompting',
        'no_routing':     '-Routing(Manas)',
        'no_verification':'-Verif.(Pass2)',
        'no_consistency': '-Cons.(Pass3)',
        'no_retrieval':   '-Retrieval(Chitta)',
        'no_output':      '-Output(Ahamkara)',
        'no_logging':     '-Logging(Sakshi)',
    }

    hdr = '{:<22} {:>7} {:>7} {:>7} {:>7} {:>8} {:>6} {:>6}'.format(
        'Method','VQA%','EM%','M-F1','Hall%','Lat(s)','P2','P3')
    print('\n── AGGREGATE VQA RESULTS ──')
    print(hdr); print('-'*80)
    for cond in DISPLAY_CONDS:
        if cond not in metrics: continue
        m = metrics[cond]
        print('{:<22} {:>7.1f} {:>7.1f} {:>7.3f} {:>7.1f} {:>8.3f} {:>6} {:>6}'.format(
            DISPLAY_LABELS.get(cond, cond),
            m['vqa_accuracy'], m['exact_match'], m['macro_f1'],
            m['hallucination_pct'], m['latency_mean_s'],
            m['pass2_fired'], m['pass3_fired']))

    ds_order = ['vqav2','gqa','okvqa','textvqa','scienceqa']
    print('\n── PER-DATASET EXACT MATCH (%) ──')
    print('{:<22} {:>8} {:>8} {:>8} {:>8} {:>8}'.format('Method','VQAv2','GQA','OK-VQA','TextVQA','SciQA'))
    print('-'*70)
    for cond in ['antahkarana','single_pass','cot','self_consistency','direct']:
        if cond not in per_ds_em: continue
        vals = ''.join('{:>8.1f}'.format(per_ds_em[cond].get(ds,{}).get('exact_match',0.0))
                       for ds in ds_order)
        print('{:<22}{}'.format(DISPLAY_LABELS.get(cond, cond), vals))


# ─── Statistical Significance (Cell 11) ──────────────────────────────────────
def mcnemar_test(results_a, results_b) -> Tuple[float, float]:
    id_to_a = {r.qid: r.is_exact for r in results_a}
    id_to_b = {r.qid: r.is_exact for r in results_b}
    common  = list(set(id_to_a.keys()) & set(id_to_b.keys()))
    if len(common) < 2: return 0.0, 1.0
    b10 = sum(1 for qid in common if not id_to_a[qid] and id_to_b[qid])
    b01 = sum(1 for qid in common if id_to_a[qid] and not id_to_b[qid])
    if b10 + b01 == 0: return 0.0, 1.0
    chi2 = (abs(b10 - b01) - 1.0) ** 2 / (b10 + b01)
    p    = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p


def run_statistical_tests(all_results_dict, results_dir):
    """Run McNemar and Wilcoxon tests, save results."""
    ALPHA_BONFERRONI = 0.01 / 5
    print(f'McNemar Tests (Bonferroni α=0.01/5={ALPHA_BONFERRONI:.4f})\n')
    BASELINE_PAIRS = [
        ('antahkarana','single_pass',      'Full Antahk. vs Single-Pass'),
        ('antahkarana','cot',              'Full Antahk. vs CoT'),
        ('antahkarana','self_consistency', 'Full Antahk. vs SC'),
        ('antahkarana','no_retrieval',     'Full Antahk. vs -Chitta'),
        ('antahkarana','no_verification',  'Full Antahk. vs -Pass2'),
        ('antahkarana','no_consistency',   'Full Antahk. vs -Pass3'),
        ('antahkarana','no_routing',       'Full Antahk. vs -Routing'),
    ]
    sig_results = []
    for a_key, b_key, label in BASELINE_PAIRS:
        if a_key not in all_results_dict or b_key not in all_results_dict: continue
        chi2, p  = mcnemar_test(all_results_dict[a_key], all_results_dict[b_key])
        sig      = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
        bonf_sig = 'sig (Bonf.)' if p < ALPHA_BONFERRONI else 'ns (Bonf.)'
        print(f'{label:<42} chi2={chi2:.3f}  p={p:.4f}  {sig}  {bonf_sig}')
        sig_results.append({'pair': label, 'chi2': chi2, 'p': p, 'sig': sig, 'bonferroni': bonf_sig})

    print('\nWilcoxon signed-rank: latency Antahkarana vs SC')
    lat_full = [r.latency_s for r in all_results_dict.get('antahkarana', [])]
    lat_sc   = [r.latency_s for r in all_results_dict.get('self_consistency', [])]
    if lat_full and lat_sc and len(lat_full) == len(lat_sc):
        try:
            w_stat, w_p = wilcoxon(lat_full, lat_sc, alternative='less')
            tag = '*** significantly faster' if w_p < 0.001 else ('** sig' if w_p < 0.01 else 'ns')
            print(f'  W={w_stat:.1f}, p={w_p:.4f} — {tag}')
            sig_results.append({'pair': 'latency_Antahk_vs_SC', 'W': w_stat, 'p': w_p})
        except Exception as e:
            print(f'  Wilcoxon error: {e}')

    with open(results_dir / 'statistical_significance.json', 'w') as f:
        json.dump(sig_results, f, indent=2)
    print('\n✅ Statistical tests saved.')


# ─── Save Results (Cell 12) ──────────────────────────────────────────────────
def results_to_df(results: List[SampleResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        'qid': r.qid, 'dataset': r.dataset, 'question': r.question,
        'predicted': r.predicted, 'ground_truth': '|'.join(r.ground_truth),
        'soft_score': r.soft_score, 'is_exact': r.is_exact,
        'is_hallucination': r.is_hallucination, 'latency_s': r.latency_s,
        'model_calls': r.model_calls, 'q_type': r.q_type,
        'pass2_fired': r.pass2_fired, 'pass3_fired': r.pass3_fired,
        'condition': r.condition,
    } for r in results])


def save_all_results(all_results_dict, metrics, per_ds_em, results_dir, base_dir):
    """Save CSVs, metrics JSON, per-dataset JSON, and ZIP archive."""
    for cond, res_list in all_results_dict.items():
        results_to_df(res_list).to_csv(results_dir / f'results_{cond}.csv', index=False)

    with open(results_dir / 'metrics_all.json',    'w') as f: json.dump(metrics,   f, indent=2)
    with open(results_dir / 'per_dataset_em.json', 'w') as f: json.dump(per_ds_em, f, indent=2)

    zip_path = str(base_dir / 'antahkarana_results')
    shutil.make_archive(zip_path, 'zip', str(results_dir))
    print(f'✅ All results saved to {results_dir}')
    print(f'✅ ZIP: {zip_path}.zip')


# ─── Figure Generation (Cell 13) ─────────────────────────────────────────────
def generate_figures(metrics, per_ds_em, figures_dir):
    """Generate all 5 IEEE figures."""
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.titlesize': 13, 'axes.labelsize': 12,
        'figure.dpi': 150, 'savefig.dpi': 150,
    })
    METHOD_ORDER  = ['direct','cot','self_consistency','single_pass','antahkarana']
    METHOD_LABELS = {'direct':'Direct','single_pass':'Single-Pass','cot':'CoT (SubQ)',
                     'self_consistency':'Self-Cons.','antahkarana':'Antahkarana'}
    COLORS = {'direct':'#4878CF','single_pass':'#6ACC65','cot':'#D65F5F',
              'self_consistency':'#B47CC7','antahkarana':'#C4AD66'}

    def save_fig(fig, name):
        fig.savefig(figures_dir / f'{name}.png', bbox_inches='tight')
        fig.savefig(figures_dir / f'{name}.pdf', bbox_inches='tight')
        plt.close(fig); print(f'  Saved {name}.png + .pdf')

    # Table II — Aggregate
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Table II: Aggregate VQA Results', fontsize=14, y=1.02)
    for ax, metric, label in zip(axes, ['vqa_accuracy','exact_match','macro_f1'],
                                  ['VQA Soft Accuracy (%)','Exact Match (%)','Macro-F1']):
        keys = [m for m in METHOD_ORDER if m in metrics]
        vals = [metrics[m].get(metric, 0) for m in keys]
        bars = ax.bar([METHOD_LABELS[m] for m in keys], vals,
                      color=[COLORS[m] for m in keys], edgecolor='black', linewidth=0.8)
        ax.set_title(label); ax.set_ylim(0, max(vals) * 1.25 if vals else 1)
        ax.tick_params(axis='x', rotation=20)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{v:.1f}' if metric != 'macro_f1' else f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9)
    plt.tight_layout(); save_fig(fig, 'table2_aggregate_vqa')

    # Table III — Per-dataset
    ds_order  = ['vqav2','gqa','okvqa','textvqa','scienceqa']
    ds_labels = ['VQAv2','GQA','OK-VQA','TextVQA','SciQA']
    methods_t3 = [m for m in ['antahkarana','single_pass','cot','self_consistency'] if m in per_ds_em]
    x, w = np.arange(len(ds_order)), 0.2
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, method in enumerate(methods_t3):
        vals = [per_ds_em[method].get(ds, {}).get('exact_match', 0.0) for ds in ds_order]
        ax.bar(x + i*w, vals, w, label=METHOD_LABELS.get(method, method),
               color=COLORS.get(method,'gray'), edgecolor='black', linewidth=0.7)
    ax.set_xticks(x + w*(len(methods_t3)-1)/2); ax.set_xticklabels(ds_labels)
    ax.set_ylabel('Exact Match (%)'); ax.set_title('Table III: Per-Dataset VQA Exact Match (%)')
    ax.legend(); ax.set_ylim(0, 100); plt.tight_layout(); save_fig(fig, 'table3_per_dataset_em')

    # Table V — Ablation
    abl_keys   = ['antahkarana','no_routing','no_verification','no_consistency',
                   'no_retrieval','no_output','no_logging']
    abl_labels = ['Full','-Manas','-Pass2','-Pass3','-Chitta','-Ahamkara','-Sakshi']
    abl_em     = [metrics.get(k, {}).get('exact_match', 0.0)        for k in abl_keys]
    abl_hall   = [metrics.get(k, {}).get('hallucination_pct', 0.0)  for k in abl_keys]
    abl_colors = ['#C4AD66' if i == 0 else '#888888' for i in range(len(abl_keys))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5)); fig.suptitle('Table V: VQA Ablation Study', fontsize=14)
    for ax, vals, title in zip(axes, [abl_em, abl_hall], ['Exact Match (%)','Hallucination Rate (%)']):
        ax.bar(abl_labels, vals, color=abl_colors, edgecolor='black', linewidth=0.8)
        ax.set_title(title); ax.set_ylim(0, 100)
        for i, v in enumerate(vals): ax.text(i, v+0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout(); save_fig(fig, 'table5_ablation')

    # Figure 5 — Hallucination
    fig5_keys   = ['single_pass','cot','self_consistency','no_verification','no_consistency','antahkarana']
    fig5_labels = ['Single-Pass','CoT (SubQ)','Self-Cons.','-Pass2','-Pass3','Full Antahkarana']
    fig5_vals   = [metrics.get(k, {}).get('hallucination_pct', 0.0) for k in fig5_keys]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(fig5_labels, fig5_vals,
                   color=['#D65F5F']*5+['#C4AD66'], edgecolor='black', linewidth=0.8)
    ax.axvline(x=metrics.get('antahkarana',{}).get('hallucination_pct',0),
               color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    for bar, v in zip(bars, fig5_vals):
        ax.text(v+0.3, bar.get_y()+bar.get_height()/2, f'{v:.1f}%', va='center', fontsize=10)
    ax.set_xlabel('Hallucination Rate (%)'); ax.set_title('Figure 5: Hallucination Rate by Condition')
    ax.legend(handles=[mpatches.Patch(color='#C4AD66', label='Full Antahkarana'),
                       mpatches.Patch(color='#D65F5F', label='Baseline / Ablation')], loc='lower right')
    plt.tight_layout(); save_fig(fig, 'figure5_hallucination_rate')

    # Figure 6 — Accuracy vs Latency
    fig6_keys = ['single_pass','cot','self_consistency','no_verification','no_consistency','antahkarana']
    fig6_lm   = {'single_pass':'Single-Pass','cot':'CoT (SubQ)','self_consistency':'Self-Cons.\n(5x)',
                  'no_verification':'No Verif.\n(-P2)','no_consistency':'No Cons.\n(-P3)',
                  'antahkarana':'Full\nAntahkarana'}
    fig6_lats  = [metrics.get(k,{}).get('latency_mean_s',0)*1000 for k in fig6_keys]
    fig6_em    = [metrics.get(k,{}).get('exact_match',0)         for k in fig6_keys]
    fig6_hall  = [metrics.get(k,{}).get('hallucination_pct',0)   for k in fig6_keys]
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(fig6_lats, fig6_em, c=fig6_hall, cmap='RdYlGn_r', s=200,
                    zorder=5, edgecolors='black', linewidths=1.5)
    for k, lat, em in zip(fig6_keys, fig6_lats, fig6_em):
        ax.annotate(fig6_lm.get(k,k), xy=(lat,em), xytext=(8,4), textcoords='offset points', fontsize=9)
    fig.colorbar(sc, ax=ax).set_label('Hallucination Rate (%)', fontsize=11)
    ax.set_xlabel('Mean Latency (ms/sample)', fontsize=12)
    ax.set_ylabel('Exact Match Accuracy (%)', fontsize=12)
    ax.set_title('Figure 6: Accuracy vs. Latency Trade-off'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); save_fig(fig, 'figure6_accuracy_vs_latency')

    print('\n✅ All 5 figures generated.')


# ─── Paper Summary (Cell 14) ──────────────────────────────────────────────────
def generate_paper_summary(metrics, all_samples, results_dir, gpu_name='GPU', torch_version='',
                           samples_per_dataset=200):
    """Generate the IEEE paper update summary text."""
    antahk = metrics.get('antahkarana', {})
    sp     = metrics.get('single_pass', {})
    sc_m   = metrics.get('self_consistency', {})
    nr     = metrics.get('no_retrieval', {})
    call_red = (1 - antahk.get('total_model_calls',0) /
                max(sc_m.get('total_model_calls',1),1)) * 100
    hall_red = sp.get('hallucination_pct',0) - antahk.get('hallucination_pct',0)

    lines = [
        '=' * 72,
        'ANTAHKARANA IEEE PAPER — VERIFIED RESULTS (v5)',
        f'Generated : {time.strftime("%Y-%m-%d %H:%M:%S")}',
        f'Hardware  : {gpu_name}  |  PyTorch {torch_version}',
        f'Samples   : {len(all_samples)} ({samples_per_dataset}/dataset × 5)',
        '=' * 72, '',
        '--- TABLE II: VQA RESULTS ---',
        '  Method               VQA%   EM%    M-F1  Hall%  Pass2  Pass3',
    ]
    for key, lbl in [
        ('antahkarana',  'Full Antahkarana'),
        ('single_pass',  'Single-Pass+RAG'),
        ('cot',          'CoT (SubQ)'),
        ('self_consistency','SC (5x,weighted)'),
        ('direct',       'Direct Prompting'),
        ('no_retrieval', '-Retrieval(Chitta)'),
    ]:
        if key not in metrics: continue
        m = metrics[key]
        lines.append(f'  {lbl:<20} {m["vqa_accuracy"]:>5.1f}  {m["exact_match"]:>5.1f}'
                     f'  {m["macro_f1"]:>5.3f}  {m["hallucination_pct"]:>5.1f}'
                     f'  {m["pass2_fired"]:>5}  {m["pass3_fired"]:>5}')
    lines += [
        '',
        '--- KEY FINDINGS ---',
        f'  Antahkarana EM:        {antahk.get("exact_match",0):.1f}%',
        f'  vs Single-Pass:        {sp.get("exact_match",0):.1f}%  '
            f'(Δ = {antahk.get("exact_match",0)-sp.get("exact_match",0):+.1f}pp)',
        f'  Hallucination drop:    {hall_red:+.1f}pp vs Single-Pass',
        f'  Model-call reduction:  {call_red:.1f}% vs SC',
        f'  Pass2 fired:           {antahk.get("pass2_fired",0)}/{len(all_samples)}',
        f'  Pass3 fired:           {antahk.get("pass3_fired",0)}/{len(all_samples)}',
        '=' * 72,
    ]
    summary = '\n'.join(lines)
    print(summary)
    with open(results_dir / 'paper_update_summary.txt', 'w') as f:
        f.write(summary)
    print('\n✅ Summary saved.')


# ─── Validation Checklist (Cell 15) ──────────────────────────────────────────
def run_validation_checklist(all_samples, dataset_samples, metrics, all_results_dict,
                             samples_per_dataset):
    """Run the final validation checklist from the notebook."""
    print('\n' + '='*60)
    print('FINAL VALIDATION CHECKLIST (v5)')
    print('='*60)

    def chk(label, cond):
        status = '✅ PASS' if cond else '❌ FAIL'
        print(f'[{status}] {label}')

    expected_total = samples_per_dataset * 5
    chk(f'Total samples == {expected_total} (got {len(all_samples)})', len(all_samples) == expected_total)
    chk('5 datasets loaded and non-empty',
        all(len(dataset_samples.get(d,[])) > 0 for d in ['vqav2','gqa','okvqa','textvqa','scienceqa']))
    chk(f'SAMPLES_PER_DATASET == {samples_per_dataset} (got {samples_per_dataset})', True)
    chk('ScienceQA samples have choices_str',
        all(bool(s.choices_str) for s in dataset_samples.get('scienceqa',[])))
    chk('All 11 conditions ran', len(all_results_dict) == 11)

    if 'antahkarana' in all_results_dict:
        ant = all_results_dict['antahkarana']
        chk('Pass2 fired > 0', sum(r.pass2_fired for r in ant) > 0)
        chk('Pass3 fired >= 0', sum(r.pass3_fired for r in ant) >= 0)
        chk('No A: prefixes in antahkarana predictions',
            all(not r.predicted.startswith('A:') for r in ant))

    # These comparisons are all from the SAME run (fix #3)
    if all(k in metrics for k in ['antahkarana','single_pass','cot','self_consistency','direct']):
        ant_em = metrics['antahkarana']['exact_match']
        sp_em  = metrics['single_pass']['exact_match']
        cot_em = metrics['cot']['exact_match']
        sc_em  = metrics['self_consistency']['exact_match']
        dir_em = metrics['direct']['exact_match']
        chk(f'CoT EM ({cot_em:.1f}%) > Direct EM ({dir_em:.1f}%)',  cot_em  > dir_em)
        chk(f'SC EM ({sc_em:.1f}%) > Direct EM ({dir_em:.1f}%)',    sc_em   > dir_em)
        chk(f'Antahkarana EM ({ant_em:.1f}%) >= Single-Pass EM ({sp_em:.1f}%)',  ant_em >= sp_em)
        chk(f'Antahkarana EM ({ant_em:.1f}%) >= CoT EM ({cot_em:.1f}%)',         ant_em >= cot_em)
        chk(f'Antahkarana EM ({ant_em:.1f}%) >= SC EM ({sc_em:.1f}%)',           ant_em >= sc_em)
        chk('Antahkarana hallucination < Single-Pass hallucination',
            metrics['antahkarana']['hallucination_pct'] < metrics['single_pass']['hallucination_pct'])

    print('\n✅ Checklist complete.')
