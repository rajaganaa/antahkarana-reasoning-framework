"""
ANTAHKARANA v7 — Main Entry Point
Runs the full pipeline: load data → load models → run 11 experiments → evaluate → save.
Reproduces the exact execution flow of the original notebook.

Usage:
    python main.py
    python main.py --samples_per_dataset 200
    python main.py --samples_per_dataset 500
"""

import sys
import os
import time
import argparse
import warnings
import logging

warnings.filterwarnings('ignore')

# ─── Ensure project root is on path ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    SAMPLES_PER_DATASET_DEFAULT, BATCH_SIZE, RESULTS_DIR, FIGURES_DIR,
    BASE_DIR, GPU_NAME, GPU_MEM_GB, TORCH_VERSION, N_GPU, DEVICE,
    HF_TOKEN
)
from src.utils import free_gpu
from src.data_loader import load_all_datasets
from src.preprocessing import init_models
from src.manas_module import build_chitta_index
from src.antahkarana_model import run_buddhi_full, sakshi_logger, SampleResult
from src.training import (
    run_direct_prompting, run_cot_baseline, run_self_consistency,
    run_single_pass_blip2, run_no_routing, run_no_verification,
    run_no_consistency, run_no_retrieval, run_no_output, run_no_logging,
)
from src.evaluation import (
    compute_metrics, compute_per_dataset, print_aggregate_results,
    run_statistical_tests, save_all_results, generate_figures,
    generate_paper_summary, run_validation_checklist,
    # V9/V10 additions
    compute_metrics_with_ci, compute_per_qtype, compute_efficiency_pareto,
    generate_pareto_plot, bootstrap_ci, compute_bleu1,
)

from tqdm.auto import tqdm
from typing import List, Tuple


# ─── Pipeline Runner (from Cell 9) ──────────────────────────────────────────────
def run_pipeline(fn, label, samples, all_samples, chitta) -> Tuple[List[SampleResult], float]:
    SEP = '=' * 60
    print(f'\n{SEP}\nRunning: {label} ({len(samples)} samples)\n{SEP}')
    qid_to_global = {s.qid: i for i, s in enumerate(all_samples)}
    t_total, all_results = time.time(), []
    for i in tqdm(range(0, len(samples), BATCH_SIZE), desc=label):
        batch   = samples[i:i+BATCH_SIZE]
        indices = [qid_to_global.get(s.qid, -1) for s in batch]
        res = fn(
            [s.image    for s in batch], [s.question    for s in batch],
            [s.qid      for s in batch], [s.answers     for s in batch],
            [s.dataset_name for s in batch], sample_indices=indices,
            all_samples=all_samples, chitta=chitta,
        )
        all_results.extend(res)
        free_gpu()
    return all_results, time.time() - t_total


def main():
    parser = argparse.ArgumentParser(description='ANTAHKARANA v7 — Full Experiment Pipeline')
    parser.add_argument('--samples_per_dataset', type=int, default=SAMPLES_PER_DATASET_DEFAULT,
                        help=f'Samples per dataset (default: {SAMPLES_PER_DATASET_DEFAULT})')
    args = parser.parse_args()
    samples_per_dataset = args.samples_per_dataset

    # ── Print hardware info ──────────────────────────────────────────────────
    if DEVICE.type == 'cuda':
        print(f'GPU : {GPU_NAME} x {N_GPU}  |  {GPU_MEM_GB:.0f} GB VRAM  |  PyTorch {TORCH_VERSION}')
        print(f'Auto-selected BATCH_SIZE = {BATCH_SIZE}')
    else:
        print('⚠️  No GPU detected — CPU mode')
    print(f'HF token set: {HF_TOKEN[:8]}{"*" * max(0, len(HF_TOKEN)-8)}')
    print(f'Samples : {samples_per_dataset}/dataset × 5 = {samples_per_dataset*5} total')
    print(f'Output  : {BASE_DIR}')

    # ── Step 1: Load datasets ────────────────────────────────────────────────
    all_samples, dataset_samples, full_dataset = load_all_datasets(samples_per_dataset)

    # ── Step 2: Load models ──────────────────────────────────────────────────
    init_models()

    # ── Step 3: Build Chitta index ───────────────────────────────────────────
    chitta = build_chitta_index(all_samples)

    # ── Step 4: Run all 11 experiments (Cell 9) ──────────────────────────────
    # FIX #2: ALL 11 experiments must run — antahkarana was missing in v4
    all_results_dict, wall_times = {}, {}
    experiment_list = [
        ('direct',           run_direct_prompting,  'Direct Prompting'),
        ('cot',              run_cot_baseline,       'CoT (Visual Sub-Q Decomp.)'),
        ('self_consistency', run_self_consistency,   'Self-Consistency (n=5, weighted)'),
        ('single_pass',      run_single_pass_blip2,  'Single-Pass BLIP-2+RAG'),
        ('antahkarana',      run_buddhi_full,        'Full Antahkarana'),
        ('no_routing',       run_no_routing,         'Ablation: -Routing (Manas)'),
        ('no_verification',  run_no_verification,    'Ablation: -Verification (Pass2)'),
        ('no_consistency',   run_no_consistency,     'Ablation: -Consistency (Pass3)'),
        ('no_retrieval',     run_no_retrieval,       'Ablation: -Retrieval (Chitta)'),
        ('no_output',        run_no_output,          'Ablation: -Output (Ahamkara)'),
        ('no_logging',       run_no_logging,         'Ablation: -Logging (Sakshi)'),
    ]
    for key, fn, label in experiment_list:
        try:
            res, wt = run_pipeline(fn, label, all_samples, all_samples, chitta)
            all_results_dict[key] = res
            wall_times[key]       = wt
            em = round(sum(r.is_exact for r in res)/len(res)*100, 1) if res else 0
            print(f'  ✓ {label}: EM={em:.1f}%  ({wt/60:.1f} min)')
        except Exception as e:
            print(f'  ❌ {label} failed: {e}')
            import traceback; traceback.print_exc()

    sakshi_logger.save(RESULTS_DIR / 'sakshi_log.jsonl')
    print(f'\n✅ All {len(all_results_dict)}/11 experiments complete.')

    # ── Step 5: Compute metrics ──────────────────────────────────────────────
    metrics, per_ds_em = {}, {}
    for cond, res_list in all_results_dict.items():
        metrics[cond]   = compute_metrics(res_list, cond)
        per_ds_em[cond] = compute_per_dataset(res_list)
    print_aggregate_results(metrics, per_ds_em)

    # ── Step 5b: V9/V10 — Enhanced metrics for IEEE ──────────────────────────
    import json
    metrics_ci = {}
    for cond, res_list in all_results_dict.items():
        metrics_ci[cond] = compute_metrics_with_ci(res_list, cond)
    with open(RESULTS_DIR / 'metrics_with_ci.json', 'w') as f:
        json.dump(metrics_ci, f, indent=2)
    print('  ✅ Saved metrics_with_ci.json (95% bootstrap CIs)')

    # V10-B: Per-question-type breakdown
    if 'antahkarana' in all_results_dict:
        qtype_breakdown = compute_per_qtype(all_results_dict['antahkarana'])
        with open(RESULTS_DIR / 'per_qtype_breakdown.json', 'w') as f:
            json.dump(qtype_breakdown, f, indent=2)
        print('  ✅ Saved per_qtype_breakdown.json')

    # V10-B: Efficiency Pareto table
    pareto = compute_efficiency_pareto(metrics)
    with open(RESULTS_DIR / 'efficiency_pareto.json', 'w') as f:
        json.dump(pareto, f, indent=2)
    print('  ✅ Saved efficiency_pareto.json')

    # ── Step 6: Statistical significance ─────────────────────────────────────
    run_statistical_tests(all_results_dict, RESULTS_DIR)

    # ── Step 7: Save all results ─────────────────────────────────────────────
    save_all_results(all_results_dict, metrics, per_ds_em, RESULTS_DIR, BASE_DIR)

    # ── Step 8: Generate figures ─────────────────────────────────────────────
    generate_figures(metrics, per_ds_em, FIGURES_DIR)
    # V10-B: Pareto plot (IEEE Figure 6)
    generate_pareto_plot(metrics, str(FIGURES_DIR))

    # ── Step 9: Paper summary ────────────────────────────────────────────────
    generate_paper_summary(metrics, all_samples, RESULTS_DIR,
                          gpu_name=GPU_NAME, torch_version=TORCH_VERSION,
                          samples_per_dataset=samples_per_dataset)

    # ── Step 10: Validation checklist ────────────────────────────────────────
    run_validation_checklist(all_samples, dataset_samples, metrics,
                            all_results_dict, samples_per_dataset)

    # ── Save metrics.json to outputs/ ─────────────────────────────────────────
    metrics_out = BASE_DIR / 'metrics.json'
    with open(metrics_out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\n✅ Final metrics saved to {metrics_out}')
    print('🎉 ANTAHKARANA v7 pipeline complete.')


if __name__ == '__main__':
    main()
