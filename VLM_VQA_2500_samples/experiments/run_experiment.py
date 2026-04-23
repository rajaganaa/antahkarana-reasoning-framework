"""
ANTAHKARANA v7 — Experiment Runner
Runs the full experiment at scale: 5 datasets × 500 samples = 2500 total.
Includes Python logging to experiments/logs/run.log.

Usage:
    python -m experiments.run_experiment
    python -m experiments.run_experiment --samples_per_dataset 500
    python -m experiments.run_experiment --samples_per_dataset 200
"""

import sys
import os
import time
import json
import argparse
import warnings
import logging
from pathlib import Path

warnings.filterwarnings('ignore')

# ─── Ensure project root is on path ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    SAMPLES_PER_DATASET, BATCH_SIZE, RESULTS_DIR, FIGURES_DIR,
    BASE_DIR, GPU_NAME, GPU_MEM_GB, TORCH_VERSION, N_GPU, DEVICE,
    EXP_LOGS, EXP_RESULTS, HF_TOKEN
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
)

from tqdm.auto import tqdm
from typing import List, Tuple


def setup_logging():
    """Configure Python logging to file and console."""
    EXP_LOGS.mkdir(parents=True, exist_ok=True)
    log_path = EXP_LOGS / 'run.log'

    logger = logging.getLogger('antahkarana')
    logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh_fmt = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fh_fmt)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_fmt = logging.Formatter('%(levelname)-8s | %(message)s')
    ch.setFormatter(ch_fmt)
    logger.addHandler(ch)

    return logger


def run_pipeline_logged(fn, label, samples, all_samples, chitta, logger):
    """Run a pipeline with logging for each batch."""
    SEP = '=' * 60
    logger.info(f'{SEP}')
    logger.info(f'Running: {label} ({len(samples)} samples)')
    logger.info(f'{SEP}')
    print(f'\n{SEP}\nRunning: {label} ({len(samples)} samples)\n{SEP}')

    qid_to_global = {s.qid: i for i, s in enumerate(all_samples)}
    t_total, all_results = time.time(), []

    for batch_start in tqdm(range(0, len(samples), BATCH_SIZE), desc=label):
        batch   = samples[batch_start:batch_start+BATCH_SIZE]
        indices = [qid_to_global.get(s.qid, -1) for s in batch]

        try:
            res = fn(
                [s.image    for s in batch], [s.question    for s in batch],
                [s.qid      for s in batch], [s.answers     for s in batch],
                [s.dataset_name for s in batch], sample_indices=indices,
                all_samples=all_samples, chitta=chitta,
            )
            all_results.extend(res)

            # Log per-sample results
            for r in res:
                logger.debug(
                    f'  sample={r.qid} | dataset={r.dataset} | '
                    f'predicted="{r.predicted}" | exact={r.is_exact} | '
                    f'score={r.soft_score:.3f} | latency={r.latency_s:.4f}s'
                )
        except Exception as e:
            logger.error(f'  Batch {batch_start} error: {e}')
            import traceback
            logger.error(traceback.format_exc())

        free_gpu()

    wall_time = time.time() - t_total
    em = round(sum(r.is_exact for r in all_results)/len(all_results)*100, 1) if all_results else 0
    logger.info(f'  ✓ {label}: EM={em:.1f}%  ({wall_time/60:.1f} min)')
    return all_results, wall_time


def main():
    parser = argparse.ArgumentParser(description='ANTAHKARANA v7 — Scaled Experiment Runner')
    parser.add_argument('--samples_per_dataset', type=int, default=SAMPLES_PER_DATASET,
                        help=f'Samples per dataset (default: {SAMPLES_PER_DATASET} → {SAMPLES_PER_DATASET*5} total)')
    args = parser.parse_args()
    samples_per_dataset = args.samples_per_dataset

    logger = setup_logging()
    logger.info('='*60)
    logger.info('ANTAHKARANA v7 — Experiment Runner')
    logger.info(f'Samples per dataset: {samples_per_dataset}')
    logger.info(f'Total samples: {samples_per_dataset * 5}')
    logger.info('='*60)

    # ── Print hardware info ──────────────────────────────────────────────────
    if DEVICE.type == 'cuda':
        hw = f'GPU: {GPU_NAME} x {N_GPU} | {GPU_MEM_GB:.0f}GB | PyTorch {TORCH_VERSION} | Batch={BATCH_SIZE}'
    else:
        hw = f'CPU mode | PyTorch {TORCH_VERSION} | Batch={BATCH_SIZE}'
    logger.info(hw)

    # ── Step 1: Load datasets ────────────────────────────────────────────────
    logger.info('Loading datasets...')
    all_samples, dataset_samples, full_dataset = load_all_datasets(samples_per_dataset)
    for name, subset in dataset_samples.items():
        logger.info(f'  {name}: {len(subset)} samples')

    # ── Step 2: Load models ──────────────────────────────────────────────────
    logger.info('Loading models...')
    init_models()

    # ── Step 3: Build Chitta index ───────────────────────────────────────────
    logger.info('Building Chitta retrieval index...')
    chitta = build_chitta_index(all_samples)

    # ── Step 4: Run all 11 experiments ───────────────────────────────────────
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
            res, wt = run_pipeline_logged(fn, label, all_samples, all_samples, chitta, logger)
            all_results_dict[key] = res
            wall_times[key]       = wt
        except Exception as e:
            logger.error(f'❌ {label} failed: {e}')
            import traceback; traceback.print_exc()

    sakshi_logger.save(RESULTS_DIR / 'sakshi_log.jsonl')
    logger.info(f'All {len(all_results_dict)}/11 experiments complete.')

    # ── Step 5: Compute metrics ──────────────────────────────────────────────
    metrics, per_ds_em = {}, {}
    for cond, res_list in all_results_dict.items():
        metrics[cond]   = compute_metrics(res_list, cond)
        per_ds_em[cond] = compute_per_dataset(res_list)
    print_aggregate_results(metrics, per_ds_em)

    # ── Step 6: Statistical significance ─────────────────────────────────────
    run_statistical_tests(all_results_dict, RESULTS_DIR)

    # ── Step 7: Save all results ─────────────────────────────────────────────
    save_all_results(all_results_dict, metrics, per_ds_em, RESULTS_DIR, BASE_DIR)

    # ── Step 8: Generate figures ─────────────────────────────────────────────
    generate_figures(metrics, per_ds_em, FIGURES_DIR)

    # ── Step 9: Paper summary ────────────────────────────────────────────────
    generate_paper_summary(metrics, all_samples, RESULTS_DIR,
                          gpu_name=GPU_NAME, torch_version=TORCH_VERSION,
                          samples_per_dataset=samples_per_dataset)

    # ── Step 10: Validation checklist ────────────────────────────────────────
    run_validation_checklist(all_samples, dataset_samples, metrics,
                            all_results_dict, samples_per_dataset)

    # ── Save metrics to outputs/ and experiments/results/ ─────────────────────
    EXP_RESULTS.mkdir(parents=True, exist_ok=True)
    metrics_out = BASE_DIR / 'metrics.json'
    with open(metrics_out, 'w') as f:
        json.dump(metrics, f, indent=2)
    exp_metrics = EXP_RESULTS / 'metrics.json'
    with open(exp_metrics, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f'Metrics saved to {metrics_out} and {exp_metrics}')
    logger.info(f'Logs saved to {EXP_LOGS / "run.log"}')
    logger.info('🎉 ANTAHKARANA v7 experiment complete.')
    print(f'\n🎉 Experiment complete. Log: {EXP_LOGS / "run.log"}')


if __name__ == '__main__':
    main()
