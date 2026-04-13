"""
evaluation/visualize.py — Auto-generate IEEE-ready plots.

Generates:
  - Bar chart: EM/F1 comparison across methods
  - Latency vs Accuracy scatter
  - Ablation impact chart
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

PLOTS_DIR = Path("results/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Publication style
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams.update({
        "font.family":     "serif",
        "font.size":       11,
        "axes.titlesize":  13,
        "axes.labelsize":  11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi":      150,
        "axes.grid":       True,
        "grid.alpha":      0.3,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    logger.warning("matplotlib not available; skipping plots")


METHOD_COLORS = {
    "direct":           "#6c757d",
    "cot":              "#0d6efd",
    "self_consistency": "#198754",
    "tot":              "#fd7e14",
    "antahkarana":      "#dc3545",
}

METHOD_LABELS = {
    "direct":           "Direct",
    "cot":              "CoT",
    "self_consistency": "Self-Consistency",
    "tot":              "ToT",
    "antahkarana":      "Antahkarana",
}

DATASET_LABELS = {
    "hotpotqa":   "HotpotQA",
    "mmlu":       "MMLU",
    "truthfulqa": "TruthfulQA",
    "fever":      "FEVER",
    "svamp":      "SVAMP",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _get_mean(agg: Dict, metric: str) -> float:
    return agg.get(metric, {}).get("mean", 0.0)


def _get_ci(agg: Dict, metric: str) -> float:
    d = agg.get(metric, {})
    return (d.get("ci_high", d.get("mean", 0.0)) - d.get("mean", 0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: EM / F1 comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_em_f1_comparison(
    summary: Dict[str, Dict[str, Dict]],
    metric: str = "f1",
    output_path: str = None,
):
    """
    summary: {dataset: {method: aggregated_scores}}
    """
    if not MPL_AVAILABLE:
        return None

    datasets = [d for d in summary if any(summary[d].values())]
    methods  = list(METHOD_LABELS.keys())

    # Filter to methods that have data
    methods = [m for m in methods if any(m in summary.get(d, {}) for d in datasets)]

    if not datasets or not methods:
        return None

    n_datasets = len(datasets)
    n_methods  = len(methods)
    x          = np.arange(n_datasets)
    width      = 0.8 / n_methods
    offsets    = np.linspace(-(n_methods - 1) / 2 * width, (n_methods - 1) / 2 * width, n_methods)

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_datasets), 5))

    for i, method in enumerate(methods):
        means = []
        errs  = []
        for ds in datasets:
            agg = summary.get(ds, {}).get(method, {})
            means.append(_get_mean(agg, metric))
            errs.append(_get_ci(agg, metric))

        bars = ax.bar(
            x + offsets[i],
            means,
            width,
            label=METHOD_LABELS.get(method, method),
            color=METHOD_COLORS.get(method, "#888"),
            yerr=errs,
            capsize=3,
            error_kw={"elinewidth": 1.2, "alpha": 0.7},
            alpha=0.88,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS.get(d, d) for d in datasets], rotation=15, ha="right")
    ax.set_ylabel(metric.upper() + " Score")
    ax.set_title(f"Answer {metric.upper()} Comparison Across Datasets")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)

    plt.tight_layout()
    path = output_path or str(PLOTS_DIR / f"em_f1_{metric}_comparison.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Latency vs Accuracy scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_latency_vs_accuracy(
    summary: Dict[str, Dict[str, Dict]],
    dataset: str = "hotpotqa",
    metric: str = "f1",
    output_path: str = None,
):
    if not MPL_AVAILABLE:
        return None

    methods_data = summary.get(dataset, {})
    if not methods_data:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))

    plotted = []
    for method, agg in methods_data.items():
        latency  = _get_mean(agg, "mean_latency_s") or _get_mean(agg, "latency")
        accuracy = _get_mean(agg, metric)
        if latency == 0 and accuracy == 0:
            continue
        ax.scatter(
            latency, accuracy,
            s=180,
            c=METHOD_COLORS.get(method, "#888"),
            label=METHOD_LABELS.get(method, method),
            zorder=5,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(
            METHOD_LABELS.get(method, method),
            (latency, accuracy),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
        plotted.append(method)

    if not plotted:
        plt.close()
        return None

    ax.set_xlabel("Mean Latency per Sample (s)")
    ax.set_ylabel(f"{metric.upper()} Score")
    ax.set_title(f"Latency vs {metric.upper()} — {DATASET_LABELS.get(dataset, dataset)}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = output_path or str(PLOTS_DIR / f"latency_vs_{metric}_{dataset}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Ablation impact
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation(
    ablation_summary: Dict,
    metric: str = "f1",
    output_path: str = None,
):
    """
    ablation_summary: {dataset: {config_name: aggregated_scores_dict}}
    Produces one grouped horizontal bar chart per dataset, stacked vertically.
    """
    if not MPL_AVAILABLE:
        return None

    # Support both old flat structure (single dataset) and new nested structure
    # Detect by checking if inner values are dicts-of-dicts or dicts-of-scores
    first_val = next(iter(ablation_summary.values())) if ablation_summary else {}
    is_nested = isinstance(next(iter(first_val.values()), {}), dict) and \
                "mean" not in next(iter(first_val.values()), {})

    if not is_nested:
        # Legacy flat: wrap in a single "hotpotqa" key
        ablation_summary = {"hotpotqa": ablation_summary}

    datasets = list(ablation_summary.keys())
    n_datasets = len(datasets)
    if n_datasets == 0:
        return None

    fig, axes = plt.subplots(
        1, n_datasets,
        figsize=(6 * n_datasets, 4),
        sharey=True,
    )
    if n_datasets == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, datasets):
        ds_configs  = ablation_summary[ds_name]
        configs     = list(ds_configs.keys())
        means       = [_get_mean(ds_configs[c], metric) for c in configs]
        errs        = [_get_ci(ds_configs[c], metric)   for c in configs]

        if not configs:
            continue

        # Sort: full model first
        order   = sorted(range(len(configs)), key=lambda i: -means[i])
        configs = [configs[i] for i in order]
        means   = [means[i]   for i in order]
        errs    = [errs[i]    for i in order]

        colors = ["#dc3545" if "Full" in c else "#6c757d" for c in configs]

        bars = ax.barh(
            range(len(configs)), means, xerr=errs,
            color=colors, capsize=4, alpha=0.88,
            error_kw={"elinewidth": 1.2},
        )
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs, fontsize=8)
        ax.set_xlabel(f"{metric.upper()} Score")
        ax.set_title(f"{ds_name.upper()} (n=50)", fontsize=10)
        ax.set_xlim(0, (max(means) * 1.3 if means else 1.0))

        for bar, m in zip(bars, means):
            ax.text(
                m + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{m:.3f}", va="center", fontsize=8,
            )

    fig.suptitle(f"Ablation Study — {metric.upper()} across datasets", fontsize=11, y=1.02)
    plt.tight_layout()
    path = output_path or str(PLOTS_DIR / f"ablation_{metric}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Plot all
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(
    summary:          Dict,
    ablation_summary: Dict,
) -> List[str]:
    paths = []

    for metric in ("f1", "em"):
        p = plot_em_f1_comparison(summary, metric=metric)
        if p:
            paths.append(p)

    for dataset in ("hotpotqa", "mmlu", "svamp"):
        p = plot_latency_vs_accuracy(summary, dataset=dataset)
        if p:
            paths.append(p)

    if ablation_summary:
        for metric in ("f1", "em"):
            p = plot_ablation(ablation_summary, metric=metric)
            if p:
                paths.append(p)

    return paths
