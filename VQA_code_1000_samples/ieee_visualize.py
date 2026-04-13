"""
ieee_visualize.py — Antahkarana Cognitive Architecture
IEEE-standard publication-quality visualizations.

Figures generated:
  Fig 1. Ablation bar chart — VQA Accuracy + Hallucination % (dual-axis)
  Fig 2. Radar / spider chart — multi-metric comparison across conditions
  Fig 3. Latency CDF — cumulative distribution of per-sample latency
  Fig 4. Consistency distribution — histogram per condition
  Fig 5. Statistical significance heatmap — McNemar p-values (corrected)
  Fig 6. Per-answer-type F1 grouped bar chart
  Fig 7. Accuracy vs Throughput scatter — Pareto frontier

All figures use ACM/IEEE-compatible styling:
  - 8pt font (matches two-column paper body text)
  - Single-column width = 3.5 in, double-column = 7.25 in
  - Black-and-white safe (pattern fills + marker shapes)
  - PDF vector output + PNG 300 dpi fallback
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# IEEE style config
# ─────────────────────────────────────────────────────────────────────────────

IEEE_STYLE = {
    "font.family":       "serif",
    "font.size":         8,
    "axes.titlesize":    9,
    "axes.labelsize":    8,
    "xtick.labelsize":   7,
    "ytick.labelsize":   7,
    "legend.fontsize":   7,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.02,
    "lines.linewidth":   1.2,
    "axes.linewidth":    0.8,
    "grid.linewidth":    0.5,
    "grid.alpha":        0.4,
}

# IEEE-safe colour palette (distinguishable in B&W via patterns)
COLOURS = {
    "baseline":         "#555555",
    "cot_baseline":     "#888888",
    "self_consistency": "#AAAAAA",
    "no_pass2":         "#3399CC",
    "no_pass3":         "#FF7700",
    "full":             "#CC2222",
}

HATCHES = {
    "baseline":         "////",
    "cot_baseline":     "xxxx",
    "self_consistency": "....",
    "no_pass2":         "----",
    "no_pass3":         "||||",
    "full":             "",
}

LABELS = {
    "baseline":         "Single-Pass Baseline",
    "cot_baseline":     "CoT Baseline",
    "self_consistency": "Self-Consistency (3×)",
    "no_pass2":         "No Verification (–P2)",
    "no_pass3":         "No Consistency (–P3)",
    "full":             "Full Antahkarana ★",
}

CONDITION_ORDER = [
    "baseline", "cot_baseline", "self_consistency",
    "no_pass2", "no_pass3", "full",
]

_W1 = 3.5   # single-column inches
_W2 = 7.25  # double-column inches


def _apply_style():
    if _MATPLOTLIB_AVAILABLE:
        plt.rcParams.update(IEEE_STYLE)


def _save(fig, path: Path, suffix: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.stem + suffix
    pdf_path = path.with_name(stem + ".pdf")
    png_path = path.with_name(stem + ".png")
    try:
        fig.savefig(str(pdf_path))
    except Exception:
        pass
    fig.savefig(str(png_path), dpi=300)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: Ablation bar chart (accuracy + hallucination dual-axis)
# ─────────────────────────────────────────────────────────────────────────────

def fig_ablation_bar(ablation_data: List[Dict], out_dir: Path) -> None:
    """
    Dual-axis grouped bar chart:
      Left  axis → VQA Accuracy (%)
      Right axis → Hallucination Rate (%)
    Each condition gets one accuracy bar + one hallucination bar.
    """
    if not _MATPLOTLIB_AVAILABLE:
        return
    _apply_style()

    cond_key_map = {
        "Single-Pass Baseline":  "baseline",
        "CoT Baseline":          "cot_baseline",
        "Self-Consistency (3x)": "self_consistency",
        "No Verification (–P2)": "no_pass2",
        "No Consistency (–P3)":  "no_pass3",
        "Full Antahkarana ★":    "full",
    }

    rows = []
    for d in ablation_data:
        key = cond_key_map.get(d.get("condition", ""), d.get("condition", ""))
        rows.append({
            "key":  key,
            "acc":  d.get("vqa_accuracy", 0) * 100,
            "hall": d.get("hallucination_pct", 0),
        })
    rows.sort(key=lambda r: CONDITION_ORDER.index(r["key"]) if r["key"] in CONDITION_ORDER else 99)

    keys   = [r["key"] for r in rows]
    accs   = [r["acc"]  for r in rows]
    halls  = [r["hall"] for r in rows]
    labels = [LABELS.get(k, k) for k in keys]

    x   = np.arange(len(keys))
    w   = 0.35
    fig, ax1 = plt.subplots(figsize=(_W2, 2.6))
    ax2 = ax1.twinx()

    bars_acc  = ax1.bar(x - w/2, accs,  w, label="VQA Accuracy (%)",
                        color=[COLOURS.get(k, "#999") for k in keys],
                        hatch=[HATCHES.get(k, "") for k in keys],
                        edgecolor="black", linewidth=0.7, zorder=3)
    bars_hall = ax2.bar(x + w/2, halls, w, label="Hallucination (%)",
                        color="white",
                        hatch=[HATCHES.get(k, "") for k in keys],
                        edgecolor="#CC2222", linewidth=0.9, zorder=3)

    # Value annotations
    for bar, v in zip(bars_acc, accs):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.8,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=6)
    for bar, v in zip(bars_hall, halls):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=6, color="#CC2222")

    ax1.set_xlabel("Method")
    ax1.set_ylabel("VQA Accuracy (%)")
    ax2.set_ylabel("Hallucination Rate (%)", color="#CC2222")
    ax2.tick_params(axis="y", colors="#CC2222")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=18, ha="right")
    ax1.set_ylim(0, 105)
    ax2.set_ylim(0, 30)
    ax1.yaxis.grid(True, zorder=0)
    ax1.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(facecolor="#555", edgecolor="black", label="VQA Accuracy (left)"),
        mpatches.Patch(facecolor="white", edgecolor="#CC2222", label="Hallucination % (right)"),
    ]
    ax1.legend(handles=legend_patches, loc="upper left", framealpha=0.9)
    ax1.set_title("Fig. 1: Ablation Study — VQA Accuracy and Hallucination Rate", pad=4)
    fig.tight_layout()
    _save(fig, out_dir / "fig1_ablation_bar")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Radar chart — multi-metric comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig_radar(ablation_data: List[Dict], out_dir: Path) -> None:
    """Radar/spider chart: accuracy, 1-hallucination, consistency, 1/latency (normalised)."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    _apply_style()

    metrics_def = [
        ("VQA Acc",      lambda d: d.get("vqa_accuracy", 0)),
        ("1-Hall",       lambda d: 1 - d.get("hallucination_pct", 0) / 100),
        ("Consistency",  lambda d: d.get("consistency", 0)),
        ("1/Latency",    lambda d: 1 / max(d.get("mean_latency_s", 1), 0.01)),
        ("Throughput",   lambda d: d.get("throughput_sps", 0)),
    ]

    m_labels = [m[0] for m in metrics_def]
    N        = len(m_labels)
    angles   = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles  += angles[:1]

    cond_key_map = {
        "Single-Pass Baseline":  "baseline",
        "CoT Baseline":          "cot_baseline",
        "Self-Consistency (3x)": "self_consistency",
        "No Verification (–P2)": "no_pass2",
        "No Consistency (–P3)":  "no_pass3",
        "Full Antahkarana ★":    "full",
    }

    # Collect raw values, then normalise to [0,1]
    rows = []
    for d in ablation_data:
        key  = cond_key_map.get(d.get("condition",""), d.get("condition",""))
        vals = [fn(d) for _, fn in metrics_def]
        rows.append({"key": key, "vals": vals})

    # Normalise each metric column
    for col in range(N):
        col_vals = [r["vals"][col] for r in rows]
        vmin, vmax = min(col_vals), max(col_vals)
        rng = vmax - vmin if vmax > vmin else 1
        for r in rows:
            r["vals"][col] = (r["vals"][col] - vmin) / rng

    fig, ax = plt.subplots(figsize=(_W1, _W1), subplot_kw={"polar": True})

    for r in rows:
        key   = r["key"]
        vals  = r["vals"] + r["vals"][:1]
        color = COLOURS.get(key, "#999999")
        ax.plot(angles, vals, linewidth=1.2, color=color, label=LABELS.get(key, key))
        ax.fill(angles, vals, alpha=0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(m_labels, size=7)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"], size=6)
    ax.set_title("Fig. 2: Multi-Metric Radar", pad=14, size=9)
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.1), fontsize=6)
    _save(fig, out_dir / "fig2_radar")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Latency CDF
# ─────────────────────────────────────────────────────────────────────────────

def fig_latency_cdf(results_by_condition: Dict[str, Any], out_dir: Path) -> None:
    """Empirical CDF of per-sample latency for each condition."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    _apply_style()

    fig, ax = plt.subplots(figsize=(_W2, 2.4))
    linestyles = ["-", "--", "-.", ":", (0,(3,1,1,1)), (0,(5,1))]
    markers    = ["o", "s", "D", "^", "v", "*"]

    for i, cond in enumerate(CONDITION_ORDER):
        results = results_by_condition.get(cond, [])
        if not results:
            continue
        lats = sorted([r.latency_total_s for r in results if r.latency_total_s > 0])
        if not lats:
            continue
        cdf = np.arange(1, len(lats)+1) / len(lats)
        ax.plot(lats, cdf,
                color=COLOURS.get(cond, "#999"),
                linestyle=linestyles[i % len(linestyles)],
                linewidth=1.2,
                label=LABELS.get(cond, cond),
                markevery=max(1, len(lats)//10),
                marker=markers[i % len(markers)],
                markersize=3)

    ax.set_xlabel("Latency per sample (s)")
    ax.set_ylabel("Cumulative Fraction")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.yaxis.grid(True, alpha=0.4)
    ax.legend(loc="lower right", fontsize=6)
    ax.set_title("Fig. 3: Latency CDF by Condition", pad=4)
    _save(fig, out_dir / "fig3_latency_cdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Consistency histogram
# ─────────────────────────────────────────────────────────────────────────────

def fig_consistency_hist(results_by_condition: Dict[str, Any], out_dir: Path) -> None:
    """Histogram of consistency scores (0.33, 0.67, 1.0) per condition."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    _apply_style()

    multi_pass = [c for c in CONDITION_ORDER
                  if c not in ("baseline", "cot_baseline") and results_by_condition.get(c)]
    if not multi_pass:
        return

    n_cond = len(multi_pass)
    fig, axes = plt.subplots(1, n_cond, figsize=(_W2, 2.0), sharey=True)
    if n_cond == 1:
        axes = [axes]

    bins = np.array([0.0, 0.4, 0.7, 1.01])
    bin_labels = ["0.33\n(no agree)", "0.67\n(partial)", "1.00\n(full)"]

    for ax, cond in zip(axes, multi_pass):
        results = results_by_condition[cond]
        scores  = [r.consistency_score for r in results if r.consistency_score >= 0]
        if not scores:
            continue
        counts, _ = np.histogram(scores, bins=bins)
        fracs      = counts / max(counts.sum(), 1)
        x          = np.arange(len(bin_labels))
        ax.bar(x, fracs, color=COLOURS.get(cond,"#999"),
               hatch=HATCHES.get(cond,""), edgecolor="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=6)
        ax.set_title(LABELS.get(cond, cond)[:18], fontsize=7)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("Fraction of samples")
    fig.suptitle("Fig. 4: Consistency Score Distribution", fontsize=9, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir / "fig4_consistency_hist")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: Statistical significance heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig_significance_heatmap(sig_results: Dict, out_dir: Path) -> None:
    """Heatmap of Bonferroni-corrected McNemar p-values."""
    if not _MATPLOTLIB_AVAILABLE or not sig_results:
        return
    _apply_style()

    comparisons = sig_results.get("comparisons", {})
    if not comparisons:
        return

    cond_names = [k.replace("full_vs_", "") for k in comparisons]
    p_vals_raw = []
    p_vals_bon = []
    deltas     = []

    for k in comparisons:
        mc = comparisons[k]["accuracy"]["mcnemar"]
        p_vals_raw.append(mc.get("p_value", 1.0))
        p_vals_bon.append(mc.get("p_value_bonferroni", 1.0))
        deltas.append(comparisons[k]["accuracy"].get("delta", 0.0))

    n = len(cond_names)
    fig, axes = plt.subplots(1, 2, figsize=(_W2, 1.8 + 0.3*n))

    for ax, vals, title in [
        (axes[0], p_vals_raw, "Raw McNemar p-value"),
        (axes[1], p_vals_bon, "Bonferroni-corrected p-value"),
    ]:
        mat = np.array(vals).reshape(-1, 1)
        im  = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.1)
        ax.set_xticks([0])
        ax.set_xticklabels(["vs Full Antahkarana"], fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels([LABELS.get(c, c) for c in cond_names], fontsize=7)
        ax.set_title(title, fontsize=8)
        for i, v in enumerate(vals):
            sig = "***" if v < 0.001 else ("**" if v < 0.01 else ("*" if v < 0.05 else "ns"))
            ax.text(0, i, f"{v:.3f}\n({sig})", ha="center", va="center", fontsize=6,
                    color="white" if v < 0.03 else "black")
        plt.colorbar(im, ax=ax, fraction=0.15, pad=0.04).ax.tick_params(labelsize=6)

    fig.suptitle("Fig. 5: Statistical Significance (McNemar χ²)", fontsize=9)
    fig.tight_layout()
    _save(fig, out_dir / "fig5_significance_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: Per-answer-type F1 grouped bar
# ─────────────────────────────────────────────────────────────────────────────

def fig_f1_by_type(sig_results: Dict, out_dir: Path) -> None:
    """Grouped bar: macro-F1 per answer type for each condition."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    _apply_style()

    comparisons = sig_results.get("comparisons", {})
    if not comparisons:
        return

    # Collect per-type F1 for each condition (including full)
    conds  = {}
    types  = ["yes_no", "number", "other"]

    # full condition
    full_f1_found = False
    for k, v in comparisons.items():
        mf = v.get("macro_f1", {})
        if not full_f1_found:
            # We only have macro_f1 scalars here, not per-type; skip per-type breakdown
            full_f1_found = True

    # Fall back to just plotting macro_f1 scalars
    labels_cond = []
    full_vals   = []
    other_vals  = []

    for k in comparisons:
        other_name = k.replace("full_vs_", "")
        mf = comparisons[k]["macro_f1"]
        labels_cond.append(LABELS.get(other_name, other_name))
        full_vals.append(mf["full"])
        other_vals.append(mf["other"])

    x = np.arange(len(labels_cond))
    w = 0.35
    fig, ax = plt.subplots(figsize=(_W2, 2.4))
    ax.bar(x - w/2, full_vals,  w, label="Full Antahkarana",
           color=COLOURS["full"], edgecolor="black", linewidth=0.7)
    ax.bar(x + w/2, other_vals, w, label="Compared Method",
           color=COLOURS["baseline"], hatch="////", edgecolor="black", linewidth=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_cond, rotation=18, ha="right", fontsize=7)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.0)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_title("Fig. 6: Macro-F1 Comparison (Full Antahkarana vs Baselines)", pad=4)
    ax.legend(fontsize=7)
    _save(fig, out_dir / "fig6_macro_f1")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Accuracy vs Throughput scatter (Pareto)
# ─────────────────────────────────────────────────────────────────────────────

def fig_accuracy_throughput(ablation_data: List[Dict], out_dir: Path) -> None:
    """Scatter plot: VQA Accuracy vs Throughput (sps). Pareto frontier highlighted."""
    if not _MATPLOTLIB_AVAILABLE:
        return
    _apply_style()

    cond_key_map = {
        "Single-Pass Baseline":  "baseline",
        "CoT Baseline":          "cot_baseline",
        "Self-Consistency (3x)": "self_consistency",
        "No Verification (–P2)": "no_pass2",
        "No Consistency (–P3)":  "no_pass3",
        "Full Antahkarana ★":    "full",
    }
    markers = {"baseline":"o","cot_baseline":"s","self_consistency":"D",
               "no_pass2":"^","no_pass3":"v","full":"*"}

    fig, ax = plt.subplots(figsize=(_W1 + 0.5, _W1))
    pareto  = []

    for d in ablation_data:
        key   = cond_key_map.get(d.get("condition",""), d.get("condition",""))
        acc   = d.get("vqa_accuracy", 0) * 100
        tput  = d.get("throughput_sps", 0)
        color = COLOURS.get(key, "#999")
        mk    = markers.get(key, "o")
        ms    = 10 if key == "full" else 6
        ax.scatter(tput, acc, color=color, marker=mk, s=ms**2,
                   edgecolors="black", linewidths=0.6, zorder=4,
                   label=LABELS.get(key, key))
        ax.annotate(LABELS.get(key, key)[:12],
                    xy=(tput, acc), xytext=(3, 3), textcoords="offset points",
                    fontsize=5.5, color=color)
        pareto.append((tput, acc))

    # Pareto frontier (maximise both axes)
    pareto_sorted = sorted(pareto, key=lambda x: x[0])
    frontier = []
    best_acc = -1
    for tput, acc in pareto_sorted:
        if acc > best_acc:
            frontier.append((tput, acc))
            best_acc = acc
    if len(frontier) > 1:
        fx, fy = zip(*frontier)
        ax.plot(fx, fy, "k--", linewidth=0.9, label="Pareto frontier", zorder=3)

    ax.set_xlabel("Throughput (samples/sec) ↑")
    ax.set_ylabel("VQA Accuracy (%) ↑")
    ax.yaxis.grid(True, alpha=0.4)
    ax.xaxis.grid(True, alpha=0.4)
    ax.legend(fontsize=5.5, loc="lower right")
    ax.set_title("Fig. 7: Accuracy–Throughput Trade-off", pad=4)
    _save(fig, out_dir / "fig7_accuracy_throughput")


# ─────────────────────────────────────────────────────────────────────────────
# Master runner
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(
    ablation_json_path: str,
    significance_json_path: str,
    results_by_condition: Optional[Dict] = None,
    out_dir: str = "antahkarana/results/figures",
) -> None:
    """
    Load ablation + significance JSONs and generate all 7 IEEE figures.
    results_by_condition: optional dict of {condition: List[AntahkaranaResult]}
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not _MATPLOTLIB_AVAILABLE:
        print("[Viz] matplotlib not available — skipping figure generation")
        return

    ablation_data = []
    try:
        with open(ablation_json_path) as f:
            ablation_data = json.load(f)
    except Exception as e:
        print(f"[Viz] Could not load ablation JSON: {e}")

    sig_results = {}
    try:
        with open(significance_json_path) as f:
            sig_results = json.load(f)
    except Exception as e:
        print(f"[Viz] Could not load significance JSON: {e}")

    print("[Viz] Generating Fig 1: Ablation bar chart…")
    fig_ablation_bar(ablation_data, out)

    print("[Viz] Generating Fig 2: Radar chart…")
    fig_radar(ablation_data, out)

    if results_by_condition:
        print("[Viz] Generating Fig 3: Latency CDF…")
        fig_latency_cdf(results_by_condition, out)

        print("[Viz] Generating Fig 4: Consistency histogram…")
        fig_consistency_hist(results_by_condition, out)

    print("[Viz] Generating Fig 5: Significance heatmap…")
    fig_significance_heatmap(sig_results, out)

    print("[Viz] Generating Fig 6: Macro-F1 grouped bar…")
    fig_f1_by_type(sig_results, out)

    print("[Viz] Generating Fig 7: Accuracy–Throughput scatter…")
    fig_accuracy_throughput(ablation_data, out)

    print(f"[Viz] All figures saved to {out}/")


if __name__ == "__main__":
    import sys
    ablation_path = sys.argv[1] if len(sys.argv) > 1 else "antahkarana/results/ablation_comparison.json"
    sig_path      = sys.argv[2] if len(sys.argv) > 2 else "antahkarana/results/statistical_significance.json"
    out_dir       = sys.argv[3] if len(sys.argv) > 3 else "antahkarana/results/figures"
    generate_all_figures(ablation_path, sig_path, out_dir=out_dir)
