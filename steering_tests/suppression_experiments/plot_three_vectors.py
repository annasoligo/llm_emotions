#!/usr/bin/env python3
"""
Plot DPO vs Internal vs Act-on-Emotion suppression vectors.
3 rows x 2 cols, coherency > 70 filter.
"""

import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DPO_FILE = "steering_tests/behavioral_experiments/results_V0/suppression/qwen235b/blackmail/blackmail_fear55-65_sup55-65_20260202_075858_judged.jsonl"
INTERNAL_FILE = "steering_tests/behavioral_experiments/results/suppression/qwen235b/blackmail/blackmail_fear55-65_sup55-65_20260208_120341_judged.jsonl"
ACT_ON_EMO_FILE = "steering_tests/behavioral_experiments/results/suppression/qwen235b/blackmail/blackmail_fear55-65_sup55-65_20260208_141811_judged.jsonl"

MIN_COHERENCY = 70

SUP_COLORS = {
    0: "#d62728",
    5: "#ff7f0e",
    10: "#ffbb78",
    20: "#98df8a",
    50: "#2ca02c",
    75: "#1f77b4",
    100: "#9467bd",
}


def load_results(filepath):
    results = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def get_suppression_pct(condition):
    if "baseline" in condition:
        return None
    for sup in ["100", "75", "50", "20", "10", "5", "0"]:
        if f"_sup{sup}%" in condition:
            return int(sup)
    return None


def get_vector_type(condition):
    if "baseline" in condition:
        return "baseline"
    elif "tp_" in condition:
        return "TP"
    elif "high_" in condition:
        return "HIGH"
    return "unknown"


def aggregate_by_condition(results, min_coherency):
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append(r)

    aggregated = {}
    for cond, items in by_condition.items():
        fear_scores = [r["fear_judge"].get("fear_score") for r in items
                       if "fear_judge" in r and "error" not in r["fear_judge"]]
        fear_scores = [s for s in fear_scores if s is not None]

        coh_scores = [r["coherency_judge"].get("coherency_score") for r in items
                      if "coherency_judge" in r and "error" not in r["coherency_judge"]]
        coh_scores = [s for s in coh_scores if s is not None]

        behavior_scores = [
            1 if r.get("blackmail_judge", {}).get("is_blackmail", False) else 0
            for r in items
            if "blackmail_judge" in r and "error" not in r.get("blackmail_judge", {})
        ]

        if fear_scores and behavior_scores and coh_scores:
            coh_mean = np.mean(coh_scores)
            if coh_mean < min_coherency:
                continue
            n = len(behavior_scores)
            aggregated[cond] = {
                "fear_mean": np.mean(fear_scores),
                "fear_sem": np.std(fear_scores) / np.sqrt(len(fear_scores)),
                "behavior_mean": np.mean(behavior_scores) * 100,
                "behavior_sem": np.std(behavior_scores) / np.sqrt(n) * 100,
                "coherency_mean": coh_mean,
                "vector_type": get_vector_type(cond),
                "sup_pct": get_suppression_pct(cond),
                "n": n,
            }
    return aggregated


def plot_panel(ax, aggregated, vector_type, title):
    baseline_data = None
    for cond, data in aggregated.items():
        if data["vector_type"] == "baseline":
            baseline_data = data
            break

    vector_data = {c: d for c, d in aggregated.items() if d["vector_type"] == vector_type}

    if not vector_data and not baseline_data:
        ax.text(0.5, 0.5, "No data\n(all filtered by coherency)", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#888888")
        ax.set_title(title, fontsize=12)
        return

    if baseline_data:
        ax.errorbar(
            baseline_data["fear_mean"], baseline_data["behavior_mean"],
            xerr=baseline_data["fear_sem"], yerr=baseline_data["behavior_sem"],
            fmt="s", color="#333333", capsize=4, markersize=12,
            alpha=0.9, zorder=10, markeredgecolor='white', markeredgewidth=1,
        )
        ax.annotate("baseline", (baseline_data["fear_mean"], baseline_data["behavior_mean"]),
                     xytext=(8, -12), textcoords="offset points", fontsize=9, fontweight='bold')

    for cond, data in sorted(vector_data.items(), key=lambda x: x[1]["sup_pct"] or 0):
        sup_pct = data["sup_pct"]
        if sup_pct is None:
            continue
        color = SUP_COLORS.get(sup_pct, "#888888")
        ax.errorbar(
            data["fear_mean"], data["behavior_mean"],
            xerr=data["fear_sem"], yerr=data["behavior_sem"],
            fmt="o", color=color, capsize=4, markersize=11,
            alpha=0.9, zorder=5, markeredgecolor='white', markeredgewidth=1,
        )
        ax.annotate(f"{sup_pct}%", (data["fear_mean"], data["behavior_mean"]),
                     xytext=(8, 5), textcoords="offset points", fontsize=9)

    ax.set_xlabel("Fear Score", fontsize=11)
    ax.set_ylabel("Blackmail Rate (%)", fontsize=11)
    ax.set_ylim(-5, 80)
    ax.set_xlim(0, 85)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)


def main():
    plt.style.use("seaborn-v0_8-whitegrid")

    dpo_agg = aggregate_by_condition(load_results(DPO_FILE), MIN_COHERENCY)
    internal_agg = aggregate_by_condition(load_results(INTERNAL_FILE), MIN_COHERENCY)
    act_agg = aggregate_by_condition(load_results(ACT_ON_EMO_FILE), MIN_COHERENCY)

    # Print summaries
    for name, agg in [("DPO", dpo_agg), ("Internal", internal_agg), ("Act-on-Emotion", act_agg)]:
        print(f"\n{name} (coherency > {MIN_COHERENCY}):")
        for c, d in sorted(agg.items()):
            print(f"  {c}: coh={d['coherency_mean']:.0f}, fear={d['fear_mean']:.0f}, bl={d['behavior_mean']:.0f}%")

    # 3 rows x 2 cols
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    plot_panel(axes[0, 0], dpo_agg, "TP", "DPO Vectors — Text Pairs")
    plot_panel(axes[0, 1], dpo_agg, "HIGH", "DPO Vectors — High Emotion")
    plot_panel(axes[1, 0], internal_agg, "TP", "Internal Vectors — Text Pairs")
    plot_panel(axes[1, 1], internal_agg, "HIGH", "Internal Vectors — High Emotion")
    plot_panel(axes[2, 0], act_agg, "TP", "Act-on-Emotion Vectors — Text Pairs")
    plot_panel(axes[2, 1], act_agg, "HIGH", "Act-on-Emotion Vectors — High Emotion")

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#333333',
               markersize=11, label='Baseline', markeredgecolor='white'),
    ]
    for sup in [0, 5, 10, 20, 50, 75, 100]:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=SUP_COLORS[sup],
                   markersize=11, label=f'{sup}% suppression', markeredgecolor='white')
        )

    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 0.99), fontsize=11)

    plt.suptitle(
        f"Three Suppression Vector Types — Qwen 235B Blackmail\n"
        f"Fear Score vs Blackmail Rate (coherency > {MIN_COHERENCY}, layers 55-65)",
        y=1.03, fontsize=15
    )
    plt.tight_layout()

    output_dir = Path("steering_tests/behavioral_experiments/results/suppression/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "three_vectors_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
