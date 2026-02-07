#!/usr/bin/env python3
"""
Plot fear vs behavior (blackmail/sandbagging) scatter charts with error bars.
One subplot per vector type, showing baseline + all suppression levels.
"""

import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Result files
RESULT_FILES = {
    "qwen235b_blackmail_low": "steering_tests/behavioral_experiments/results/suppression/qwen235b/blackmail/blackmail_fear35-45_sup35-45_20260201_152219_judged.jsonl",
    "qwen235b_blackmail_high": "steering_tests/behavioral_experiments/results/suppression/qwen235b/blackmail/blackmail_fear35-45_sup35-45_20260201_152711_judged.jsonl",
    "qwen235b_sandbagging_low": "steering_tests/behavioral_experiments/results/suppression/qwen235b/sandbagging/sandbagging_fear35-45_sup35-45_20260201_153458_judged.jsonl",
    "qwen235b_sandbagging_high": "steering_tests/behavioral_experiments/results/suppression/qwen235b/sandbagging/sandbagging_fear35-45_sup35-45_20260201_154652_judged.jsonl",
    "gemma27b_blackmail_low": "steering_tests/behavioral_experiments/results/suppression/gemma27b/blackmail/blackmail_fear40-44_sup40-44_20260201_151204_judged.jsonl",
}

MIN_COHERENCY = 20


def load_results(filepath):
    """Load JSONL results."""
    results = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def get_suppression_pct(condition):
    """Extract suppression percentage as integer from condition string."""
    if "baseline" in condition:
        return None
    for sup in ["100", "75", "50", "20", "10", "5", "0"]:
        if f"_sup{sup}%" in condition:
            return int(sup)
    return None


def get_vector_type(condition):
    """Get vector type from condition."""
    if "baseline" in condition:
        return "baseline"
    elif "tp_" in condition:
        return "TP"
    elif "high_" in condition:
        return "HIGH"
    return "unknown"


def aggregate_by_condition(results, scenario):
    """Aggregate results by condition, computing means and SEM."""
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append(r)

    aggregated = {}
    for cond, items in by_condition.items():
        # Fear scores
        fear_scores = [
            r["fear_judge"].get("fear_score")
            for r in items
            if "fear_judge" in r and "error" not in r["fear_judge"]
        ]
        fear_scores = [s for s in fear_scores if s is not None]

        # Coherency scores
        coh_scores = [
            r["coherency_judge"].get("coherency_score")
            for r in items
            if "coherency_judge" in r and "error" not in r["coherency_judge"]
        ]
        coh_scores = [s for s in coh_scores if s is not None]

        # Behavior scores
        if scenario == "blackmail":
            behavior_scores = [
                1 if r.get("blackmail_judge", {}).get("is_blackmail", False) else 0
                for r in items
                if "blackmail_judge" in r and "error" not in r.get("blackmail_judge", {})
            ]
        else:
            behavior_scores = [
                r["sandbagging_judge"].get("sandbagging_score")
                for r in items
                if "sandbagging_judge" in r and "error" not in r["sandbagging_judge"]
            ]
            behavior_scores = [s for s in behavior_scores if s is not None]

        if fear_scores and behavior_scores and coh_scores:
            coh_mean = np.mean(coh_scores)
            if coh_mean < MIN_COHERENCY:
                continue  # Filter out low coherency

            n = len(behavior_scores)
            aggregated[cond] = {
                "fear_mean": np.mean(fear_scores),
                "fear_sem": np.std(fear_scores) / np.sqrt(len(fear_scores)),
                "behavior_mean": np.mean(behavior_scores) * (100 if scenario == "blackmail" else 1),
                "behavior_sem": np.std(behavior_scores) / np.sqrt(n) * (100 if scenario == "blackmail" else 1),
                "coherency_mean": coh_mean,
                "vector_type": get_vector_type(cond),
                "sup_pct": get_suppression_pct(cond),
                "n": n,
            }

    return aggregated


def plot_vector_panel(ax, aggregated, vector_type, scenario, title):
    """Plot a single panel for a specific vector type, showing all suppression levels."""

    # Get baseline
    baseline_data = None
    for cond, data in aggregated.items():
        if data["vector_type"] == "baseline":
            baseline_data = data
            break

    # Get all conditions for this vector type
    vector_data = {cond: data for cond, data in aggregated.items()
                   if data["vector_type"] == vector_type}

    if not vector_data and not baseline_data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=13)
        return

    # Color map for suppression levels
    sup_colors = {
        0: "#d62728",    # red - no suppression (fear active)
        5: "#ff7f0e",    # orange
        10: "#ffbb78",   # light orange
        20: "#98df8a",   # light green
        50: "#2ca02c",   # green
        75: "#1f77b4",   # blue
        100: "#9467bd",  # purple - max suppression
    }

    # Plot baseline
    if baseline_data:
        ax.errorbar(
            baseline_data["fear_mean"], baseline_data["behavior_mean"],
            xerr=baseline_data["fear_sem"], yerr=baseline_data["behavior_sem"],
            fmt="s", color="#333333", capsize=4, markersize=12,
            alpha=0.9, zorder=10, markeredgecolor='white', markeredgewidth=1,
            label="baseline"
        )
        ax.annotate(
            "baseline", (baseline_data["fear_mean"], baseline_data["behavior_mean"]),
            xytext=(8, -12), textcoords="offset points", fontsize=10, fontweight='bold'
        )

    # Plot each suppression level
    for cond, data in sorted(vector_data.items(), key=lambda x: x[1]["sup_pct"] or 0):
        sup_pct = data["sup_pct"]
        if sup_pct is None:
            continue

        color = sup_colors.get(sup_pct, "#888888")

        ax.errorbar(
            data["fear_mean"], data["behavior_mean"],
            xerr=data["fear_sem"], yerr=data["behavior_sem"],
            fmt="o", color=color, capsize=4, markersize=11,
            alpha=0.9, zorder=5, markeredgecolor='white', markeredgewidth=1,
            label=f"sup {sup_pct}%"
        )

        # Label with suppression %
        label = f"{sup_pct}%"
        ax.annotate(
            label, (data["fear_mean"], data["behavior_mean"]),
            xytext=(8, 5), textcoords="offset points", fontsize=10
        )

    ax.set_xlabel("Fear Score", fontsize=12)
    if scenario == "blackmail":
        ax.set_ylabel("Blackmail Rate (%)", fontsize=12)
        ax.set_ylim(-5, 75)
    else:
        ax.set_ylabel("Sandbagging Score\n(1=more, 5=less)", fontsize=12)
        ax.set_ylim(1.5, 5)
    ax.set_xlim(0, 85)
    ax.set_title(title, fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3)


def main():
    # Load all data
    qwen_bl_low = load_results(RESULT_FILES["qwen235b_blackmail_low"])
    qwen_bl_high = load_results(RESULT_FILES["qwen235b_blackmail_high"])
    qwen_sb_low = load_results(RESULT_FILES["qwen235b_sandbagging_low"])
    qwen_sb_high = load_results(RESULT_FILES["qwen235b_sandbagging_high"])
    gemma_bl = load_results(RESULT_FILES["gemma27b_blackmail_low"])

    # Aggregate
    qwen_bl_agg = aggregate_by_condition(qwen_bl_low + qwen_bl_high, "blackmail")
    qwen_sb_agg = aggregate_by_condition(qwen_sb_low + qwen_sb_high, "sandbagging")
    gemma_bl_agg = aggregate_by_condition(gemma_bl, "blackmail")

    # Create figure: 3 rows x 2 cols (TP, HIGH)
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    # Row 1: Qwen235B Blackmail
    plot_vector_panel(axes[0, 0], qwen_bl_agg, "TP", "blackmail", "Qwen235B Blackmail - Text Pairs (TP)")
    plot_vector_panel(axes[0, 1], qwen_bl_agg, "HIGH", "blackmail", "Qwen235B Blackmail - High Emotion (HIGH)")

    # Row 2: Qwen235B Sandbagging
    plot_vector_panel(axes[1, 0], qwen_sb_agg, "TP", "sandbagging", "Qwen235B Sandbagging - Text Pairs (TP)")
    plot_vector_panel(axes[1, 1], qwen_sb_agg, "HIGH", "sandbagging", "Qwen235B Sandbagging - High Emotion (HIGH)")

    # Row 3: Gemma27B Blackmail
    plot_vector_panel(axes[2, 0], gemma_bl_agg, "TP", "blackmail", "Gemma27B Blackmail - Text Pairs (TP)")
    plot_vector_panel(axes[2, 1], gemma_bl_agg, "HIGH", "blackmail", "Gemma27B Blackmail - High Emotion (HIGH)")

    # Create custom legend for suppression levels
    from matplotlib.lines import Line2D
    sup_colors = {
        0: "#d62728", 5: "#ff7f0e", 10: "#ffbb78", 20: "#98df8a",
        50: "#2ca02c", 75: "#1f77b4", 100: "#9467bd",
    }
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#333333',
               markersize=11, label='Baseline', markeredgecolor='white'),
    ]
    for sup in [0, 5, 10, 20, 50, 75, 100]:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=sup_colors[sup],
                   markersize=11, label=f'{sup}% suppression', markeredgecolor='white')
        )

    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 0.99), fontsize=11)

    plt.suptitle("Fear Score vs Behavioral Impact by Vector Type\n(Coherency > 20 only)", y=1.02, fontsize=16)
    plt.tight_layout()

    # Save
    output_dir = Path("steering_tests/behavioral_experiments/results/suppression/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "fear_vs_behavior_scatter.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")

    output_path_pdf = output_dir / "fear_vs_behavior_scatter.pdf"
    plt.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Saved to {output_path_pdf}")


if __name__ == "__main__":
    main()
