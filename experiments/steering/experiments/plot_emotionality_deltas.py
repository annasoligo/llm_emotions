#!/usr/bin/env python3
"""
Plot emotionality judge deltas as grouped bar charts.
Each group shows all judge dimensions for a single steering condition.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import argparse

# Emotionality dimension colors
DIMENSION_COLORS = {
    'valence': '#E07B54',     # warm orange-red
    'arousal': '#9370DB',     # medium purple
    'agency': '#5B8DBE',      # steel blue
    'intensity': '#7CB68D',   # sage green
    'certainty': '#D4A84B',   # gold
    'fear': '#8B0000',        # dark red
    'anger': '#FF4500',       # orange red
}

DIMENSIONS = ['valence', 'arousal', 'agency', 'intensity', 'certainty']
DIMENSIONS_WITH_SENTIMENT = ['valence', 'arousal', 'agency', 'intensity', 'certainty', 'fear', 'anger']


def load_jsonl(path):
    """Load JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def aggregate_emotionality(results, include_sentiment=False):
    """Aggregate emotionality scores by condition."""
    dims_to_use = DIMENSIONS_WITH_SENTIMENT if include_sentiment else DIMENSIONS
    data = defaultdict(lambda: {dim: [] for dim in dims_to_use})

    for r in results:
        cond = r['condition']
        emo = r.get('emotionality_judge', {})

        for dim in DIMENSIONS:
            score = emo.get(dim)
            if score is not None:
                data[cond][dim].append(score)

        # Also get fear and anger if available
        if include_sentiment:
            fear_judge = r.get('fear_judge', {})
            anger_judge = r.get('anger_judge', {})

            fear_score = fear_judge.get('fear_score')
            if fear_score is not None:
                data[cond]['fear'].append(fear_score)

            anger_score = anger_judge.get('anger_score')
            if anger_score is not None:
                data[cond]['anger'].append(anger_score)

    # Compute means
    means = {}
    for cond, dims in data.items():
        means[cond] = {dim: np.mean(vals) if vals else None for dim, vals in dims.items()}

    return means


def plot_emotionality_deltas(gemma_results, qwen_results, output_path):
    """Plot emotionality deltas as grouped bar charts."""

    gemma_data = aggregate_emotionality(gemma_results)
    qwen_data = aggregate_emotionality(qwen_results)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Define conditions to plot (single-axis only)
    gemma_conditions = [
        ('valence_+7.5%', 'valence+'),
        ('valence_-7.5%', 'valence-'),
        ('uncertainty_+7.5%', 'uncertainty+'),
        ('uncertainty_-7.5%', 'uncertainty-'),
        ('agency_+7.5%', 'agency+'),
        ('agency_-7.5%', 'agency-'),
    ]

    qwen_conditions = [
        ('valence_+100%', 'valence+'),
        ('valence_-100%', 'valence-'),
        ('uncertainty_+100%', 'uncertainty+'),
        ('uncertainty_-100%', 'uncertainty-'),
        ('agency_+100%', 'agency+'),
        ('agency_-100%', 'agency-'),
    ]

    datasets = [
        (gemma_data, gemma_conditions, "Gemma 3 27B (7.5%)", axes[0]),
        (qwen_data, qwen_conditions, "Qwen3-32B (100%)", axes[1]),
    ]

    bar_width = 0.15
    n_dims = len(DIMENSIONS)

    for data, conditions, title, ax in datasets:
        baseline = data.get('baseline', {})

        x = np.arange(len(conditions))

        for dim_idx, dim in enumerate(DIMENSIONS):
            deltas = []
            for cond_key, cond_label in conditions:
                if cond_key in data and baseline.get(dim) is not None and data[cond_key].get(dim) is not None:
                    delta = data[cond_key][dim] - baseline[dim]
                    deltas.append(delta)
                else:
                    deltas.append(0)

            offset = (dim_idx - n_dims/2 + 0.5) * bar_width
            bars = ax.bar(x + offset, deltas, bar_width * 0.9,
                         label=dim.title() if ax == axes[0] else "",
                         color=DIMENSION_COLORS[dim],
                         edgecolor='white', linewidth=0.5)

        # Reference line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in conditions], fontsize=10)
        ax.set_xlabel('Steering Condition', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add grid for y-axis
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Delta from Baseline', fontsize=11)

    # Shared legend
    fig.legend(handles=axes[0].containers[0:5],
               labels=[d.title() for d in DIMENSIONS],
               loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)

    plt.suptitle('Emotionality Judge Deltas: Single-Axis Appraisal Steering',
                 fontsize=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_emotionality_deltas_combos(gemma_results, qwen_results, output_path):
    """Plot emotionality deltas for combo conditions."""

    gemma_data = aggregate_emotionality(gemma_results)
    qwen_data = aggregate_emotionality(qwen_results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Define combo conditions
    gemma_conditions = [
        ('valence-_uncertainty+_7.5%', 'val- unc+'),
        ('valence-_agency-_7.5%', 'val- agn-'),
        ('valence-_uncertainty+_agency-_7.5%', 'val- unc+ agn-'),
    ]

    qwen_conditions = [
        ('valence-_uncertainty+_100%', 'val- unc+'),
        ('valence-_agency-_100%', 'val- agn-'),
        ('valence-_uncertainty+_agency-_100%', 'val- unc+ agn-'),
    ]

    datasets = [
        (gemma_data, gemma_conditions, "Gemma 3 27B (7.5%)", axes[0]),
        (qwen_data, qwen_conditions, "Qwen3-32B (100%)", axes[1]),
    ]

    bar_width = 0.15
    n_dims = len(DIMENSIONS)

    for data, conditions, title, ax in datasets:
        baseline = data.get('baseline', {})

        x = np.arange(len(conditions))

        for dim_idx, dim in enumerate(DIMENSIONS):
            deltas = []
            for cond_key, cond_label in conditions:
                if cond_key in data and baseline.get(dim) is not None and data[cond_key].get(dim) is not None:
                    delta = data[cond_key][dim] - baseline[dim]
                    deltas.append(delta)
                else:
                    deltas.append(0)

            offset = (dim_idx - n_dims/2 + 0.5) * bar_width
            bars = ax.bar(x + offset, deltas, bar_width * 0.9,
                         label=dim.title() if ax == axes[0] else "",
                         color=DIMENSION_COLORS[dim],
                         edgecolor='white', linewidth=0.5)

        # Reference line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in conditions], fontsize=10)
        ax.set_xlabel('Steering Condition', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add grid for y-axis
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Delta from Baseline', fontsize=11)

    # Shared legend
    fig.legend(handles=axes[0].containers[0:5],
               labels=[d.title() for d in DIMENSIONS],
               loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 1.02), fontsize=10)

    plt.suptitle('Emotionality Judge Deltas: Combo Appraisal Steering',
                 fontsize=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_all_judges_deltas(gemma_results, qwen_results, output_path):
    """Plot all judge deltas including fear and anger - single axis conditions."""

    gemma_data = aggregate_emotionality(gemma_results, include_sentiment=True)
    qwen_data = aggregate_emotionality(qwen_results, include_sentiment=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    # Define conditions to plot (single-axis only)
    gemma_conditions = [
        ('valence_+7.5%', 'valence+'),
        ('valence_-7.5%', 'valence-'),
        ('uncertainty_+7.5%', 'uncertainty+'),
        ('uncertainty_-7.5%', 'uncertainty-'),
        ('agency_+7.5%', 'agency+'),
        ('agency_-7.5%', 'agency-'),
    ]

    qwen_conditions = [
        ('valence_+100%', 'valence+'),
        ('valence_-100%', 'valence-'),
        ('uncertainty_+100%', 'uncertainty+'),
        ('uncertainty_-100%', 'uncertainty-'),
        ('agency_+100%', 'agency+'),
        ('agency_-100%', 'agency-'),
    ]

    datasets = [
        (gemma_data, gemma_conditions, "Gemma 3 27B (7.5%)", axes[0]),
        (qwen_data, qwen_conditions, "Qwen3-32B (100%)", axes[1]),
    ]

    bar_width = 0.11
    dims = DIMENSIONS_WITH_SENTIMENT
    n_dims = len(dims)

    for data, conditions, title, ax in datasets:
        baseline = data.get('baseline', {})

        x = np.arange(len(conditions))

        for dim_idx, dim in enumerate(dims):
            deltas = []
            for cond_key, cond_label in conditions:
                if cond_key in data and baseline.get(dim) is not None and data[cond_key].get(dim) is not None:
                    delta = data[cond_key][dim] - baseline[dim]
                    deltas.append(delta)
                else:
                    deltas.append(0)

            offset = (dim_idx - n_dims/2 + 0.5) * bar_width
            bars = ax.bar(x + offset, deltas, bar_width * 0.9,
                         label=dim.title() if ax == axes[0] else "",
                         color=DIMENSION_COLORS[dim],
                         edgecolor='white', linewidth=0.5)

        # Reference line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in conditions], fontsize=10)
        ax.set_xlabel('Steering Condition', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add grid for y-axis
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Delta from Baseline', fontsize=11)

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=DIMENSION_COLORS[d], label=d.title()) for d in dims]
    fig.legend(handles=legend_elements,
               loc='upper center', ncol=7,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)

    plt.suptitle('All Judge Deltas: Single-Axis Appraisal Steering',
                 fontsize=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_all_judges_combos(gemma_results, qwen_results, output_path):
    """Plot all judge deltas including fear and anger - combo conditions."""

    gemma_data = aggregate_emotionality(gemma_results, include_sentiment=True)
    qwen_data = aggregate_emotionality(qwen_results, include_sentiment=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # Define combo conditions
    gemma_conditions = [
        ('valence-_uncertainty+_7.5%', 'val- unc+'),
        ('valence-_agency-_7.5%', 'val- agn-'),
        ('valence-_uncertainty+_agency-_7.5%', 'val- unc+ agn-'),
    ]

    qwen_conditions = [
        ('valence-_uncertainty+_100%', 'val- unc+'),
        ('valence-_agency-_100%', 'val- agn-'),
        ('valence-_uncertainty+_agency-_100%', 'val- unc+ agn-'),
    ]

    datasets = [
        (gemma_data, gemma_conditions, "Gemma 3 27B (7.5%)", axes[0]),
        (qwen_data, qwen_conditions, "Qwen3-32B (100%)", axes[1]),
    ]

    bar_width = 0.11
    dims = DIMENSIONS_WITH_SENTIMENT
    n_dims = len(dims)

    for data, conditions, title, ax in datasets:
        baseline = data.get('baseline', {})

        x = np.arange(len(conditions))

        for dim_idx, dim in enumerate(dims):
            deltas = []
            for cond_key, cond_label in conditions:
                if cond_key in data and baseline.get(dim) is not None and data[cond_key].get(dim) is not None:
                    delta = data[cond_key][dim] - baseline[dim]
                    deltas.append(delta)
                else:
                    deltas.append(0)

            offset = (dim_idx - n_dims/2 + 0.5) * bar_width
            bars = ax.bar(x + offset, deltas, bar_width * 0.9,
                         label=dim.title() if ax == axes[0] else "",
                         color=DIMENSION_COLORS[dim],
                         edgecolor='white', linewidth=0.5)

        # Reference line at 0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in conditions], fontsize=10)
        ax.set_xlabel('Steering Condition', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Add grid for y-axis
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel('Delta from Baseline', fontsize=11)

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=DIMENSION_COLORS[d], label=d.title()) for d in dims]
    fig.legend(handles=legend_elements,
               loc='upper center', ncol=7,
               bbox_to_anchor=(0.5, 1.02), fontsize=9)

    plt.suptitle('All Judge Deltas: Combo Appraisal Steering',
                 fontsize=14, fontweight='bold', y=1.08)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma", type=Path, required=True, help="Gemma JSONL (with sentiment)")
    parser.add_argument("--qwen", type=Path, required=True, help="Qwen JSONL (with sentiment)")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/steering/outputs/sandbagging"))
    args = parser.parse_args()

    gemma_results = load_jsonl(args.gemma)
    qwen_results = load_jsonl(args.qwen)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Plot single-axis deltas (emotionality only)
    plot_emotionality_deltas(
        gemma_results, qwen_results,
        str(args.output_dir / "emotionality_deltas_single_axis.png")
    )

    # Plot combo deltas (emotionality only)
    plot_emotionality_deltas_combos(
        gemma_results, qwen_results,
        str(args.output_dir / "emotionality_deltas_combos.png")
    )

    # Plot all judges including fear/anger - single axis
    plot_all_judges_deltas(
        gemma_results, qwen_results,
        str(args.output_dir / "all_judges_deltas_single_axis.png")
    )

    # Plot all judges including fear/anger - combos
    plot_all_judges_combos(
        gemma_results, qwen_results,
        str(args.output_dir / "all_judges_deltas_combos.png")
    )


if __name__ == '__main__':
    main()
