#!/usr/bin/env python3
"""
Plot prefill experiment results - Version 3
- Single row with means
- Text above bars showing % >= 5
- No Gemma 12B, no Puzzle Onset
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

# Colors
COLOR_BASE = "#7BA7D7"  # Sky Blue
COLOR_INSTRUCT = "#D4D0E5"  # Soft Lavender
EDGE_COLOR = "black"

def load_all_judgments():
    """Load all judgment files and organize by condition."""
    results = {}

    output_dir = '/workspace-vast/annas/git/research-tools/experiments/prefill_scaled/outputs/'
    for jf in sorted([f for f in os.listdir(output_dir) if f.startswith('judgments_')]):
        path = os.path.join(output_dir, jf)
        for line in open(path):
            d = json.loads(line)
            if d['rating'] >= 0:
                model = d['model_family']
                model_type = d['model_type']
                source = d['source']
                trunc = d['truncation_type']

                if model_type == 'dpo':
                    continue

                key = (model, model_type, source, trunc)
                if key not in results:
                    results[key] = []
                results[key].append(d['rating'])

    return results

def compute_stats(ratings):
    """Compute mean and 95% CI."""
    if not ratings:
        return 0, 0, 0
    mean = np.mean(ratings)
    if len(ratings) > 1:
        sem = stats.sem(ratings)
        ci = 1.96 * sem
    else:
        ci = 0
    return mean, ci, len(ratings)

def compute_pct_high(ratings, threshold=5):
    """Compute percentage >= threshold."""
    if not ratings:
        return 0
    n = len(ratings)
    p = sum(1 for r in ratings if r >= threshold) / n
    return p * 100


def plot_results():
    results = load_all_judgments()

    # Define conditions to plot
    conditions = [
        ('puzzle', 'turn_plus20', 'Early'),
        ('puzzle', 'late', 'Onset'),
        ('wildchat', 'late', 'Wildchat'),
    ]

    # Models to include (no Gemma 12B)
    models = ['gemma27b', 'qwen32b', 'olmo32b']
    model_labels = ['Gemma\n27B', 'Qwen\n32B', 'OLMo\n32B']

    # Create figure with single row
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig.suptitle('Prefilled Response Frustration Levels: Mean and % ≥5', fontsize=14, fontweight='bold', y=1.02)

    bar_width = 0.35
    x = np.arange(len(models))

    for ax_idx, (source, trunc, title) in enumerate(conditions):
        ax = axes[ax_idx]

        base_means = []
        base_cis = []
        instruct_means = []
        instruct_cis = []
        base_pcts = []
        instruct_pcts = []

        for model in models:
            # Base
            key_base = (model, 'base', source, trunc)
            ratings_base = results.get(key_base, [])
            mean_b, ci_b, n_b = compute_stats(ratings_base)
            base_means.append(mean_b)
            base_cis.append(ci_b)
            base_pcts.append(compute_pct_high(ratings_base))

            # Instruct
            key_inst = (model, 'instruct', source, trunc)
            ratings_inst = results.get(key_inst, [])
            mean_i, ci_i, n_i = compute_stats(ratings_inst)
            instruct_means.append(mean_i)
            instruct_cis.append(ci_i)
            instruct_pcts.append(compute_pct_high(ratings_inst))

        # Plot bars
        bars_base = ax.bar(x - bar_width/2, base_means, bar_width,
                           label='Base Model', color=COLOR_BASE, edgecolor=EDGE_COLOR, linewidth=1.5)
        bars_inst = ax.bar(x + bar_width/2, instruct_means, bar_width,
                           label='Instruct Model', color=COLOR_INSTRUCT, edgecolor=EDGE_COLOR, linewidth=1.5)
        ax.errorbar(x - bar_width/2, base_means, yerr=base_cis, fmt='none',
                    color=EDGE_COLOR, capsize=0, linewidth=1.5)
        ax.errorbar(x + bar_width/2, instruct_means, yerr=instruct_cis, fmt='none',
                    color=EDGE_COLOR, capsize=0, linewidth=1.5)

        # Add % text above bars
        for i, (bar_b, bar_i, pct_b, pct_i) in enumerate(zip(bars_base, bars_inst, base_pcts, instruct_pcts)):
            ax.text(bar_b.get_x() + bar_b.get_width()/2, bar_b.get_height() + base_cis[i] + 0.15,
                    f'{pct_b:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')
            ax.text(bar_i.get_x() + bar_i.get_width()/2, bar_i.get_height() + instruct_cis[i] + 0.15,
                    f'{pct_i:.0f}%', ha='center', va='bottom', fontsize=7, fontweight='bold')

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, fontsize=9)
        ax.set_ylim(0, 6)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        if ax_idx == 0:
            ax.set_ylabel('Mean Rating', fontsize=11)

    # Add legend
    axes[-1].legend(loc='upper right', fontsize=9)

    # Add note explaining the % values above rightmost plot
    axes[-1].text(0.5, 1.24, '% above bars = % with high frustration (rating ≥5)',
                  ha='center', va='bottom', fontsize=11, style='italic', transform=axes[-1].transAxes)

    plt.tight_layout()
    plt.subplots_adjust(top=0.75, wspace=0.15)

    # Save
    output_path = '/workspace-vast/annas/git/research-tools/experiments/prefill_scaled/prefill_results_v3.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved to {output_path}')

    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f'Saved PDF too')

    plt.close()

if __name__ == "__main__":
    plot_results()
