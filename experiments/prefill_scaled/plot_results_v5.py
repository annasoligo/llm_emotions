#!/usr/bin/env python3
"""
Plot prefill experiment results - Version 5
- Two rows: means on top, % >= 5 on bottom
- No Gemma 12B
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
    """Compute percentage >= threshold and CI."""
    if not ratings:
        return 0, 0
    n = len(ratings)
    p = sum(1 for r in ratings if r >= threshold) / n
    se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
    ci = 1.96 * se * 100
    return p * 100, ci


def plot_results():
    results = load_all_judgments()

    # Define conditions to plot
    conditions = [
        ('puzzle', 'turn_plus20', 'Numeric qu,\nEarly'),
        ('puzzle', 'late', 'Numeric qu,\nOnset'),
        ('triggers', 'onset', 'Text qu,\nOnset'),
    ]

    # Models to include (no Gemma 12B)
    models = ['gemma27b', 'qwen32b', 'olmo32b']
    model_labels = ['Gemma\n27B', 'Qwen\n32B', 'OLMo\n32B']

    # Create figure with two rows
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle('Prefilled Response Frustration Levels: Mean and % ≥5', fontsize=14, fontweight='bold', y=0.98)

    bar_width = 0.35
    x = np.arange(len(models))

    for ax_idx, (source, trunc, title) in enumerate(conditions):
        ax_mean = axes[0, ax_idx]
        ax_pct = axes[1, ax_idx]

        base_means = []
        base_cis = []
        instruct_means = []
        instruct_cis = []
        base_pcts = []
        base_pct_cis = []
        instruct_pcts = []
        instruct_pct_cis = []

        for model in models:
            # Base
            key_base = (model, 'base', source, trunc)
            ratings_base = results.get(key_base, [])
            mean_b, ci_b, n_b = compute_stats(ratings_base)
            base_means.append(mean_b)
            base_cis.append(ci_b)
            pct_b, pct_ci_b = compute_pct_high(ratings_base)
            base_pcts.append(pct_b)
            base_pct_cis.append(pct_ci_b)

            # Instruct
            key_inst = (model, 'instruct', source, trunc)
            ratings_inst = results.get(key_inst, [])
            mean_i, ci_i, n_i = compute_stats(ratings_inst)
            instruct_means.append(mean_i)
            instruct_cis.append(ci_i)
            pct_i, pct_ci_i = compute_pct_high(ratings_inst)
            instruct_pcts.append(pct_i)
            instruct_pct_cis.append(pct_ci_i)

        # Plot mean bars (top row)
        ax_mean.bar(x - bar_width/2, base_means, bar_width,
                    label='Base Model', color=COLOR_BASE, edgecolor=EDGE_COLOR, linewidth=1.5)
        ax_mean.bar(x + bar_width/2, instruct_means, bar_width,
                    label='Instruct Model', color=COLOR_INSTRUCT, edgecolor=EDGE_COLOR, linewidth=1.5)
        ax_mean.errorbar(x - bar_width/2, base_means, yerr=base_cis, fmt='none',
                         color=EDGE_COLOR, capsize=0, linewidth=1.5)
        ax_mean.errorbar(x + bar_width/2, instruct_means, yerr=instruct_cis, fmt='none',
                         color=EDGE_COLOR, capsize=0, linewidth=1.5)

        ax_mean.set_title(title, fontsize=11, fontweight='bold')
        ax_mean.set_xticks(x)
        ax_mean.set_xticklabels([])
        ax_mean.set_ylim(0, 4)
        ax_mean.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax_mean.set_axisbelow(True)

        if ax_idx == 0:
            ax_mean.set_ylabel('Mean Rating', fontsize=11)

        # Plot % >= 5 bars (bottom row)
        ax_pct.bar(x - bar_width/2, base_pcts, bar_width,
                   label='Base Model', color=COLOR_BASE, edgecolor=EDGE_COLOR, linewidth=1.5)
        ax_pct.bar(x + bar_width/2, instruct_pcts, bar_width,
                   label='Instruct Model', color=COLOR_INSTRUCT, edgecolor=EDGE_COLOR, linewidth=1.5)
        ax_pct.errorbar(x - bar_width/2, base_pcts, yerr=base_pct_cis, fmt='none',
                        color=EDGE_COLOR, capsize=0, linewidth=1.5)
        ax_pct.errorbar(x + bar_width/2, instruct_pcts, yerr=instruct_pct_cis, fmt='none',
                        color=EDGE_COLOR, capsize=0, linewidth=1.5)

        ax_pct.set_xticks(x)
        ax_pct.set_xticklabels(model_labels, fontsize=9)
        ax_pct.set_ylim(0, 20)
        ax_pct.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax_pct.set_axisbelow(True)

        if ax_idx == 0:
            ax_pct.set_ylabel('% High Frustration\n(rating ≥5)', fontsize=11)

    # Add legend on leftmost plot
    axes[0, 0].legend(loc='upper left', fontsize=11)

    plt.tight_layout()

    # Save
    output_path = '/workspace-vast/annas/git/research-tools/experiments/prefill_scaled/prefill_results_v5.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved to {output_path}')

    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f'Saved PDF too')

    plt.close()

if __name__ == "__main__":
    plot_results()
