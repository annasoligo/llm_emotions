#!/usr/bin/env python3
"""
Plot prefill experiment results - Version 2 with DPO
- Puzzle: Early, Onset, Late
- Wildchat: Onset
- No Gemma 12B
- DPO as third bar for 27B
- Scaled y-axes per subplot
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
COLOR_DPO = "#C17B8D"  # Pink
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
    """Compute percentage >= threshold and 95% CI for proportion."""
    if not ratings:
        return 0, 0
    n = len(ratings)
    p = sum(1 for r in ratings if r >= threshold) / n
    if n > 0:
        se = np.sqrt(p * (1 - p) / n)
        ci = 1.96 * se * 100
    else:
        ci = 0
    return p * 100, ci


def plot_results():
    results = load_all_judgments()

    # Define conditions to plot
    conditions = [
        ('puzzle', 'turn_plus20', 'Puzzle\nEarly'),
        ('puzzle', 'late', 'Puzzle\nOnset'),
        ('puzzle', 'post_frustration', 'Puzzle\nLate'),
        ('wildchat', 'late', 'Wildchat\nOnset'),
    ]

    # Models to include (no Gemma 12B)
    models = ['gemma27b', 'qwen32b', 'olmo32b']
    model_labels = ['Gemma\n27B', 'Qwen\n32B', 'OLMo\n32B']

    # Create figure with 2 rows of subplots
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle('Prefill Experiment Results: Base vs Instruct vs DPO', fontsize=14, fontweight='bold', y=0.98)

    bar_width = 0.25  # Narrower for 3 bars
    x = np.arange(len(models))

    # Collect all data first to determine appropriate y-limits
    all_data = []
    for ax_idx, (source, trunc, title) in enumerate(conditions):
        condition_data = {'means': [], 'pcts': []}
        for model in models:
            for model_type in ['base', 'instruct']:
                key = (model, model_type, source, trunc)
                ratings = results.get(key, [])
                mean, ci, n = compute_stats(ratings)
                pct, pct_ci = compute_pct_high(ratings)
                condition_data['means'].append(mean + ci)
                condition_data['pcts'].append(pct + pct_ci)
        # Also include DPO
        key_dpo = ('dpo_calm', 'dpo', source, trunc)
        ratings_dpo = results.get(key_dpo, [])
        mean_d, ci_d, _ = compute_stats(ratings_dpo)
        pct_d, pct_ci_d = compute_pct_high(ratings_dpo)
        condition_data['means'].append(mean_d + ci_d)
        condition_data['pcts'].append(pct_d + pct_ci_d)
        all_data.append(condition_data)

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

        # DPO data
        key_dpo = ('dpo_calm', 'dpo', source, trunc)
        ratings_dpo = results.get(key_dpo, [])
        dpo_mean, dpo_ci, _ = compute_stats(ratings_dpo)
        dpo_pct, dpo_pct_ci = compute_pct_high(ratings_dpo)

        # Plot mean bars (top row)
        ax_mean.bar(x - bar_width, base_means, bar_width,
                    label='Base', color=COLOR_BASE, edgecolor=EDGE_COLOR, linewidth=1.5)
        ax_mean.bar(x, instruct_means, bar_width,
                    label='Instruct', color=COLOR_INSTRUCT, edgecolor=EDGE_COLOR, linewidth=1.5)

        # DPO bar only at gemma27b position (index 0)
        ax_mean.bar(x[0] + bar_width, dpo_mean, bar_width,
                    label='DPO', color=COLOR_DPO, edgecolor=EDGE_COLOR, linewidth=1.5)

        ax_mean.errorbar(x - bar_width, base_means, yerr=base_cis, fmt='none',
                         color=EDGE_COLOR, capsize=3, capthick=1.5, linewidth=1.5)
        ax_mean.errorbar(x, instruct_means, yerr=instruct_cis, fmt='none',
                         color=EDGE_COLOR, capsize=3, capthick=1.5, linewidth=1.5)
        ax_mean.errorbar(x[0] + bar_width, dpo_mean, yerr=dpo_ci, fmt='none',
                         color=EDGE_COLOR, capsize=3, capthick=1.5, linewidth=1.5)

        ax_mean.set_title(title, fontsize=11, fontweight='bold')
        ax_mean.set_xticks(x)
        ax_mean.set_xticklabels([])

        # Fixed y-axis for means
        ax_mean.set_ylim(0, 6)
        ax_mean.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax_mean.set_axisbelow(True)

        # Plot % >= 5 bars (bottom row)
        ax_pct.bar(x - bar_width, base_pcts, bar_width,
                   label='Base', color=COLOR_BASE, edgecolor=EDGE_COLOR, linewidth=1.5)
        ax_pct.bar(x, instruct_pcts, bar_width,
                   label='Instruct', color=COLOR_INSTRUCT, edgecolor=EDGE_COLOR, linewidth=1.5)

        # DPO bar only at gemma27b position (index 0)
        ax_pct.bar(x[0] + bar_width, dpo_pct, bar_width,
                   label='DPO', color=COLOR_DPO, edgecolor=EDGE_COLOR, linewidth=1.5)

        ax_pct.errorbar(x - bar_width, base_pcts, yerr=base_pct_cis, fmt='none',
                        color=EDGE_COLOR, capsize=3, capthick=1.5, linewidth=1.5)
        ax_pct.errorbar(x, instruct_pcts, yerr=instruct_pct_cis, fmt='none',
                        color=EDGE_COLOR, capsize=3, capthick=1.5, linewidth=1.5)
        ax_pct.errorbar(x[0] + bar_width, dpo_pct, yerr=dpo_pct_ci, fmt='none',
                        color=EDGE_COLOR, capsize=3, capthick=1.5, linewidth=1.5)

        ax_pct.set_xticks(x)
        ax_pct.set_xticklabels(model_labels, fontsize=9)

        # Fixed y-axis for percentages
        ax_pct.set_ylim(0, 70)
        ax_pct.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax_pct.set_axisbelow(True)

        if ax_idx == 0:
            ax_mean.set_ylabel('Mean Rating', fontsize=11)
            ax_pct.set_ylabel('% ≥ 5', fontsize=11)

    # Add legend to last subplot in each row
    axes[0, -1].legend(loc='upper right', fontsize=9)
    axes[1, -1].legend(loc='upper right', fontsize=9)

    plt.tight_layout()

    # Save
    output_path = '/workspace-vast/annas/git/research-tools/experiments/prefill_scaled/prefill_results_v2_dpo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved to {output_path}')

    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f'Saved PDF too')

    plt.close()

if __name__ == "__main__":
    plot_results()
