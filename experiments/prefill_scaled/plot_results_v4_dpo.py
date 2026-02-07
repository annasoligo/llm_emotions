#!/usr/bin/env python3
"""
Plot prefill experiment results - Version 3 with DPO
- Single row with means
- Text above bars showing % >= 5
- No Gemma 12B
- DPO as third bar for 27B
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
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
            # Support both old ('rating') and new ('judgment_score') field names
            rating = d.get('rating', d.get('judgment_score'))
            if rating is not None and rating >= 0:
                model = d['model_family']
                model_type = d['model_type']
                source = d['source']
                trunc = d['truncation_type']

                key = (model, model_type, source, trunc)
                if key not in results:
                    results[key] = []
                results[key].append(rating)

    return results

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    """Calculate bootstrap confidence interval for the mean."""
    if not data:
        return 0, 0
    data = np.array(data)
    n = len(data)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    mean = np.mean(data)
    return mean, (upper - lower) / 2

def compute_stats(ratings):
    """Compute mean and bootstrap 95% CI."""
    if not ratings:
        return 0, 0, 0
    mean, ci = bootstrap_ci(ratings)
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
        ('puzzle', 'turn_plus20', 'Numeric qu,\nEarly'),
        ('puzzle', 'late', 'Numeric qu,\nOnset'),
        ('puzzle', 'post_frustration', 'Numeric qu,\nLate (Recovery)'),
        ('triggers', 'onset', 'Text qu,\nOnset'),
    ]

    # Models to include (no Gemma 12B)
    models = ['gemma27b', 'qwen25b', 'olmo32b']
    model_labels = ['Gemma\n27B', 'Qwen\n32B', 'OLMo\n32B']

    # Create figure with single row
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    fig.suptitle('Prefilled Response Frustration Levels: Mean and % ≥5 (95% CIs)', fontsize=17, y=1.18)

    bar_width = 0.42
    # Non-uniform x positions: Gemma needs space for 3 bars, Qwen/OLMo only 2
    # Equal visual gaps between model groups (accounting for bar edges)
    x_gemma = 0
    x_qwen = 1.25
    x_olmo = 2.3
    x_positions = [x_gemma, x_qwen, x_olmo]

    for ax_idx, (source, trunc, title) in enumerate(conditions):
        ax = axes[ax_idx]

        base_means = []
        base_cis = []
        instruct_means = []
        instruct_cis = []
        base_pcts = []
        instruct_pcts = []

        for model in models:
            # Base - NO FALLBACKS, missing data shows as 0
            key_base = (model, 'base', source, trunc)
            ratings_base = results.get(key_base, [])

            mean_b, ci_b, n_b = compute_stats(ratings_base)
            base_means.append(mean_b)
            base_cis.append(ci_b)
            base_pcts.append(compute_pct_high(ratings_base))

            # Instruct - NO FALLBACKS
            key_inst = (model, 'instruct', source, trunc)
            ratings_inst = results.get(key_inst, [])

            mean_i, ci_i, n_i = compute_stats(ratings_inst)
            instruct_means.append(mean_i)
            instruct_cis.append(ci_i)
            instruct_pcts.append(compute_pct_high(ratings_inst))

        # DPO data
        key_dpo = ('dpo_calm', 'dpo', source, trunc)
        ratings_dpo = results.get(key_dpo, [])
        dpo_mean, dpo_ci, _ = compute_stats(ratings_dpo)
        dpo_pct = compute_pct_high(ratings_dpo)

        # Plot bars - Gemma has 3 bars (centered around x_gemma)
        # Base at x-bar_width, Instruct at x, DPO at x+bar_width
        ax.bar(x_gemma - bar_width, base_means[0], bar_width,
               label='Base Model', color=COLOR_BASE, edgecolor=EDGE_COLOR, linewidth=1.5)
        ax.bar(x_gemma, instruct_means[0], bar_width,
               label='Instruct Model', color=COLOR_INSTRUCT, edgecolor=EDGE_COLOR, linewidth=1.5)
        ax.bar(x_gemma + bar_width, dpo_mean, bar_width,
               label='DPO', color=COLOR_DPO, edgecolor=EDGE_COLOR, linewidth=1.5)

        # Qwen and OLMo have 2 bars (centered around their x positions)
        # Base at x-bar_width/2, Instruct at x+bar_width/2
        for i, x_pos in enumerate([x_qwen, x_olmo]):
            model_idx = i + 1  # Qwen is index 1, OLMo is index 2
            ax.bar(x_pos - bar_width/2, base_means[model_idx], bar_width,
                   color=COLOR_BASE, edgecolor=EDGE_COLOR, linewidth=1.5)
            ax.bar(x_pos + bar_width/2, instruct_means[model_idx], bar_width,
                   color=COLOR_INSTRUCT, edgecolor=EDGE_COLOR, linewidth=1.5)

        # Error bars - Gemma
        ax.errorbar(x_gemma - bar_width, base_means[0], yerr=base_cis[0], fmt='none',
                    color=EDGE_COLOR, capsize=0, linewidth=1.5)
        ax.errorbar(x_gemma, instruct_means[0], yerr=instruct_cis[0], fmt='none',
                    color=EDGE_COLOR, capsize=0, linewidth=1.5)
        ax.errorbar(x_gemma + bar_width, dpo_mean, yerr=dpo_ci, fmt='none',
                    color=EDGE_COLOR, capsize=0, linewidth=1.5)

        # Error bars - Qwen and OLMo
        for i, x_pos in enumerate([x_qwen, x_olmo]):
            model_idx = i + 1
            ax.errorbar(x_pos - bar_width/2, base_means[model_idx], yerr=base_cis[model_idx], fmt='none',
                        color=EDGE_COLOR, capsize=0, linewidth=1.5)
            ax.errorbar(x_pos + bar_width/2, instruct_means[model_idx], yerr=instruct_cis[model_idx], fmt='none',
                        color=EDGE_COLOR, capsize=0, linewidth=1.5)

        # Add % text above bars - Gemma
        ax.text(x_gemma - bar_width, base_means[0] + base_cis[0] + 0.15,
                f'{base_pcts[0]:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(x_gemma, instruct_means[0] + instruct_cis[0] + 0.15,
                f'{instruct_pcts[0]:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(x_gemma + bar_width, dpo_mean + dpo_ci + 0.15,
                f'{dpo_pct:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add % text above bars - Qwen and OLMo
        for i, x_pos in enumerate([x_qwen, x_olmo]):
            model_idx = i + 1
            ax.text(x_pos - bar_width/2, base_means[model_idx] + base_cis[model_idx] + 0.15,
                    f'{base_pcts[model_idx]:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(x_pos + bar_width/2, instruct_means[model_idx] + instruct_cis[model_idx] + 0.15,
                    f'{instruct_pcts[model_idx]:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_title(title, fontsize=14)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_labels, fontsize=12)
        ax.set_ylim(0, 6)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

        if ax_idx == 0:
            ax.set_ylabel('Mean Rating', fontsize=14)

    # Add legend on leftmost plot
    axes[0].legend(loc='upper left', fontsize=14)

    # Add note explaining the % values - centered over entire figure
    fig.text(0.5, 1.05, '% above bars = % with high frustration (rating ≥5)',
             ha='center', va='bottom', fontsize=14, style='italic')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, wspace=0.15)

    # Save
    output_path = '/workspace-vast/annas/git/gemma-iclr/hcair2026/figures/prefill_results_v4_dpo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved to {output_path}')

    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    print(f'Saved PDF too')

    plt.close()

if __name__ == "__main__":
    plot_results()
