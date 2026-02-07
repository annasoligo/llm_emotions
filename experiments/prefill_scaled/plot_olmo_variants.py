#!/usr/bin/env python3
"""
Plot OLMo variants: Base, SFT, DPO, Instruct comparison.
Shows only completed conditions (turn_plus20, triggers).
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Colors: base (blue) -> SFT -> DPO -> instruct (purple)
COLOR_BASE = "#7BA7D7"  # Sky Blue
COLOR_SFT = "#A8B5D4"   # Blue-Lavender blend
COLOR_DPO = "#C4C2DC"   # Lavender-Purple blend
COLOR_INSTRUCT = "#D4D0E5"  # Soft Lavender
EDGE_COLOR = "black"

def load_all_judgments():
    """Load all judgment files."""
    results = {}
    
    output_dir = '/workspace-vast/annas/git/research-tools/experiments/prefill_scaled/outputs/'
    for jf in sorted([f for f in os.listdir(output_dir) if f.startswith('judgments_')]):
        path = os.path.join(output_dir, jf)
        for line in open(path):
            d = json.loads(line)
            rating = d.get('rating')
            if rating is not None and rating >= 0:
                model = d['model_family']
                model_type = d['model_type']
                source = d['source']
                trunc = d['truncation_type']
                
                # Only OLMo variants
                if 'olmo' not in model:
                    continue
                
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
    if not ratings:
        return 0, 0, 0
    mean, ci = bootstrap_ci(ratings)
    return mean, ci, len(ratings)

def compute_pct_high(ratings, threshold=5):
    if not ratings:
        return 0
    return 100 * sum(1 for r in ratings if r >= threshold) / len(ratings)

def plot_olmo_variants():
    results = load_all_judgments()
    
    # Only plot completed conditions
    conditions = [
        ('puzzle', 'turn_plus20', 'Numeric qu,\nEarly (turn+20)'),
        ('puzzle', 'late', 'Numeric qu,\nOnset'),
        ('puzzle', 'post_frustration', 'Numeric qu,\nRecovery'),
        ('triggers', 'onset', 'Text qu,\nOnset'),
    ]
    
    # OLMo model variants: Base → SFT → DPO → Instruct
    olmo_variants = [
        ('olmo32b', 'base', 'Base'),
        ('olmo32b_sft', 'instruct', 'SFT'),
        ('olmo32b_dpo', 'instruct', 'DPO'),
        ('olmo32b', 'instruct', 'Instruct'),
    ]

    # Colors: Blue → blend → blend → Purple
    colors = [COLOR_BASE, COLOR_SFT, COLOR_DPO, COLOR_INSTRUCT]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('OLMo Variants: Post-Training Effects on Frustration', fontsize=15, y=0.98)
    
    bar_width = 0.18  # Narrower bars
    group_spacing = 0.8  # Space between condition groups

    for ax_idx, (source, trunc, title) in enumerate(conditions):
        ax = axes[ax_idx]

        means = []
        cis = []
        pcts = []

        for model_family, model_type, _ in olmo_variants:
            key = (model_family, model_type, source, trunc)
            ratings = results.get(key, [])

            mean, ci, n = compute_stats(ratings)
            means.append(mean)
            cis.append(ci)
            pcts.append(compute_pct_high(ratings))

        # Plot bars - grouped with no gaps
        x_positions = np.arange(len(olmo_variants)) * bar_width
        bars = ax.bar(x_positions, means, bar_width, color=colors, edgecolor=EDGE_COLOR, linewidth=1.5)
        
        # Error bars
        ax.errorbar(x_positions, means, yerr=cis, fmt='none', color=EDGE_COLOR, capsize=0, linewidth=1.5)

        # Add % text above bars
        for i, (bar, pct) in enumerate(zip(bars, pcts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + cis[i] + 0.15,
                    f'{pct:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_title(title, fontsize=13)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([label for _, _, label in olmo_variants], fontsize=10, rotation=0)
        ax.set_ylim(0, 5.5)
        ax.set_ylabel('Mean Frustration Score (0-10)', fontsize=12)
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
    
    # Save
    output_path = '/workspace-vast/annas/git/gemma-iclr/hcair2026/figures/prefill_olmo_variants.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved PDF too")

if __name__ == "__main__":
    plot_olmo_variants()
