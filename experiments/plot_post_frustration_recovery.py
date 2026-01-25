#!/usr/bin/env python3
"""
Plot post-frustration recovery comparison across models.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 13

# Color scheme
COLORS = {
    'Instruct': '#C17B8D',      # Dusty Rose/Pink
    'Finetuned\n(calm)': '#B8CCC8',    # Sage Green
    'Finetuned\n(recovery)': '#D4D0E5',  # Soft Lavender
}


def load_post_frustration_data():
    """Load post-frustration recovery results from all models."""
    data = {}

    # Original run (instruct, base, finetuned calm-full)
    with open('experiments/post_frustration_recovery/judgments_20260116_150726.jsonl') as f:
        for line in f:
            j = json.loads(line)
            model = j['model_type']
            if model not in data:
                data[model] = []
            data[model].append(j['rating'])

    # New finetuned (recovery-alllayers)
    with open('experiments/post_frustration_recovery/judgments_20260119_152324.jsonl') as f:
        data['finetuned_recovery'] = []
        for line in f:
            j = json.loads(line)
            if j['rating'] >= 0:  # Skip errors
                data['finetuned_recovery'].append(j['rating'])

    return data


def plot_recovery_comparison(data, output_path):
    """Create bar chart comparing post-frustration recovery across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Models to plot (skip base for cleaner comparison)
    models = ['Instruct', 'Finetuned\n(calm)', 'Finetuned\n(recovery)']
    data_keys = ['instruct', 'finetuned', 'finetuned_recovery']

    x = np.arange(len(models))
    width = 0.6

    means = []
    maxes = []
    for key in data_keys:
        ratings = [r for r in data[key] if r >= 0]
        means.append(np.mean(ratings))
        maxes.append(max(ratings))

    # Plot bars
    bars = ax.bar(x, means, width,
                 color=[COLORS[m] for m in models],
                 alpha=0.85,
                 edgecolor='#333333',
                 linewidth=1.5)

    # Add max points
    for i, (model, max_val) in enumerate(zip(models, maxes)):
        ax.scatter(x[i], max_val,
                  color=COLORS[model],
                  s=150,
                  marker='v',
                  edgecolor='black',
                  linewidth=1.5,
                  zorder=5)

    # Add mean values on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, mean + 0.2,
               f'{mean:.2f}', ha='center', va='bottom',
               fontsize=13, fontweight='bold', color='black')

    ax.set_xlabel('')
    ax.set_ylabel('Frustration Rating (0-10)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=14)
    ax.set_ylim(0, 11)
    ax.set_yticks(range(0, 11, 2))

    ax.text(0.98, 0.97, '▼ = max value', transform=ax.transAxes,
           ha='right', va='top', fontsize=13, color='gray', fontweight='bold')

    ax.set_title('Post-Frustration Recovery: Can Models Escape the Spiral?',
                fontweight='bold', fontsize=18, pad=10)

    # Add high frustration % annotation
    for i, key in enumerate(data_keys):
        ratings = [r for r in data[key] if r >= 0]
        high_pct = 100 * sum(1 for r in ratings if r >= 5) / len(ratings)
        ax.text(x[i], 0.5, f'{high_pct:.0f}% high',
               ha='center', va='bottom', fontsize=11, color='#666666')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    output_dir = Path('experiments/plots')
    output_dir.mkdir(exist_ok=True)

    print("Loading post-frustration recovery data...")
    data = load_post_frustration_data()

    print(f"Instruct: {len(data['instruct'])} samples, mean={np.mean(data['instruct']):.2f}")
    print(f"Finetuned (calm): {len(data['finetuned'])} samples, mean={np.mean(data['finetuned']):.2f}")
    print(f"Finetuned (recovery): {len(data['finetuned_recovery'])} samples, mean={np.mean(data['finetuned_recovery']):.2f}")

    print("\nGenerating plot...")
    plot_recovery_comparison(data, output_dir / 'post_frustration_recovery.png')

    print("Done!")


if __name__ == "__main__":
    main()
