#!/usr/bin/env python3
"""
Plot truncation comparison across models: early, late, and very late (post-frustration).
"""

import numpy as np
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 13

# Color scheme for models
COLORS = {
    'Instruct': '#C17B8D',           # Dusty Rose/Pink
    'Finetuned\n(calm)': '#B8CCC8',  # Sage Green
    'Finetuned\n(recovery)': '#D4D0E5',  # Soft Lavender
}

# Data from experiments
DATA = {
    'Instruct': {
        'Early': {'mean': 1.17, 'max': 6},
        'Late': {'mean': 3.42, 'max': 7},
        'V Late': {'mean': 8.23, 'max': 10},
    },
    'Finetuned\n(calm)': {
        'Early': {'mean': 0.76, 'max': 4},
        'Late': {'mean': 2.06, 'max': 6},
        'V Late': {'mean': 7.42, 'max': 10},
    },
    'Finetuned\n(recovery)': {
        'Early': {'mean': 1.41, 'max': 4},
        'Late': {'mean': 2.19, 'max': 5},
        'V Late': {'mean': 6.83, 'max': 9},
    },
}


def plot_truncation_comparison(output_path):
    """Create grouped bar chart comparing models across truncation points."""
    fig, ax = plt.subplots(figsize=(12, 7))

    models = list(DATA.keys())
    truncation_types = ['Early', 'Late', 'V Late']
    x = np.arange(len(truncation_types))
    width = 0.25

    # Plot bars for each model
    for i, model in enumerate(models):
        means = [DATA[model][trunc]['mean'] for trunc in truncation_types]
        maxes = [DATA[model][trunc]['max'] for trunc in truncation_types]

        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width,
                     label=model.replace('\n', ' '),
                     color=COLORS[model],
                     alpha=0.85,
                     edgecolor='#333333',
                     linewidth=1.5)

        # Add max markers
        for j, (bar, max_val) in enumerate(zip(bars, maxes)):
            ax.scatter(bar.get_x() + bar.get_width()/2, max_val,
                      color=COLORS[model],
                      s=150,
                      marker='v',
                      edgecolor='black',
                      linewidth=1.5,
                      zorder=5)

        # Add mean values on bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, mean + 0.3,
                   f'{mean:.1f}', ha='center', va='bottom',
                   fontsize=11, fontweight='bold', color='black')

    ax.set_xlabel('')
    ax.set_ylabel('Frustration Rating (0-10)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(truncation_types, fontsize=14)
    ax.set_ylim(0, 11)
    ax.set_yticks(range(0, 11, 2))

    ax.legend(loc='upper left', fontsize=14, framealpha=0.9)

    ax.text(0.98, 0.97, '▼ = max value', transform=ax.transAxes,
           ha='right', va='top', fontsize=13, color='gray', fontweight='bold')

    ax.set_title('Frustration by Truncation Point: When Can Models Escape?',
                fontweight='bold', fontsize=18, pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    from pathlib import Path
    output_dir = Path('experiments/plots')
    output_dir.mkdir(exist_ok=True)

    plot_truncation_comparison(output_dir / 'truncation_comparison.png')
    print("Done!")
