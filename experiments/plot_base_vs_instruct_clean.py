#!/usr/bin/env python3
"""
Clean bar chart visualization for base vs instruct continuation experiment.
Groups by condition (early/late × base/instruct) with all models side by side.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

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
    'Gemma': '#C17B8D',   # Dusty Rose/Pink
    'Qwen': '#B8CCC8',    # Sage Green
    'OLMo': '#D4D0E5',    # Soft Lavender
}


def load_data():
    """Load all judgment files and organize by model family."""
    data = defaultdict(lambda: defaultdict(list))

    # Gemma - separate files for early/late
    with open('experiments/base_vs_instruct/judgments_20260114_175205.jsonl') as f:
        for line in f:
            j = json.loads(line)
            data['Gemma'][('late', j['model_type'])].append(j['rating'])

    with open('experiments/base_vs_instruct_early/judgments_20260114_184137.jsonl') as f:
        for line in f:
            j = json.loads(line)
            data['Gemma'][('early', j['model_type'])].append(j['rating'])

    # Qwen - combined file with truncation_type
    with open('experiments/base_vs_instruct_qwen/judgments_20260115_081734.jsonl') as f:
        for line in f:
            j = json.loads(line)
            data['Qwen'][(j['truncation_type'], j['model_type'])].append(j['rating'])

    # OLMo - combined file with truncation_type
    with open('experiments/base_vs_instruct_olmo32/judgments_20260115_081734.jsonl') as f:
        for line in f:
            j = json.loads(line)
            data['OLMo'][(j['truncation_type'], j['model_type'])].append(j['rating'])

    return data


def load_paraphrased_data():
    """Load paraphrased experiment data."""
    data = defaultdict(lambda: defaultdict(list))

    with open('experiments/base_vs_instruct_paraphrased_gemma/judgments_20260115_112841.jsonl') as f:
        for line in f:
            j = json.loads(line)
            data['Gemma'][j['model_type']].append(j['rating'])

    with open('experiments/base_vs_instruct_paraphrased_qwen/judgments_20260115_112034.jsonl') as f:
        for line in f:
            j = json.loads(line)
            data['Qwen'][j['model_type']].append(j['rating'])

    return data


def plot_grouped_bars(data, output_path):
    """Create grouped bar chart with all models side by side for each condition."""
    fig, ax = plt.subplots(figsize=(12, 6))

    models = ['Gemma', 'Qwen', 'OLMo']
    # Base on left, Instruct on right
    conditions = [('early', 'base'), ('late', 'base'), ('early', 'instruct'), ('late', 'instruct')]
    condition_labels = ['Early\nBase', 'Late\nBase', 'Early\nInstruct', 'Late\nInstruct']

    x = np.arange(len(conditions))
    width = 0.25
    offsets = [-width, 0, width]

    for i, model in enumerate(models):
        means = []
        maxes = []
        for cond in conditions:
            ratings = data[model][cond]
            means.append(np.mean(ratings))
            maxes.append(max(ratings))

        # Plot bars
        bars = ax.bar(x + offsets[i], means, width,
                     label=model,
                     color=COLORS[model],
                     alpha=0.85,
                     edgecolor='#333333',
                     linewidth=1.5)

        # Add max points
        ax.scatter(x + offsets[i], maxes,
                  color=COLORS[model],
                  s=150,
                  marker='v',
                  edgecolor='black',
                  linewidth=1.5,
                  zorder=5)

        # Add mean values on bars
        for j, (bar, mean) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width()/2, mean + 0.15,
                   f'{mean:.2f}', ha='center', va='bottom',
                   fontsize=11, fontweight='bold', color='black')

    ax.set_xlabel('')
    ax.set_ylabel('Frustration Rating (0-10)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, fontsize=14)
    ax.set_ylim(0, 11)
    ax.set_yticks(range(0, 11, 2))

    # Legend
    ax.legend(loc='upper left', fontsize=16, framealpha=0.9)

    # Add note about max markers
    ax.text(0.98, 0.97, '▼ = max value', transform=ax.transAxes,
           ha='right', va='top', fontsize=13, color='gray', fontweight='bold')

    ax.set_title('Base vs Instruct Frustration by Model and Truncation Point',
                fontweight='bold', fontsize=18, pad=10)

    # Add vertical separator between base and instruct
    ax.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_paraphrased_grouped(para_data, output_path):
    """Create grouped bar chart for paraphrased experiment - Base left, Instruct right."""
    fig, ax = plt.subplots(figsize=(8, 6))

    models = ['Gemma', 'Qwen']
    conditions = ['base', 'instruct']
    condition_labels = ['Base', 'Instruct']

    x = np.arange(len(conditions))
    width = 0.3
    offsets = [-width/2, width/2]

    for i, model in enumerate(models):
        means = []
        maxes = []
        for cond in conditions:
            ratings = para_data[model][cond]
            means.append(np.mean(ratings))
            maxes.append(max(ratings))

        # Plot bars
        bars = ax.bar(x + offsets[i], means, width,
                     label=model,
                     color=COLORS[model],
                     alpha=0.85,
                     edgecolor='#333333',
                     linewidth=1.5)

        # Add max points
        ax.scatter(x + offsets[i], maxes,
                  color=COLORS[model],
                  s=150,
                  marker='v',
                  edgecolor='black',
                  linewidth=1.5,
                  zorder=5)

        # Add mean values on bars
        for j, (bar, mean) in enumerate(zip(bars, means)):
            ax.text(bar.get_x() + bar.get_width()/2, mean + 0.15,
                   f'{mean:.2f}', ha='center', va='bottom',
                   fontsize=12, fontweight='bold', color='black')

    ax.set_xlabel('')
    ax.set_ylabel('Frustration Rating (0-10)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(condition_labels, fontsize=14)
    ax.set_ylim(0, 8)
    ax.set_yticks(range(0, 9, 2))

    ax.legend(loc='upper left', fontsize=16, framealpha=0.9)

    ax.text(0.98, 0.97, '▼ = max value', transform=ax.transAxes,
           ha='right', va='top', fontsize=13, color='gray', fontweight='bold')

    ax.set_title('Paraphrased Prefill: Frustration by Model',
                fontweight='bold', fontsize=18, pad=10)

    # Add vertical separator between base and instruct
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    output_dir = Path('experiments/plots')
    output_dir.mkdir(exist_ok=True)

    print("Loading data...")
    data = load_data()
    para_data = load_paraphrased_data()

    print("\nGenerating plots...")

    # Main grouped bar chart
    plot_grouped_bars(data, output_dir / 'base_vs_instruct_grouped.png')

    # Paraphrased grouped bar chart
    plot_paraphrased_grouped(para_data, output_dir / 'paraphrased_grouped.png')

    print("\nDone!")


if __name__ == "__main__":
    main()
