#!/usr/bin/env python3
"""Visualize multi-seed probe results with confidence intervals."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Emotion color scheme
COLORS = {
    'raw': '#5A9BD5',       # Clear Blue
    'nc5': '#D4876A',       # Orange/Coral
    'nc10': '#7D9B7D',      # Green/Olive
    'nc20': '#B8A8D4',      # Purple/Lavender
}

LABELS = {
    'raw': 'Raw (5376-dim)',
    'nc5': '5 PCs',
    'nc10': '10 PCs',
    'nc20': '20 PCs',
}

MARKERS = {
    'raw': 's',
    'nc5': 's',
    'nc10': 's',
    'nc20': 's',
}


def plot_accuracy_with_ci(results: dict, output_dir: Path):
    """Plot accuracy by layer with confidence intervals."""

    # User accuracy plot
    fig, ax = plt.subplots(figsize=(12, 7))

    for nc_key in ['raw', 'nc5', 'nc10', 'nc20']:
        if nc_key not in results:
            continue

        layers = sorted([int(l) for l in results[nc_key].keys()])
        means = [results[nc_key][str(l)]['user_accuracy_mean'] for l in layers]
        ci95s = [results[nc_key][str(l)]['user_accuracy_ci95'] for l in layers]

        ax.plot(layers, means, marker=MARKERS[nc_key], color=COLORS[nc_key],
                label=LABELS[nc_key], linewidth=2.5, markersize=8, alpha=0.9)
        ax.fill_between(layers,
                         [m - ci for m, ci in zip(means, ci95s)],
                         [m + ci for m, ci in zip(means, ci95s)],
                         color=COLORS[nc_key], alpha=0.2)

    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('User Emotion Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('User Emotion Prediction Accuracy (Mean ± 95% CI, n=10 seeds)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / 'user_accuracy_with_ci.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'user_accuracy_with_ci.png'}")

    # Assistant accuracy plot
    fig, ax = plt.subplots(figsize=(12, 7))

    for nc_key in ['raw', 'nc5', 'nc10', 'nc20']:
        if nc_key not in results:
            continue

        layers = sorted([int(l) for l in results[nc_key].keys()])
        means = [results[nc_key][str(l)]['asst_accuracy_mean'] for l in layers]
        ci95s = [results[nc_key][str(l)]['asst_accuracy_ci95'] for l in layers]

        ax.plot(layers, means, marker=MARKERS[nc_key], color=COLORS[nc_key],
                label=LABELS[nc_key], linewidth=2.5, markersize=8, alpha=0.9)
        ax.fill_between(layers,
                         [m - ci for m, ci in zip(means, ci95s)],
                         [m + ci for m, ci in zip(means, ci95s)],
                         color=COLORS[nc_key], alpha=0.2)

    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Assistant Emotion Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Assistant Emotion Prediction Accuracy (Mean ± 95% CI, n=10 seeds)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / 'asst_accuracy_with_ci.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'asst_accuracy_with_ci.png'}")

    # Overall accuracy plot
    fig, ax = plt.subplots(figsize=(12, 7))

    for nc_key in ['raw', 'nc5', 'nc10', 'nc20']:
        if nc_key not in results:
            continue

        layers = sorted([int(l) for l in results[nc_key].keys()])
        means = [results[nc_key][str(l)]['overall_accuracy_mean'] for l in layers]
        ci95s = [results[nc_key][str(l)]['overall_accuracy_ci95'] for l in layers]

        ax.plot(layers, means, marker=MARKERS[nc_key], color=COLORS[nc_key],
                label=LABELS[nc_key], linewidth=2.5, markersize=8, alpha=0.9)
        ax.fill_between(layers,
                         [m - ci for m, ci in zip(means, ci95s)],
                         [m + ci for m, ci in zip(means, ci95s)],
                         color=COLORS[nc_key], alpha=0.2)

    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Overall Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Overall Emotion Prediction Accuracy (Mean ± 95% CI, n=10 seeds)',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_accuracy_with_ci.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'overall_accuracy_with_ci.png'}")


def plot_dimensionality_comparison(results: dict, output_dir: Path, layer: int = 30):
    """Plot accuracy by dimensionality at a specific layer."""

    nc_keys = ['raw', 'nc5', 'nc10', 'nc20']
    labels_list = ['Raw', '5', '10', '20']

    user_means = []
    user_ci95s = []
    asst_means = []
    asst_ci95s = []

    for nc_key in nc_keys:
        if nc_key not in results or str(layer) not in results[nc_key]:
            user_means.append(0)
            user_ci95s.append(0)
            asst_means.append(0)
            asst_ci95s.append(0)
            continue

        layer_data = results[nc_key][str(layer)]
        user_means.append(layer_data['user_accuracy_mean'])
        user_ci95s.append(layer_data['user_accuracy_ci95'])
        asst_means.append(layer_data['asst_accuracy_mean'])
        asst_ci95s.append(layer_data['asst_accuracy_ci95'])

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(labels_list))
    width = 0.35

    bars1 = ax.bar(x - width/2, user_means, width, yerr=user_ci95s,
                   label='User', color='#7BA7D7', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, asst_means, width, yerr=asst_ci95s,
                   label='Assistant', color='#D4876A', alpha=0.8, capsize=5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.2%}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

    ax.set_xlabel('Dimensionality (PCs or Raw)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'Accuracy by Dimensionality at Layer {layer} (Mean ± 95% CI, n=10 seeds)',
                 fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_dir / f'dimensionality_comparison_layer{layer}.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / f'dimensionality_comparison_layer{layer}.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str,
                       default="results/conversation_eval/multiseed_results.json")
    parser.add_argument("--output-dir", type=str,
                       default="results/conversation_eval/multiseed_plots")
    args = parser.parse_args()

    results_path = Path(args.results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    with open(results_path, 'r') as f:
        results = json.load(f)

    print("Generating plots...")
    plot_accuracy_with_ci(results, output_dir)
    plot_dimensionality_comparison(results, output_dir, layer=30)

    print("\n✓ All visualizations saved to", output_dir)


if __name__ == "__main__":
    main()
