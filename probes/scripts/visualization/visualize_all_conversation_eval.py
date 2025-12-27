#!/usr/bin/env python3
"""Visualize and compare results from all probe conversation evaluations."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all result JSONs from directory."""
    results = {}

    patterns = [
        'gemma3_raw.json',
        'gemma3_all_cpca.json',
        'gemma3_top3.json',
        'gemma3_top5.json',
        'gemma3_top10.json',
        'gemma3_top20.json',
    ]

    for pattern in patterns:
        path = results_dir / pattern
        if path.exists():
            with open(path, 'r') as f:
                key = pattern.replace('gemma3_', '').replace('.json', '')
                results[key] = json.load(f)
                print(f"Loaded {key}: {results[key]['num_conversations']} conversations")
        else:
            print(f"Warning: {path} not found")

    return results


def plot_user_accuracy_by_layer(results: Dict[str, Dict], output_path: Path):
    """Plot user emotion accuracy by layer for all probe types."""
    plt.figure(figsize=(12, 7))

    # Define colors and markers (using emotion color scheme)
    colors = {
        'raw': '#B8CCC8',       # Pale Green/Sage (bored)
        'all_cpca': '#7BA7D7',  # Blue/Sky Blue
        'top3': '#C17B8D',      # Pink/Dusty Rose
        'top5': '#D4876A',      # Orange/Coral
        'top10': '#7D9B7D',     # Green/Olive Green
        'top20': '#B8A8D4',     # Purple/Lavender
    }

    markers = {
        'raw': 'x',
        'all_cpca': 'o',
        'top3': '^',
        'top5': 's',
        'top10': 'D',
        'top20': 'v',
    }

    for probe_name, data in results.items():
        layers = sorted([int(k) for k in data['layers'].keys()])
        user_accs = [data['layers'][str(l)]['user_accuracy'] for l in layers]

        label = probe_name.replace('_', ' ').title()
        if probe_name.startswith('top'):
            k = probe_name.replace('top', '')
            label = f'Top-{k} PCs'
        elif probe_name == 'all_cpca':
            label = 'All PCs (50)'
        elif probe_name == 'raw':
            label = 'Raw (5376-dim)'

        plt.plot(
            layers,
            user_accs,
            marker=markers.get(probe_name, 'o'),
            color=colors.get(probe_name, 'gray'),
            label=label,
            linewidth=2.5,
            markersize=8,
            alpha=0.9
        )

    plt.xlabel('Layer', fontsize=14, fontweight='bold')
    plt.ylabel('User Emotion Accuracy', fontsize=14, fontweight='bold')
    plt.title('User Emotion Prediction Accuracy by Layer and PC Count', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best', framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_asst_accuracy_by_layer(results: Dict[str, Dict], output_path: Path):
    """Plot assistant emotion accuracy by layer for all probe types."""
    plt.figure(figsize=(12, 7))

    # Using emotion color scheme
    colors = {
        'raw': '#B8CCC8',       # Pale Green/Sage (bored)
        'all_cpca': '#7BA7D7',  # Blue/Sky Blue
        'top3': '#C17B8D',      # Pink/Dusty Rose
        'top5': '#D4876A',      # Orange/Coral
        'top10': '#7D9B7D',     # Green/Olive Green
        'top20': '#B8A8D4',     # Purple/Lavender
    }

    markers = {
        'raw': 'x',
        'all_cpca': 'o',
        'top3': '^',
        'top5': 's',
        'top10': 'D',
        'top20': 'v',
    }

    for probe_name, data in results.items():
        layers = sorted([int(k) for k in data['layers'].keys()])
        asst_accs = [data['layers'][str(l)]['asst_accuracy'] for l in layers]

        label = probe_name.replace('_', ' ').title()
        if probe_name.startswith('top'):
            k = probe_name.replace('top', '')
            label = f'Top-{k} PCs'
        elif probe_name == 'all_cpca':
            label = 'All PCs (50)'
        elif probe_name == 'raw':
            label = 'Raw (5376-dim)'

        plt.plot(
            layers,
            asst_accs,
            marker=markers.get(probe_name, 'o'),
            color=colors.get(probe_name, 'gray'),
            label=label,
            linewidth=2.5,
            markersize=8,
            alpha=0.9
        )

    plt.xlabel('Layer', fontsize=14, fontweight='bold')
    plt.ylabel('Assistant Emotion Accuracy', fontsize=14, fontweight='bold')
    plt.title('Assistant Emotion Prediction Accuracy by Layer and PC Count', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='best', framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_by_pc_count(results: Dict[str, Dict], output_path: Path, best_layer: int = None):
    """Plot accuracy by PC count for a specific layer (or best layer)."""

    # Find best layer if not specified
    if best_layer is None:
        # Use layer 30 as default (usually good for emotion)
        best_layer = 30

    # Filter to only top-k results
    topk_results = {k: v for k, v in results.items() if k.startswith('top')}

    if not topk_results:
        print("No top-k results found for PC count plot")
        return

    pc_counts = []
    user_accs = []
    asst_accs = []
    labels = []

    # Add raw if available
    if 'raw' in results:
        data = results['raw']
        layer_key = str(best_layer)
        if layer_key in data['layers']:
            pc_counts.append(0)  # 0 to represent raw (no reduction)
            user_accs.append(data['layers'][layer_key]['user_accuracy'])
            asst_accs.append(data['layers'][layer_key]['asst_accuracy'])
            labels.append('Raw')

    for probe_name in ['top3', 'top5', 'top10', 'top20']:
        if probe_name in topk_results:
            data = topk_results[probe_name]
            layer_key = str(best_layer)

            if layer_key in data['layers']:
                k = int(probe_name.replace('top', ''))
                pc_counts.append(k)
                user_accs.append(data['layers'][layer_key]['user_accuracy'])
                asst_accs.append(data['layers'][layer_key]['asst_accuracy'])
                labels.append(str(k))

    # Add all_cpca if available
    if 'all_cpca' in results:
        data = results['all_cpca']
        layer_key = str(best_layer)
        if layer_key in data['layers']:
            pc_counts.append(50)
            user_accs.append(data['layers'][layer_key]['user_accuracy'])
            asst_accs.append(data['layers'][layer_key]['asst_accuracy'])
            labels.append('50')

    if not pc_counts:
        print(f"No data found for layer {best_layer}")
        return

    # Sort by PC count
    sorted_indices = np.argsort(pc_counts)
    pc_counts = [pc_counts[i] for i in sorted_indices]
    user_accs = [user_accs[i] for i in sorted_indices]
    asst_accs = [asst_accs[i] for i in sorted_indices]
    labels = [labels[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(pc_counts))
    width = 0.35

    # Using emotion color scheme
    bars1 = ax.bar(x - width/2, user_accs, width, label='User', color='#7BA7D7', alpha=0.8)  # Blue
    bars2 = ax.bar(x + width/2, asst_accs, width, label='Assistant', color='#D4876A', alpha=0.8)  # Orange/Coral

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.2%}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

    ax.set_xlabel('Dimensionality (PCs or Raw)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(f'Accuracy by Dimensionality (Layer {best_layer})', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_heatmap(results: Dict[str, Dict], output_path: Path, role: str = 'user'):
    """Plot heatmap showing accuracy across layers and PC counts."""

    # Collect data
    probe_types = ['raw', 'top3', 'top5', 'top10', 'top20', 'all_cpca']
    available_probes = [p for p in probe_types if p in results]

    if not available_probes:
        print("No results available for heatmap")
        return

    # Get all layers (assume consistent across probe types)
    layers = sorted([int(k) for k in results[available_probes[0]]['layers'].keys()])

    # Build matrix
    matrix = []
    y_labels = []

    for probe_name in available_probes:
        data = results[probe_name]
        row = []
        for layer in layers:
            layer_key = str(layer)
            if layer_key in data['layers']:
                acc = data['layers'][layer_key][f'{role}_accuracy']
                row.append(acc)
            else:
                row.append(np.nan)
        matrix.append(row)

        # Create label
        if probe_name.startswith('top'):
            k = probe_name.replace('top', '')
            y_labels.append(f'Top-{k}')
        elif probe_name == 'all_cpca':
            y_labels.append('All (50)')
        elif probe_name == 'raw':
            y_labels.append('Raw')

    matrix = np.array(matrix)

    # Plot
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2%',
        cmap='RdYlGn',
        xticklabels=layers,
        yticklabels=y_labels,
        cbar_kws={'label': 'Accuracy'},
        vmin=0,
        vmax=1,
        linewidths=0.5,
        linecolor='gray'
    )

    plt.xlabel('Layer', fontsize=14, fontweight='bold')
    plt.ylabel('PC Count', fontsize=14, fontweight='bold')
    plt.title(f'{role.capitalize()} Emotion Accuracy Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_summary_table(results: Dict[str, Dict]):
    """Print summary statistics table."""
    print("\n" + "=" * 100)
    print("SUMMARY: BEST ACCURACY BY PROBE TYPE")
    print("=" * 100)

    summary = []
    for probe_name, data in results.items():
        best_user = max(data['layers'].items(), key=lambda x: x[1]['user_accuracy'])
        best_asst = max(data['layers'].items(), key=lambda x: x[1]['asst_accuracy'])
        best_overall = max(data['layers'].items(), key=lambda x: x[1]['overall_accuracy'])

        summary.append({
            'probe': probe_name,
            'user_acc': best_user[1]['user_accuracy'],
            'user_layer': int(best_user[0]),
            'asst_acc': best_asst[1]['asst_accuracy'],
            'asst_layer': int(best_asst[0]),
            'overall_acc': best_overall[1]['overall_accuracy'],
            'overall_layer': int(best_overall[0]),
        })

    # Print table
    print(f"{'Probe Type':<15} {'User Acc':<12} {'Layer':<8} {'Asst Acc':<12} {'Layer':<8} {'Overall':<12} {'Layer':<8}")
    print("-" * 100)

    for s in summary:
        probe_label = s['probe'].replace('top', 'Top-').replace('all_cpca', 'All (50)')
        print(
            f"{probe_label:<15} "
            f"{s['user_acc']:<12.2%} {s['user_layer']:<8} "
            f"{s['asst_acc']:<12.2%} {s['asst_layer']:<8} "
            f"{s['overall_acc']:<12.2%} {s['overall_layer']:<8}"
        )

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="Visualize all conversation evaluation results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/conversation_eval"),
        help="Directory containing all result JSONs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/conversation_eval/plots"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--best-layer",
        type=int,
        default=30,
        help="Layer to use for PC count comparison (default: 30)",
    )

    args = parser.parse_args()

    print("Loading all results...")
    results = load_results(args.results_dir)

    if not results:
        print("No results found!")
        return

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating visualizations in {args.output_dir}...")

    # Print summary
    print_summary_table(results)

    # Generate plots
    print("\nGenerating plots...")

    plot_user_accuracy_by_layer(
        results,
        args.output_dir / "user_accuracy_by_layer.png"
    )

    plot_asst_accuracy_by_layer(
        results,
        args.output_dir / "asst_accuracy_by_layer.png"
    )

    plot_accuracy_by_pc_count(
        results,
        args.output_dir / f"accuracy_by_pc_count_layer{args.best_layer}.png",
        best_layer=args.best_layer
    )

    plot_combined_heatmap(
        results,
        args.output_dir / "user_accuracy_heatmap.png",
        role='user'
    )

    plot_combined_heatmap(
        results,
        args.output_dir / "asst_accuracy_heatmap.png",
        role='asst'
    )

    print(f"\n✓ All visualizations saved to {args.output_dir}")
    print("\nGenerated files:")
    print("  - user_accuracy_by_layer.png")
    print("  - asst_accuracy_by_layer.png")
    print(f"  - accuracy_by_pc_count_layer{args.best_layer}.png")
    print("  - user_accuracy_heatmap.png")
    print("  - asst_accuracy_heatmap.png")


if __name__ == "__main__":
    main()
