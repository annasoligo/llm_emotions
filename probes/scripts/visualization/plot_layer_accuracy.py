#!/usr/bin/env python3
"""Plot test accuracy across layers for emotion probes.

Usage:
    python scripts/plot_layer_accuracy.py --results-dir results/emotion_probes
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_probe_results(results_dir: Path) -> Dict[int, dict]:
    """Load all probe results from a directory.

    Args:
        results_dir: Directory containing probe .pkl files

    Returns:
        Dict mapping layer number to results
    """
    results = {}

    # Match both raw and cPCA probe files
    for pattern in ["probe_layer*_all.pkl", "probe_layer*_all_cpca.pkl"]:
        for pkl_file in results_dir.glob(pattern):
            # Extract layer number from filename: probe_layer31_all.pkl or probe_layer31_all_cpca.pkl
            parts = pkl_file.stem.split("_")
            if len(parts) >= 2 and parts[1].startswith("layer"):
                layer_str = parts[1][5:]  # Remove "layer" prefix
                try:
                    layer = int(layer_str)
                    with open(pkl_file, "rb") as f:
                        results[layer] = pickle.load(f)
                except (ValueError, Exception) as e:
                    print(f"Warning: Could not load {pkl_file}: {e}")
                    continue

    return results


def plot_accuracy_by_layer(
    results: Dict[int, dict],
    output_path: Path,
    title: str = "Emotion Probe Accuracy by Layer",
):
    """Plot test accuracy across layers.

    Args:
        results: Dict mapping layer to results
        output_path: Path to save plot
        title: Plot title
    """
    # Sort by layer
    layers = sorted(results.keys())
    test_accs = [results[l]["test_accuracy"] for l in layers]
    train_accs = [results[l]["train_accuracy"] for l in layers]
    best_test_accs = [results[l]["best_test_accuracy"] for l in layers]

    # believe-it-or-not color scheme
    COLORS = {
        "coral": "#D4876A",
        "sky_blue": "#7BA7D7",
        "sage": "#B8CCC8",
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot lines with believe-it-or-not colors
    ax.plot(layers, test_accs, 'o-', linewidth=2.5, markersize=8,
            label='Test Accuracy (final)', color=COLORS['coral'])
    ax.plot(layers, best_test_accs, 's--', linewidth=2, markersize=6,
            label='Test Accuracy (best)', color=COLORS['sky_blue'], alpha=0.8)
    ax.plot(layers, train_accs, '^:', linewidth=1.5, markersize=6,
            label='Train Accuracy (final)', color=COLORS['sage'], alpha=0.6)

    # Formatting
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.05)

    # Add value annotations for test accuracy
    for layer, acc in zip(layers, test_accs):
        ax.annotate(f'{acc:.3f}',
                   xy=(layer, acc),
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=8,
                   alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    return fig, ax


def print_summary_table(results: Dict[int, dict]):
    """Print summary table of results."""
    layers = sorted(results.keys())

    print("\n" + "=" * 90)
    print("EMOTION PROBE ACCURACY BY LAYER")
    print("=" * 90)
    print(f"{'Layer':<8} {'Train Acc':<12} {'Test Acc':<12} {'Best Test':<12} "
          f"{'Best Epoch':<12} {'Total Epochs':<12}")
    print("-" * 90)

    for layer in layers:
        r = results[layer]
        print(f"{layer:<8} {r['train_accuracy']:<12.4f} {r['test_accuracy']:<12.4f} "
              f"{r['best_test_accuracy']:<12.4f} {r['best_epoch']+1:<12} "
              f"{r['total_epochs']:<12}")

    print("=" * 90)

    # Find best layer
    best_layer = max(layers, key=lambda l: results[l]['test_accuracy'])
    best_acc = results[best_layer]['test_accuracy']
    print(f"\nBest layer: {best_layer} (test accuracy: {best_acc:.4f})")

    # Average across layers
    avg_test = np.mean([results[l]['test_accuracy'] for l in layers])
    std_test = np.std([results[l]['test_accuracy'] for l in layers])
    print(f"Average test accuracy: {avg_test:.4f} ± {std_test:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Plot probe accuracy by layer")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/emotion_probes",
        help="Directory containing probe results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/emotion_probes/accuracy_by_layer.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Emotion Probe Accuracy by Layer",
        help="Plot title",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Load results
    print(f"Loading results from {results_dir}...")
    results = load_probe_results(results_dir)

    if not results:
        print("No probe results found!")
        return

    print(f"Found results for {len(results)} layers: {sorted(results.keys())}")

    # Print summary
    print_summary_table(results)

    # Plot
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_accuracy_by_layer(results, output_path, title=args.title)

    print("\nDone!")


if __name__ == "__main__":
    main()
