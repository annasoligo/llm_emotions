#!/usr/bin/env python3
"""Plot test accuracy across layers for conversation emotion probes.

Evaluates probes on both user and assistant regional activations to compare performance.

Usage:
    python scripts/plot_conversation_probe_accuracy.py --results-dir probes/results/conversation_probes
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_probe_results(results_dir: Path, representation: str = "raw") -> Dict[int, dict]:
    """Load probe results for a specific representation.

    Args:
        results_dir: Directory containing probe .pkl files
        representation: Representation type (raw, global_cpca, regional_cpca)

    Returns:
        Dict mapping layer number to results
    """
    results = {}

    # Match probe files for this representation
    pattern = f"probe_layer*_{representation}*.pkl"

    for pkl_file in results_dir.glob(pattern):
        # Extract layer number from filename: probe_layer10_raw_l10.1.pkl
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


def plot_accuracy_by_layer_and_representation(
    results_by_rep: Dict[str, Dict[int, dict]],
    output_path: Path,
    title: str = "Conversation Probe Accuracy by Layer",
):
    """Plot test accuracy across layers for different representations.

    Args:
        results_by_rep: Dict mapping representation name to {layer: results}
        output_path: Path to save plot
        title: Plot title
    """
    # Color scheme
    COLORS = {
        "raw": "#D4876A",           # coral
        "global_cpca_3": "#7BA7D7", # sky blue
        "global_cpca_5": "#6B9BC7",
        "global_cpca_10": "#5B8BB7",
        "regional_cpca_3": "#B8CCC8", # sage
        "regional_cpca_5": "#A8BCB8",
        "regional_cpca_10": "#98ACA8",
    }

    LABELS = {
        "raw": "Raw Mean",
        "global_cpca_3": "Global cPCA top 3",
        "global_cpca_5": "Global cPCA top 5",
        "global_cpca_10": "Global cPCA top 10",
        "regional_cpca_3": "Regional cPCA top 3 (×4)",
        "regional_cpca_5": "Regional cPCA top 5 (×4)",
        "regional_cpca_10": "Regional cPCA top 10 (×4)",
    }

    MARKERS = {
        "raw": "o",
        "global_cpca_3": "s",
        "global_cpca_5": "^",
        "global_cpca_10": "D",
        "regional_cpca_3": "v",
        "regional_cpca_5": "<",
        "regional_cpca_10": ">",
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each representation
    for rep_name, results in sorted(results_by_rep.items()):
        if not results:
            continue

        layers = sorted(results.keys())
        test_accs = [results[l]["test_accuracy"] for l in layers]

        color = COLORS.get(rep_name, "#666666")
        label = LABELS.get(rep_name, rep_name)
        marker = MARKERS.get(rep_name, "o")

        ax.plot(layers, test_accs, marker=marker, linewidth=2.5, markersize=8,
                label=label, color=color, alpha=0.85)

    # Formatting
    ax.set_xlabel('Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Accuracy', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.legend(fontsize=10, framealpha=0.95, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    return fig, ax


def print_summary_table(results_by_rep: Dict[str, Dict[int, dict]]):
    """Print summary table comparing representations."""

    print("\n" + "=" * 100)
    print("CONVERSATION PROBE ACCURACY COMPARISON")
    print("=" * 100)

    # Get all layers
    all_layers = set()
    for results in results_by_rep.values():
        all_layers.update(results.keys())
    layers = sorted(all_layers)

    print(f"{'Layer':<8}", end='')
    for rep in sorted(results_by_rep.keys()):
        print(f"{rep:<15}", end='')
    print()
    print("-" * 100)

    for layer in layers:
        print(f"{layer:<8}", end='')
        for rep in sorted(results_by_rep.keys()):
            if layer in results_by_rep[rep]:
                acc = results_by_rep[rep][layer]['test_accuracy']
                print(f"{acc:<15.4f}", end='')
            else:
                print(f"{'N/A':<15}", end='')
        print()

    print("=" * 100)

    # Best per representation
    print("\nBest accuracy per representation:")
    for rep, results in sorted(results_by_rep.items()):
        if results:
            best_layer = max(results.keys(), key=lambda l: results[l]['test_accuracy'])
            best_acc = results[best_layer]['test_accuracy']
            print(f"  {rep:<20}: Layer {best_layer:2d} = {best_acc:.4f}")

    print()


def main():
    parser = argparse.ArgumentParser(description="Plot conversation probe accuracy by layer")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/workspace-vast/annas/git/research-tools/probes/results/conversation_probes",
        help="Directory containing probe results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output plot path (default: results_dir/accuracy_by_layer.png)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Conversation Probe Accuracy by Layer",
        help="Plot title",
    )
    parser.add_argument(
        "--representations",
        type=str,
        nargs="+",
        default=["raw", "global_cpca", "regional_cpca"],
        help="Representations to include",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Load results for each representation
    print(f"Loading results from {results_dir}...")
    results_by_rep = {}

    # Load raw
    if "raw" in args.representations:
        raw_results = load_probe_results(results_dir, "raw")
        if raw_results:
            results_by_rep["raw"] = raw_results
            print(f"  raw: {len(raw_results)} layers")

    # Load global cPCA variants
    if "global_cpca" in args.representations:
        for n_comp in [3, 5, 10]:
            # Try both with and without _top prefix
            global_results = {}
            for pattern in [f"global_cpca_top{n_comp}", f"global_cpca"]:
                temp_results = load_probe_results(results_dir, pattern)
                # Filter by n_components
                filtered = {k: v for k, v in temp_results.items()
                           if v.get('n_components') == n_comp}
                if filtered:
                    global_results = filtered
                    break

            if global_results:
                results_by_rep[f"global_cpca_{n_comp}"] = global_results
                print(f"  global_cpca top {n_comp}: {len(global_results)} layers")

    # Load regional cPCA variants
    if "regional_cpca" in args.representations:
        for n_comp in [3, 5, 10]:
            # Try both with and without _top prefix
            regional_results = {}
            for pattern in [f"regional_cpca_top{n_comp}", f"regional_cpca"]:
                temp_results = load_probe_results(results_dir, pattern)
                # Filter by n_components
                filtered = {k: v for k, v in temp_results.items()
                           if v.get('n_components') == n_comp}
                if filtered:
                    regional_results = filtered
                    break

            if regional_results:
                results_by_rep[f"regional_cpca_{n_comp}"] = regional_results
                print(f"  regional_cpca top {n_comp}: {len(regional_results)} layers")

    if not results_by_rep:
        print("No probe results found!")
        return

    # Print summary
    print_summary_table(results_by_rep)

    # Plot
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_dir / "accuracy_by_layer_comparison.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_accuracy_by_layer_and_representation(results_by_rep, output_path, title=args.title)

    print("\nDone!")


if __name__ == "__main__":
    main()
