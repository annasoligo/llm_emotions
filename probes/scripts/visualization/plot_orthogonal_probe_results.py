#!/usr/bin/env python3
"""Plot orthogonal probe results and compare with regular probes.

Usage:
    python scripts/plot_orthogonal_probe_results.py
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_orthogonal_probe_results(results_dir: Path) -> Dict[Tuple[int, str], dict]:
    """Load all orthogonal probe results from a directory."""
    results = {}

    for pkl_file in results_dir.glob("probe_layer*_ortho*.pkl"):
        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            layer = data["layer"]
            representation = data["representation"]
            n_components = data.get("n_components")

            # Create representation key
            if representation == "raw":
                rep_key = "raw"
            elif representation == "global_cpca":
                rep_key = f"global_top{n_components}"
            elif representation == "regional_cpca":
                rep_key = f"regional_top{n_components}"
            else:
                continue

            results[(layer, rep_key)] = data

        except Exception as e:
            print(f"Warning: Could not load {pkl_file}: {e}")
            continue

    return results


def plot_orthogonality_metrics(results: Dict[Tuple[int, str], dict], output_dir: Path):
    """Plot orthogonality metrics across layers and representations."""

    # Group by representation
    rep_to_results = {}
    for (layer, rep), data in results.items():
        if rep not in rep_to_results:
            rep_to_results[rep] = {}
        rep_to_results[rep][layer] = data

    if not rep_to_results:
        print("No orthogonal results found!")
        return

    # Color scheme - organized by representation type
    COLORS = {
        "raw": "#000000",  # Black
        "global_top3": "#E74C3C",  # Red
        "global_top5": "#F39C12",  # Orange
        "global_top10": "#F1C40F",  # Yellow
        "global_top20": "#E67E22",  # Dark orange
        "regional_top3": "#3498DB",  # Blue
        "regional_top5": "#1ABC9C",  # Teal
        "regional_top10": "#2ECC71",  # Green
        "regional_top20": "#16A085",  # Dark teal
    }

    MARKERS = {
        "raw": "o",
        "global_top3": "s",
        "global_top5": "^",
        "global_top10": "D",
        "global_top20": "p",
        "regional_top3": "v",
        "regional_top5": "<",
        "regional_top10": ">",
        "regional_top20": "h",
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: User accuracy (full scale)
    ax = axes[0, 0]
    for rep in sorted(rep_to_results.keys()):
        layer_results = rep_to_results[rep]
        layers = sorted(layer_results.keys())

        user_acc = [layer_results[l]["final_val_metrics"]["user_accuracy"] for l in layers]

        ax.plot(layers, user_acc, marker=MARKERS.get(rep, "o"), linewidth=2.5, markersize=8,
                label=rep, color=COLORS.get(rep, "#666"), alpha=0.85)

    ax.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax.set_ylabel("User Accuracy", fontsize=14, fontweight="bold")
    ax.set_title("User Emotion Prediction Accuracy", fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.05)

    # Plot 2: Assistant accuracy (full scale)
    ax = axes[0, 1]
    for rep in sorted(rep_to_results.keys()):
        layer_results = rep_to_results[rep]
        layers = sorted(layer_results.keys())

        asst_acc = [layer_results[l]["final_val_metrics"]["asst_accuracy"] for l in layers]

        ax.plot(layers, asst_acc, marker=MARKERS.get(rep, "o"), linewidth=2.5, markersize=8,
                label=rep, color=COLORS.get(rep, "#666"), alpha=0.85)

    ax.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax.set_ylabel("Assistant Accuracy", fontsize=14, fontweight="bold")
    ax.set_title("Assistant Emotion Prediction Accuracy", fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.05)

    # Plot 3: User accuracy (zoomed to 0.8-1.0)
    ax = axes[1, 0]
    for rep in sorted(rep_to_results.keys()):
        layer_results = rep_to_results[rep]
        layers = sorted(layer_results.keys())

        user_acc = [layer_results[l]["final_val_metrics"]["user_accuracy"] for l in layers]

        ax.plot(layers, user_acc, marker=MARKERS.get(rep, "o"), linewidth=2.5, markersize=8,
                label=rep, color=COLORS.get(rep, "#666"), alpha=0.85)

    ax.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax.set_ylabel("User Accuracy", fontsize=14, fontweight="bold")
    ax.set_title("User Accuracy (Zoomed)", fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0.8, 1.0)

    # Plot 4: Assistant accuracy (zoomed to 0.8-1.0)
    ax = axes[1, 1]
    for rep in sorted(rep_to_results.keys()):
        layer_results = rep_to_results[rep]
        layers = sorted(layer_results.keys())

        asst_acc = [layer_results[l]["final_val_metrics"]["asst_accuracy"] for l in layers]

        ax.plot(layers, asst_acc, marker=MARKERS.get(rep, "o"), linewidth=2.5, markersize=8,
                label=rep, color=COLORS.get(rep, "#666"), alpha=0.85)

    ax.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax.set_ylabel("Assistant Accuracy", fontsize=14, fontweight="bold")
    ax.set_title("Assistant Accuracy (Zoomed)", fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0.8, 1.0)

    plt.tight_layout()
    output_path = output_dir / "orthogonal_probe_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved orthogonal metrics plot to {output_path}")


def print_summary_table(results: Dict[Tuple[int, str], dict]):
    """Print summary table of orthogonal probe results."""

    # Group by layer
    layer_results = {}
    for (layer, rep), data in results.items():
        if layer not in layer_results:
            layer_results[layer] = {}
        layer_results[layer][rep] = data

    print("\n" + "=" * 140)
    print("ORTHOGONAL PROBE RESULTS SUMMARY")
    print("=" * 140)

    for layer in sorted(layer_results.keys()):
        print(f"\nLAYER {layer}")
        print("-" * 140)
        print(f"{'Representation':<20} {'User Acc':<12} {'Asst Acc':<12} {'Cosine Sim':<12} {'User→Asst':<12} {'Asst→User':<12}")
        print("-" * 140)

        for rep in sorted(layer_results[layer].keys()):
            data = layer_results[layer][rep]
            user_acc = data["final_val_metrics"]["user_accuracy"]
            asst_acc = data["final_val_metrics"]["asst_accuracy"]
            cosine_sim = data["final_ortho_metrics"]["cross_dots_mean"]
            user_on_asst = data["final_val_metrics"]["user_on_asst_accuracy"]
            asst_on_user = data["final_val_metrics"]["asst_on_user_accuracy"]

            print(f"{rep:<20} {user_acc:<12.4f} {asst_acc:<12.4f} {cosine_sim:<12.4f} {user_on_asst:<12.4f} {asst_on_user:<12.4f}")

    print("\n" + "=" * 140)


def main():
    parser = argparse.ArgumentParser(description="Plot orthogonal probe results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/workspace-vast/annas/git/research-tools/probes/results/conversation_probes_orthogonal",
        help="Directory containing orthogonal probe results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as results-dir)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading orthogonal probe results from {results_dir}...")
    results = load_orthogonal_probe_results(results_dir)

    if not results:
        print("No orthogonal probe results found!")
        return

    print(f"Found {len(results)} orthogonal probe results")

    # Print summary
    print_summary_table(results)

    # Create plots
    plot_orthogonality_metrics(results, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
