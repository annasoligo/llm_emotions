#!/usr/bin/env python3
"""Plot overlayed user and assistant probe accuracies across layers.

Creates two plots:
1. All user probes overlayed (different representations as different lines)
2. All assistant probes overlayed (different representations as different lines)

Usage:
    python scripts/plot_conversation_overlayed_accuracy.py --results-dir results/conversation_probes
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_dual_probe_results(results_dir: Path) -> Dict[Tuple[int, str], dict]:
    """Load all dual probe results from a directory."""
    results = {}

    for pkl_file in results_dir.glob("probe_layer*_l1*.pkl"):
        # Skip single-target probes
        if pkl_file.stem.endswith("_user") or pkl_file.stem.endswith("_assistant"):
            continue

        try:
            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            # Only process dual probe results
            if "user" not in data or "assistant" not in data:
                continue

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


def plot_overlayed_accuracy(
    results: Dict[Tuple[int, str], dict],
    output_path: Path,
    target: str = "user",
    title: str = "User Probe Accuracy Across Representations",
):
    """Plot accuracy across layers with all representations overlayed.

    Args:
        results: Dict mapping (layer, representation) to results
        output_path: Path to save plot
        target: "user" or "assistant"
        title: Plot title
    """
    # Group by representation
    rep_to_results = {}
    for (layer, rep), data in results.items():
        if rep not in rep_to_results:
            rep_to_results[rep] = {}
        rep_to_results[rep][layer] = data

    if not rep_to_results:
        print(f"No results found for {target}!")
        return None

    # Color scheme - expanded for more representations
    COLORS = {
        "raw": "#D4876A",           # coral
        "global_top3": "#9BA7D7",   # light blue
        "global_top5": "#7BA7D7",   # sky blue
        "global_top10": "#5B87C7",  # darker blue
        "regional_top3": "#D8CCC8", # light sage
        "regional_top5": "#B8CCC8", # sage
        "regional_top10": "#98ACA8", # darker sage
    }

    MARKERS = {
        "raw": "o",
        "global_top3": "s",
        "global_top5": "^",
        "global_top10": "D",
        "regional_top3": "v",
        "regional_top5": "<",
        "regional_top10": ">",
    }

    LABELS = {
        "raw": "Raw Mean",
        "global_top3": "Global cPCA (top 3)",
        "global_top5": "Global cPCA (top 5)",
        "global_top10": "Global cPCA (top 10)",
        "regional_top3": "Regional cPCA (top 3 × 4)",
        "regional_top5": "Regional cPCA (top 5 × 4)",
        "regional_top10": "Regional cPCA (top 10 × 4)",
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot each representation
    for rep in sorted(rep_to_results.keys()):
        layer_results = rep_to_results[rep]
        layers = sorted(layer_results.keys())

        # Get accuracies for this target
        accuracies = [layer_results[l][target]["test_accuracy"] for l in layers]

        color = COLORS.get(rep, "#666666")
        marker = MARKERS.get(rep, "o")
        label = LABELS.get(rep, rep)

        ax.plot(
            layers,
            accuracies,
            marker=marker,
            linewidth=2.5,
            markersize=8,
            label=label,
            color=color,
            alpha=0.85,
        )

    # Formatting
    ax.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax.set_ylabel("Test Accuracy", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=11, framealpha=0.95, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.05)

    # Add layer ticks
    all_layers = set()
    for layer_results in rep_to_results.values():
        all_layers.update(layer_results.keys())
    ax.set_xticks(sorted(all_layers))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Plot overlayed probe accuracy by representation"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="/workspace-vast/annas/git/research-tools/probes/results/conversation_probes",
        help="Directory containing probe results",
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
    print(f"Loading results from {results_dir}...")
    results = load_dual_probe_results(results_dir)

    if not results:
        print("No dual probe results found!")
        return

    print(f"Found {len(results)} dual probe results")

    # Get unique representations
    representations = sorted(set(rep for _, rep in results.keys()))
    print(f"Representations: {representations}")

    # Plot user probes overlayed
    user_output = output_dir / "user_accuracy_overlayed_all_representations.png"
    plot_overlayed_accuracy(
        results,
        user_output,
        target="user",
        title="User Emotion Probe Accuracy (All Representations)",
    )

    # Plot assistant probes overlayed
    asst_output = output_dir / "asst_accuracy_overlayed_all_representations.png"
    plot_overlayed_accuracy(
        results,
        asst_output,
        target="assistant",
        title="Assistant Emotion Probe Accuracy (All Representations)",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
