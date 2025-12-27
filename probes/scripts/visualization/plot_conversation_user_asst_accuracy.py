#!/usr/bin/env python3
"""Plot user vs assistant probe accuracy across layers for conversation data.

Usage:
    python scripts/plot_conversation_user_asst_accuracy.py --results-dir results/conversation_probes
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_dual_probe_results(results_dir: Path) -> Dict[Tuple[int, str], dict]:
    """Load all dual probe results from a directory.

    Args:
        results_dir: Directory containing probe .pkl files

    Returns:
        Dict mapping (layer, representation) to results
    """
    results = {}

    # Match probe files with target="both" (no target suffix) or explicit _both suffix
    for pkl_file in results_dir.glob("probe_layer*_l1*.pkl"):
        # Skip single-target probes (_user or _assistant suffix)
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


def plot_user_vs_assistant_by_layer(
    results: Dict[Tuple[int, str], dict],
    output_path: Path,
    representation: str = "raw",
    title: str = "User vs Assistant Probe Accuracy by Layer",
):
    """Plot user vs assistant test accuracy across layers for a specific representation.

    Args:
        results: Dict mapping (layer, representation) to results
        output_path: Path to save plot
        representation: Which representation to plot (e.g., "raw", "global_top10")
        title: Plot title
    """
    # Filter results for this representation
    layer_results = {}
    for (layer, rep), data in results.items():
        if rep == representation:
            layer_results[layer] = data

    if not layer_results:
        print(f"No results found for representation: {representation}")
        return None

    # Sort by layer
    layers = sorted(layer_results.keys())
    user_test_accs = [layer_results[l]["user"]["test_accuracy"] for l in layers]
    asst_test_accs = [layer_results[l]["assistant"]["test_accuracy"] for l in layers]
    user_best_accs = [layer_results[l]["user"]["best_test_accuracy"] for l in layers]
    asst_best_accs = [layer_results[l]["assistant"]["best_test_accuracy"] for l in layers]

    # believe-it-or-not color scheme
    COLORS = {
        "coral": "#D4876A",
        "sky_blue": "#7BA7D7",
        "sage": "#B8CCC8",
        "coral_light": "#E5A589",
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot user probes
    ax.plot(layers, user_test_accs, "o-", linewidth=2.5, markersize=8,
            label="User (final)", color=COLORS["coral"])
    ax.plot(layers, user_best_accs, "s--", linewidth=2, markersize=6,
            label="User (best)", color=COLORS["coral_light"], alpha=0.7)

    # Plot assistant probes
    ax.plot(layers, asst_test_accs, "o-", linewidth=2.5, markersize=8,
            label="Assistant (final)", color=COLORS["sky_blue"])
    ax.plot(layers, asst_best_accs, "s--", linewidth=2, markersize=6,
            label="Assistant (best)", color=COLORS["sage"], alpha=0.7)

    # Formatting
    ax.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.legend(fontsize=11, framealpha=0.95, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(0, 1.05)

    # Add value annotations for final test accuracy
    for layer, user_acc, asst_acc in zip(layers, user_test_accs, asst_test_accs):
        # User annotations (above)
        ax.annotate(
            f"{user_acc:.3f}",
            xy=(layer, user_acc),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            alpha=0.7,
            color=COLORS["coral"],
        )
        # Assistant annotations (below)
        ax.annotate(
            f"{asst_acc:.3f}",
            xy=(layer, asst_acc),
            xytext=(0, -15),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            alpha=0.7,
            color=COLORS["sky_blue"],
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    return fig, ax


def plot_all_representations_combined(
    results: Dict[Tuple[int, str], dict],
    output_path: Path,
    title: str = "User vs Assistant Probe Accuracy (All Representations)",
):
    """Plot user vs assistant for all representations in subplots.

    Args:
        results: Dict mapping (layer, representation) to results
        output_path: Path to save plot
        title: Plot title
    """
    # Group by representation
    rep_to_results = {}
    for (layer, rep), data in results.items():
        if rep not in rep_to_results:
            rep_to_results[rep] = {}
        rep_to_results[rep][layer] = data

    representations = sorted(rep_to_results.keys())
    if not representations:
        print("No results found!")
        return None

    # Create subplots
    n_reps = len(representations)
    n_cols = min(3, n_reps)
    n_rows = (n_reps + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_reps == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes

    COLORS = {
        "coral": "#D4876A",
        "sky_blue": "#7BA7D7",
    }

    for idx, rep in enumerate(representations):
        ax = axes[idx]
        layer_results = rep_to_results[rep]
        layers = sorted(layer_results.keys())

        user_accs = [layer_results[l]["user"]["test_accuracy"] for l in layers]
        asst_accs = [layer_results[l]["assistant"]["test_accuracy"] for l in layers]

        # Plot
        ax.plot(layers, user_accs, "o-", linewidth=2.5, markersize=7,
                label="User", color=COLORS["coral"])
        ax.plot(layers, asst_accs, "o-", linewidth=2.5, markersize=7,
                label="Assistant", color=COLORS["sky_blue"])

        # Formatting
        ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
        ax.set_ylabel("Test Accuracy", fontsize=12, fontweight="bold")
        ax.set_title(rep, fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(0, 1.05)

    # Hide unused subplots
    for idx in range(len(representations), len(axes)):
        axes[idx].axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved combined plot to {output_path}")

    return fig, axes


def print_summary_table(results: Dict[Tuple[int, str], dict]):
    """Print summary table of user vs assistant results."""
    # Group by layer
    layer_to_reps = {}
    for (layer, rep), data in results.items():
        if layer not in layer_to_reps:
            layer_to_reps[layer] = {}
        layer_to_reps[layer][rep] = data

    print("\n" + "=" * 120)
    print("USER VS ASSISTANT PROBE ACCURACY SUMMARY")
    print("=" * 120)

    for layer in sorted(layer_to_reps.keys()):
        print(f"\n{'=' * 120}")
        print(f"LAYER {layer}")
        print(f"{'=' * 120}")
        print(
            f"{'Representation':<20} {'User Test':<12} {'Asst Test':<12} {'User Best':<12} {'Asst Best':<12} {'Cross User→Asst':<18} {'Cross Asst→User':<18}"
        )
        print("-" * 120)

        for rep in sorted(layer_to_reps[layer].keys()):
            data = layer_to_reps[layer][rep]
            user_test = data["user"]["test_accuracy"]
            asst_test = data["assistant"]["test_accuracy"]
            user_best = data["user"]["best_test_accuracy"]
            asst_best = data["assistant"]["best_test_accuracy"]
            user_on_asst = data.get("user_on_asst_accuracy", 0.0)
            asst_on_user = data.get("asst_on_user_accuracy", 0.0)

            print(
                f"{rep:<20} {user_test:<12.4f} {asst_test:<12.4f} {user_best:<12.4f} {asst_best:<12.4f} {user_on_asst:<18.4f} {asst_on_user:<18.4f}"
            )

    print("=" * 120)

    # Overall statistics
    print("\n" + "=" * 120)
    print("OVERALL STATISTICS")
    print("=" * 120)

    all_user_accs = [data["user"]["test_accuracy"] for data in results.values()]
    all_asst_accs = [data["assistant"]["test_accuracy"] for data in results.values()]

    print(f"User probes:      Mean={np.mean(all_user_accs):.4f}, Std={np.std(all_user_accs):.4f}")
    print(f"Assistant probes: Mean={np.mean(all_asst_accs):.4f}, Std={np.std(all_asst_accs):.4f}")

    # Disentanglement
    cross_accs = [
        (data.get("user_on_asst_accuracy", 0), data.get("asst_on_user_accuracy", 0))
        for data in results.values()
    ]
    if any(u > 0 or a > 0 for u, a in cross_accs):
        user_cross = [u for u, _ in cross_accs if u > 0]
        asst_cross = [a for _, a in cross_accs if a > 0]
        if user_cross:
            print(f"User→Asst cross:  Mean={np.mean(user_cross):.4f}, Std={np.std(user_cross):.4f}")
        if asst_cross:
            print(f"Asst→User cross:  Mean={np.mean(asst_cross):.4f}, Std={np.std(asst_cross):.4f}")

    print("=" * 120)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Plot user vs assistant probe accuracy by layer"
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
    parser.add_argument(
        "--representation",
        type=str,
        default=None,
        help="Plot specific representation (e.g., 'raw', 'global_top10'). If None, plots all.",
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

    # Print summary
    print_summary_table(results)

    # Plot
    if args.representation:
        # Single representation
        output_path = output_dir / f"user_vs_asst_accuracy_{args.representation}.png"
        plot_user_vs_assistant_by_layer(
            results,
            output_path,
            representation=args.representation,
            title=f"User vs Assistant Probe Accuracy - {args.representation}",
        )
    else:
        # All representations in subplots
        output_path = output_dir / "user_vs_asst_accuracy_all.png"
        plot_all_representations_combined(
            results, output_path, title="User vs Assistant Probe Accuracy (All Representations)"
        )

        # Also create individual plots for each representation
        representations = set(rep for _, rep in results.keys())
        for rep in sorted(representations):
            output_path = output_dir / f"user_vs_asst_accuracy_{rep}.png"
            plot_user_vs_assistant_by_layer(
                results,
                output_path,
                representation=rep,
                title=f"User vs Assistant Probe Accuracy - {rep}",
            )

    print("\nDone!")


if __name__ == "__main__":
    main()
