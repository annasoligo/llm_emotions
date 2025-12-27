#!/usr/bin/env python3
"""Compare the effect of different orthogonality weights."""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_ortho_weight_results(results_dir: Path, representations: list):
    """Load results for different orthogonality weights."""

    results = {}

    for rep in representations:
        results[rep] = {}

        for ortho in [1.0, 10.0, 100.0, 1000.0]:
            results[rep][ortho] = {}

            for layer in [10, 20, 30, 40, 50]:
                pkl_file = results_dir / f"probe_layer{layer}_{rep}_ortho{ortho}.pkl"

                if not pkl_file.exists():
                    continue

                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)

                results[rep][ortho][layer] = {
                    "user_acc": data["final_val_metrics"]["user_accuracy"],
                    "asst_acc": data["final_val_metrics"]["asst_accuracy"],
                    "cosine_sim": data["final_ortho_metrics"]["cross_dots_mean"],
                    "cross_acc": (data["final_val_metrics"]["user_on_asst_accuracy"] +
                                 data["final_val_metrics"]["asst_on_user_accuracy"]) / 2,
                }

    return results


def plot_ortho_weight_comparison(results, output_dir: Path):
    """Plot comparison of orthogonality weights."""

    representations = {
        "raw": "Raw Activations",
        "global_cpca_top10": "Global cPCA (top 10)",
        "regional_cpca_top10": "Regional cPCA (top 10)",
    }

    colors = {
        1.0: "#2ECC71",   # Green
        10.0: "#F39C12",  # Orange
        100.0: "#E74C3C", # Red
        1000.0: "#8E44AD", # Purple
    }

    markers = {
        1.0: "o",
        10.0: "s",
        100.0: "^",
        1000.0: "D",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, (rep_key, rep_name) in enumerate(representations.items()):
        if rep_key not in results or not results[rep_key]:
            continue

        # Plot 1: User accuracy
        ax = axes[0, col]
        for ortho in [1.0, 10.0, 100.0, 1000.0]:
            if ortho not in results[rep_key]:
                continue

            layers = sorted(results[rep_key][ortho].keys())
            user_acc = [results[rep_key][ortho][l]["user_acc"] for l in layers]

            ax.plot(layers, user_acc, marker=markers[ortho], linewidth=2.5, markersize=8,
                   label=f"ortho={ortho}", color=colors[ortho], alpha=0.85)

        ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
        ax.set_ylabel("User Accuracy", fontsize=12, fontweight="bold")
        ax.set_title(f"{rep_name}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_ylim(0.5, 1.05)

        # Plot 2: Cosine similarity (orthogonality)
        ax = axes[1, col]
        for ortho in [1.0, 10.0, 100.0, 1000.0]:
            if ortho not in results[rep_key]:
                continue

            layers = sorted(results[rep_key][ortho].keys())
            cosine_sim = [results[rep_key][ortho][l]["cosine_sim"] for l in layers]

            ax.plot(layers, cosine_sim, marker=markers[ortho], linewidth=2.5, markersize=8,
                   label=f"ortho={ortho}", color=colors[ortho], alpha=0.85)

        ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
        ax.set_ylabel("Mean |Cosine Similarity|", fontsize=12, fontweight="bold")
        ax.set_title(f"Orthogonality (Lower = Better)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_yscale("log")

    plt.tight_layout()
    output_path = output_dir / "ortho_weight_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare orthogonality weights")
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

    print("Loading orthogonality weight comparison data...")
    representations = ["raw", "global_cpca_top10", "regional_cpca_top10"]
    results = load_ortho_weight_results(results_dir, representations)

    print(f"Creating comparison plot...")
    plot_ortho_weight_comparison(results, output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
