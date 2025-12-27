#!/usr/bin/env python3
"""Analyze regional feature importance for orthogonal conversation probes.

Shows which regional features (user/asst/special1/special2) are most important
for predicting user vs assistant emotions in orthogonal probes.
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def analyze_regional_weights(
    results_dir: Path,
    layers: List[int] = [10, 20, 30, 40, 50],
    n_components_list: List[int] = [3, 5, 10, 20],
    ortho_weight: float = 1.0,
):
    """Analyze regional weight importance across layers for orthogonal probes."""

    region_names = ["User", "Assistant", "Special1", "Special2"]
    n_regions = 4

    # Storage for results
    results = {}

    for n_comp in n_components_list:
        results[n_comp] = {
            "user_importance": [],
            "asst_importance": [],
            "layers": [],
        }

        for layer in layers:
            pkl_file = results_dir / f"probe_layer{layer}_regional_cpca_top{n_comp}_ortho{ortho_weight}.pkl"

            if not pkl_file.exists():
                print(f"Warning: {pkl_file} not found, skipping")
                continue

            with open(pkl_file, "rb") as f:
                data = pickle.load(f)

            # Get probe weights
            user_probes = data["final_user_probes"]  # [n_emotions, n_features]
            asst_probes = data["final_asst_probes"]  # [n_emotions, n_features]

            # Calculate L1 importance per region
            # Each region has n_comp features
            user_importance = []
            asst_importance = []

            for i in range(n_regions):
                start_idx = i * n_comp
                end_idx = start_idx + n_comp

                # Sum absolute weights for this region across all emotions
                user_l1 = np.abs(user_probes[:, start_idx:end_idx]).sum()
                asst_l1 = np.abs(asst_probes[:, start_idx:end_idx]).sum()

                user_importance.append(user_l1)
                asst_importance.append(asst_l1)

            # Normalize to percentages
            user_total = sum(user_importance)
            asst_total = sum(asst_importance)

            user_pct = [100 * x / user_total for x in user_importance]
            asst_pct = [100 * x / asst_total for x in asst_importance]

            results[n_comp]["user_importance"].append(user_pct)
            results[n_comp]["asst_importance"].append(asst_pct)
            results[n_comp]["layers"].append(layer)

    return results, region_names


def plot_regional_importance(results, region_names, output_dir: Path):
    """Create stacked bar chart showing regional importance per layer."""

    # We have top 3, 5, 10, 20
    # Focus on top 5 as requested (using top 5 if available, else closest)
    n_comp = 5 if 5 in results else list(results.keys())[0]

    if n_comp not in results or not results[n_comp]["layers"]:
        print(f"No results found for regional top {n_comp}")
        return

    data = results[n_comp]
    layers = data["layers"]
    n_layers = len(layers)

    # Convert to arrays
    user_importance = np.array(data["user_importance"])  # [n_layers, n_regions]
    asst_importance = np.array(data["asst_importance"])  # [n_layers, n_regions]

    # Create stacked bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(n_layers)
    width = 0.6

    # Color scheme matching regular probes
    COLORS = {
        "User": "#D4876A",        # coral
        "Assistant": "#7BA7D7",   # sky blue
        "Special1": "#B8CCC8",    # sage
        "Special2": "#E5A589",    # light coral
    }

    # User probe importance - stacked bars per layer
    bottom = np.zeros(n_layers)
    for i, region in enumerate(region_names):
        values = user_importance[:, i]
        ax1.bar(x, values, width, label=region, color=COLORS[region],
                bottom=bottom, edgecolor="white", linewidth=0.5)

        # Add percentage labels in the middle of each segment
        for j in range(n_layers):
            pct = values[j]
            if pct > 5:  # Only show label if segment is large enough
                y_pos = bottom[j] + pct / 2
                ax1.text(j, y_pos, f"{pct:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")

        bottom += values

    ax1.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Importance (%)", fontsize=14, fontweight="bold")
    ax1.set_title(f"User Probe: Regional Feature Importance by Layer\n(Regional cPCA top {n_comp}, ortho=1.0)",
                  fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in layers], fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax1.grid(True, alpha=0.3, linestyle="--", axis="y")

    # Assistant probe importance - stacked bars per layer
    bottom = np.zeros(n_layers)
    for i, region in enumerate(region_names):
        values = asst_importance[:, i]
        ax2.bar(x, values, width, label=region, color=COLORS[region],
                bottom=bottom, edgecolor="white", linewidth=0.5)

        # Add percentage labels in the middle of each segment
        for j in range(n_layers):
            pct = values[j]
            if pct > 5:  # Only show label if segment is large enough
                y_pos = bottom[j] + pct / 2
                ax2.text(j, y_pos, f"{pct:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")

        bottom += values

    ax2.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Importance (%)", fontsize=14, fontweight="bold")
    ax2.set_title(f"Assistant Probe: Regional Feature Importance by Layer\n(Regional cPCA top {n_comp}, ortho=1.0)",
                  fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(l) for l in layers], fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    output_path = output_dir / f"orthogonal_regional_importance_stacked_top{n_comp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved regional importance plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze regional importance for orthogonal probes")
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
    parser.add_argument(
        "--ortho-weight",
        type=float,
        default=1.0,
        help="Orthogonality weight to analyze",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing regional importance for orthogonal probes (ortho={args.ortho_weight})...")
    results, region_names = analyze_regional_weights(results_dir, ortho_weight=args.ortho_weight)

    if not any(results.values()):
        print("No results found!")
        return

    print(f"\nFound results for n_components: {list(results.keys())}")
    plot_regional_importance(results, region_names, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
