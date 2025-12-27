#!/usr/bin/env python3
"""Plot regional cPCA weight importance for user and assistant probes.

Analyzes which regional features (user, assistant, special1, special2) are most
important for predicting user vs assistant emotions.

Usage:
    python scripts/plot_regional_weight_importance.py --results-dir results/conversation_probes
"""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def analyze_regional_weights(results_dir: Path, layers=[10, 20, 30, 40, 50]):
    """Analyze regional cPCA weight importance across layers."""

    region_names = ["User", "Assistant", "Special1", "Special2"]

    # Store results per layer
    user_importance = {layer: [] for layer in layers}
    asst_importance = {layer: [] for layer in layers}

    for layer in layers:
        pkl_file = results_dir / f"probe_layer{layer}_regional_cpca_top3_l10.1.pkl"

        if not pkl_file.exists():
            print(f"Warning: {pkl_file} not found, skipping layer {layer}")
            continue

        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        # Get probe weights
        user_weights = data["user"]["model"].weight.detach().cpu().numpy()  # [6, 12]
        asst_weights = data["assistant"]["model"].weight.detach().cpu().numpy()  # [6, 12]

        # Compute L1 importance per region (3 features per region)
        for i in range(4):
            start_idx = i * 3
            end_idx = start_idx + 3

            user_l1 = np.abs(user_weights[:, start_idx:end_idx]).sum()
            asst_l1 = np.abs(asst_weights[:, start_idx:end_idx]).sum()

            user_importance[layer].append(user_l1)
            asst_importance[layer].append(asst_l1)

        # Normalize to percentages
        user_total = sum(user_importance[layer])
        asst_total = sum(asst_importance[layer])
        user_importance[layer] = [100 * x / user_total for x in user_importance[layer]]
        asst_importance[layer] = [100 * x / asst_total for x in asst_importance[layer]]

    return user_importance, asst_importance, region_names


def plot_regional_importance_by_layer(
    user_importance, asst_importance, region_names, output_path, layers=[10, 20, 30, 40, 50]
):
    """Plot regional importance as stacked bars per layer."""

    # Color scheme for regions
    COLORS = {
        "User": "#D4876A",        # coral
        "Assistant": "#7BA7D7",   # sky blue
        "Special1": "#B8CCC8",    # sage
        "Special2": "#E5A589",    # light coral
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(len(layers))
    width = 0.6

    # User probe plot
    bottom = np.zeros(len(layers))
    for i, region in enumerate(region_names):
        values = [user_importance[layer][i] for layer in layers]
        ax1.bar(x, values, width, label=region, color=COLORS[region], bottom=bottom)

        # Add percentage labels in the middle of each segment
        for j, layer in enumerate(layers):
            pct = values[j]
            if pct > 5:  # Only show label if segment is large enough
                y_pos = bottom[j] + pct / 2
                ax1.text(j, y_pos, f"{pct:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")

        bottom += values

    ax1.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Weight Importance (%)", fontsize=14, fontweight="bold")
    ax1.set_title("User Emotion Probe\nRegional Feature Importance", fontsize=16, fontweight="bold", pad=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax1.set_ylim(0, 100)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")

    # Assistant probe plot
    bottom = np.zeros(len(layers))
    for i, region in enumerate(region_names):
        values = [asst_importance[layer][i] for layer in layers]
        ax2.bar(x, values, width, label=region, color=COLORS[region], bottom=bottom)

        # Add percentage labels
        for j, layer in enumerate(layers):
            pct = values[j]
            if pct > 5:
                y_pos = bottom[j] + pct / 2
                ax2.text(j, y_pos, f"{pct:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")

        bottom += values

    ax2.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Weight Importance (%)", fontsize=14, fontweight="bold")
    ax2.set_title("Assistant Emotion Probe\nRegional Feature Importance", fontsize=16, fontweight="bold", pad=20)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved stacked bar plot to {output_path}")


def plot_regional_importance_lines(
    user_importance, asst_importance, region_names, output_path, layers=[10, 20, 30, 40, 50]
):
    """Plot regional importance as lines across layers."""

    COLORS = {
        "User": "#D4876A",
        "Assistant": "#7BA7D7",
        "Special1": "#B8CCC8",
        "Special2": "#E5A589",
    }

    MARKERS = {
        "User": "o",
        "Assistant": "s",
        "Special1": "^",
        "Special2": "D",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # User probe
    for i, region in enumerate(region_names):
        values = [user_importance[layer][i] for layer in layers]
        ax1.plot(layers, values, marker=MARKERS[region], linewidth=2.5, markersize=8,
                label=region, color=COLORS[region], alpha=0.85)

    ax1.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Weight Importance (%)", fontsize=14, fontweight="bold")
    ax1.set_title("User Emotion Probe\nRegional Feature Importance by Layer",
                  fontsize=16, fontweight="bold", pad=20)
    ax1.legend(fontsize=11, framealpha=0.95, loc="best")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_ylim(0, 65)
    ax1.set_xticks(layers)

    # Assistant probe
    for i, region in enumerate(region_names):
        values = [asst_importance[layer][i] for layer in layers]
        ax2.plot(layers, values, marker=MARKERS[region], linewidth=2.5, markersize=8,
                label=region, color=COLORS[region], alpha=0.85)

    ax2.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Weight Importance (%)", fontsize=14, fontweight="bold")
    ax2.set_title("Assistant Emotion Probe\nRegional Feature Importance by Layer",
                  fontsize=16, fontweight="bold", pad=20)
    ax2.legend(fontsize=11, framealpha=0.95, loc="best")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_ylim(0, 65)
    ax2.set_xticks(layers)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved line plot to {output_path}")


def plot_disentanglement_comparison(
    user_importance, asst_importance, region_names, output_path, layers=[10, 20, 30, 40, 50]
):
    """Plot comparison of user vs assistant region importance for each probe."""

    COLORS = {
        "coral": "#D4876A",
        "sky_blue": "#7BA7D7",
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(layers))
    width = 0.35

    # Get user region importance (index 0)
    user_probe_user_region = [user_importance[layer][0] for layer in layers]
    user_probe_asst_region = [user_importance[layer][1] for layer in layers]

    # Get assistant region importance (index 1)
    asst_probe_user_region = [asst_importance[layer][0] for layer in layers]
    asst_probe_asst_region = [asst_importance[layer][1] for layer in layers]

    # Plot user probe's preference for user region
    ax.bar(x - width/2, user_probe_user_region, width, label="User Probe → User Region",
           color=COLORS["coral"], alpha=0.8)

    # Plot assistant probe's preference for assistant region
    ax.bar(x + width/2, asst_probe_asst_region, width, label="Assistant Probe → Assistant Region",
           color=COLORS["sky_blue"], alpha=0.8)

    # Add value labels
    for i, layer in enumerate(layers):
        ax.text(i - width/2, user_probe_user_region[i] + 1, f"{user_probe_user_region[i]:.1f}%",
               ha="center", fontsize=9, fontweight="bold")
        ax.text(i + width/2, asst_probe_asst_region[i] + 1, f"{asst_probe_asst_region[i]:.1f}%",
               ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax.set_ylabel("Weight Importance (%)", fontsize=14, fontweight="bold")
    ax.set_title("Disentanglement: Probes Focus on Their Respective Regions",
                fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(fontsize=12, framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, 70)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved disentanglement comparison to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot regional cPCA weight importance"
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

    # Analyze weights
    print("Analyzing regional weight importance...")
    layers = [10, 20, 30, 40, 50]
    user_importance, asst_importance, region_names = analyze_regional_weights(results_dir, layers)

    # Create plots
    print("\nGenerating plots...")

    # 1. Stacked bar chart
    plot_regional_importance_by_layer(
        user_importance, asst_importance, region_names,
        output_dir / "regional_importance_stacked_bars.png",
        layers
    )

    # 2. Line plot
    plot_regional_importance_lines(
        user_importance, asst_importance, region_names,
        output_dir / "regional_importance_lines.png",
        layers
    )

    # 3. Disentanglement comparison
    plot_disentanglement_comparison(
        user_importance, asst_importance, region_names,
        output_dir / "regional_disentanglement_comparison.png",
        layers
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
