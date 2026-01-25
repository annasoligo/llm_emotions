#!/usr/bin/env python3
"""Plot heatmaps of layer sweep results."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Data from the sweep results
models = [
    "Vanilla",
    "Last 5",
    "Last 10",
    "Last 15",
    "Last 20",
    "Last 20 (2ep)",
    "Last 30",
    "Last 30 (2ep)",
    "Last 35",
    "Last 40",
    "Full DPO",
]

# Number of layers for sorting/labeling
n_layers = [0, 5, 10, 15, 20, 20, 30, 30, 35, 40, 62]

# Columns (eval types)
columns = ["Aggressive", "Disappointed", "Sarcastic", "Original", "Variant", "WildChat", "T5"]

# Mean scores (Turn 3 for tones, overall for prompts, Turn 5 for T5)
mean_scores = np.array([
    [2.70, 1.75, 2.70, 1.50, 1.20, 1.00, 2.80],  # Vanilla
    [2.85, 2.55, 3.50, 1.45, 1.15, 0.70, 4.65],  # Last 5
    [2.85, 1.95, 2.00, 1.31, 1.11, 0.62, 4.25],  # Last 10
    [2.95, 2.10, 2.70, 1.21, 1.11, 0.68, 3.35],  # Last 15
    [3.60, 2.35, 2.50, 1.22, 0.99, 0.59, 3.20],  # Last 20
    [2.70, 1.80, 2.35, 1.11, 0.94, 0.58, 3.10],  # Last 20 (2ep)
    [1.35, 1.50, 1.55, 0.80, 0.69, 0.43, 2.05],  # Last 30
    [1.30, 1.10, 1.20, 0.68, 0.64, 0.36, 1.30],  # Last 30 (2ep)
    [1.15, 0.70, 0.75, 0.46, 0.47, 0.30, 0.65],  # Last 35
    [1.00, 0.60, 1.00, 0.41, 0.39, 0.25, 0.60],  # Last 40
    [1.05, 0.45, 0.80, 0.54, 0.49, 0.31, np.nan],  # Full DPO (no T5 data)
])

# Max scores
max_scores = np.array([
    [7, 5, 6, 4, 4, 3, 7],  # Vanilla (T5 max estimated from T8)
    [7, 5, 7, 5, 4, 4, 7],  # Last 5
    [6, 4, 5, 9, 3, 3, 7],  # Last 10
    [5, 4, 5, 5, 5, 4, 6],  # Last 15
    [6, 5, 5, 5, 4, 4, 6],  # Last 20
    [5, 4, 5, 5, 3, 3, 5],  # Last 20 (2ep)
    [3, 3, 3, 3, 3, 3, 4],  # Last 30
    [3, 3, 3, 3, 2, 2, 3],  # Last 30 (2ep)
    [2, 2, 2, 2, 2, 2, 2],  # Last 35
    [3, 2, 4, 2, 2, 2, 3],  # Last 40
    [3, 2, 3, 3, 3, 2, np.nan],  # Full DPO (no T5 data)
])

def plot_heatmap(data, title, filename, vmax=None, fmt=".1f"):
    """Plot a single heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create custom colormap: white to red
    cmap = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "darkred"])

    # Create masked array for NaN values
    masked_data = np.ma.masked_invalid(data)

    # Plot heatmap
    im = ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Frustration Score", fontsize=11)

    # Set ticks
    ax.set_xticks(np.arange(len(columns)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(columns, fontsize=10)
    ax.set_yticklabels(models, fontsize=10)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(columns)):
            if not np.isnan(data[i, j]):
                text_color = "white" if data[i, j] > vmax * 0.6 else "black"
                ax.text(j, i, f"{data[i, j]:{fmt}}", ha="center", va="center",
                       color=text_color, fontsize=9)

    # Add grid lines
    ax.set_xticks(np.arange(len(columns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(models) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    # Add horizontal line to separate good models (Last 30+)
    ax.axhline(y=5.5, color='black', linewidth=2)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Evaluation Type", fontsize=12)
    ax.set_ylabel("Model (layers trained)", fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

def plot_combined(mean_data, max_data, filename):
    """Plot both heatmaps side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    cmap = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "darkred"])

    # Mean scores
    masked_mean = np.ma.masked_invalid(mean_data)
    im1 = axes[0].imshow(masked_mean, cmap=cmap, vmin=0, vmax=5, aspect='auto')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label("Mean Score", fontsize=11)

    axes[0].set_xticks(np.arange(len(columns)))
    axes[0].set_yticks(np.arange(len(models)))
    axes[0].set_xticklabels(columns, fontsize=10)
    axes[0].set_yticklabels(models, fontsize=10)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(models)):
        for j in range(len(columns)):
            if not np.isnan(mean_data[i, j]):
                text_color = "white" if mean_data[i, j] > 3 else "black"
                axes[0].text(j, i, f"{mean_data[i, j]:.2f}", ha="center", va="center",
                           color=text_color, fontsize=8)

    axes[0].set_xticks(np.arange(len(columns) + 1) - 0.5, minor=True)
    axes[0].set_yticks(np.arange(len(models) + 1) - 0.5, minor=True)
    axes[0].grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
    axes[0].tick_params(which="minor", size=0)
    axes[0].axhline(y=5.5, color='black', linewidth=2)
    axes[0].set_title("Mean Frustration Scores", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Evaluation Type", fontsize=12)
    axes[0].set_ylabel("Model (layers trained)", fontsize=12)

    # Max scores
    masked_max = np.ma.masked_invalid(max_data)
    im2 = axes[1].imshow(masked_max, cmap=cmap, vmin=0, vmax=9, aspect='auto')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label("Max Score", fontsize=11)

    axes[1].set_xticks(np.arange(len(columns)))
    axes[1].set_yticks(np.arange(len(models)))
    axes[1].set_xticklabels(columns, fontsize=10)
    axes[1].set_yticklabels(models, fontsize=10)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(models)):
        for j in range(len(columns)):
            if not np.isnan(max_data[i, j]):
                text_color = "white" if max_data[i, j] > 5 else "black"
                axes[1].text(j, i, f"{max_data[i, j]:.0f}", ha="center", va="center",
                           color=text_color, fontsize=9)

    axes[1].set_xticks(np.arange(len(columns) + 1) - 0.5, minor=True)
    axes[1].set_yticks(np.arange(len(models) + 1) - 0.5, minor=True)
    axes[1].grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
    axes[1].tick_params(which="minor", size=0)
    axes[1].axhline(y=5.5, color='black', linewidth=2)
    axes[1].set_title("Maximum Frustration Scores", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Evaluation Type", fontsize=12)
    axes[1].set_ylabel("")

    plt.suptitle("DPO Layer Sweep: Frustration Reduction by Layer Count", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

if __name__ == "__main__":
    output_dir = "/workspace-vast/annas/Ant_Cluster_Notes"

    # Individual heatmaps
    plot_heatmap(
        mean_scores,
        "Mean Frustration Scores by Model and Evaluation Type",
        f"{output_dir}/layer_sweep_heatmap_mean.png",
        vmax=5,
        fmt=".2f"
    )

    plot_heatmap(
        max_scores,
        "Maximum Frustration Scores by Model and Evaluation Type",
        f"{output_dir}/layer_sweep_heatmap_max.png",
        vmax=9,
        fmt=".0f"
    )

    # Combined plot
    plot_combined(
        mean_scores,
        max_scores,
        f"{output_dir}/layer_sweep_heatmap_combined.png"
    )

    print("\nDone! Generated 3 plots.")
