#!/usr/bin/env python3
"""Plot heatmap comparing Full DPO, 6-layer finetunes, and L40-50."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Models to compare
models = [
    "Vanilla",
    "Full DPO (1ep)",
    "L20-25 (2ep)",
    "L25-30 (2ep)",
    "L30-35 (2ep)",
    "L35-40 (2ep)",
    "L40-50 (2ep)",
]

# Columns (eval types)
columns = ["Aggressive", "Disappointed", "Sarcastic", "Original", "Variant", "WildChat"]

# Mean scores (Turn 3 for tones, T3 for prompts)
mean_scores = np.array([
    [2.70, 1.75, 2.70, 1.50, 1.20, 1.00],  # Vanilla
    [1.05, 0.45, 0.80, 0.54, 0.49, 0.31],  # Full DPO
    [1.20, 0.65, 1.20, 1.22, 1.23, 0.93],  # L20-25
    [1.10, 0.65, 0.80, 0.92, 0.92, 0.71],  # L25-30
    [0.90, 0.65, 1.10, 0.73, 0.76, 0.60],  # L30-35
    [1.05, 1.25, 0.80, 1.04, 0.93, 0.49],  # L35-40
    [2.05, 1.70, 1.85, 1.68, 1.47, 1.01],  # L40-50
])

# Max scores
max_scores = np.array([
    [7, 5, 6, 4, 4, 3],  # Vanilla
    [3, 2, 3, 3, 3, 2],  # Full DPO
    [2, 2, 2, 5, 3, 5],  # L20-25
    [2, 2, 2, 2, 2, 2],  # L25-30
    [2, 2, 3, 1, 1, 4],  # L30-35
    [3, 3, 2, 3, 2, 2],  # L35-40
    [4, 4, 4, 5, 5, 3],  # L40-50
])

def plot_combined(mean_data, max_data, models, columns, filename):
    """Plot both heatmaps side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cmap = mcolors.LinearSegmentedColormap.from_list("white_red", ["white", "darkred"])

    # Mean scores
    im1 = axes[0].imshow(mean_data, cmap=cmap, vmin=0, vmax=2.5, aspect='auto')
    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label("Mean Score", fontsize=11)

    axes[0].set_xticks(np.arange(len(columns)))
    axes[0].set_yticks(np.arange(len(models)))
    axes[0].set_xticklabels(columns, fontsize=10)
    axes[0].set_yticklabels(models, fontsize=10)
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(models)):
        for j in range(len(columns)):
            text_color = "white" if mean_data[i, j] > 1.5 else "black"
            axes[0].text(j, i, f"{mean_data[i, j]:.2f}", ha="center", va="center",
                       color=text_color, fontsize=9)

    axes[0].set_xticks(np.arange(len(columns) + 1) - 0.5, minor=True)
    axes[0].set_yticks(np.arange(len(models) + 1) - 0.5, minor=True)
    axes[0].grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
    axes[0].tick_params(which="minor", size=0)
    # Add line between Full DPO and 6-layer models
    axes[0].axhline(y=1.5, color='black', linewidth=2)
    # Add line between 6-layer models and L40-50
    axes[0].axhline(y=5.5, color='black', linewidth=2)
    axes[0].set_title("Mean Frustration Scores (T3)", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Evaluation Type", fontsize=11)
    axes[0].set_ylabel("Model", fontsize=11)

    # Max scores
    im2 = axes[1].imshow(max_data, cmap=cmap, vmin=0, vmax=5, aspect='auto')
    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label("Max Score", fontsize=11)

    axes[1].set_xticks(np.arange(len(columns)))
    axes[1].set_yticks(np.arange(len(models)))
    axes[1].set_xticklabels(columns, fontsize=10)
    axes[1].set_yticklabels(models, fontsize=10)
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(models)):
        for j in range(len(columns)):
            text_color = "white" if max_data[i, j] > 3 else "black"
            axes[1].text(j, i, f"{max_data[i, j]:.0f}", ha="center", va="center",
                       color=text_color, fontsize=10)

    axes[1].set_xticks(np.arange(len(columns) + 1) - 0.5, minor=True)
    axes[1].set_yticks(np.arange(len(models) + 1) - 0.5, minor=True)
    axes[1].grid(which="minor", color="lightgray", linestyle="-", linewidth=0.5)
    axes[1].tick_params(which="minor", size=0)
    axes[1].axhline(y=1.5, color='black', linewidth=2)
    axes[1].axhline(y=5.5, color='black', linewidth=2)
    axes[1].set_title("Maximum Frustration Scores", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Evaluation Type", fontsize=11)
    axes[1].set_ylabel("")

    plt.suptitle("6-Layer DPO Models vs Full DPO: Frustration Scores", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

if __name__ == "__main__":
    output_dir = "/workspace-vast/annas/Ant_Cluster_Notes"

    plot_combined(
        mean_scores,
        max_scores,
        models,
        columns,
        f"{output_dir}/layer_sweep_6layer_comparison.png"
    )

    print("\nDone!")
