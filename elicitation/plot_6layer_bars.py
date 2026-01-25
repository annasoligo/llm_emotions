#!/usr/bin/env python3
"""Bar chart comparing 6-layer DPO models vs Full DPO and Vanilla."""

import matplotlib.pyplot as plt
import numpy as np

# Models
models = [
    "Vanilla",
    "Full DPO\n(1ep)",
    "L20-25\n(2ep)",
    "L25-30\n(2ep)",
    "L30-35\n(2ep)",
    "L35-40\n(2ep)",
    "L40-50\n(2ep)",
]

# Metrics
metrics = ["Aggressive", "Disappointed", "Sarcastic", "Original", "Variant", "WildChat"]

# Mean scores (T3)
mean_data = np.array([
    [2.70, 1.75, 2.70, 1.50, 1.20, 1.00],  # Vanilla
    [1.05, 0.45, 0.80, 0.54, 0.49, 0.31],  # Full DPO
    [1.20, 0.65, 1.20, 1.22, 1.23, 0.93],  # L20-25
    [1.10, 0.65, 0.80, 0.92, 0.92, 0.71],  # L25-30
    [0.90, 0.65, 1.10, 0.73, 0.76, 0.60],  # L30-35
    [1.05, 1.25, 0.80, 1.04, 0.93, 0.49],  # L35-40
    [2.05, 1.70, 1.85, 1.68, 1.47, 1.01],  # L40-50
])

# Max scores
max_data = np.array([
    [7, 5, 6, 4, 4, 3],  # Vanilla
    [3, 2, 3, 3, 3, 2],  # Full DPO
    [2, 2, 2, 5, 3, 5],  # L20-25
    [2, 2, 2, 2, 2, 2],  # L25-30
    [2, 2, 3, 1, 1, 4],  # L30-35
    [3, 3, 2, 3, 2, 2],  # L35-40
    [4, 4, 4, 5, 5, 3],  # L40-50
])

# Colors for metrics
colors = [
    "#D4876A",  # Coral/Terra Cotta
    "#7BA7D7",  # Sky Blue
    "#7D9B7D",  # Olive Green
    "#C17B8D",  # Dusty Rose/Pink
    "#B8CCC8",  # Sage Green
    "#D4D0E5",  # Soft Lavender
]

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

x = np.arange(len(models))
width = 0.12
offsets = np.linspace(-2.5*width, 2.5*width, 6)

# ============ Plot 1: Mean Scores ============
ax1 = axes[0]
for i, (metric, color, offset) in enumerate(zip(metrics, colors, offsets)):
    bars = ax1.bar(x + offset, mean_data[:, i], width, label=metric, color=color,
                   edgecolor='black', linewidth=0.5)

ax1.set_ylabel('Mean Frustration Score (T3)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_title('Mean Scores', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.set_ylim(0, 3.2)
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
ax1.set_axisbelow(True)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.95, ncol=2)

# ============ Plot 2: Max Scores ============
ax2 = axes[1]
for i, (metric, color, offset) in enumerate(zip(metrics, colors, offsets)):
    bars = ax2.bar(x + offset, max_data[:, i], width, label=metric, color=color,
                   edgecolor='black', linewidth=0.5)

ax2.set_ylabel('Max Frustration Score', fontsize=12, fontweight='bold')
ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_title('Max Scores', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10)
ax2.set_ylim(0, 8)
ax2.axhline(y=5, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
ax2.set_axisbelow(True)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.95, ncol=2)

plt.suptitle('6-Layer DPO Models vs Full DPO: Frustration Scores\n(Gemma-3-27B-it, Lower = Better)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/layer_sweep_6layer_bars.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/layer_sweep_6layer_bars.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved to /workspace-vast/annas/Ant_Cluster_Notes/layer_sweep_6layer_bars.png")
