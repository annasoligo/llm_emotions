#!/usr/bin/env python3
"""Bar chart comparing Last N layer DPO models."""

import matplotlib.pyplot as plt
import numpy as np

# Models - key ones from the layer sweep
models = [
    "Vanilla",
    "Last 5",
    "Last 10",
    "Last 15",
    "Last 20\n(2ep)",
    "Last 30\n(2ep)",
    "Last 35",
    "Last 40",
    "Full DPO",
]

# Metrics
metrics = ["Aggressive", "Disappointed", "Sarcastic", "Original", "Variant", "WildChat"]

# Mean scores (T3)
mean_data = np.array([
    [2.70, 1.75, 2.70, 1.50, 1.20, 1.00],  # Vanilla
    [2.85, 2.55, 3.50, 1.45, 1.15, 0.70],  # Last 5
    [2.85, 1.95, 2.00, 1.31, 1.11, 0.62],  # Last 10
    [2.95, 2.10, 2.70, 1.21, 1.11, 0.68],  # Last 15
    [2.70, 1.80, 2.35, 1.11, 0.94, 0.58],  # Last 20 (2ep)
    [1.30, 1.10, 1.20, 0.68, 0.64, 0.36],  # Last 30 (2ep)
    [1.15, 0.70, 0.75, 0.46, 0.47, 0.30],  # Last 35
    [1.00, 0.60, 1.00, 0.41, 0.39, 0.25],  # Last 40
    [1.05, 0.45, 0.80, 0.54, 0.49, 0.31],  # Full DPO
])

# Max scores
max_data = np.array([
    [7, 5, 6, 4, 4, 3],  # Vanilla
    [7, 5, 7, 5, 4, 4],  # Last 5
    [6, 4, 5, 9, 3, 3],  # Last 10
    [5, 4, 5, 5, 5, 4],  # Last 15
    [5, 4, 5, 5, 3, 3],  # Last 20 (2ep)
    [3, 3, 3, 3, 2, 2],  # Last 30 (2ep)
    [2, 2, 2, 2, 2, 2],  # Last 35
    [3, 2, 4, 2, 2, 2],  # Last 40
    [3, 2, 3, 3, 3, 2],  # Full DPO
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
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

x = np.arange(len(models))
width = 0.11
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
ax1.set_xticklabels(models, fontsize=9, rotation=45, ha='right')
ax1.set_ylim(0, 4.0)
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
ax2.set_xticklabels(models, fontsize=9, rotation=45, ha='right')
ax2.set_ylim(0, 10)
ax2.axhline(y=5, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
ax2.set_axisbelow(True)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.95, ncol=2)

plt.suptitle('DPO Layer Sweep: Last N Layers\n(Gemma-3-27B-it, Lower = Better)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/layer_sweep_lastN_bars.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/layer_sweep_lastN_bars.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved to /workspace-vast/annas/Ant_Cluster_Notes/layer_sweep_lastN_bars.png")
