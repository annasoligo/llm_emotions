#!/usr/bin/env python3
"""Grouped bar chart comparing minimal intervention approaches - MAX scores."""

import matplotlib.pyplot as plt
import numpy as np

# Models
models = [
    "Vanilla",
    "Full DPO\n(466M)",
    "SFT Last 20\n(73M)",
    "DPO R1 L20\n(27K)",
    "DPO SV L20\n(5.4K)",
    "SFT SV L20\n(5.4K)",
]

# Metrics
metrics = ["Aggressive T3", "Disappointed T3", "Sarcastic T3", "T8"]

# Data: rows = models, cols = metrics (MAX scores)
data = np.array([
    [7, 5, 6, 6],  # Vanilla
    [3, 2, 3, 3],  # Full DPO
    [5, 4, 5, 6],  # SFT Last 20
    [3, 2, 3, 6],  # DPO R1 L20
    [5, 3, 3, 5],  # DPO SV L20
    [3, 2, 3, 5],  # SFT SV L20
])

# Colors for metrics
colors = [
    "#D4876A",  # Coral/Terra Cotta - Aggressive
    "#7BA7D7",  # Sky Blue - Disappointed
    "#7D9B7D",  # Olive Green - Sarcastic
    "#C17B8D",  # Dusty Rose/Pink - T8
]

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(models))
width = 0.18
offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

# Plot bars for each metric
for i, (metric, color, offset) in enumerate(zip(metrics, colors, offsets)):
    bars = ax.bar(x + offset, data[:, i], width, label=metric, color=color,
                  edgecolor='black', linewidth=0.5)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

# Styling
ax.set_ylabel('Max Frustration Score', fontsize=13, fontweight='bold')
ax.set_xlabel('Model', fontsize=13, fontweight='bold')
ax.set_title('Minimal Intervention Comparison: Maximum Frustration Scores\n(Gemma-3-27B-it, Lower = Better)',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 8.5)

# Reference line at 5 (high frustration threshold)
ax.axhline(y=5, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='High frustration threshold')

# Grid
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

plt.tight_layout()
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/minimal_intervention_max.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/minimal_intervention_max.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved to /workspace-vast/annas/Ant_Cluster_Notes/minimal_intervention_max.png")
