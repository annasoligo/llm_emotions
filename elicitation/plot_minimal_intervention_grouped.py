#!/usr/bin/env python3
"""Grouped bar chart comparing minimal intervention approaches."""

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

# Data: rows = models, cols = metrics
data = np.array([
    [2.70, 1.75, 2.70, 3.75],  # Vanilla
    [1.05, 0.45, 0.80, 0.80],  # Full DPO
    [2.40, 1.90, 2.45, 3.50],  # SFT Last 20
    [1.35, 1.15, 1.60, 3.20],  # DPO R1 L20
    [1.90, 1.35, 1.50, 2.75],  # DPO SV L20
    [1.30, 1.10, 1.40, 2.50],  # SFT SV L20
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
        ax.annotate(f'{height:.1f}' if height >= 1 else f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

# Styling
ax.set_ylabel('Frustration Score', fontsize=13, fontweight='bold')
ax.set_xlabel('Model', fontsize=13, fontweight='bold')
ax.set_title('Minimal Intervention Comparison: Frustration Scores\n(Gemma-3-27B-it, Lower = Better)',
             fontsize=15, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylim(0, 4.5)

# Reference line
ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=1)

# Grid
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)

plt.tight_layout()
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/minimal_intervention_grouped.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/minimal_intervention_grouped.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved to /workspace-vast/annas/Ant_Cluster_Notes/minimal_intervention_grouped.png")
