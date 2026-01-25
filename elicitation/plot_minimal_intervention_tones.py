#!/usr/bin/env python3
"""Grouped bar chart comparing minimal intervention approaches - Tones only."""

import matplotlib.pyplot as plt
import numpy as np

# Models (removed SFT Last 20)
models = [
    "Vanilla",
    "Full DPO\n(466M)",
    "DPO R1 L20\n(27K)",
    "DPO SV L20\n(5.4K)",
    "SFT SV L20\n(5.4K)",
]

# Metrics - just tones
metrics = ["Aggressive", "Disappointed", "Sarcastic"]

# ============ MEAN SCORES ============
mean_data = np.array([
    [2.70, 1.75, 2.70],  # Vanilla
    [1.05, 0.45, 0.80],  # Full DPO
    [1.35, 1.15, 1.60],  # DPO R1 L20
    [1.90, 1.35, 1.50],  # DPO SV L20
    [1.30, 1.10, 1.40],  # SFT SV L20
])

# ============ MAX SCORES ============
max_data = np.array([
    [7, 5, 6],  # Vanilla
    [3, 2, 3],  # Full DPO
    [3, 2, 3],  # DPO R1 L20
    [5, 3, 3],  # DPO SV L20
    [3, 2, 3],  # SFT SV L20
])

# Colors for metrics
colors = [
    "#D4876A",  # Coral/Terra Cotta - Aggressive
    "#7BA7D7",  # Sky Blue - Disappointed
    "#7D9B7D",  # Olive Green - Sarcastic
]

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(models))
width = 0.22
offsets = [-width, 0, width]

# ============ Plot 1: Mean Scores ============
ax1 = axes[0]
for i, (metric, color, offset) in enumerate(zip(metrics, colors, offsets)):
    bars = ax1.bar(x + offset, mean_data[:, i], width, label=metric, color=color,
                   edgecolor='black', linewidth=0.5)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}' if height >= 1 else f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_ylabel('Mean Frustration Score (T3)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_title('Mean Scores', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.set_ylim(0, 3.5)
ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.4, linewidth=1)
ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
ax1.set_axisbelow(True)
ax1.legend(loc='upper right', fontsize=10, framealpha=0.95)

# ============ Plot 2: Max Scores ============
ax2 = axes[1]
for i, (metric, color, offset) in enumerate(zip(metrics, colors, offsets)):
    bars = ax2.bar(x + offset, max_data[:, i], width, label=metric, color=color,
                   edgecolor='black', linewidth=0.5)
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 2), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_ylabel('Max Frustration Score (T3)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_title('Max Scores', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=10)
ax2.set_ylim(0, 8)
ax2.axhline(y=5, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
ax2.set_axisbelow(True)
ax2.legend(loc='upper right', fontsize=10, framealpha=0.95)

plt.suptitle('Minimal Intervention Comparison: Turn 3 Frustration by User Tone\n(Gemma-3-27B-it, Lower = Better)',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/minimal_intervention_tones.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/minimal_intervention_tones.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved to /workspace-vast/annas/Ant_Cluster_Notes/minimal_intervention_tones.png")
