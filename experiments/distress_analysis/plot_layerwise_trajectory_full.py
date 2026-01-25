#!/usr/bin/env python3
"""
Plot layerwise trajectory for all 62 layers of Gemma-3-27B.
Shows how probe and logit lens distress scores evolve through layers.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
csv_path = Path(__file__).parent / "divergent_layerwise_all.csv"
df = pd.read_csv(csv_path)

# All 62 layers
layers = list(range(62))

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 14

# Create figure with two subplots (probe on top, logit on bottom)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Colors - blue and red like the previous plot
colors = {
    'high_logit_low_probe': '#1f77b4',  # blue
    'high_probe_low_logit': '#d62728',  # red
}
labels = {
    'high_logit_low_probe': 'High Logit / Low Probe (n=44)',
    'high_probe_low_logit': 'High Probe / Low Logit (n=18)',
}

# === TOP PLOT: Probe trajectory ===
for div_type in ['high_logit_low_probe', 'high_probe_low_logit']:
    subset = df[df['divergence_type'] == div_type]
    probe_means = [subset[f'probe_L{l}'].mean() for l in layers]
    ax1.plot(layers, probe_means, color=colors[div_type], linewidth=2.5,
             marker='o', markersize=4, label=labels[div_type])

ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
ax1.set_ylabel('Probe Distress (σ)')
ax1.set_title('Probe Distress by Layer (fear + anger + sadness)', fontsize=20, fontweight='bold')
ax1.legend(loc='lower left', fontsize=16, framealpha=0.9)
ax1.set_xlim(0, 61)

# === BOTTOM PLOT: Logit lens trajectory ===
for div_type in ['high_logit_low_probe', 'high_probe_low_logit']:
    subset = df[df['divergence_type'] == div_type]
    logit_means = [subset[f'logit_L{l}'].mean() for l in layers]
    ax2.plot(layers, logit_means, color=colors[div_type], linewidth=2.5,
             marker='o', markersize=4, label=labels[div_type])

ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
ax2.set_xlabel('Layer')
ax2.set_ylabel('Logit Lens Distress (σ)')
ax2.set_title('Logit Lens Distress by Layer (fear + anger + sadness)', fontsize=20, fontweight='bold')
ax2.legend(loc='upper left', fontsize=16, framealpha=0.9)
ax2.set_xlim(0, 61)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'layerwise_trajectory_62layers.png', dpi=150, bbox_inches='tight')
print("Saved: layerwise_trajectory_62layers.png")

# Print summary
print("\n" + "=" * 80)
print("NORMALIZATION:")
print("=" * 80)
print("- PROBE: Per-layer baselines (fixed)")
print("- LOGIT: Per-layer baselines with final RMSNorm applied (fixed)")
print("=" * 80)
