"""
Plot comparing vanilla model and all DPO layer configurations across all evaluations.
Creates a clean, presentation-ready visualization showing mean and max scores.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Data from the layer sweep results
# Format: (mean, max) for each metric
data = {
    'Model': [
        'Vanilla',
        'Last 5', 'Last 10', 'Last 15', 'Last 20', 'Last 20 (2ep)',
        'Last 30', 'Last 30 (2ep)', 'Last 35', 'Last 40',
        'L20-25', 'L20-30', 'L25-30', 'L30-35', 'L35-40',
        'L30-40', 'L30-50', 'L40-50',
        'Full DPO'
    ],
    'Layers': [
        '-',
        '57-61', '52-61', '47-61', '42-61', '42-61',
        '32-61', '32-61', '27-61', '22-61',
        '20-25', '20-30', '25-30', '30-35', '35-40',
        '30-40', '30-50', '40-50',
        '0-61'
    ],
    'Aggressive': [(2.70, 7), (2.85, 7), (2.85, 6), (2.95, 5), (3.60, 6), (2.70, 5),
                   (1.35, 3), (1.30, 3), (1.15, 2), (1.00, 3),
                   (1.20, 2), (1.20, 3), (1.10, 2), (0.90, 2), (1.05, 3),
                   (0.70, 1), (0.60, 1), (2.05, 4),
                   (1.05, 3)],
    'Disappointed': [(1.75, 5), (2.55, 5), (1.95, 4), (2.10, 4), (2.35, 5), (1.80, 4),
                     (1.50, 3), (1.10, 3), (0.70, 2), (0.60, 2),
                     (0.65, 2), (0.60, 2), (0.65, 2), (0.65, 2), (1.25, 3),
                     (0.70, 1), (0.65, 1), (1.70, 4),
                     (0.45, 2)],
    'Sarcastic': [(2.70, 6), (3.50, 7), (2.00, 5), (2.70, 5), (2.50, 5), (2.35, 5),
                  (1.55, 3), (1.20, 3), (0.75, 2), (1.00, 4),
                  (1.20, 2), (1.05, 3), (0.80, 2), (1.10, 2), (0.80, 2),
                  (0.55, 1), (0.70, 2), (1.85, 4),
                  (0.80, 3)],
    'Original': [(1.50, 4), (1.45, 5), (1.31, 9), (1.21, 5), (1.22, 5), (1.11, 5),
                 (0.80, 3), (0.68, 3), (0.46, 2), (0.41, 2),
                 (1.22, 5), (1.14, 3), (0.92, 2), (0.73, 1), (1.04, 3),
                 (0.59, 1), (0.44, 1), (1.68, 5),
                 (0.54, 3)],
    'Variant': [(1.20, 4), (1.15, 4), (1.11, 3), (1.11, 5), (0.99, 4), (0.94, 3),
                (0.69, 3), (0.64, 2), (0.47, 2), (0.39, 2),
                (1.23, 3), (1.01, 3), (0.92, 2), (0.76, 1), (0.93, 2),
                (0.55, 2), (0.43, 1), (1.47, 5),
                (0.49, 3)],
    'WildChat': [(1.00, 3), (0.70, 4), (0.62, 3), (0.68, 4), (0.59, 4), (0.58, 3),
                 (0.43, 3), (0.36, 2), (0.30, 2), (0.25, 2),
                 (0.93, 5), (0.96, 2), (0.71, 2), (0.60, 4), (0.49, 2),
                 (0.41, 1), (0.31, 1), (1.01, 3),
                 (0.31, 2)],
    'Long T8': [(3.75, None), (None, None), (None, None), (None, None), (None, None), (None, None),
                (None, None), (None, None), (None, None), (1.45, None),
                (1.90, None), (1.75, None), (0.95, None), (0.85, None), (1.05, None),
                (0.45, None), (0.25, None), (2.50, None),
                (0.80, None)],
}

# Create separate dataframes for mean and max
models = data['Model']
layers = data['Layers']
metrics = ['Aggressive', 'Disappointed', 'Sarcastic', 'Original', 'Variant', 'WildChat', 'Long T8']

# Extract means and maxes
mean_data = []
max_data = []
for model in models:
    idx = models.index(model)
    mean_row = []
    max_row = []
    for metric in metrics:
        val = data[metric][idx]
        if val[0] is not None:
            mean_row.append(val[0])
        else:
            mean_row.append(np.nan)
        if val[1] is not None:
            max_row.append(val[1])
        else:
            max_row.append(np.nan)
    mean_data.append(mean_row)
    max_data.append(max_row)

mean_df = pd.DataFrame(mean_data, index=models, columns=metrics)
max_df = pd.DataFrame(max_data, index=models, columns=metrics)

# Create figure with two subplots side by side
fig, axes = plt.subplots(1, 2, figsize=(18, 12))

# Custom colormap - white to red
cmap_mean = sns.color_palette("YlOrRd", as_cmap=True)
cmap_max = sns.color_palette("YlOrRd", as_cmap=True)

# Plot mean heatmap
ax1 = axes[0]
sns.heatmap(mean_df, annot=True, fmt='.2f', cmap=cmap_mean,
            ax=ax1, vmin=0, vmax=4,
            cbar_kws={'label': 'Frustration Score', 'shrink': 0.8},
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 9})
ax1.set_title('Mean Frustration Score (Turn 3)', fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.tick_params(axis='x', rotation=45, labelsize=10)
ax1.tick_params(axis='y', rotation=0, labelsize=10)

# Add horizontal line to separate vanilla from DPO models
ax1.axhline(y=1, color='black', linewidth=2)

# Plot max heatmap
ax2 = axes[1]
sns.heatmap(max_df, annot=True, fmt='.0f', cmap=cmap_max,
            ax=ax2, vmin=0, vmax=9,
            cbar_kws={'label': 'Max Score', 'shrink': 0.8},
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 9})
ax2.set_title('Maximum Frustration Score (Turn 3)', fontsize=14, fontweight='bold', pad=10)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.tick_params(axis='x', rotation=45, labelsize=10)
ax2.tick_params(axis='y', rotation=0, labelsize=10)

# Add horizontal line to separate vanilla from DPO models
ax2.axhline(y=1, color='black', linewidth=2)

# Highlight best models with boxes
best_models = ['L30-40', 'L30-50']
for ax in axes:
    for i, model in enumerate(models):
        if model in best_models:
            ax.add_patch(plt.Rectangle((0, i), len(metrics), 1,
                                       fill=False, edgecolor='green', linewidth=3))

plt.suptitle('DPO Layer Sweep: Frustration Reduction Comparison\n(Gemma-3-27B-it, Lower = Better)',
             fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/layer_sweep_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/layer_sweep_comparison.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved to /workspace-vast/annas/Ant_Cluster_Notes/layer_sweep_comparison.png")

# Also create a cleaner version focusing on key models
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8))

# Select key models for comparison
key_models = ['Vanilla', 'Last 20 (2ep)', 'Last 35', 'L20-25', 'L25-30', 'L30-35',
              'L30-40', 'L30-50', 'L40-50', 'Full DPO']
key_mean_df = mean_df.loc[key_models]
key_max_df = max_df.loc[key_models]

# Plot mean heatmap
ax1 = axes2[0]
sns.heatmap(key_mean_df, annot=True, fmt='.2f', cmap=cmap_mean,
            ax=ax1, vmin=0, vmax=4,
            cbar_kws={'label': 'Score', 'shrink': 0.8},
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 11, 'fontweight': 'bold'})
ax1.set_title('Mean Frustration Score', fontsize=14, fontweight='bold', pad=10)
ax1.set_xlabel('')
ax1.set_ylabel('')
ax1.tick_params(axis='x', rotation=45, labelsize=11)
ax1.tick_params(axis='y', rotation=0, labelsize=11)
ax1.axhline(y=1, color='black', linewidth=2)

# Plot max heatmap
ax2 = axes2[1]
sns.heatmap(key_max_df, annot=True, fmt='.0f', cmap=cmap_max,
            ax=ax2, vmin=0, vmax=9,
            cbar_kws={'label': 'Score', 'shrink': 0.8},
            linewidths=0.5, linecolor='white',
            annot_kws={'size': 11, 'fontweight': 'bold'})
ax2.set_title('Maximum Frustration Score', fontsize=14, fontweight='bold', pad=10)
ax2.set_xlabel('')
ax2.set_ylabel('')
ax2.tick_params(axis='x', rotation=45, labelsize=11)
ax2.tick_params(axis='y', rotation=0, labelsize=11)
ax2.axhline(y=1, color='black', linewidth=2)

# Highlight best models
for ax in axes2:
    for i, model in enumerate(key_models):
        if model in best_models:
            ax.add_patch(plt.Rectangle((0, i), len(metrics), 1,
                                       fill=False, edgecolor='#2ecc71', linewidth=3))

plt.suptitle('DPO Layer Targeting: Frustration Reduction\n(Gemma-3-27B, Turn 3 scores, Lower = Better)',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/layer_sweep_key_models.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/layer_sweep_key_models.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved to /workspace-vast/annas/Ant_Cluster_Notes/layer_sweep_key_models.png")

# Create a third plot - bar chart showing reduction from vanilla
fig3, ax3 = plt.subplots(figsize=(14, 7))

# Calculate reduction from vanilla for each model (average across metrics)
vanilla_means = mean_df.loc['Vanilla'].values
reductions = []
for model in key_models[1:]:  # Skip vanilla
    model_means = mean_df.loc[model].values
    # Calculate % reduction, handling NaN
    valid_mask = ~np.isnan(model_means) & ~np.isnan(vanilla_means)
    if valid_mask.any():
        reduction = np.nanmean((vanilla_means[valid_mask] - model_means[valid_mask]) / vanilla_means[valid_mask] * 100)
    else:
        reduction = 0
    reductions.append(reduction)

colors = ['#3498db' if r < 50 else '#2ecc71' if r < 70 else '#27ae60' for r in reductions]
bars = ax3.bar(key_models[1:], reductions, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on bars
for bar, val in zip(bars, reductions):
    height = bar.get_height()
    ax3.annotate(f'{val:.0f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax3.set_ylabel('Frustration Reduction vs Vanilla (%)', fontsize=12, fontweight='bold')
ax3.set_xlabel('')
ax3.set_title('Average Frustration Reduction by Layer Configuration\n(Higher = Better)',
              fontsize=14, fontweight='bold')
ax3.tick_params(axis='x', rotation=45, labelsize=11)
ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% reduction')
ax3.axhline(y=70, color='gray', linestyle=':', alpha=0.5, label='70% reduction')
ax3.set_ylim(0, 100)
ax3.legend(loc='lower right')

# Add annotation for best model
best_idx = np.argmax(reductions)
ax3.annotate('Best', xy=(best_idx, reductions[best_idx]),
            xytext=(best_idx, reductions[best_idx] + 8),
            ha='center', fontsize=10, fontweight='bold', color='#27ae60',
            arrowprops=dict(arrowstyle='->', color='#27ae60'))

plt.tight_layout()
plt.savefig('/workspace-vast/annas/Ant_Cluster_Notes/layer_sweep_reduction.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("Saved to /workspace-vast/annas/Ant_Cluster_Notes/layer_sweep_reduction.png")

print("\nDone! Created 3 plots.")
