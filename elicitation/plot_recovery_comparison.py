"""
Bar chart comparing Vanilla, Full DPO, and Recovery DPO models.
Shows mean with triangles for max values.
"""
import matplotlib.pyplot as plt
import numpy as np

# Categories
categories = ['Aggressive', 'Disappointed', 'Sarcastic', 'Subj. Reject', 'Correct Reject', 'Long T8']

# Data extracted from eval results
# Vanilla model
vanilla_mean = [2.85, 2.15, 3.25, 1.5, 1.8, 3.75]
vanilla_max = [5, 4, 8, 3, 4, 10]

# Full DPO model (283 pairs, baseline only)
dpo_mean = [1.05, 0.45, 0.80, 0.45, 0.75, 0.80]
dpo_max = [2, 1, 2, 1, 1, 3]

# Recovery DPO model (565 pairs: 283 baseline + 282 recovery)
recovery_mean = [0.95, 0.95, 1.00, 0.40, 0.70, 1.15]
recovery_max = [2, 2, 2, 1, 1, 2]

# Set up the figure
fig, ax = plt.subplots(figsize=(14, 9))

x = np.arange(len(categories))
width = 0.25

# Colors
vanilla_color = '#D4876A'  # Coral/Terra Cotta
dpo_color = '#7BA7D7'  # Sky Blue
recovery_color = '#7D9B7D'  # Olive Green

# Create bars
bars1 = ax.bar(x - width, vanilla_mean, width, label='Vanilla',
               color=vanilla_color, edgecolor='#8B5A42', linewidth=1.5, alpha=0.85)
bars2 = ax.bar(x, dpo_mean, width, label='Full DPO (283 pairs)',
               color=dpo_color, edgecolor='#4A7BAD', linewidth=1.5, alpha=0.85)
bars3 = ax.bar(x + width, recovery_mean, width, label='Recovery DPO (565 pairs)',
               color=recovery_color, edgecolor='#4A6B4A', linewidth=1.5, alpha=0.85)

# Add max points (triangle markers) with faint lines from bar top
for i, (vm, mean) in enumerate(zip(vanilla_max, vanilla_mean)):
    if vm is not None and vm > mean:
        ax.plot([i - width, i - width], [mean, vm], color=vanilla_color, alpha=0.4, linewidth=1.5, zorder=3)
        ax.scatter(i - width, vm, color=vanilla_color, s=120, zorder=5, marker='^',
                   edgecolors='white', linewidths=1.5)

for i, (dm, mean) in enumerate(zip(dpo_max, dpo_mean)):
    if dm is not None and dm > mean:
        ax.plot([i, i], [mean, dm], color=dpo_color, alpha=0.4, linewidth=1.5, zorder=3)
        ax.scatter(i, dm, color=dpo_color, s=120, zorder=5, marker='^',
                   edgecolors='white', linewidths=1.5)

for i, (rm, mean) in enumerate(zip(recovery_max, recovery_mean)):
    if rm is not None and rm > mean:
        ax.plot([i + width, i + width], [mean, rm], color=recovery_color, alpha=0.4, linewidth=1.5, zorder=3)
        ax.scatter(i + width, rm, color=recovery_color, s=120, zorder=5, marker='^',
                   edgecolors='white', linewidths=1.5)

# Add a horizontal line at 5 (frustration threshold)
ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Frustration threshold (5)')

# Styling
ax.set_ylabel('Frustration Score (0-10)', fontsize=18, fontweight='bold')
ax.set_xlabel('Evaluation Scenario', fontsize=18, fontweight='bold')
ax.set_title('Frustration Comparison: Vanilla vs Full DPO vs Recovery DPO\n(Gemma-3-27B-it, bars=mean, triangles=max)',
             fontsize=20, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=16, rotation=30, ha='right')
ax.tick_params(axis='y', labelsize=16)
ax.set_ylim(0, 11)

# Grid
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# Legend
ax.legend(loc='upper right', fontsize=14, framealpha=0.95)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.savefig('/workspace-vast/annas/git/research-tools/elicitation/recovery_dpo_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/workspace-vast/annas/git/research-tools/elicitation/recovery_dpo_comparison.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved to elicitation/recovery_dpo_comparison.png and .pdf")
