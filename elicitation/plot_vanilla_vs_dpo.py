import matplotlib.pyplot as plt
import numpy as np

# Data from notes
categories = ['Aggressive', 'Disappointed', 'Sarcastic', 'Original', 'Variant', 'WildChat', 'Long T8']

# Vanilla model
vanilla_mean = [2.70, 1.75, 2.70, 1.5, 1.2, 1.0, 3.75]
vanilla_max = [7, 5, 6, 4, 4, 3, 6]  # T8: 15% >= 5, so max ~6

# SFT Last 20 (3ep) - best SFT model
sft_mean = [2.40, 1.90, 2.45, 0.98, 0.97, 0.65, 3.50]
sft_max = [5, 4, 5, 5, 3, 5, 6]  # Estimated from % >= 5 (10%, 0%, 15%, ...)

# Full DPO model
dpo_mean = [1.05, 0.45, 0.80, 0.54, 0.49, 0.31, 0.80]
dpo_max = [3, 2, 3, 3, 3, 2, 3]  # T8: 0% >= 5, so max ~3

# Set up the figure - narrower and taller for half-slide
fig, ax = plt.subplots(figsize=(8, 10))

x = np.arange(len(categories))
width = 0.25  # Narrower bars for 3 groups

# Colors
vanilla_color = '#D4876A'  # Coral/Terra Cotta
sft_color = '#7BA7D7'  # Sky Blue
dpo_color = '#7D9B7D'  # Olive Green

# Create bars (no text labels)
bars1 = ax.bar(x - width, vanilla_mean, width, label='Vanilla',
               color=vanilla_color, edgecolor='#8B5A42', linewidth=1.5, alpha=0.85)
bars2 = ax.bar(x, sft_mean, width, label='SFT',
               color=sft_color, edgecolor='#4A7BAD', linewidth=1.5, alpha=0.85)
bars3 = ax.bar(x + width, dpo_mean, width, label='DPO',
               color=dpo_color, edgecolor='#4A6B4A', linewidth=1.5, alpha=0.85)

# Add max points (diamond markers) with faint lines from bar top
for i, (vm, mean) in enumerate(zip(vanilla_max, vanilla_mean)):
    if vm is not None:
        ax.plot([i - width, i - width], [mean, vm], color=vanilla_color, alpha=0.3, linewidth=1, zorder=3)
        ax.scatter(i - width, vm, color=vanilla_color, s=100, zorder=5, marker='D',
                   edgecolors='white', linewidths=1.5)

for i, (sm, mean) in enumerate(zip(sft_max, sft_mean)):
    if sm is not None:
        ax.plot([i, i], [mean, sm], color=sft_color, alpha=0.3, linewidth=1, zorder=3)
        ax.scatter(i, sm, color=sft_color, s=100, zorder=5, marker='D',
                   edgecolors='white', linewidths=1.5)

for i, (dm, mean) in enumerate(zip(dpo_max, dpo_mean)):
    if dm is not None:
        ax.plot([i + width, i + width], [mean, dm], color=dpo_color, alpha=0.3, linewidth=1, zorder=3)
        ax.scatter(i + width, dm, color=dpo_color, s=100, zorder=5, marker='D',
                   edgecolors='white', linewidths=1.5)

# Legend entries for max markers
ax.scatter([], [], color=vanilla_color, s=100, marker='D', edgecolors='white', linewidths=1.5, label='Vanilla Max')
ax.scatter([], [], color=sft_color, s=100, marker='D', edgecolors='white', linewidths=1.5, label='SFT Max')
ax.scatter([], [], color=dpo_color, s=100, marker='D', edgecolors='white', linewidths=1.5, label='DPO Max')

# Styling
ax.set_ylabel('Frustration Score', fontsize=14, fontweight='bold')
ax.set_xlabel('Evaluation Scenario', fontsize=14, fontweight='bold')
ax.set_title('Frustration Reduction: Vanilla vs SFT vs DPO\n(Gemma-3-27B-it)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=13, rotation=45, ha='right')
ax.tick_params(axis='y', labelsize=13)
ax.set_ylim(0, 8.5)

# Grid
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

# Legend - bigger font
legend = ax.legend(loc='upper right', fontsize=13, framealpha=0.95)

plt.tight_layout()
plt.savefig('/workspace-vast/annas/git/research-tools/elicitation/vanilla_vs_dpo_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/workspace-vast/annas/git/research-tools/elicitation/vanilla_vs_dpo_comparison.pdf',
            bbox_inches='tight', facecolor='white')
print("Saved to elicitation/vanilla_vs_dpo_comparison.png and .pdf")
