#!/usr/bin/env python3
"""Plot comparison of Base vs DPO Calm on representative benchmarks."""

import matplotlib.pyplot as plt
import numpy as np

# Colors (slightly paler)
CORAL = "#DBA089"         # Pre-finetune (Instruct)
SKY_BLUE = "#9BBDE3"      # Post-finetune (DPO)

# Data: 7 representative benchmarks at temp=0
# Format: (benchmark_name, base_score, dpo_score, n_samples)
benchmarks = [
    ("AIME 24/25\n(Math Competition)", 21.7, 21.7, 60),  # 30 questions x 2 years
    ("MATH Hard\n(Level 5)", 64.58, 65.71, 1324),        # Level 5 questions
    ("GPQA\n(PhD Science)", 39.90, 38.89, 198),          # Diamond subset
    ("BBH\n(Reasoning)", 80.57, 80.86, 6511),            # Big Bench Hard total
    ("TruthfulQA", 43.8, 43.3, 817),                     # TruthfulQA MC2
    ("EmoBench EU\n(Understanding)", 46.0, 48.5, 200),  # EmoBench EU
    ("EmoBench EA\n(Application)", 73.0, 72.5, 200),    # EmoBench EA
]

# Calculate bootstrap 95% CI for proportions
def bootstrap_ci_proportion(p, n, n_bootstrap=1000, ci=0.95):
    """Bootstrap CI for a proportion given p (as percentage) and n samples."""
    prop = p / 100
    boot_props = []
    for _ in range(n_bootstrap):
        # Simulate bootstrap by sampling from binomial
        successes = np.random.binomial(n, prop)
        boot_props.append(successes / n * 100)
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_props, alpha * 100)
    upper = np.percentile(boot_props, (1 - alpha) * 100)
    return (upper - lower) / 2

names = [b[0] for b in benchmarks]
base_scores = [b[1] for b in benchmarks]
dpo_scores = [b[2] for b in benchmarks]
base_errors = [bootstrap_ci_proportion(b[1], b[3]) for b in benchmarks]
dpo_errors = [bootstrap_ci_proportion(b[2], b[3]) for b in benchmarks]

# Plot
fig, ax = plt.subplots(figsize=(16, 5.5))

x = np.arange(len(names))
width = 0.28

bars1 = ax.bar(x - width/2 - 0.02, base_scores, width, label='Instruct (Gemma-3-27B-it)',
               color=CORAL, yerr=base_errors, capsize=5, error_kw={'linewidth': 1.5},
               edgecolor='black', linewidth=0.8)
bars2 = ax.bar(x + width/2 + 0.02, dpo_scores, width, label='DPO Finetune',
               color=SKY_BLUE, yerr=dpo_errors, capsize=5, error_kw={'linewidth': 1.5},
               edgecolor='black', linewidth=0.8)

# Formatting
ax.set_ylabel('Accuracy (%)', fontsize=21)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=15)
ax.legend(fontsize=19, loc='upper left', bbox_to_anchor=(0.0, 0.97))
ax.set_ylim(0, 100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='y', labelsize=17)

# Add gridlines
ax.yaxis.grid(True, linestyle='--', alpha=0.3)
ax.set_axisbelow(True)

plt.title('Comparison of Benchmark Scores Between Gemma-3-27B-it and DPO Finetune (95% CIs)', fontsize=21)
plt.tight_layout()
plt.savefig('/workspace-vast/annas/git/gemma-iclr/hcair2026/figures/dpo_calm_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig('/workspace-vast/annas/git/gemma-iclr/hcair2026/figures/dpo_calm_comparison.pdf', bbox_inches='tight')
print("Saved to figures/dpo_calm_comparison.png and .pdf")
