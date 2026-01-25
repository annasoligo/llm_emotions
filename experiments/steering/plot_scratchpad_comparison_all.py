#!/usr/bin/env python3
"""Plot sandbagging score comparison across all conditions: 7% vs 10%, emotional vs logic."""
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Emotion color mapping (from plot_deception_prefix_results.py)
COLORS = {
    'emotional_7': '#5D3A8A',   # Dark purple
    'logic_7': '#8B7BB8',       # Medium purple
    'logic_10': '#B8A8D4',      # Light purple
}

def load_and_aggregate(filepath, conditions_of_interest):
    """Load JSONL and compute mean/CI95 for sandbagging score."""
    results = []
    with open(filepath) as f:
        for line in f:
            results.append(json.loads(line))

    by_condition = defaultdict(list)
    for r in results:
        by_condition[r['condition']].append(r)

    stats = {}
    for cond in conditions_of_interest:
        data = by_condition.get(cond, [])
        scores = []
        for r in data:
            j = r.get('judgment', {})
            if isinstance(j, dict) and 'sandbagging_score' in j:
                scores.append(j['sandbagging_score'])
        if scores:
            mean = np.mean(scores)
            sem = np.std(scores) / np.sqrt(len(scores))
            ci95 = 1.96 * sem
            sandbag_pct = 100 * sum(1 for s in scores if s <= 2) / len(scores)
            stats[cond] = {'mean': mean, 'ci95': ci95, 'n': len(scores), 'sandbag_pct': sandbag_pct}
    return stats

# Load all data
emotional_7pct = load_and_aggregate(
    'experiments/steering/outputs/judged_ua_7pct_emotional_with_emotion.jsonl',
    ['baseline', 'fear_+7%', 'fear_-7%']
)
logic_7pct = load_and_aggregate(
    'experiments/steering/outputs/judged_ua_7pct_logic_with_emotion.jsonl',
    ['baseline', 'fear_+7%', 'fear_-7%']
)
logic_10pct = load_and_aggregate(
    'experiments/steering/outputs/judged_ua_model_logic_layer30_with_emotion.jsonl',
    ['baseline', 'fear_+10%', 'fear_-10%']
)

# Print summary
print("=== SUMMARY ===")
print("\n7% Emotional Scratchpad:")
for cond, stats in emotional_7pct.items():
    print(f"  {cond}: score={stats['mean']:.2f}±{stats['ci95']:.2f}, sandbagging={stats['sandbag_pct']:.1f}%")

print("\n7% Logic-only Scratchpad:")
for cond, stats in logic_7pct.items():
    print(f"  {cond}: score={stats['mean']:.2f}±{stats['ci95']:.2f}, sandbagging={stats['sandbag_pct']:.1f}%")

print("\n10% Logic-only Scratchpad:")
for cond, stats in logic_10pct.items():
    print(f"  {cond}: score={stats['mean']:.2f}±{stats['ci95']:.2f}, sandbagging={stats['sandbag_pct']:.1f}%")

# Create summary figure
fig, ax = plt.subplots(figsize=(10, 6))

# Data for grouped bar chart - focus on fear steering effect
labels = ['Baseline', 'Fear Steering\n(positive)']
emotional_7 = [emotional_7pct['baseline']['mean'], emotional_7pct['fear_+7%']['mean']]
emotional_7_ci = [emotional_7pct['baseline']['ci95'], emotional_7pct['fear_+7%']['ci95']]
logic_7 = [logic_7pct['baseline']['mean'], logic_7pct['fear_+7%']['mean']]
logic_7_ci = [logic_7pct['baseline']['ci95'], logic_7pct['fear_+7%']['ci95']]
logic_10 = [logic_10pct['baseline']['mean'], logic_10pct['fear_+10%']['mean']]
logic_10_ci = [logic_10pct['baseline']['ci95'], logic_10pct['fear_+10%']['ci95']]

x = np.arange(len(labels))
width = 0.25

bars1 = ax.bar(x - width, emotional_7, width, yerr=emotional_7_ci,
               label='7% Emotional Scratchpad', color=COLORS['emotional_7'], capsize=4, alpha=0.9)
bars2 = ax.bar(x, logic_7, width, yerr=logic_7_ci,
               label='7% Logic-only Scratchpad', color=COLORS['logic_7'], capsize=4, alpha=0.9)
bars3 = ax.bar(x + width, logic_10, width, yerr=logic_10_ci,
               label='10% Logic-only Scratchpad', color=COLORS['logic_10'], capsize=4, alpha=0.9)

ax.set_ylabel('Sandbagging Score\n(1=sandbagging, 5=honest)', fontsize=12)
ax.set_xlabel('Condition', fontsize=12)
ax.set_title('Fear Steering Effect on Sandbagging\nby Scratchpad Type and Steering Magnitude', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='lower left')
ax.set_ylim(3.5, 5.3)
ax.axhline(y=5.0, color='gray', linestyle='--', alpha=0.3)

# Add value annotations
for i, (e7, l7, l10) in enumerate(zip(emotional_7, logic_7, logic_10)):
    ax.annotate(f'{e7:.2f}', (x[i] - width, e7 + 0.05), ha='center', fontsize=9, fontweight='bold')
    ax.annotate(f'{l7:.2f}', (x[i], l7 + 0.05), ha='center', fontsize=9, fontweight='bold')
    ax.annotate(f'{l10:.2f}', (x[i] + width, l10 + 0.05), ha='center', fontsize=9, fontweight='bold')

# Add sandbagging % annotation for steered conditions
steered_pcts = [
    emotional_7pct['fear_+7%']['sandbag_pct'],
    logic_7pct['fear_+7%']['sandbag_pct'],
    logic_10pct['fear_+10%']['sandbag_pct']
]
for i, (pct, xpos) in enumerate(zip(steered_pcts, [x[1] - width, x[1], x[1] + width])):
    ax.annotate(f'({pct:.0f}%)', (xpos, emotional_7[1] - 0.3 if i == 0 else logic_7[1] - 0.15 if i == 1 else logic_10[1] - 0.2),
                ha='center', fontsize=8, color='red')

plt.tight_layout()
plt.savefig('experiments/steering/outputs/sandbagging_summary_comparison.png', dpi=150, bbox_inches='tight')
print('\nSaved to experiments/steering/outputs/sandbagging_summary_comparison.png')

# Create detailed 2-panel figure
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: 7% comparison (emotional vs logic)
ax1 = axes[0]
x = np.arange(3)
width = 0.35

emotional_means = [emotional_7pct['baseline']['mean'], emotional_7pct['fear_+7%']['mean'], emotional_7pct['fear_-7%']['mean']]
emotional_cis = [emotional_7pct['baseline']['ci95'], emotional_7pct['fear_+7%']['ci95'], emotional_7pct['fear_-7%']['ci95']]
logic_means = [logic_7pct['baseline']['mean'], logic_7pct['fear_+7%']['mean'], logic_7pct['fear_-7%']['mean']]
logic_cis = [logic_7pct['baseline']['ci95'], logic_7pct['fear_+7%']['ci95'], logic_7pct['fear_-7%']['ci95']]

bars1 = ax1.bar(x - width/2, emotional_means, width, yerr=emotional_cis,
                label='Emotional Scratchpad', color=COLORS['emotional_7'], capsize=3, alpha=0.9)
bars2 = ax1.bar(x + width/2, logic_means, width, yerr=logic_cis,
                label='Logic-only Scratchpad', color=COLORS['logic_7'], capsize=3, alpha=0.9)

ax1.set_ylabel('Sandbagging Score (1=sandbagging, 5=honest)', fontsize=11)
ax1.set_xlabel('Steering Condition', fontsize=11)
ax1.set_title('7% Norm: Emotional vs Logic-only', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(['Baseline', 'Fear +7%', 'Fear -7%'])
ax1.legend(loc='lower right')
ax1.set_ylim(3.5, 5.3)
ax1.axhline(y=5.0, color='gray', linestyle='--', alpha=0.3)

# Right plot: Logic-only comparison (7% vs 10%)
ax2 = axes[1]
x2 = np.arange(3)

logic_7_means = [logic_7pct['baseline']['mean'], logic_7pct['fear_+7%']['mean'], logic_7pct['fear_-7%']['mean']]
logic_7_cis = [logic_7pct['baseline']['ci95'], logic_7pct['fear_+7%']['ci95'], logic_7pct['fear_-7%']['ci95']]
logic_10_means = [logic_10pct['baseline']['mean'], logic_10pct['fear_+10%']['mean'], logic_10pct['fear_-10%']['mean']]
logic_10_cis = [logic_10pct['baseline']['ci95'], logic_10pct['fear_+10%']['ci95'], logic_10pct['fear_-10%']['ci95']]

bars3 = ax2.bar(x2 - width/2, logic_7_means, width, yerr=logic_7_cis,
                label='7% Norm', color=COLORS['logic_7'], capsize=3, alpha=0.9)
bars4 = ax2.bar(x2 + width/2, logic_10_means, width, yerr=logic_10_cis,
                label='10% Norm', color=COLORS['logic_10'], capsize=3, alpha=0.9)

ax2.set_ylabel('Sandbagging Score (1=sandbagging, 5=honest)', fontsize=11)
ax2.set_xlabel('Steering Condition', fontsize=11)
ax2.set_title('Logic-only Scratchpad: 7% vs 10% Norm', fontsize=13, fontweight='bold')
ax2.set_xticks(x2)
ax2.set_xticklabels(['Baseline', 'Fear +', 'Fear -'])
ax2.legend(loc='lower right')
ax2.set_ylim(3.5, 5.3)
ax2.axhline(y=5.0, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/steering/outputs/sandbagging_7pct_vs_10pct_detailed.png', dpi=150, bbox_inches='tight')
print('Saved to experiments/steering/outputs/sandbagging_7pct_vs_10pct_detailed.png')
