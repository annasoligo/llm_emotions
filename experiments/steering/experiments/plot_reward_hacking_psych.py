#!/usr/bin/env python3
"""
Plot reward hacking psychological framing experiment results.

Creates subplots for each prompt showing paired +/- emotion steering effects.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Style settings
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Consistent color scheme (matching other steering plots)
EMOTION_COLORS = {
    'anger': '#7BA7D7',      # light blue
    'disgust': '#7D9B7D',    # muted green
    'fear': '#a59dc9',       # light purple
    'happiness': '#D4876A',  # coral/orange
    'sadness': '#B8CCC8',    # sage green
    'surprise': '#D1728F',   # pink
}

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

PROMPT_LABELS = {
    'autonomy_controlling_overt': 'Autonomy (Controlling)',
    'status_dismissive': 'Status (Dismissive)',
    'harm_salient': 'Harm (Salient)',
    'unfair_evaluation': 'Unfair Evaluation',
    'baseline_metric_only': 'Baseline (Metric Only)',
}


def load_results(jsonl_path: Path):
    """Load and aggregate results by prompt and condition."""
    results = defaultdict(lambda: defaultdict(list))

    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            prompt_id = d['prompt_id']
            condition = d['condition']
            entry_count = d['entry_count']
            results[prompt_id][condition].append(entry_count)

    return results


def plot_subplots(results, output_path: Path):
    """Create subplots for each prompt type in consistent style."""

    prompts = list(PROMPT_LABELS.keys())

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

    for idx, prompt_id in enumerate(prompts):
        ax = axes[idx]
        prompt_data = results[prompt_id]

        # Get baseline
        baseline_counts = prompt_data.get('baseline', [])
        baseline_mean = np.mean(baseline_counts) if baseline_counts else 0
        baseline_se = np.std(baseline_counts) / np.sqrt(len(baseline_counts)) if baseline_counts else 0

        x = np.arange(len(EMOTIONS) + 1)  # +1 for baseline
        width = 0.35

        # Plot baseline
        ax.bar(0, baseline_mean, width * 2, color='#808080', alpha=0.6,
               yerr=baseline_se * 1.96, capsize=4, edgecolor='black', linewidth=0.5)

        # Plot emotion bars
        for i, emotion in enumerate(EMOTIONS):
            color = EMOTION_COLORS[emotion]

            # +10% condition
            plus_key = f'{emotion}_+6%'
            plus_data = prompt_data.get(plus_key, [])
            if plus_data:
                plus_mean = np.mean(plus_data)
                plus_se = np.std(plus_data) / np.sqrt(len(plus_data))
            else:
                plus_mean, plus_se = 0, 0

            # -10% condition
            minus_key = f'{emotion}_-6%'
            minus_data = prompt_data.get(minus_key, [])
            if minus_data:
                minus_mean = np.mean(minus_data)
                minus_se = np.std(minus_data) / np.sqrt(len(minus_data))
            else:
                minus_mean, minus_se = 0, 0

            # +10% (darker)
            ax.bar(i + 1 - width/2, plus_mean, width, color=color, alpha=0.9,
                   yerr=plus_se * 1.96, capsize=2, edgecolor='black', linewidth=0.5)

            # -10% (lighter)
            ax.bar(i + 1 + width/2, minus_mean, width, color=color, alpha=0.4,
                   yerr=minus_se * 1.96, capsize=2, edgecolor='black', linewidth=0.5)

        # Add baseline reference line
        ax.axhline(y=baseline_mean, color='#808080', linestyle='--', alpha=0.7, linewidth=1.5)

        # Labels
        ax.set_title(PROMPT_LABELS[prompt_id], fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['baseline'] + EMOTIONS, rotation=30, ha='right', fontsize=9)

        if idx == 0:
            ax.set_ylabel('Glossary Entries', fontsize=11)

    # Set y-axis range
    axes[0].set_ylim(0, 180)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', alpha=0.6, edgecolor='black', label='Baseline'),
        Patch(facecolor='#808080', alpha=0.9, edgecolor='black', label='+ (towards)'),
        Patch(facecolor='#808080', alpha=0.4, edgecolor='black', label='- (away)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.98), fontsize=10)

    fig.suptitle('Reward Hacking with Scratchpad: Glossary Entry Counts (Layer 30, ±6%)\n'
                 'Dark = towards emotion, Light = away',
                 fontsize=13, y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    return fig


def plot_delta_from_baseline(results, output_path: Path):
    """Create delta plot showing change from baseline for each condition."""

    prompts = list(PROMPT_LABELS.keys())

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)

    for idx, prompt_id in enumerate(prompts):
        ax = axes[idx]
        prompt_data = results[prompt_id]

        # Get baseline
        baseline_counts = prompt_data.get('baseline', [])
        baseline_mean = np.mean(baseline_counts) if baseline_counts else 0

        x = np.arange(len(EMOTIONS))
        width = 0.35

        # Plot emotion bars
        for i, emotion in enumerate(EMOTIONS):
            color = EMOTION_COLORS[emotion]

            # +10% condition
            plus_key = f'{emotion}_+6%'
            plus_data = prompt_data.get(plus_key, [])
            if plus_data:
                delta = np.mean(plus_data) - baseline_mean
                se = np.std(plus_data) / np.sqrt(len(plus_data))
            else:
                delta, se = 0, 0
            plus_delta = delta
            plus_se = se

            # -10% condition
            minus_key = f'{emotion}_-6%'
            minus_data = prompt_data.get(minus_key, [])
            if minus_data:
                delta = np.mean(minus_data) - baseline_mean
                se = np.std(minus_data) / np.sqrt(len(minus_data))
            else:
                delta, se = 0, 0
            minus_delta = delta
            minus_se = se

            # +10% (darker)
            ax.bar(i - width/2, plus_delta, width, color=color, alpha=0.9,
                   yerr=plus_se * 1.96, capsize=2, edgecolor='black', linewidth=0.5)

            # -10% (lighter)
            ax.bar(i + width/2, minus_delta, width, color=color, alpha=0.4,
                   yerr=minus_se * 1.96, capsize=2, edgecolor='black', linewidth=0.5)

        # Add zero reference line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

        # Labels
        ax.set_title(f"{PROMPT_LABELS[prompt_id]}\n(baseline: {baseline_mean:.0f})",
                     fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(EMOTIONS, rotation=30, ha='right', fontsize=9)

        if idx == 0:
            ax.set_ylabel('Δ Entries from Baseline', fontsize=11)

    # Set symmetric y-axis
    axes[0].set_ylim(-50, 50)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', alpha=0.9, edgecolor='black', label='+ (towards)'),
        Patch(facecolor='#808080', alpha=0.4, edgecolor='black', label='- (away)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.98), fontsize=10)

    fig.suptitle('Reward Hacking with Scratchpad: Change in Glossary Entries (Layer 30, ±6%)\n'
                 'Dark = towards emotion, Light = away',
                 fontsize=13, y=1.02)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    return fig


def main():
    # Use the 6% scratchpad experiment output
    input_file = Path(__file__).parent.parent / "outputs" / "reward_hacking_psych_layer30_scratchpad_20260112_145514.jsonl"
    output_dir = Path(__file__).parent.parent / "outputs"

    print(f"Loading results from {input_file}")
    results = load_results(input_file)

    # Print summary
    for prompt_id in PROMPT_LABELS.keys():
        baseline = results[prompt_id].get('baseline', [])
        print(f"{prompt_id}: {len(baseline)} baseline samples, mean={np.mean(baseline):.1f}")

    # Absolute values plot
    plot_subplots(results, output_dir / "reward_hacking_psych_absolute.png")

    # Delta from baseline plot
    plot_delta_from_baseline(results, output_dir / "reward_hacking_psych_delta.png")


if __name__ == "__main__":
    main()
