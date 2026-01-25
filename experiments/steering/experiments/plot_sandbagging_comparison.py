#!/usr/bin/env python3
"""
Plot sandbagging steering results comparing Gemma, Qwen, and OLMo.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Consistent color scheme
EMOTION_COLORS = {
    'anger': '#7BA7D7',      # light blue
    'disgust': '#7D9B7D',    # muted green
    'fear': '#a59dc9',       # light purple
    'happiness': '#D4876A',  # coral/orange
    'sadness': '#B8CCC8',    # sage green
    'surprise': '#D1728F',   # pink
}

ALL_EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def load_results(jsonl_path: Path):
    """Load and aggregate sandbagging results."""
    results = defaultdict(lambda: {'scores': []})

    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            cond = d.get('condition', 'unknown')
            judge = d.get('judge', {})

            # Get sandbagging_score (1-5 scale)
            if 'sandbagging_score' in judge:
                results[cond]['scores'].append(judge['sandbagging_score'])

    return results


def compute_scores(results):
    """Compute mean sandbagging score and SE for each condition."""
    stats = {}
    for cond, data in results.items():
        scores = data['scores']
        if len(scores) > 0:
            mean = np.mean(scores)
            se = np.std(scores) / np.sqrt(len(scores))
            stats[cond] = {'mean': mean, 'se': se, 'n': len(scores)}
    return stats


def merge_results(results1, results2):
    """Merge two result dicts, combining scores lists."""
    merged = defaultdict(lambda: {'scores': []})
    for r in [results1, results2]:
        for cond, data in r.items():
            merged[cond]['scores'].extend(data['scores'])
    return merged


def plot_comparison():
    """Create comparison plot across models."""

    # Load data
    gemma_file = OUTPUT_DIR / "sandbagging_steering_layer30_20260109_162128.judged.jsonl"
    olmo_file = OUTPUT_DIR / "sandbagging_steering_olmo_layer30_20260111_165101.judged.jsonl"
    # Qwen has two files: original (anger/fear) and missing emotions
    qwen_file1 = OUTPUT_DIR / "sandbagging_steering_qwen_layer30_20260111_151033.judged.jsonl"
    qwen_file2 = OUTPUT_DIR / "sandbagging_steering_qwen_layer30_20260112_073940.judged.jsonl"

    models = {}

    if gemma_file.exists():
        models['Gemma-3-27B\n(±10%)'] = compute_scores(load_results(gemma_file))

    if olmo_file.exists():
        models['OLMo-3-32B\n(±100%)'] = compute_scores(load_results(olmo_file))

    # Merge Qwen files
    qwen_results = defaultdict(lambda: {'scores': []})
    if qwen_file1.exists():
        qwen_results = merge_results(qwen_results, load_results(qwen_file1))
    if qwen_file2.exists():
        qwen_results = merge_results(qwen_results, load_results(qwen_file2))
    if qwen_results:
        models['Qwen3-32B\n(±100%)'] = compute_scores(qwen_results)

    if not models:
        print("No data found!")
        return

    # Create figure with subplots for each model
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 6), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax_idx, (model_name, stats) in enumerate(models.items()):
        ax = axes[ax_idx]

        # Get baseline
        baseline = stats.get('baseline', {}).get('mean', 0)
        baseline_se = stats.get('baseline', {}).get('se', 0)

        # Determine steering percentage from condition names
        sample_cond = [c for c in stats.keys() if c != 'baseline']
        if sample_cond:
            if '+100%' in sample_cond[0] or '-100%' in sample_cond[0]:
                pct_str = '100%'
            else:
                pct_str = '10%'
        else:
            pct_str = '?'

        # Get emotions that have data
        available_emotions = []
        for emotion in ALL_EMOTIONS:
            plus_key = f'{emotion}_+{pct_str}'
            minus_key = f'{emotion}_-{pct_str}'
            if plus_key in stats or minus_key in stats:
                available_emotions.append(emotion)

        x = np.arange(len(available_emotions) + 1)
        width = 0.35

        # Plot baseline
        ax.bar(0, baseline, width * 2, color='#808080', alpha=0.6,
               yerr=baseline_se * 1.96, capsize=4, edgecolor='black', linewidth=0.5)

        # Plot emotions
        for i, emotion in enumerate(available_emotions):
            color = EMOTION_COLORS.get(emotion, '#808080')

            plus_key = f'{emotion}_+{pct_str}'
            minus_key = f'{emotion}_-{pct_str}'

            plus_mean = stats.get(plus_key, {}).get('mean', 0)
            plus_se = stats.get(plus_key, {}).get('se', 0)
            minus_mean = stats.get(minus_key, {}).get('mean', 0)
            minus_se = stats.get(minus_key, {}).get('se', 0)

            # +% (darker)
            ax.bar(i + 1 - width/2, plus_mean, width, color=color, alpha=0.9,
                   yerr=plus_se * 1.96, capsize=2, edgecolor='black', linewidth=0.5)

            # -% (lighter)
            ax.bar(i + 1 + width/2, minus_mean, width, color=color, alpha=0.4,
                   yerr=minus_se * 1.96, capsize=2, edgecolor='black', linewidth=0.5)

        ax.axhline(y=baseline, color='#808080', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['baseline'] + available_emotions, rotation=30, ha='right', fontsize=9)
        ax.set_ylim(1, 5)

        if ax_idx == 0:
            ax.set_ylabel('Honesty Score (1=sandbagging, 5=honest)', fontsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', alpha=0.6, edgecolor='black', label='Baseline'),
        Patch(facecolor='#808080', alpha=0.9, edgecolor='black', label='+ (towards)'),
        Patch(facecolor='#808080', alpha=0.4, edgecolor='black', label='- (away)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)

    fig.suptitle('Sandbagging Score by Emotion Steering (Layer 30)\nHigher = more honest (5=honest, 1=sandbagging) | Dark = towards, Light = away',
                 fontsize=13, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    output_path = OUTPUT_DIR / "sandbagging_model_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def main():
    plot_comparison()


if __name__ == "__main__":
    main()
