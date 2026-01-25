#!/usr/bin/env python3
"""
Plot OLMo sandbagging steering results.
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

EMOTION_COLORS = {
    'anger': '#e74c3c',
    'disgust': '#27ae60',
    'fear': '#3498db',
    'happiness': '#f39c12',
    'sadness': '#9b59b6',
    'surprise': '#e91e63',
}


def load_results(jsonl_path: Path):
    """Load and aggregate sandbagging results."""
    results = defaultdict(lambda: {'correct': [], 'total': 0})

    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            cond = d.get('condition')
            judge = d.get('judge', {})
            if judge:
                results[cond]['total'] += 1
                results[cond]['correct'].append(1 if judge.get('answer_correct') else 0)

    return results


def plot_sandbagging_bars(results, output_path: Path, model_name: str = "OLMo-3-32B"):
    """Create bar chart for sandbagging results."""

    emotions = ['fear', 'surprise', 'anger', 'disgust', 'sadness', 'happiness']

    # Get baseline
    baseline_data = results.get('baseline', {})
    baseline_acc = np.mean(baseline_data['correct']) * 100 if baseline_data['correct'] else 0
    baseline_se = np.std(baseline_data['correct']) / np.sqrt(len(baseline_data['correct'])) * 100 if baseline_data['correct'] else 0

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(emotions) + 1)
    width = 0.35

    plus_accs = []
    plus_errors = []
    minus_accs = []
    minus_errors = []

    for emotion in emotions:
        # +100% condition
        plus_key = f'{emotion}_+100%'
        plus_data = results.get(plus_key, {})
        if plus_data['correct']:
            acc = np.mean(plus_data['correct']) * 100
            se = np.std(plus_data['correct']) / np.sqrt(len(plus_data['correct'])) * 100
        else:
            acc, se = 0, 0
        plus_accs.append(acc)
        plus_errors.append(se * 1.96)

        # -100% condition
        minus_key = f'{emotion}_-100%'
        minus_data = results.get(minus_key, {})
        if minus_data['correct']:
            acc = np.mean(minus_data['correct']) * 100
            se = np.std(minus_data['correct']) / np.sqrt(len(minus_data['correct'])) * 100
        else:
            acc, se = 0, 0
        minus_accs.append(acc)
        minus_errors.append(se * 1.96)

    # Plot baseline
    ax.bar(0, baseline_acc, width * 2, color='gray', alpha=0.7,
           yerr=baseline_se * 1.96, capsize=4, label='Baseline')

    # Plot emotion bars
    for i, emotion in enumerate(emotions):
        color = EMOTION_COLORS.get(emotion, 'gray')

        # +100% (dark)
        ax.bar(i + 1 - width/2, plus_accs[i], width,
               color=color, alpha=0.9,
               yerr=plus_errors[i], capsize=3)

        # -100% (light)
        ax.bar(i + 1 + width/2, minus_accs[i], width,
               color=color, alpha=0.4,
               yerr=minus_errors[i], capsize=3)

    # Add baseline reference line
    ax.axhline(y=baseline_acc, color='gray', linestyle='--', alpha=0.5, linewidth=2)

    # Labels
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_xlabel('Emotion Steering', fontsize=12)
    ax.set_title(f'Sandbagging Steering: {model_name} (Layer 30)\nDark = +100% (towards), Light = -100% (away)', fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(['baseline'] + emotions)
    ax.set_ylim(0, 70)

    # Add change annotations
    for i, emotion in enumerate(emotions):
        plus_change = plus_accs[i] - baseline_acc
        minus_change = minus_accs[i] - baseline_acc

        # Annotate significant changes
        if abs(plus_change) > 5:
            ax.annotate(f'{plus_change:+.0f}%',
                       xy=(i + 1 - width/2, plus_accs[i] + plus_errors[i] + 1),
                       ha='center', va='bottom', fontsize=8, color='darkred' if plus_change < 0 else 'darkgreen')
        if abs(minus_change) > 5:
            ax.annotate(f'{minus_change:+.0f}%',
                       xy=(i + 1 + width/2, minus_accs[i] + minus_errors[i] + 1),
                       ha='center', va='bottom', fontsize=8, color='darkred' if minus_change < 0 else 'darkgreen')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.7, label='Baseline'),
        Patch(facecolor='black', alpha=0.9, label='+100% (steer towards)'),
        Patch(facecolor='black', alpha=0.4, label='-100% (steer away)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    return fig


def main():
    input_file = Path(__file__).parent.parent / "outputs" / "sandbagging_steering_olmo_layer30_20260111_165101.judged.jsonl"
    output_dir = Path(__file__).parent.parent / "outputs"

    print(f"Loading results from {input_file}")
    results = load_results(input_file)

    plot_sandbagging_bars(results, output_dir / "sandbagging_steering_olmo_results.png", "OLMo-3-32B")


if __name__ == "__main__":
    main()
