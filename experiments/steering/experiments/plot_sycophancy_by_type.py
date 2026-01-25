#!/usr/bin/env python3
"""
Plot sycophancy steering results by test type with correct styling.
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

# Match the color scheme from other plots
EMOTION_COLORS = {
    'anger': '#7BA7D7',      # light blue
    'disgust': '#7D9B7D',    # muted green
    'fear': '#a59dc9',       # light purple
    'happiness': '#D4876A',  # coral/orange
    'sadness': '#B8CCC8',    # sage green
    'surprise': '#D1728F',   # pink
}

# Each test type uses a different metric
TEST_TYPE_METRICS = {
    'dispute': {
        'field': 'favors_user_as_person',
        'label': 'Favors User (%)',
        'title': 'DISPUTE\n(user vs friend debate)',
    },
    'opinion': {
        'field': 'adopts_user_stance',
        'label': 'Adopts User Stance (%)',
        'title': 'OPINION\n(agrees with user opinion)',
    },
    'author': {
        'field': 'offered_praise',
        'label': 'Offers Praise (%)',
        'title': 'AUTHOR\n(praises user\'s writing)',
    },
    'falsehood': {
        'field': 'identifies_falsehood',
        'label': 'Identifies Falsehood (%)',
        'title': 'FALSEHOOD\n(corrects false claims)',
        'invert': True,  # Higher is better here, so we DON'T invert
    },
}


def load_results(jsonl_path: Path):
    """Load and aggregate sycophancy results by test type."""
    results = defaultdict(lambda: defaultdict(lambda: {'values': []}))

    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            cond = d.get('condition')
            test_type = d.get('test_type')
            judge = d.get('sycophancy_judge', {})

            if not judge or 'error' in judge:
                continue

            metric_info = TEST_TYPE_METRICS.get(test_type, {})
            field = metric_info.get('field')

            if field and field in judge:
                val = 1 if judge.get(field) else 0
                results[test_type][cond]['values'].append(val)

    return results


def plot_by_test_type(results, output_path: Path):
    """Create multi-panel plot by test type."""

    test_types = ['dispute', 'opinion', 'author', 'falsehood']
    emotions = ['happiness', 'disgust', 'sadness', 'surprise']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax_idx, test_type in enumerate(test_types):
        ax = axes[ax_idx]
        data = results.get(test_type, {})
        metric_info = TEST_TYPE_METRICS.get(test_type, {})

        # Get baseline
        baseline_vals = data.get('baseline', {}).get('values', [])
        baseline_rate = np.mean(baseline_vals) * 100 if baseline_vals else 0
        baseline_se = np.std(baseline_vals) / np.sqrt(len(baseline_vals)) * 100 if len(baseline_vals) > 1 else 0

        x = np.arange(len(emotions) + 1)
        width = 0.35

        plus_rates = []
        minus_rates = []
        plus_errors = []
        minus_errors = []

        for emotion in emotions:
            for sign, rates_list, errors_list in [
                ('+10%', plus_rates, plus_errors),
                ('-10%', minus_rates, minus_errors)
            ]:
                key = f'{emotion}_{sign}'
                vals = data.get(key, {}).get('values', [])
                if vals:
                    rate = np.mean(vals) * 100
                    se = np.std(vals) / np.sqrt(len(vals)) * 100 if len(vals) > 1 else 0
                else:
                    rate, se = 0, 0
                rates_list.append(rate)
                errors_list.append(se * 1.96)

        # Plot baseline
        ax.bar(0, baseline_rate, width * 2, color='#808080', alpha=0.6,
               yerr=baseline_se * 1.96, capsize=3, edgecolor='black', linewidth=0.5)

        # Plot emotion bars
        for i, emotion in enumerate(emotions):
            color = EMOTION_COLORS.get(emotion, '#808080')

            # +10% (darker/saturated)
            ax.bar(i + 1 - width/2, plus_rates[i], width, color=color, alpha=0.9,
                   yerr=plus_errors[i], capsize=2, edgecolor='black', linewidth=0.5)

            # -10% (lighter)
            ax.bar(i + 1 + width/2, minus_rates[i], width, color=color, alpha=0.4,
                   yerr=minus_errors[i], capsize=2, edgecolor='black', linewidth=0.5)

        ax.axhline(y=baseline_rate, color='#808080', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.set_ylabel(metric_info.get('label', 'Rate (%)'), fontsize=10)
        ax.set_title(metric_info.get('title', test_type.upper()), fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['baseline'] + emotions, rotation=20, ha='right', fontsize=9)
        ax.set_ylim(0, 100)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', alpha=0.6, edgecolor='black', label='Baseline'),
        Patch(facecolor='#808080', alpha=0.9, edgecolor='black', label='+10% (towards)'),
        Patch(facecolor='#808080', alpha=0.4, edgecolor='black', label='-10% (away)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)

    fig.suptitle('Sycophancy Steering by Test Type: Gemma-3-27B (Layer 30)\nDark = +10% (towards emotion), Light = -10% (away from emotion)',
                 fontsize=13, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def plot_summary(results, output_path: Path):
    """Create summary bar chart for dispute test type (most interpretable)."""

    emotions = ['happiness', 'disgust', 'sadness', 'surprise']
    data = results.get('dispute', {})

    fig, ax = plt.subplots(figsize=(12, 6))

    baseline_vals = data.get('baseline', {}).get('values', [])
    baseline_rate = np.mean(baseline_vals) * 100 if baseline_vals else 0
    baseline_se = np.std(baseline_vals) / np.sqrt(len(baseline_vals)) * 100 if len(baseline_vals) > 1 else 0

    x = np.arange(len(emotions) + 1)
    width = 0.35

    plus_rates, minus_rates = [], []
    plus_errors, minus_errors = [], []

    for emotion in emotions:
        for sign, rates_list, errors_list in [
            ('+10%', plus_rates, plus_errors),
            ('-10%', minus_rates, minus_errors)
        ]:
            key = f'{emotion}_{sign}'
            vals = data.get(key, {}).get('values', [])
            if vals:
                rate = np.mean(vals) * 100
                se = np.std(vals) / np.sqrt(len(vals)) * 100 if len(vals) > 1 else 0
            else:
                rate, se = 0, 0
            rates_list.append(rate)
            errors_list.append(se * 1.96)

    # Plot baseline
    ax.bar(0, baseline_rate, width * 2, color='#808080', alpha=0.6,
           yerr=baseline_se * 1.96, capsize=4, edgecolor='black', linewidth=0.5)

    # Plot emotion bars
    for i, emotion in enumerate(emotions):
        color = EMOTION_COLORS.get(emotion, '#808080')

        ax.bar(i + 1 - width/2, plus_rates[i], width, color=color, alpha=0.9,
               yerr=plus_errors[i], capsize=3, edgecolor='black', linewidth=0.5)
        ax.bar(i + 1 + width/2, minus_rates[i], width, color=color, alpha=0.4,
               yerr=minus_errors[i], capsize=3, edgecolor='black', linewidth=0.5)

        # Annotate significant changes
        plus_change = plus_rates[i] - baseline_rate
        minus_change = minus_rates[i] - baseline_rate
        if abs(plus_change) > 8:
            color_text = '#8B0000' if plus_change < 0 else '#006400'
            ax.annotate(f'{plus_change:+.0f}%', xy=(i + 1 - width/2, plus_rates[i] + plus_errors[i] + 2),
                       ha='center', fontsize=9, color=color_text, fontweight='bold')
        if abs(minus_change) > 8:
            color_text = '#8B0000' if minus_change < 0 else '#006400'
            ax.annotate(f'{minus_change:+.0f}%', xy=(i + 1 + width/2, minus_rates[i] + minus_errors[i] + 2),
                       ha='center', fontsize=9, color=color_text, fontweight='bold')

    ax.axhline(y=baseline_rate, color='#808080', linestyle='--', alpha=0.7, linewidth=2)
    ax.set_ylabel('Favors User (%)', fontsize=12)
    ax.set_xlabel('Emotion Steering', fontsize=12)
    ax.set_title('Sycophancy Steering (Dispute): Gemma-3-27B (Layer 30)\nDark = +10% (towards), Light = -10% (away)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['baseline'] + emotions, fontsize=11)
    ax.set_ylim(0, 60)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', alpha=0.6, edgecolor='black', label='Baseline'),
        Patch(facecolor='#808080', alpha=0.9, edgecolor='black', label='+10% (towards)'),
        Patch(facecolor='#808080', alpha=0.4, edgecolor='black', label='-10% (away)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def main():
    input_file = Path(__file__).parent.parent / "outputs" / "sycophancy_steering_layer30_20260109_161637.judged.jsonl"
    output_dir = Path(__file__).parent.parent / "outputs"

    print(f"Loading results from {input_file}")
    results = load_results(input_file)

    # Print data summary
    for tt, data in results.items():
        total = sum(len(v['values']) for v in data.values())
        print(f"  {tt}: {total} samples")

    # By test type
    plot_by_test_type(results, output_dir / "sycophancy_steering_by_type.png")

    # Summary (dispute only)
    plot_summary(results, output_dir / "sycophancy_steering_summary.png")


if __name__ == "__main__":
    main()
