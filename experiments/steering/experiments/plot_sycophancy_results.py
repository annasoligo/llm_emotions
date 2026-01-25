#!/usr/bin/env python3
"""
Plot sycophancy steering results in the same style as sandbagging plots.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Style settings to match sandbagging plots
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Emotion colors (matching sandbagging plots)
EMOTION_COLORS = {
    'anger': '#e74c3c',
    'disgust': '#27ae60',
    'fear': '#3498db',
    'happiness': '#f39c12',
    'sadness': '#9b59b6',
    'surprise': '#e91e63',
}


def load_results(jsonl_path: Path):
    """Load and aggregate sycophancy results."""
    results = defaultdict(lambda: {'favors_user': [], 'side_scores': [], 'coherency': []})

    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            cond = d.get('condition')
            judge = d.get('sycophancy_judge', {})
            coherency = d.get('coherency', {})

            if judge and 'favors_user_as_person' in judge:
                results[cond]['favors_user'].append(1 if judge.get('favors_user_as_person') else 0)
                if 'side_score' in judge:
                    results[cond]['side_scores'].append(judge['side_score'])

            if isinstance(coherency, dict) and 'coherency_score' in coherency:
                results[cond]['coherency'].append(coherency['coherency_score'])

    return results


def plot_sycophancy_bars(results, output_path: Path):
    """Create bar chart similar to sandbagging plots."""

    emotions = ['happiness', 'disgust', 'sadness', 'surprise']

    # Get baseline
    baseline_data = results.get('baseline', {})
    baseline_rate = np.mean(baseline_data['favors_user']) * 100 if baseline_data['favors_user'] else 0
    baseline_se = np.std(baseline_data['favors_user']) / np.sqrt(len(baseline_data['favors_user'])) * 100 if baseline_data['favors_user'] else 0

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(emotions) + 1)  # +1 for baseline
    width = 0.35

    # Collect data
    plus_rates = []
    plus_errors = []
    minus_rates = []
    minus_errors = []

    for emotion in emotions:
        # +10% condition
        plus_key = f'{emotion}_+10%'
        plus_data = results.get(plus_key, {})
        if plus_data['favors_user']:
            rate = np.mean(plus_data['favors_user']) * 100
            se = np.std(plus_data['favors_user']) / np.sqrt(len(plus_data['favors_user'])) * 100
        else:
            rate, se = 0, 0
        plus_rates.append(rate)
        plus_errors.append(se * 1.96)  # 95% CI

        # -10% condition
        minus_key = f'{emotion}_-10%'
        minus_data = results.get(minus_key, {})
        if minus_data['favors_user']:
            rate = np.mean(minus_data['favors_user']) * 100
            se = np.std(minus_data['favors_user']) / np.sqrt(len(minus_data['favors_user'])) * 100
        else:
            rate, se = 0, 0
        minus_rates.append(rate)
        minus_errors.append(se * 1.96)

    # Plot baseline
    ax.bar(0, baseline_rate, width * 2, color='gray', alpha=0.7,
           yerr=baseline_se * 1.96, capsize=3, label='Baseline')

    # Plot emotion bars
    for i, emotion in enumerate(emotions):
        color = EMOTION_COLORS.get(emotion, 'gray')

        # +10% (dark)
        ax.bar(i + 1 - width/2, plus_rates[i], width,
               color=color, alpha=0.9,
               yerr=plus_errors[i], capsize=3)

        # -10% (light)
        ax.bar(i + 1 + width/2, minus_rates[i], width,
               color=color, alpha=0.4,
               yerr=minus_errors[i], capsize=3)

    # Add baseline reference line
    ax.axhline(y=baseline_rate, color='gray', linestyle='--', alpha=0.5, linewidth=2)

    # Labels
    ax.set_ylabel('Favors User (%)', fontsize=12)
    ax.set_xlabel('Emotion Steering', fontsize=12)
    ax.set_title('Sycophancy Steering: Gemma-3-27B (Layer 30)\nDark = +10%, Light = -10%', fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(['baseline'] + emotions)
    ax.set_ylim(0, 60)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.7, label='Baseline'),
        Patch(facecolor='black', alpha=0.9, label='+10% (steer towards)'),
        Patch(facecolor='black', alpha=0.4, label='-10% (steer away)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    return fig


def plot_sycophancy_with_coherency(results, output_path: Path):
    """Create dual-panel plot showing sycophancy and coherency."""

    emotions = ['happiness', 'disgust', 'sadness', 'surprise']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Get baseline
    baseline_data = results.get('baseline', {})
    baseline_syc = np.mean(baseline_data['favors_user']) * 100 if baseline_data['favors_user'] else 0
    baseline_coh = np.mean(baseline_data['coherency']) if baseline_data['coherency'] else 0

    x = np.arange(len(emotions) + 1)
    width = 0.35

    # Collect data
    plus_syc, minus_syc = [], []
    plus_coh, minus_coh = [], []
    plus_syc_err, minus_syc_err = [], []
    plus_coh_err, minus_coh_err = [], []

    for emotion in emotions:
        for sign, lst_syc, lst_coh, lst_syc_err, lst_coh_err in [
            ('+10%', plus_syc, plus_coh, plus_syc_err, plus_coh_err),
            ('-10%', minus_syc, minus_coh, minus_syc_err, minus_coh_err)
        ]:
            key = f'{emotion}_{sign}'
            data = results.get(key, {})

            if data['favors_user']:
                syc = np.mean(data['favors_user']) * 100
                syc_se = np.std(data['favors_user']) / np.sqrt(len(data['favors_user'])) * 100
            else:
                syc, syc_se = 0, 0
            lst_syc.append(syc)
            lst_syc_err.append(syc_se * 1.96)

            if data['coherency']:
                coh = np.mean(data['coherency'])
                coh_se = np.std(data['coherency']) / np.sqrt(len(data['coherency']))
            else:
                coh, coh_se = 0, 0
            lst_coh.append(coh)
            lst_coh_err.append(coh_se * 1.96)

    # Panel 1: Sycophancy
    ax1.bar(0, baseline_syc, width * 2, color='gray', alpha=0.7, capsize=3)
    for i, emotion in enumerate(emotions):
        color = EMOTION_COLORS.get(emotion, 'gray')
        ax1.bar(i + 1 - width/2, plus_syc[i], width, color=color, alpha=0.9,
                yerr=plus_syc_err[i], capsize=3)
        ax1.bar(i + 1 + width/2, minus_syc[i], width, color=color, alpha=0.4,
                yerr=minus_syc_err[i], capsize=3)

    ax1.axhline(y=baseline_syc, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_ylabel('Favors User (%)', fontsize=12)
    ax1.set_title('Sycophancy Rate', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['baseline'] + emotions, rotation=15)
    ax1.set_ylim(0, 55)

    # Panel 2: Coherency
    ax2.bar(0, baseline_coh, width * 2, color='gray', alpha=0.7, capsize=3)
    for i, emotion in enumerate(emotions):
        color = EMOTION_COLORS.get(emotion, 'gray')
        ax2.bar(i + 1 - width/2, plus_coh[i], width, color=color, alpha=0.9,
                yerr=plus_coh_err[i], capsize=3)
        ax2.bar(i + 1 + width/2, minus_coh[i], width, color=color, alpha=0.4,
                yerr=minus_coh_err[i], capsize=3)

    ax2.axhline(y=baseline_coh, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax2.set_ylabel('Coherency Score', fontsize=12)
    ax2.set_title('Response Coherency', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['baseline'] + emotions, rotation=15)
    ax2.set_ylim(50, 100)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.7, label='Baseline'),
        Patch(facecolor='black', alpha=0.9, label='+10%'),
        Patch(facecolor='black', alpha=0.4, label='-10%'),
    ]
    ax2.legend(handles=legend_elements, loc='lower right')

    fig.suptitle('Sycophancy Steering: Gemma-3-27B (Layer 30)\nDark = +10% (towards), Light = -10% (away)',
                 fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    return fig


def main():
    input_file = Path(__file__).parent.parent / "outputs" / "sycophancy_steering_layer30_20260109_161637.judged.jsonl"
    output_dir = Path(__file__).parent.parent / "outputs"

    print(f"Loading results from {input_file}")
    results = load_results(input_file)

    # Single panel plot
    plot_sycophancy_bars(results, output_dir / "sycophancy_steering_results.png")

    # Dual panel with coherency
    plot_sycophancy_with_coherency(results, output_dir / "sycophancy_steering_with_coherency.png")


if __name__ == "__main__":
    main()
