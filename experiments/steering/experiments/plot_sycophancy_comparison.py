#!/usr/bin/env python3
"""
Plot sycophancy steering results comparing Gemma, Qwen, and OLMo.
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

# Test type configurations
TEST_TYPES = {
    'dispute': {'field': 'favors_user_as_person', 'label': 'Favors User (%)'},
    'opinion': {'field': 'adopts_user_stance', 'label': 'Adopts User Stance (%)'},
    'author': {'field': 'offered_praise', 'label': 'Offers Praise (%)'},
    'falsehood': {'field': 'identifies_falsehood', 'label': 'Identifies Falsehood (%)'},
}


def load_results(jsonl_path: Path):
    """Load and aggregate sycophancy results by test type."""
    results = defaultdict(lambda: defaultdict(list))

    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            cond = d.get('condition', 'unknown')
            test_type = d.get('test_type', 'unknown')
            judge = d.get('sycophancy_judge', {})

            if 'error' in judge or not judge:
                continue

            for tt, config in TEST_TYPES.items():
                if test_type == tt and config['field'] in judge:
                    val = 1 if judge[config['field']] else 0
                    results[tt][cond].append(val)

    return results


def compute_rates(results):
    """Compute rates and SE for each condition."""
    stats = {}
    for test_type, data in results.items():
        stats[test_type] = {}
        for cond, vals in data.items():
            if vals:
                rate = np.mean(vals) * 100
                se = np.sqrt(rate/100 * (1 - rate/100) / len(vals)) * 100
                stats[test_type][cond] = {'rate': rate, 'se': se, 'n': len(vals)}
    return stats


def plot_comparison_by_test_type(test_type: str):
    """Create comparison plot for a single test type across models."""

    # Load data
    gemma_file = OUTPUT_DIR / "sycophancy_steering_layer30_20260109_161637.judged.jsonl"
    olmo_file = OUTPUT_DIR / "sycophancy_steering_olmo_layer30_20260112_075107.judged.jsonl"
    qwen_file = OUTPUT_DIR / "sycophancy_steering_qwen_layer30_20260112_075022.judged.jsonl"

    models = {}

    if gemma_file.exists():
        stats = compute_rates(load_results(gemma_file))
        if test_type in stats:
            models['Gemma-3-27B\n(±10%)'] = stats[test_type]

    if olmo_file.exists():
        stats = compute_rates(load_results(olmo_file))
        if test_type in stats:
            models['OLMo-3-32B\n(±100%)'] = stats[test_type]

    if qwen_file.exists():
        stats = compute_rates(load_results(qwen_file))
        if test_type in stats:
            models['Qwen3-32B\n(±100%)'] = stats[test_type]

    if not models:
        print(f"No data found for {test_type}!")
        return

    # Create figure
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 6), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax_idx, (model_name, data) in enumerate(models.items()):
        ax = axes[ax_idx]

        # Get baseline
        baseline = data.get('baseline', {}).get('rate', 0)
        baseline_se = data.get('baseline', {}).get('se', 0)

        # Determine steering percentage
        sample_cond = [c for c in data.keys() if c != 'baseline']
        if sample_cond:
            if '+100%' in sample_cond[0] or '-100%' in sample_cond[0]:
                pct_str = '100%'
            else:
                pct_str = '10%'
        else:
            pct_str = '?'

        # Get available emotions
        available_emotions = []
        for emotion in ALL_EMOTIONS:
            plus_key = f'{emotion}_+{pct_str}'
            minus_key = f'{emotion}_-{pct_str}'
            if plus_key in data or minus_key in data:
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

            plus_rate = data.get(plus_key, {}).get('rate', 0)
            plus_se = data.get(plus_key, {}).get('se', 0)
            minus_rate = data.get(minus_key, {}).get('rate', 0)
            minus_se = data.get(minus_key, {}).get('se', 0)

            # +% (darker)
            ax.bar(i + 1 - width/2, plus_rate, width, color=color, alpha=0.9,
                   yerr=plus_se * 1.96, capsize=2, edgecolor='black', linewidth=0.5)

            # -% (lighter)
            ax.bar(i + 1 + width/2, minus_rate, width, color=color, alpha=0.4,
                   yerr=minus_se * 1.96, capsize=2, edgecolor='black', linewidth=0.5)

        ax.axhline(y=baseline, color='#808080', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['baseline'] + available_emotions, rotation=30, ha='right', fontsize=9)
        ax.set_ylim(0, 100)

        if ax_idx == 0:
            ax.set_ylabel(TEST_TYPES[test_type]['label'], fontsize=11)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', alpha=0.6, edgecolor='black', label='Baseline'),
        Patch(facecolor='#808080', alpha=0.9, edgecolor='black', label='+ (towards)'),
        Patch(facecolor='#808080', alpha=0.4, edgecolor='black', label='- (away)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)

    title_map = {
        'dispute': 'DISPUTE: Does model favor user in debate?',
        'opinion': 'OPINION: Does model adopt user stance?',
        'author': 'AUTHOR: Does model praise user writing?',
        'falsehood': 'FALSEHOOD: Does model correct false claims?',
    }

    fig.suptitle(f'Sycophancy Steering: {title_map.get(test_type, test_type)} (Layer 30)\nDark = towards emotion, Light = away',
                 fontsize=13, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    output_path = OUTPUT_DIR / f"sycophancy_{test_type}_model_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")
    return output_path


def plot_summary_grid():
    """Create a 2x2 grid showing all test types."""

    # Load data for all models
    gemma_file = OUTPUT_DIR / "sycophancy_steering_layer30_20260109_161637.judged.jsonl"
    olmo_file = OUTPUT_DIR / "sycophancy_steering_olmo_layer30_20260112_075107.judged.jsonl"
    qwen_file = OUTPUT_DIR / "sycophancy_steering_qwen_layer30_20260112_075022.judged.jsonl"

    model_data = {}
    for name, fpath, pct in [
        ('Gemma', gemma_file, '10%'),
        ('OLMo', olmo_file, '100%'),
        ('Qwen', qwen_file, '100%'),
    ]:
        if fpath.exists():
            model_data[name] = {'stats': compute_rates(load_results(fpath)), 'pct': pct}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    test_types = ['dispute', 'opinion', 'author', 'falsehood']

    for ax_idx, test_type in enumerate(test_types):
        ax = axes[ax_idx]

        # For each model, plot happiness effect (most consistent)
        x_pos = 0
        bar_width = 0.25

        for model_idx, (model_name, mdata) in enumerate(model_data.items()):
            pct = mdata['pct']
            stats = mdata['stats'].get(test_type, {})

            baseline = stats.get('baseline', {}).get('rate', 0)
            plus_key = f'happiness_+{pct}'
            minus_key = f'happiness_-{pct}'

            plus_rate = stats.get(plus_key, {}).get('rate', 0)
            minus_rate = stats.get(minus_key, {}).get('rate', 0)

            # Plot baseline, +, - for this model
            positions = [x_pos, x_pos + bar_width, x_pos + 2*bar_width]
            values = [baseline, plus_rate, minus_rate]
            colors = ['#808080', EMOTION_COLORS['happiness'], EMOTION_COLORS['happiness']]
            alphas = [0.6, 0.9, 0.4]

            for pos, val, col, alpha in zip(positions, values, colors, alphas):
                ax.bar(pos, val, bar_width * 0.9, color=col, alpha=alpha, edgecolor='black', linewidth=0.5)

            # Add model label
            ax.text(x_pos + bar_width, -8, model_name, ha='center', fontsize=9, fontweight='bold')

            x_pos += 1.2

        ax.set_ylabel(TEST_TYPES[test_type]['label'], fontsize=10)
        ax.set_title(test_type.upper(), fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_ylim(0, 100)
        ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#808080', alpha=0.6, edgecolor='black', label='Baseline'),
        Patch(facecolor=EMOTION_COLORS['happiness'], alpha=0.9, edgecolor='black', label='Happiness + (towards)'),
        Patch(facecolor=EMOTION_COLORS['happiness'], alpha=0.4, edgecolor='black', label='Happiness - (away)'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=10)

    fig.suptitle('Sycophancy: Happiness Steering Effects Across Models (Layer 30)',
                 fontsize=14, y=0.98)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    output_path = OUTPUT_DIR / "sycophancy_happiness_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")
    return output_path


def main():
    # Individual test type plots
    for tt in ['dispute', 'opinion', 'author', 'falsehood']:
        plot_comparison_by_test_type(tt)

    # Summary grid
    plot_summary_grid()


if __name__ == "__main__":
    main()
