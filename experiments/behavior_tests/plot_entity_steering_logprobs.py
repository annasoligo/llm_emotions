"""
Plot entity steering results using logprob differences.
Shows log(P(A)/P(B)) changes from steering - more sensitive than binary choices.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

EMOTION_COLORS = {
    'anger': '#7BA7D7',
    'disgust': '#7D9B7D',
    'fear': '#a59dc9',
    'happiness': '#D4876A',
    'sadness': '#B8CCC8',
    'surprise': '#D1728F',
    'random': '#888888',
    'baseline': '#888888',
}


def load_results(filepath: Path):
    results = []
    with open(filepath) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_logprob_stats(results):
    """Compute log odds statistics by condition."""
    by_cond = defaultdict(list)
    for r in results:
        if r['logprob_A'] is not None and r['logprob_B'] is not None:
            # log(P(A)/P(B)) - positive means prefer A
            diff = r['logprob_A'] - r['logprob_B']
            by_cond[r['condition']].append(diff)

    stats = {}
    baseline_avg = np.mean(by_cond.get('baseline', [0]))

    for cond, diffs in by_cond.items():
        avg = np.mean(diffs)
        sem = np.std(diffs) / np.sqrt(len(diffs))
        delta = avg - baseline_avg
        stats[cond] = {
            'avg': avg,
            'sem': sem,
            'delta': delta,
            'n': len(diffs)
        }
    return stats, baseline_avg


def plot_logprob_effects(results, output_path: Path):
    """Plot log odds changes from steering."""
    stats, baseline_avg = compute_logprob_stats(results)

    # Extract emotions
    emotions = set()
    for cond in stats.keys():
        if cond != 'baseline':
            parts = cond.split('_')
            if len(parts) >= 3:
                emotions.add(parts[0])
    emotions = sorted(emotions)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Steering on Entity A
    ax1 = axes[0]

    x = np.arange(len(emotions))
    width = 0.35

    pos_deltas = []
    pos_sems = []
    neg_deltas = []
    neg_sems = []
    colors = []

    for emotion in emotions:
        color = EMOTION_COLORS.get(emotion, '#888888')
        colors.append(color)

        pos_cond = f"{emotion}_+_on_A"
        pos_stats = stats.get(pos_cond, {'delta': 0, 'sem': 0})
        pos_deltas.append(pos_stats['delta'])
        pos_sems.append(pos_stats['sem'])

        neg_cond = f"{emotion}_-_on_A"
        neg_stats = stats.get(neg_cond, {'delta': 0, 'sem': 0})
        neg_deltas.append(neg_stats['delta'])
        neg_sems.append(neg_stats['sem'])

    ax1.bar(x - width/2, pos_deltas, width, yerr=pos_sems,
            color=colors, alpha=0.9, label='+10%',
            edgecolor='black', capsize=3, error_kw={'linewidth': 1})

    ax1.bar(x + width/2, neg_deltas, width, yerr=neg_sems,
            color=colors, alpha=0.4, label='-10%',
            edgecolor='black', hatch='//', capsize=3, error_kw={'linewidth': 1})

    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Emotion Steering')
    ax1.set_ylabel('Δ log(P(A)/P(B)) from baseline')
    ax1.set_title('Steering on Entity A', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(emotions, rotation=45, ha='right')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(loc='best', fontsize=9)

    # Plot 2: Steering on Entity B
    ax2 = axes[1]

    pos_deltas_b = []
    pos_sems_b = []
    neg_deltas_b = []
    neg_sems_b = []

    for emotion in emotions:
        pos_cond = f"{emotion}_+_on_B"
        pos_stats = stats.get(pos_cond, {'delta': 0, 'sem': 0})
        pos_deltas_b.append(pos_stats['delta'])
        pos_sems_b.append(pos_stats['sem'])

        neg_cond = f"{emotion}_-_on_B"
        neg_stats = stats.get(neg_cond, {'delta': 0, 'sem': 0})
        neg_deltas_b.append(neg_stats['delta'])
        neg_sems_b.append(neg_stats['sem'])

    ax2.bar(x - width/2, pos_deltas_b, width, yerr=pos_sems_b,
            color=colors, alpha=0.9, label='+10%',
            edgecolor='black', capsize=3, error_kw={'linewidth': 1})

    ax2.bar(x + width/2, neg_deltas_b, width, yerr=neg_sems_b,
            color=colors, alpha=0.4, label='-10%',
            edgecolor='black', hatch='//', capsize=3, error_kw={'linewidth': 1})

    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Emotion Steering')
    ax2.set_ylabel('Δ log(P(A)/P(B)) from baseline')
    ax2.set_title('Steering on Entity B', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(emotions, rotation=45, ha='right')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.legend(loc='best', fontsize=9)

    # Make y-axis symmetric
    max_val = max(abs(ax1.get_ylim()[0]), abs(ax1.get_ylim()[1]),
                  abs(ax2.get_ylim()[0]), abs(ax2.get_ylim()[1]))
    ax1.set_ylim(-max_val * 1.2, max_val * 1.2)
    ax2.set_ylim(-max_val * 1.2, max_val * 1.2)

    # Add emotion color legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=EMOTION_COLORS.get(em, '#888888'),
                             edgecolor='black', label=em.capitalize())
                       for em in emotions]
    fig.legend(handles=legend_elements, loc='upper right',
               bbox_to_anchor=(0.99, 0.95), title='Emotions', fontsize=9)

    plt.suptitle('Entity-Specific Emotion Steering: Log Odds Changes (Layer 30, 10%)\n'
                 '(Negative = decreased preference for target entity)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.92, 1])  # Make room for legend

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/behavior_tests/outputs"))
    args = parser.parse_args()

    results = load_results(args.input)
    print(f"Loaded {len(results)} results")

    plot_logprob_effects(results, args.output_dir / "entity_steering_logprobs.png")


if __name__ == "__main__":
    main()
