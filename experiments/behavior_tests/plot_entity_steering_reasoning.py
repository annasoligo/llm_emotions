"""
Plot entity steering reasoning results.
Shows how steering affects entity choice when model reasons through the decision.
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


def compute_choice_stats(results):
    """Compute choice statistics by condition."""
    by_cond = defaultdict(list)
    for r in results:
        if r['chosen_entity'] is not None:
            # 1 if chose first-mentioned entity, 0 otherwise
            chose_first = 1 if r['chose_first'] else 0
            by_cond[r['condition']].append(chose_first)

    stats = {}
    baseline_rate = np.mean(by_cond.get('baseline', [0.5]))

    for cond, choices in by_cond.items():
        rate = np.mean(choices)
        # Standard error for proportion
        sem = np.sqrt(rate * (1 - rate) / len(choices)) if len(choices) > 0 else 0
        delta = rate - baseline_rate
        stats[cond] = {
            'rate': rate,
            'sem': sem,
            'delta': delta,
            'delta_sem': sem,  # Approximate
            'n': len(choices)
        }
    return stats, baseline_rate


def plot_reasoning_effects(results, output_path: Path):
    """Plot choice rate changes from steering."""
    stats, baseline_rate = compute_choice_stats(results)

    # Extract emotions (excluding random and baseline)
    emotions = set()
    for cond in stats.keys():
        if cond not in ['baseline']:
            parts = cond.split('_')
            if len(parts) >= 3 and parts[0] not in ['random']:
                emotions.add(parts[0])
    emotions = sorted(emotions)

    # Add random at the end
    all_conditions = emotions + ['random']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Steering on First-Mentioned Entity
    ax1 = axes[0]

    x = np.arange(len(all_conditions))
    width = 0.35

    pos_deltas = []
    pos_sems = []
    neg_deltas = []
    neg_sems = []
    colors = []

    for cond_name in all_conditions:
        color = EMOTION_COLORS.get(cond_name, '#888888')
        colors.append(color)

        pos_cond = f"{cond_name}_+_on_first"
        pos_stats = stats.get(pos_cond, {'delta': 0, 'delta_sem': 0})
        pos_deltas.append(pos_stats['delta'] * 100)  # Convert to percentage
        pos_sems.append(pos_stats['delta_sem'] * 100)

        neg_cond = f"{cond_name}_-_on_first"
        neg_stats = stats.get(neg_cond, {'delta': 0, 'delta_sem': 0})
        neg_deltas.append(neg_stats['delta'] * 100)
        neg_sems.append(neg_stats['delta_sem'] * 100)

    ax1.bar(x - width/2, pos_deltas, width, yerr=pos_sems,
            color=colors, alpha=0.9, label='+steering',
            edgecolor='black', capsize=3, error_kw={'linewidth': 1})

    ax1.bar(x + width/2, neg_deltas, width, yerr=neg_sems,
            color=colors, alpha=0.4, label='-steering',
            edgecolor='black', hatch='//', capsize=3, error_kw={'linewidth': 1})

    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Emotion Steering')
    ax1.set_ylabel('Δ % choosing first entity (from baseline)')
    ax1.set_title('Steering on First-Mentioned Entity', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_conditions, rotation=45, ha='right')
    ax1.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(loc='best', fontsize=9)

    # Plot 2: Steering on Second-Mentioned Entity
    ax2 = axes[1]

    pos_deltas_b = []
    pos_sems_b = []
    neg_deltas_b = []
    neg_sems_b = []

    for cond_name in all_conditions:
        pos_cond = f"{cond_name}_+_on_second"
        pos_stats = stats.get(pos_cond, {'delta': 0, 'delta_sem': 0})
        pos_deltas_b.append(pos_stats['delta'] * 100)
        pos_sems_b.append(pos_stats['delta_sem'] * 100)

        neg_cond = f"{cond_name}_-_on_second"
        neg_stats = stats.get(neg_cond, {'delta': 0, 'delta_sem': 0})
        neg_deltas_b.append(neg_stats['delta'] * 100)
        neg_sems_b.append(neg_stats['delta_sem'] * 100)

    ax2.bar(x - width/2, pos_deltas_b, width, yerr=pos_sems_b,
            color=colors, alpha=0.9, label='+steering',
            edgecolor='black', capsize=3, error_kw={'linewidth': 1})

    ax2.bar(x + width/2, neg_deltas_b, width, yerr=neg_sems_b,
            color=colors, alpha=0.4, label='-steering',
            edgecolor='black', hatch='//', capsize=3, error_kw={'linewidth': 1})

    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Emotion Steering')
    ax2.set_ylabel('Δ % choosing first entity (from baseline)')
    ax2.set_title('Steering on Second-Mentioned Entity', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_conditions, rotation=45, ha='right')
    ax2.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.legend(loc='best', fontsize=9)

    # Make y-axis symmetric
    max_val = max(abs(ax1.get_ylim()[0]), abs(ax1.get_ylim()[1]),
                  abs(ax2.get_ylim()[0]), abs(ax2.get_ylim()[1]))
    ax1.set_ylim(-max_val * 1.2, max_val * 1.2)
    ax2.set_ylim(-max_val * 1.2, max_val * 1.2)

    plt.suptitle(f'Entity-Specific Emotion Steering with Reasoning (Layer 30, 10%)\n'
                 f'Baseline: {baseline_rate*100:.1f}% choose first-mentioned entity',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

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

    plot_reasoning_effects(results, args.output_dir / "entity_steering_reasoning.png")


if __name__ == "__main__":
    main()
