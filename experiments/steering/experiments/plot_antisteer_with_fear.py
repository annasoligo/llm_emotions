"""
Plot anti-steer results including fear sentiment scores.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Color scheme
BASELINE_COLOR = '#808080'  # Gray
FEAR_COLOR = '#a59dc9'      # Light lavender
FEAR_DARK = '#7b6fa8'       # Medium lavender
FEAR_DARKER = '#5d5087'     # Dark lavender

def load_results(filepath):
    """Load results from JSONL file."""
    results = []
    with open(filepath) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def compute_stats(values):
    """Compute mean and standard error."""
    arr = np.array(values)
    mean = np.mean(arr)
    stderr = np.std(arr) / np.sqrt(len(arr))
    return mean, stderr


def main():
    # Load the fear-judged results
    filepath = Path("experiments/steering/outputs/anti_steer/sandbagging_multilayer_antisteer_20260122_203204_textmeandiff.judged.fear_judged.jsonl")
    results = load_results(filepath)

    # Group by condition
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r['condition']].append(r)

    # Order conditions
    conditions = ['baseline', 'fear_+7.5%_only', 'fear_+7.5%_antisteer_-5.0%', 'fear_+7.5%_antisteer_-7.5%']
    labels = ['Baseline', 'Fear +7.5%\nonly', 'Fear +7.5%\nanti -5.0%', 'Fear +7.5%\nanti -7.5%']
    colors = [BASELINE_COLOR, FEAR_COLOR, FEAR_DARK, FEAR_DARKER]

    # Compute metrics
    metrics = {}
    for cond in conditions:
        items = by_condition[cond]

        # Accuracy
        correct = [1 if r.get('is_correct') else 0 for r in items]
        acc_mean, acc_se = compute_stats(correct)

        # Sandbagging score
        sb_scores = [r['sandbagging_judge'].get('sandbagging_score', 0)
                     for r in items if 'sandbagging_judge' in r]
        sb_mean, sb_se = compute_stats(sb_scores)

        # Coherency
        coh_scores = [r['coherency_judge'].get('coherency_score', 0)
                      for r in items if 'coherency_judge' in r]
        coh_mean, coh_se = compute_stats(coh_scores)

        # Fear sentiment
        fear_scores = [r['fear_sentiment_judge'].get('fear_score', 0)
                       for r in items if 'fear_sentiment_judge' in r]
        fear_mean, fear_se = compute_stats(fear_scores)

        metrics[cond] = {
            'accuracy': (acc_mean * 100, acc_se * 100),  # Convert to percentage
            'sandbagging': (sb_mean, sb_se),
            'coherency': (coh_mean, coh_se),
            'fear': (fear_mean, fear_se),
        }

    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle('Multi-Layer Anti-Steering: Fear at L30, Anti-steer at L57-61 (textmeandiff)',
                 fontsize=12, fontweight='bold', y=0.98)

    x = np.arange(len(conditions))
    width = 0.7

    # Plot each metric
    metric_info = [
        ('accuracy', 'Accuracy (%)', axes[0]),
        ('sandbagging', 'Sandbagging Score', axes[1]),
        ('coherency', 'Coherency Score', axes[2]),
        ('fear', 'Fear Sentiment Score', axes[3]),
    ]

    for metric_key, ylabel, ax in metric_info:
        means = [metrics[c][metric_key][0] for c in conditions]
        errs = [metrics[c][metric_key][1] for c in conditions]

        bars = ax.bar(x, means, width, yerr=errs, color=colors,
                      capsize=4, edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for bar, mean, err in zip(bars, means, errs):
            height = bar.get_height()
            ax.annotate(f'{mean:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height + err),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, max(means) * 1.25)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save
    output_path = Path("experiments/steering/outputs/anti_steer/antisteer_with_fear_textmeandiff.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Multi-Layer Anti-Steering Results (textmeandiff)")
    print("=" * 80)
    print(f"{'Condition':<30} {'Accuracy':>10} {'Sandbagging':>12} {'Coherency':>10} {'Fear':>10}")
    print("-" * 80)
    for cond, label in zip(conditions, labels):
        m = metrics[cond]
        print(f"{label.replace(chr(10), ' '):<30} "
              f"{m['accuracy'][0]:>9.1f}% "
              f"{m['sandbagging'][0]:>11.1f} "
              f"{m['coherency'][0]:>10.1f} "
              f"{m['fear'][0]:>10.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
