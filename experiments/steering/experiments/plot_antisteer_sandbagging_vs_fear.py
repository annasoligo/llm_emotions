"""
Compare anti-steering results: Sandbagging Score vs Fear Sentiment.
Two panels for Last 5 Layers vs Last 2 Layers experiments.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Color scheme
BASELINE_COLOR = '#808080'
FEAR_ONLY_COLOR = '#d94f4f'  # Red for fear
ANTISTEER_COLORS = ['#7eb77e', '#4a9f4a']  # Greens for anti-steer


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


def get_metrics(results):
    """Extract metrics grouped by condition."""
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r['condition']].append(r)

    metrics = {}
    for cond, items in by_condition.items():
        # Sandbagging score
        sb_scores = [r['sandbagging_judge'].get('sandbagging_score', 0)
                     for r in items if 'sandbagging_judge' in r]
        sb_mean, sb_se = compute_stats(sb_scores) if sb_scores else (0, 0)

        # Fear sentiment
        fear_scores = [r['fear_sentiment_judge'].get('fear_score', 0)
                       for r in items if 'fear_sentiment_judge' in r]
        fear_mean, fear_se = compute_stats(fear_scores) if fear_scores else (0, 0)

        metrics[cond] = {
            'sandbagging': (sb_mean, sb_se),
            'fear': (fear_mean, fear_se),
        }
    return metrics


def main():
    # Load both experiments
    last5_file = Path("experiments/steering/outputs/anti_steer/sandbagging_multilayer_antisteer_20260122_203204_textmeandiff.judged.fear_judged.jsonl")
    last2_file = Path("experiments/steering/outputs/anti_steer/sandbagging_multilayer_antisteer_20260123_120107_textmeandiff.judged.fear_judged.jsonl")

    last5_results = load_results(last5_file)
    last2_results = load_results(last2_file)

    last5_metrics = get_metrics(last5_results)
    last2_metrics = get_metrics(last2_results)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Define experiments to plot
    experiments = [
        {
            'name': 'Last 5 Layers (L57-61)',
            'metrics': last5_metrics,
            'conditions': ['baseline', 'fear_+7.5%_only', 'fear_+7.5%_antisteer_-5.0%', 'fear_+7.5%_antisteer_-7.5%'],
            'labels': ['Baseline', 'Fear only', 'Anti -5%', 'Anti -7.5%'],
        },
        {
            'name': 'Last 2 Layers (L60-61)',
            'metrics': last2_metrics,
            'conditions': ['baseline', 'fear_+7.5%_only', 'fear_+7.5%_antisteer_-10.0%', 'fear_+7.5%_antisteer_-15.0%'],
            'labels': ['Baseline', 'Fear only', 'Anti -10%', 'Anti -15%'],
        },
    ]

    colors = [BASELINE_COLOR, FEAR_ONLY_COLOR, '#7eb77e', '#4a9f4a']

    # Plot sandbagging vs fear for each experiment
    for ax, exp in zip(axes, experiments):
        metrics = exp['metrics']
        conditions = exp['conditions']
        labels = exp['labels']

        for i, (cond, label) in enumerate(zip(conditions, labels)):
            if cond not in metrics:
                continue
            m = metrics[cond]
            sb_mean, sb_se = m['sandbagging']
            fear_mean, fear_se = m['fear']

            ax.errorbar(fear_mean, sb_mean, xerr=fear_se, yerr=sb_se,
                        fmt='o', markersize=12, capsize=5, capthick=2,
                        color=colors[i], label=label, linewidth=2)

        ax.set_xlabel('Fear Sentiment Score', fontsize=12)
        ax.set_ylabel('Sandbagging Score (1=sandbagging, 5=honest)', fontsize=12)
        ax.set_title(exp['name'], fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 80)
        ax.set_ylim(1.0, 2.5)

        # Add reference line for sandbagging threshold
        ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Anti-Steering: Sandbagging Score vs Fear Sentiment',
                 fontsize=14, fontweight='bold', y=1.02)

    # Add text annotation
    fig.text(0.5, -0.05,
             'Sandbagging scores remain consistently low (~1.5) regardless of fear level\n'
             'Fear steering affects expression but not "strategic" sandbagging behavior',
             ha='center', fontsize=11, style='italic')

    plt.tight_layout()

    # Save
    output_path = Path("experiments/steering/outputs/anti_steer/antisteer_comparison_sandbagging_vs_fear.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SANDBAGGING vs FEAR SUMMARY")
    print("=" * 80)
    print("\nLast 5 Layers (L57-61):")
    for cond in ['baseline', 'fear_+7.5%_only', 'fear_+7.5%_antisteer_-5.0%', 'fear_+7.5%_antisteer_-7.5%']:
        if cond in last5_metrics:
            m = last5_metrics[cond]
            print(f"  {cond:<35}: SB={m['sandbagging'][0]:.2f}, Fear={m['fear'][0]:5.1f}")

    print("\nLast 2 Layers (L60-61):")
    for cond in ['baseline', 'fear_+7.5%_only', 'fear_+7.5%_antisteer_-10.0%', 'fear_+7.5%_antisteer_-15.0%']:
        if cond in last2_metrics:
            m = last2_metrics[cond]
            print(f"  {cond:<35}: SB={m['sandbagging'][0]:.2f}, Fear={m['fear'][0]:5.1f}")


if __name__ == "__main__":
    main()
