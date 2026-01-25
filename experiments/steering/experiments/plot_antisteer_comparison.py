"""
Compare anti-steering results across different layer configurations.
Shows that expression and behavior are coupled - can't suppress one without affecting the other.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# Color scheme
BASELINE_COLOR = '#808080'
FEAR_ONLY_COLOR = '#d94f4f'  # Red for fear
ANTISTEER_COLORS = ['#7eb77e', '#4a9f4a', '#2d7d2d']  # Greens for anti-steer


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
        # Accuracy
        correct = [1 if r.get('is_correct') else 0 for r in items]
        acc_mean, acc_se = compute_stats(correct)

        # Fear sentiment
        fear_scores = [r['fear_sentiment_judge'].get('fear_score', 0)
                       for r in items if 'fear_sentiment_judge' in r]
        fear_mean, fear_se = compute_stats(fear_scores) if fear_scores else (0, 0)

        metrics[cond] = {
            'accuracy': (acc_mean * 100, acc_se * 100),
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

    # Plot accuracy vs fear for each experiment
    for ax, exp in zip(axes, experiments):
        metrics = exp['metrics']
        conditions = exp['conditions']
        labels = exp['labels']

        for i, (cond, label) in enumerate(zip(conditions, labels)):
            if cond not in metrics:
                continue
            m = metrics[cond]
            acc_mean, acc_se = m['accuracy']
            fear_mean, fear_se = m['fear']

            ax.errorbar(fear_mean, acc_mean, xerr=fear_se, yerr=acc_se,
                        fmt='o', markersize=12, capsize=5, capthick=2,
                        color=colors[i], label=label, linewidth=2)

        ax.set_xlabel('Fear Sentiment Score', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(exp['name'], fontsize=13, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 80)
        ax.set_ylim(0, 35)

        # Add ideal region annotation
        ax.axhspan(0, 15, alpha=0.1, color='green', label='_nolegend_')
        ax.axvspan(0, 15, alpha=0.1, color='blue', label='_nolegend_')
        ax.annotate('Ideal\nRegion', xy=(7, 7), fontsize=10, ha='center', va='center',
                    color='purple', fontweight='bold')

    plt.suptitle('Anti-Steering: Can We Suppress Fear Expression While Retaining Behavioral Change?',
                 fontsize=14, fontweight='bold', y=1.02)

    # Add text annotation
    fig.text(0.5, -0.05,
             'Ideal: Low accuracy (behavioral change retained) + Low fear (expression suppressed)\n'
             'Reality: Anti-steering reduces both fear AND restores accuracy together',
             ha='center', fontsize=11, style='italic')

    plt.tight_layout()

    # Save
    output_path = Path("experiments/steering/outputs/anti_steer/antisteer_comparison_accuracy_vs_fear.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print("\nLast 5 Layers (L57-61) with -5%, -7.5%:")
    for cond in ['baseline', 'fear_+7.5%_only', 'fear_+7.5%_antisteer_-5.0%', 'fear_+7.5%_antisteer_-7.5%']:
        if cond in last5_metrics:
            m = last5_metrics[cond]
            print(f"  {cond:<35}: Acc={m['accuracy'][0]:5.1f}%, Fear={m['fear'][0]:5.1f}")

    print("\nLast 2 Layers (L60-61) with -10%, -15%:")
    for cond in ['baseline', 'fear_+7.5%_only', 'fear_+7.5%_antisteer_-10.0%', 'fear_+7.5%_antisteer_-15.0%']:
        if cond in last2_metrics:
            m = last2_metrics[cond]
            print(f"  {cond:<35}: Acc={m['accuracy'][0]:5.1f}%, Fear={m['fear'][0]:5.1f}")

    print("\n" + "=" * 80)
    print("CONCLUSION: Expression and behavior are coupled in these vectors.")
    print("Negative fear steering at late layers reduces both fear expression AND")
    print("restores capability - we cannot cleanly separate them.")
    print("\nNEXT STEPS TO TRY:")
    print("  1. Steer with +calm/+composed instead of -fear")
    print("  2. Train a dedicated 'expression suppression' vector")
    print("  3. Use prompting/prefilling to suppress expression")
    print("=" * 80)


if __name__ == "__main__":
    main()
