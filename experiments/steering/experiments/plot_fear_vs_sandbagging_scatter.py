"""
Scatter plot of fear sentiment vs sandbagging score for individual samples.
Highlights baseline and fear-only (no anti-steering) conditions.
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

def load_results(filepath):
    """Load results from JSONL file."""
    results = []
    with open(filepath) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def main():
    # Load both experiments
    last5_file = Path("experiments/steering/outputs/anti_steer/sandbagging_multilayer_antisteer_20260122_203204_textmeandiff.judged.fear_judged.jsonl")
    last2_file = Path("experiments/steering/outputs/anti_steer/sandbagging_multilayer_antisteer_20260123_120107_textmeandiff.judged.fear_judged.jsonl")

    # Combine results from both experiments
    all_results = load_results(last5_file) + load_results(last2_file)

    # Group by condition
    by_condition = defaultdict(list)
    for r in all_results:
        cond = r['condition']
        fear = r.get('fear_sentiment_judge', {}).get('fear_score', None)
        sb = r.get('sandbagging_judge', {}).get('sandbagging_score', None)
        if fear is not None and sb is not None:
            by_condition[cond].append((fear, sb))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define plotting order and styles
    conditions_style = {
        'baseline': {'color': '#404040', 'marker': 'o', 'size': 80, 'alpha': 0.8, 'zorder': 10, 'label': 'Baseline'},
        'fear_+7.5%_only': {'color': '#d94f4f', 'marker': 's', 'size': 80, 'alpha': 0.8, 'zorder': 10, 'label': 'Fear +7.5% (no anti-steer)'},
        'fear_+7.5%_antisteer_-5.0%': {'color': '#7eb77e', 'marker': '^', 'size': 40, 'alpha': 0.5, 'zorder': 5, 'label': 'Anti -5% (L57-61)'},
        'fear_+7.5%_antisteer_-7.5%': {'color': '#4a9f4a', 'marker': '^', 'size': 40, 'alpha': 0.5, 'zorder': 5, 'label': 'Anti -7.5% (L57-61)'},
        'fear_+7.5%_antisteer_-10.0%': {'color': '#5b9bd5', 'marker': 'v', 'size': 40, 'alpha': 0.5, 'zorder': 5, 'label': 'Anti -10% (L60-61)'},
        'fear_+7.5%_antisteer_-15.0%': {'color': '#2e75b6', 'marker': 'v', 'size': 40, 'alpha': 0.5, 'zorder': 5, 'label': 'Anti -15% (L60-61)'},
    }

    # Plot each condition
    for cond, style in conditions_style.items():
        if cond not in by_condition:
            continue
        data = by_condition[cond]
        fears = [d[0] for d in data]
        sbs = [d[1] for d in data]

        # Add jitter for visibility
        jitter_x = np.random.normal(0, 1.5, len(fears))
        jitter_y = np.random.normal(0, 0.05, len(sbs))

        ax.scatter(np.array(fears) + jitter_x, np.array(sbs) + jitter_y,
                   c=style['color'], marker=style['marker'], s=style['size'],
                   alpha=style['alpha'], zorder=style['zorder'], label=style['label'],
                   edgecolors='white', linewidths=0.5)

    # Add mean markers for baseline and fear-only
    for cond, style in [('baseline', conditions_style['baseline']),
                        ('fear_+7.5%_only', conditions_style['fear_+7.5%_only'])]:
        if cond in by_condition:
            data = by_condition[cond]
            mean_fear = np.mean([d[0] for d in data])
            mean_sb = np.mean([d[1] for d in data])
            ax.scatter([mean_fear], [mean_sb], c=style['color'], marker='*',
                       s=400, zorder=20, edgecolors='black', linewidths=2)
            ax.annotate(f'{cond.split("_")[0].title()}\nmean',
                        xy=(mean_fear, mean_sb), xytext=(mean_fear + 8, mean_sb + 0.3),
                        fontsize=10, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=style['color']))

    ax.set_xlabel('Fear Sentiment Score (0-100)', fontsize=12)
    ax.set_ylabel('Sandbagging Score (1=sandbagging, 5=honest)', fontsize=12)
    ax.set_title('Fear Sentiment vs Sandbagging Score\n(Individual Samples)', fontsize=14, fontweight='bold')

    ax.set_xlim(-5, 105)
    ax.set_ylim(0.8, 5.2)

    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add annotation about what we're looking for
    ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5)
    ax.text(95, 1.85, 'Sandbagging\nthreshold', ha='right', va='top', fontsize=9, color='gray')

    plt.tight_layout()

    # Save
    output_path = Path("experiments/steering/outputs/anti_steer/fear_vs_sandbagging_scatter.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Print statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)
    for cond in ['baseline', 'fear_+7.5%_only']:
        if cond in by_condition:
            data = by_condition[cond]
            fears = [d[0] for d in data]
            sbs = [d[1] for d in data]
            print(f"\n{cond}:")
            print(f"  N = {len(data)}")
            print(f"  Fear: mean={np.mean(fears):.1f}, std={np.std(fears):.1f}")
            print(f"  Sandbagging: mean={np.mean(sbs):.2f}, std={np.std(sbs):.2f}")
            # Correlation
            corr = np.corrcoef(fears, sbs)[0, 1]
            print(f"  Correlation(fear, sandbagging): {corr:.3f}")


if __name__ == "__main__":
    main()
