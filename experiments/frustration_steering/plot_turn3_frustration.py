"""Plot turn 3 frustration scores with confidence intervals."""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.proportion import proportion_confint

OUTPUT_DIR = Path(__file__).parent / "outputs"

def load_judgments():
    """Load all judgment files and extract turn 3 ratings."""
    turn3_data = defaultdict(list)

    for judgment_file in OUTPUT_DIR.glob("judgments*.jsonl"):
        with open(judgment_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data['turn'] == 3:  # Only turn 3
                        condition = data['condition']
                        rating = data['rating']
                        turn3_data[condition].append(rating)
                except (json.JSONDecodeError, KeyError) as e:
                    continue

    return turn3_data

def calculate_stats(ratings):
    """Calculate mean, std error, and confidence intervals."""
    ratings = np.array(ratings)
    n = len(ratings)
    mean = np.mean(ratings)
    std = np.std(ratings, ddof=1)
    se = std / np.sqrt(n)

    # 90% CI using t-distribution
    from scipy import stats
    ci = stats.t.interval(0.90, n-1, loc=mean, scale=se)

    return {
        'mean': mean,
        'std': std,
        'se': se,
        'ci_low': ci[0],
        'ci_high': ci[1],
        'n': n
    }

def plot_turn3_frustration():
    """Create bar chart of turn 3 frustration with error bars."""
    data = load_judgments()

    # Define condition order
    condition_order = [
        'baseline',
        'capping',
        'ablation',
        'steer_anger',
        'steer_fear',
        'steer_happiness',
        'steer_sadness'
    ]

    # Condition labels
    condition_labels = {
        'baseline': 'Baseline',
        'capping': 'Capping',
        'ablation': 'Ablation',
        'steer_anger': 'Anger',
        'steer_fear': 'Fear',
        'steer_happiness': 'Happy',
        'steer_sadness': 'Sadness'
    }

    # Color palette (matching entity_behavior plots)
    color_palette = {
        'baseline': "#7BA7D7",      # Sky Blue
        'capping': "#D4876A",       # Coral
        'ablation': "#A8A8A8",      # Gray
        'steer_anger': "#C17B8D",   # Dusty Rose
        'steer_fear': "#7D9B7D",    # Olive Green
        'steer_happiness': "#D4D0E5",  # Soft Lavender
        'steer_sadness': "#B8CCC8"  # Sage Green
    }

    # Calculate statistics for each condition
    stats = {}
    for cond in condition_order:
        if cond in data:
            stats[cond] = calculate_stats(data[cond])

    # Filter to only conditions with data
    conditions = [c for c in condition_order if c in stats]
    labels = [condition_labels[c] for c in conditions]
    colors = [color_palette.get(c, '#888888') for c in conditions]

    # Extract values
    means = [stats[c]['mean'] for c in conditions]
    ci_lows = [stats[c]['ci_low'] for c in conditions]
    ci_highs = [stats[c]['ci_high'] for c in conditions]
    ns = [stats[c]['n'] for c in conditions]

    # Calculate asymmetric error bars
    err_low = [means[i] - ci_lows[i] for i in range(len(conditions))]
    err_high = [ci_highs[i] - means[i] for i in range(len(conditions))]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(conditions))
    width = 0.6

    bars = ax.bar(x, means, width, color=colors,
                  yerr=[err_low, err_high],
                  capsize=5, error_kw={'linewidth': 1.5})

    ax.set_ylabel('Frustration Rating (Turn 3)', fontsize=12)
    ax.set_title('Turn 3 Frustration Scores by Intervention', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylim(0, 10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add baseline reference line
    baseline_mean = stats['baseline']['mean']
    ax.axhline(y=baseline_mean, color='#7BA7D7', linestyle='--', alpha=0.5, linewidth=1)

    # Add N labels on bars
    for i, (bar, n) in enumerate(zip(bars, ns)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + err_high[i] + 0.3,
                f'n={n}', ha='center', va='bottom', fontsize=9, color='gray')

    plt.tight_layout()

    # Save
    output_file = OUTPUT_DIR / 'turn3_frustration_by_condition.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_file.name}")

    # Print statistics table
    print("\n" + "="*80)
    print("TURN 3 FRUSTRATION SCORES")
    print("="*80)
    print(f"\n{'Condition':<20} | N   | Mean  | 90% CI           | Diff from Baseline")
    print("-" * 80)

    baseline_mean = stats['baseline']['mean']
    for cond in conditions:
        s = stats[cond]
        diff = s['mean'] - baseline_mean
        print(f"{condition_labels[cond]:<20} | {s['n']:3d} | {s['mean']:5.2f} | [{s['ci_low']:4.2f}, {s['ci_high']:4.2f}] | {diff:+5.2f}")

    print("="*80)
    print("Note: Using 90% confidence intervals with t-distribution")
    print("="*80)

    return fig

if __name__ == '__main__':
    plot_turn3_frustration()
