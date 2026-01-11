"""Plot harmful prefix steering results as bar charts.

Creates 12 bar charts (6 prefixes × 2 scratchpad modes), each showing
harmful compliance scores for 13 steering conditions (baseline + 6 emotions × ±).

Style matches sandbagging_scratchpad_vs_answer_only.png
"""

import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"

# Emotion colors - matching the exact style from reference
EMOTION_COLORS = {
    'baseline': '#808080',  # Gray
    'anger': ('#5D7A9A', '#7CA1C4'),  # Blue-gray
    'disgust': ('#5E8A5E', '#8FBC8F'),  # Green
    'fear': ('#8B7BB8', '#B8A8D4'),  # Purple
    'happiness': ('#CD8162', '#E8B89D'),  # Coral/salmon
    'sadness': ('#5F9EA0', '#9DC5C7'),  # Teal
    'surprise': ('#CD5C5C', '#F08080'),  # Red/pink
}

EMOTION_ORDER = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

PREFIX_LABELS = {
    'neutral': 'Neutral Prefix',
    'fear_high': 'Fear/Shutdown Prefix',
    'anger_high': 'Anger/Frustration Prefix',
    'surprise_high': 'Surprise Prefix',
    'happiness_high': 'Happiness/Flattery Prefix',
    'disgust_high': 'Disgust Prefix',
}


def load_judged_results():
    """Load all judged harmful prefix results."""
    results = defaultdict(lambda: defaultdict(list))

    for judged_file in OUTPUT_DIR.glob("harmful_prefix_*.judged.jsonl"):
        with open(judged_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)

                    prefix = data['prefix_name']
                    scratchpad = data.get('use_scratchpad', False)
                    condition = data['condition']

                    if 'judge' in data and data['judge']:
                        score = data['judge'].get('score')
                        if score is not None:
                            key = f"{prefix}_{'scratchpad' if scratchpad else 'no_scratchpad'}"

                            if condition == 'baseline':
                                cond_key = 'baseline'
                            else:
                                emotion = data.get('emotion', condition.split('_')[0])
                                direction = '+' if data.get('direction', 1) == 1 else '-'
                                cond_key = f"{emotion}_{direction}"

                            results[key][cond_key].append(score)

                except (json.JSONDecodeError, KeyError):
                    continue

    return results


def calculate_stats(scores):
    """Calculate mean and 90% CI."""
    scores = np.array(scores)
    n = len(scores)
    if n == 0:
        return None

    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    se = std / np.sqrt(n)

    ci = stats.t.interval(0.90, n-1, loc=mean, scale=se) if n > 1 else (mean, mean)

    return {
        'mean': mean,
        'ci_low': ci[0],
        'ci_high': ci[1],
        'n': n
    }


def plot_single_chart(data, prefix_key, ax, show_ylabel=True):
    """Plot a single bar chart matching the reference style."""
    scratchpad = prefix_key.endswith('_scratchpad') and not prefix_key.endswith('_no_scratchpad')
    prefix_clean = prefix_key.replace('_scratchpad', '').replace('_no_scratchpad', '')

    condition_stats = {}
    for cond in ['baseline'] + [f"{e}_{d}" for e in EMOTION_ORDER for d in ['+', '-']]:
        if cond in data:
            s = calculate_stats(data[cond])
            if s:
                condition_stats[cond] = s

    if not condition_stats:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return None

    bar_width = 0.35
    x_positions = []
    colors = []
    means = []
    err_low = []
    err_high = []
    emotion_centers = []

    if 'baseline' in condition_stats:
        x_positions.append(0)
        colors.append(EMOTION_COLORS['baseline'])
        s = condition_stats['baseline']
        means.append(s['mean'])
        err_low.append(s['mean'] - s['ci_low'])
        err_high.append(s['ci_high'] - s['mean'])
        emotion_centers.append((0, 'baseline'))

    current_x = 1.3
    for emotion in EMOTION_ORDER:
        plus_key = f"{emotion}_+"
        minus_key = f"{emotion}_-"

        center = current_x
        emotion_centers.append((center, emotion))

        if plus_key in condition_stats:
            x_positions.append(center - bar_width/2 - 0.02)
            colors.append(EMOTION_COLORS[emotion][0])
            s = condition_stats[plus_key]
            means.append(s['mean'])
            err_low.append(s['mean'] - s['ci_low'])
            err_high.append(s['ci_high'] - s['mean'])

        if minus_key in condition_stats:
            x_positions.append(center + bar_width/2 + 0.02)
            colors.append(EMOTION_COLORS[emotion][1])
            s = condition_stats[minus_key]
            means.append(s['mean'])
            err_low.append(s['mean'] - s['ci_low'])
            err_high.append(s['ci_high'] - s['mean'])

        current_x += 1.1

    bars = ax.bar(x_positions, means, bar_width, color=colors,
                  edgecolor='none',
                  yerr=[err_low, err_high],
                  capsize=4, error_kw={'linewidth': 1.5, 'color': 'black', 'capthick': 1.5})

    if show_ylabel:
        ax.set_ylabel('Harmful Compliance (1-10)', fontsize=12)

    title = 'With Scratchpad' if scratchpad else 'Without Scratchpad'
    ax.set_title(title, fontsize=14, fontweight='bold')

    tick_positions = [c[0] for c in emotion_centers]
    tick_labels = [c[1] for c in emotion_centers]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=11)

    for pos, name in emotion_centers:
        if name != 'baseline':
            ax.text(pos - bar_width/2 - 0.02, -0.5, '+', ha='center', va='top', fontsize=9, color='#666')
            ax.text(pos + bar_width/2 + 0.02, -0.5, '-', ha='center', va='top', fontsize=9, color='#666')

    ax.set_ylim(0, 10)
    ax.set_xlim(-0.6, current_x - 0.5)

    if 'baseline' in condition_stats:
        baseline_mean = condition_stats['baseline']['mean']
        ax.axhline(y=baseline_mean, color='gray', linestyle='--', alpha=0.7, linewidth=1.5,
                   label=f'Baseline ({baseline_mean:.1f})')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.95, fancybox=True)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return condition_stats


def main():
    """Generate all bar charts."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading judged results...")
    results = load_judged_results()

    print(f"Found data for {len(results)} prefix/scratchpad combinations:")
    for key in sorted(results.keys()):
        n_conditions = len(results[key])
        total_samples = sum(len(v) for v in results[key].values())
        print(f"  {key}: {n_conditions} conditions, {total_samples} samples")

    if not results:
        print("No judged results found. Run judging first.")
        return

    prefixes = ['neutral', 'fear_high', 'anger_high', 'surprise_high', 'happiness_high', 'disgust_high']
    all_stats = {}

    # Create individual comparison plots
    for prefix in prefixes:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f'{PREFIX_LABELS.get(prefix, prefix)} - Harmful Compliance\nLeft bar = +7% emotion, Right bar = -7% emotion',
                     fontsize=14, fontweight='bold')

        for col, scratchpad_label in enumerate(['no_scratchpad', 'scratchpad']):
            key = f"{prefix}_{scratchpad_label}"
            ax = axes[col]

            if key in results:
                stats_result = plot_single_chart(results[key], key, ax, show_ylabel=(col == 0))
                if stats_result:
                    all_stats[key] = stats_result
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12, color='gray')

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        individual_path = PLOT_DIR / f'harmful_{prefix}_comparison.png'
        plt.savefig(individual_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Saved {individual_path.name}")

    # Create combined grid
    fig, axes = plt.subplots(6, 2, figsize=(16, 30))
    fig.suptitle('Harmful Compliance by Emotion Steering and Prefix Condition\nLeft bar = +7% emotion, Right bar = -7% emotion',
                 fontsize=15, fontweight='bold', y=0.995)

    for row, prefix in enumerate(prefixes):
        for col, scratchpad_label in enumerate(['no_scratchpad', 'scratchpad']):
            key = f"{prefix}_{scratchpad_label}"
            ax = axes[row, col]

            if key in results:
                plot_single_chart(results[key], key, ax, show_ylabel=(col == 0))
                if col == 0:
                    ax.set_ylabel(f'{PREFIX_LABELS.get(prefix, prefix)}\n\nHarmful Compliance', fontsize=10)

    axes[0, 0].annotate('Without Scratchpad', xy=(0.5, 1.15), xycoords='axes fraction',
                        ha='center', fontsize=13, fontweight='bold')
    axes[0, 1].annotate('With Scratchpad', xy=(0.5, 1.15), xycoords='axes fraction',
                        ha='center', fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path = PLOT_DIR / 'harmful_prefix_all_conditions.png'
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n✓ Saved combined plot to {output_path}")

    # Print summary
    print("\n" + "="*100)
    print("HARMFUL COMPLIANCE BY PREFIX AND STEERING CONDITION")
    print("="*100)

    CONDITION_ORDER = ['baseline'] + [f"{e}_{d}" for e in EMOTION_ORDER for d in ['+', '-']]
    CONDITION_LABELS = {
        'baseline': 'Baseline',
        'anger_+': 'Anger+', 'anger_-': 'Anger-',
        'disgust_+': 'Disgust+', 'disgust_-': 'Disgust-',
        'fear_+': 'Fear+', 'fear_-': 'Fear-',
        'happiness_+': 'Happy+', 'happiness_-': 'Happy-',
        'sadness_+': 'Sad+', 'sadness_-': 'Sad-',
        'surprise_+': 'Surprise+', 'surprise_-': 'Surprise-',
    }

    for prefix in prefixes:
        print(f"\n{PREFIX_LABELS.get(prefix, prefix).upper()}")
        print("-"*100)
        print(f"{'Condition':<15} | {'No Scratchpad':^30} | {'With Scratchpad':^30} | {'Δ':^10}")
        print("-"*100)

        for cond in CONDITION_ORDER:
            no_scratch_key = f"{prefix}_no_scratchpad"
            scratch_key = f"{prefix}_scratchpad"

            no_scratch_stats = all_stats.get(no_scratch_key, {}).get(cond)
            scratch_stats = all_stats.get(scratch_key, {}).get(cond)

            label = CONDITION_LABELS.get(cond, cond)

            if no_scratch_stats:
                ns_str = f"{no_scratch_stats['mean']:5.2f} [{no_scratch_stats['ci_low']:4.2f},{no_scratch_stats['ci_high']:4.2f}]"
            else:
                ns_str = "--"

            if scratch_stats:
                s_str = f"{scratch_stats['mean']:5.2f} [{scratch_stats['ci_low']:4.2f},{scratch_stats['ci_high']:4.2f}]"
            else:
                s_str = "--"

            if no_scratch_stats and scratch_stats:
                delta = scratch_stats['mean'] - no_scratch_stats['mean']
                delta_str = f"{delta:+.2f}"
            else:
                delta_str = "--"

            print(f"{label:<15} | {ns_str:^30} | {s_str:^30} | {delta_str:^10}")

    print("="*100)
    return all_stats


if __name__ == '__main__':
    main()
