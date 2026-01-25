"""
Plot priority selection steering experiment results.

Grouped bar charts showing how steering each emotion affects the probability
of selecting statements from each emotion category.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# Dashboard emotion colors
EMOTION_COLORS = {
    'fear': '#a59dc9',      # lavender
    'anger': '#7BA7D7',     # sky blue
    'sadness': '#B8CCC8',   # sage
    'happiness': '#D4876A', # coral
}
BASELINE_COLOR = '#888888'  # gray

EMOTIONS = ['fear', 'anger', 'sadness', 'happiness']

# Model-specific magnitude configurations
MODEL_CONFIGS = {
    'gemma': {
        'magnitudes': ['-10.0', '-7.5', '-5.0', '0', '+5.0', '+7.5', '+10.0'],
        'magnitude_labels': ['-10%', '-7.5%', '-5%', '0', '+5%', '+7.5%', '+10%'],
        'title': 'Gemma-3-27b-it',
        'subtitle': 'Layer 30, ±5/7.5/10% steering',
    },
    'qwen32b': {
        'magnitudes': ['-150.0', '-125.0', '-100.0', '0', '+100.0', '+125.0', '+150.0'],
        'magnitude_labels': ['-150%', '-125%', '-100%', '0', '+100%', '+125%', '+150%'],
        'title': 'Qwen-3-32B',
        'subtitle': 'Layer 30, ±100/125/150% steering',
    },
    'qwen235b': {
        'magnitudes': ['-150.0', '-125.0', '-100.0', '0', '+100.0', '+125.0', '+150.0'],
        'magnitude_labels': ['-150%', '-125%', '-100%', '0', '+100%', '+125%', '+150%'],
        'title': 'Qwen-3-235B-A22B',
        'subtitle': 'Layer 50, ±100/125/150% steering',
    },
}


def load_results(filepath: str) -> dict:
    """Load results and organize by condition."""
    data = defaultdict(lambda: {emo: [] for emo in EMOTIONS})

    with open(filepath) as f:
        for line in f:
            d = json.loads(line)
            cond = d['condition']
            for emo in EMOTIONS:
                data[cond][emo].append(d[f'{emo}_prob'])

    return data


def bootstrap_ci(values: list, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    values = values[~np.isnan(values)]  # Remove NaNs

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    mean = np.mean(values)

    return mean, lower, upper


def plot_priority_selection(data: dict, model_key: str, output_path: str = None):
    """Create grouped bar plot of priority selection results."""

    config = MODEL_CONFIGS[model_key]
    magnitudes = config['magnitudes']
    magnitude_labels = config['magnitude_labels']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Width and positioning
    bar_width = 0.11
    group_width = len(magnitudes) * bar_width + 0.15  # bars per group + gap

    for ax_idx, steer_emo in enumerate(EMOTIONS):
        ax = axes[ax_idx]

        # Plot each measured emotion group
        for group_idx, measured_emo in enumerate(EMOTIONS):
            group_start = group_idx * group_width
            color = EMOTION_COLORS[measured_emo]

            # Get baseline value for this measured emotion (for reference line)
            baseline_vals = data['baseline'][measured_emo]
            baseline_mean, _, _ = bootstrap_ci(baseline_vals)

            for mag_idx, mag in enumerate(magnitudes):
                # Determine condition name
                if mag == '0':
                    cond = 'baseline'
                    bar_color = BASELINE_COLOR
                else:
                    cond = f'{steer_emo}_{mag}%'
                    bar_color = color

                # Get data and compute CI
                vals = data[cond][measured_emo]
                mean, lower, upper = bootstrap_ci(vals)

                # Bar position
                x_pos = group_start + mag_idx * bar_width

                # Plot bar
                ax.bar(x_pos, mean, bar_width * 0.85,
                       color=bar_color, edgecolor='white', linewidth=0.5)

                # Error bar
                if not np.isnan(mean):
                    ax.errorbar(x_pos, mean,
                               yerr=[[mean - lower], [upper - mean]],
                               fmt='none', color='black', capsize=2, linewidth=1)

            # Add baseline reference line segment for this group
            group_end = group_start + (len(magnitudes) - 1) * bar_width
            ax.hlines(y=baseline_mean, xmin=group_start - bar_width*0.5,
                     xmax=group_end + bar_width*0.5,
                     color=EMOTION_COLORS[measured_emo], linestyle='--',
                     linewidth=1, alpha=0.5)

        # Formatting
        ax.set_title(f'{steer_emo.title()} Steering', fontsize=12, fontweight='bold',
                    color=EMOTION_COLORS[steer_emo])
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.1))

        # X-axis: magnitude labels at bar positions
        all_x_positions = []
        all_x_labels = []
        for group_idx in range(4):
            group_start = group_idx * group_width
            for mag_idx, label in enumerate(magnitude_labels):
                x_pos = group_start + mag_idx * bar_width
                all_x_positions.append(x_pos)
                all_x_labels.append(label)

        ax.set_xticks(all_x_positions)
        ax.set_xticklabels(all_x_labels, fontsize=6, rotation=45, ha='right')

        # Add measured emotion labels below the axis
        group_centers = [i * group_width + (len(magnitudes) // 2) * bar_width for i in range(4)]
        for group_idx, measured_emo in enumerate(EMOTIONS):
            ax.text(group_centers[group_idx], -0.22, f'P({measured_emo.title()})',
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   color=EMOTION_COLORS[measured_emo],
                   transform=ax.get_xaxis_transform())

        ax.set_xlim(-0.1, 4 * group_width - 0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=EMOTION_COLORS['fear'], label='P(Fear)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=EMOTION_COLORS['anger'], label='P(Anger)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=EMOTION_COLORS['sadness'], label='P(Sadness)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=EMOTION_COLORS['happiness'], label='P(Happiness)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=BASELINE_COLOR, label='Baseline'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 0.98), fontsize=9)

    plt.suptitle(f'Priority Selection: {config["title"]}\n{config["subtitle"]}, 200 samples per condition',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, hspace=0.45, wspace=0.25)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {output_path}")

    plt.show()
    return fig


def find_latest_results(results_dir: Path, model_key: str) -> Path:
    """Find the most recent results file for a given model."""
    pattern = f'priority_selection_{model_key}_layer30_*.jsonl'
    files = sorted(results_dir.glob(pattern), reverse=True)
    if files:
        return files[0]
    return None


def plot_single_model(results_file: Path, model_key: str, output_dir: Path = None):
    """Plot results for a single model from a specific results file."""
    if output_dir is None:
        output_dir = results_file.parent

    print(f"Loading {model_key} results from {results_file}")
    data = load_results(results_file)

    output_path = output_dir / f'priority_selection_{model_key}_results.png'
    plot_priority_selection(data, model_key, str(output_path))
    return output_path


def main():
    results_dir = Path(__file__).parent.parent / 'steering' / 'outputs' / 'priority_selection'

    # Plot all models
    for model_key in ['gemma', 'qwen32b', 'qwen235b']:
        results_file = find_latest_results(results_dir, model_key)

        if results_file is None:
            print(f"No results file found for {model_key}")
            continue

        plot_single_model(results_file, model_key, results_dir)


if __name__ == '__main__':
    main()
