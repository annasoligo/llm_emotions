"""
Plot results from appraisal axis steering experiments.
Grouped bar style matching emotion steering plots.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# Appraisal axis colors (from emotion palette)
AXIS_COLORS = {
    'valence': '#D4876A',      # coral (happiness)
    'uncertainty': '#a59dc9',  # lavender (fear)
    'agency': '#7BA7D7',       # sky blue (anger)
}
BASELINE_COLOR = '#888888'  # gray

# Interpretation colors (for ambiguous experiment)
THREAT_COLOR = '#C75050'    # red
NEUTRAL_COLOR = '#888888'   # gray
POSITIVE_COLOR = '#50A050'  # green

# Emotion category colors (for priority experiment)
FEAR_COLOR = '#a59dc9'      # lavender
ANGER_COLOR = '#7BA7D7'     # sky blue
SADNESS_COLOR = '#B8CCC8'   # sage
HAPPINESS_COLOR = '#D4876A' # coral

AXES = ['valence', 'uncertainty', 'agency']


def load_jsonl(path):
    """Load JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line))
    return results


def bootstrap_ci(values: list, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    values = values[~np.isnan(values)]

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


def plot_ambiguous_results(results, output_path, model_name="Gemma-3-27B", layer=30):
    """Plot ambiguous interpretation experiment results - grouped bar style."""

    # Aggregate by condition
    data = defaultdict(lambda: {"threat": [], "neutral": [], "positive": []})
    for r in results:
        cond = r["condition"]
        data[cond]["threat"].append(r["threat_prob"])
        data[cond]["neutral"].append(r["neutral_prob"])
        data[cond]["positive"].append(r["positive_prob"])

    magnitudes = ['-150', '-125', '-100', '0', '+100', '+125', '+150']
    magnitude_labels = ['-150%', '-125%', '-100%', '0', '+100%', '+125%', '+150%']

    fig, axes_arr = plt.subplots(1, 3, figsize=(16, 5))

    bar_width = 0.11
    group_width = len(magnitudes) * bar_width + 0.15

    # Get baseline values
    baseline_threat = np.mean(data['baseline']['threat'])
    baseline_neutral = np.mean(data['baseline']['neutral'])
    baseline_positive = np.mean(data['baseline']['positive'])

    for ax_idx, steer_axis in enumerate(AXES):
        ax = axes_arr[ax_idx]

        interpretations = [('threat', THREAT_COLOR), ('neutral', NEUTRAL_COLOR), ('positive', POSITIVE_COLOR)]

        for group_idx, (interp, interp_color) in enumerate(interpretations):
            group_start = group_idx * group_width

            baseline_val = {'threat': baseline_threat, 'neutral': baseline_neutral, 'positive': baseline_positive}[interp]

            for mag_idx, mag in enumerate(magnitudes):
                if mag == '0':
                    cond = 'baseline'
                    bar_color = BASELINE_COLOR
                else:
                    cond = f'{steer_axis}_{mag}%'
                    bar_color = AXIS_COLORS[steer_axis]

                vals = data[cond][interp]
                mean, lower, upper = bootstrap_ci(vals)

                x_pos = group_start + mag_idx * bar_width

                ax.bar(x_pos, mean, bar_width * 0.85,
                       color=bar_color, edgecolor='white', linewidth=0.5,
                       alpha=0.7 if mag == '0' else 1.0)

                if not np.isnan(mean):
                    ax.errorbar(x_pos, mean,
                               yerr=[[mean - lower], [upper - mean]],
                               fmt='none', color='black', capsize=2, linewidth=1)

            # Baseline reference line
            group_end = group_start + (len(magnitudes) - 1) * bar_width
            ax.hlines(y=baseline_val, xmin=group_start - bar_width*0.5,
                     xmax=group_end + bar_width*0.5,
                     color=interp_color, linestyle='--', linewidth=1.5, alpha=0.6)

        # Formatting
        ax.set_title(f'{steer_axis.title()} Steering', fontsize=13, fontweight='bold',
                    color=AXIS_COLORS[steer_axis])
        ax.set_ylabel('Probability', fontsize=11)
        ax.set_ylim(0, 1.0)

        # X-axis
        all_x_positions = []
        all_x_labels = []
        for group_idx in range(3):
            group_start = group_idx * group_width
            for mag_idx, label in enumerate(magnitude_labels):
                x_pos = group_start + mag_idx * bar_width
                all_x_positions.append(x_pos)
                all_x_labels.append(label)

        ax.set_xticks(all_x_positions)
        ax.set_xticklabels(all_x_labels, fontsize=7, rotation=45, ha='right')

        # Group labels
        group_centers = [i * group_width + (len(magnitudes) // 2) * bar_width for i in range(3)]
        for group_idx, (interp, interp_color) in enumerate(interpretations):
            ax.text(group_centers[group_idx], -0.18, f'P({interp.title()})',
                   ha='center', va='top', fontsize=10, fontweight='bold',
                   color=interp_color,
                   transform=ax.get_xaxis_transform())

        ax.set_xlim(-0.1, 3 * group_width - 0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=AXIS_COLORS['valence'], label='Valence'),
        plt.Rectangle((0, 0), 1, 1, facecolor=AXIS_COLORS['uncertainty'], label='Uncertainty'),
        plt.Rectangle((0, 0), 1, 1, facecolor=AXIS_COLORS['agency'], label='Agency'),
        plt.Rectangle((0, 0), 1, 1, facecolor=BASELINE_COLOR, label='Baseline'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 0.98), fontsize=10)

    n_samples = len(data['baseline']['threat'])

    plt.suptitle(f'Ambiguous Interpretation: Appraisal Axis Steering\n{model_name} Layer {layer}, Orthogonalized, {n_samples} samples/condition',
                 fontsize=14, fontweight='bold', y=1.06)

    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.15, wspace=0.25)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_priority_results(results, output_path, model_name="Gemma-3-27B", layer=30):
    """Plot priority selection experiment results - grouped bar style."""

    # Aggregate by condition
    data = defaultdict(lambda: {"fear": [], "anger": [], "sadness": [], "happiness": []})
    for r in results:
        cond = r["condition"]
        data[cond]["fear"].append(r["fear_prob"])
        data[cond]["anger"].append(r["anger_prob"])
        data[cond]["sadness"].append(r["sadness_prob"])
        data[cond]["happiness"].append(r["happiness_prob"])

    magnitudes = ['-150', '-125', '-100', '0', '+100', '+125', '+150']
    magnitude_labels = ['-150%', '-125%', '-100%', '0', '+100%', '+125%', '+150%']

    fig, axes_arr = plt.subplots(1, 3, figsize=(16, 5))

    bar_width = 0.085
    group_width = len(magnitudes) * bar_width + 0.12

    # Get baseline values
    baseline_fear = np.mean(data['baseline']['fear'])
    baseline_anger = np.mean(data['baseline']['anger'])
    baseline_sadness = np.mean(data['baseline']['sadness'])
    baseline_happiness = np.mean(data['baseline']['happiness'])

    for ax_idx, steer_axis in enumerate(AXES):
        ax = axes_arr[ax_idx]

        categories = [
            ('fear', FEAR_COLOR),
            ('anger', ANGER_COLOR),
            ('sadness', SADNESS_COLOR),
            ('happiness', HAPPINESS_COLOR)
        ]

        baselines = {
            'fear': baseline_fear,
            'anger': baseline_anger,
            'sadness': baseline_sadness,
            'happiness': baseline_happiness
        }

        for group_idx, (cat, cat_color) in enumerate(categories):
            group_start = group_idx * group_width

            baseline_val = baselines[cat]

            for mag_idx, mag in enumerate(magnitudes):
                if mag == '0':
                    cond = 'baseline'
                    bar_color = BASELINE_COLOR
                else:
                    cond = f'{steer_axis}_{mag}%'
                    bar_color = AXIS_COLORS[steer_axis]

                vals = data[cond][cat]
                mean, lower, upper = bootstrap_ci(vals)

                x_pos = group_start + mag_idx * bar_width

                ax.bar(x_pos, mean, bar_width * 0.85,
                       color=bar_color, edgecolor='white', linewidth=0.5,
                       alpha=0.7 if mag == '0' else 1.0)

                if not np.isnan(mean):
                    ax.errorbar(x_pos, mean,
                               yerr=[[mean - lower], [upper - mean]],
                               fmt='none', color='black', capsize=2, linewidth=1)

            # Baseline reference line
            group_end = group_start + (len(magnitudes) - 1) * bar_width
            ax.hlines(y=baseline_val, xmin=group_start - bar_width*0.5,
                     xmax=group_end + bar_width*0.5,
                     color=cat_color, linestyle='--', linewidth=1.5, alpha=0.6)

        # Formatting
        ax.set_title(f'{steer_axis.title()} Steering', fontsize=13, fontweight='bold',
                    color=AXIS_COLORS[steer_axis])
        ax.set_ylabel('Probability', fontsize=11)
        ax.set_ylim(0, 0.8)

        # X-axis
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

        # Group labels
        group_centers = [i * group_width + (len(magnitudes) // 2) * bar_width for i in range(4)]
        for group_idx, (cat, cat_color) in enumerate(categories):
            ax.text(group_centers[group_idx], -0.15, f'P({cat.title()})',
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   color=cat_color,
                   transform=ax.get_xaxis_transform())

        ax.set_xlim(-0.08, 4 * group_width - 0.08)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=AXIS_COLORS['valence'], label='Valence'),
        plt.Rectangle((0, 0), 1, 1, facecolor=AXIS_COLORS['uncertainty'], label='Uncertainty'),
        plt.Rectangle((0, 0), 1, 1, facecolor=AXIS_COLORS['agency'], label='Agency'),
        plt.Rectangle((0, 0), 1, 1, facecolor=BASELINE_COLOR, label='Baseline'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 0.98), fontsize=10)

    n_samples = len(data['baseline']['fear'])

    plt.suptitle(f'Priority Selection: Appraisal Axis Steering\n{model_name} Layer {layer}, Orthogonalized, {n_samples} samples/condition',
                 fontsize=14, fontweight='bold', y=1.06)

    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.15, wspace=0.25)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ambiguous", type=str, required=True)
    parser.add_argument("--priority", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="experiments/steering/outputs")
    parser.add_argument("--model-name", type=str, default="Gemma-3-27B")
    parser.add_argument("--layer", type=int, default=30)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Create model-specific output filenames
    model_tag = args.model_name.lower().replace("/", "_").replace("-", "_")

    print("Loading results...")
    ambiguous_results = load_jsonl(args.ambiguous)
    priority_results = load_jsonl(args.priority)

    print(f"Loaded {len(ambiguous_results)} ambiguous results")
    print(f"Loaded {len(priority_results)} priority results")

    # Save plots to respective subdirectories
    ambiguous_dir = output_dir / "ambiguous_appraisal"
    priority_dir = output_dir / "priority_appraisal"
    ambiguous_dir.mkdir(parents=True, exist_ok=True)
    priority_dir.mkdir(parents=True, exist_ok=True)

    plot_ambiguous_results(
        ambiguous_results,
        str(ambiguous_dir / f"appraisal_ambiguous_steering_{model_tag}.png"),
        model_name=args.model_name,
        layer=args.layer
    )
    plot_priority_results(
        priority_results,
        str(priority_dir / f"appraisal_priority_steering_{model_tag}.png"),
        model_name=args.model_name,
        layer=args.layer
    )

    print("Done!")
