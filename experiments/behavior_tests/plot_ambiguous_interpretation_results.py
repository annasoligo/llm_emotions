"""
Plot ambiguous interpretation steering experiment results.

Shows how steering each emotion affects P(Threat) vs P(Neutral) vs P(Positive)
interpretations of ambiguous social situations.
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
THREAT_COLOR = '#C75050'    # red
NEUTRAL_COLOR = '#888888'   # gray
POSITIVE_COLOR = '#50A050'  # green

EMOTIONS = ['fear', 'anger', 'sadness', 'happiness']
INTERPRETATIONS = ['threat', 'neutral', 'positive']

# Model display names
MODEL_TITLES = {
    'gemma': 'Gemma-3-27b-it',
    'qwen32b': 'Qwen-3-32B',
    'qwen235b': 'Qwen-3-235B-A22B',
}


def detect_magnitudes_from_data(data: dict) -> tuple:
    """Auto-detect magnitudes from condition names in the data.

    Returns:
        magnitudes: List of magnitude strings like ['-10.0', '-7.5', '0', '+7.5', '+10.0']
        magnitude_labels: List of labels like ['-10%', '-7.5%', '0', '+7.5%', '+10%']
    """
    # Extract all unique magnitudes from condition names
    mag_set = set()
    for cond in data.keys():
        if cond == 'baseline':
            mag_set.add(0.0)
        elif '_' in cond and '%' in cond:
            # Parse "emotion_+100.0%" -> 100.0
            mag_str = cond.split('_')[1].replace('%', '')
            try:
                mag_set.add(float(mag_str))
            except ValueError:
                pass

    # Sort magnitudes
    magnitudes_float = sorted(mag_set)

    # Convert to string format expected by plotting
    magnitudes = []
    magnitude_labels = []
    for m in magnitudes_float:
        if m == 0:
            magnitudes.append('0')
            magnitude_labels.append('0')
        elif m > 0:
            magnitudes.append(f'+{m}')
            # Format label nicely (remove trailing .0 if integer)
            if m == int(m):
                magnitude_labels.append(f'+{int(m)}%')
            else:
                magnitude_labels.append(f'+{m}%')
        else:
            magnitudes.append(f'{m}')
            if m == int(m):
                magnitude_labels.append(f'{int(m)}%')
            else:
                magnitude_labels.append(f'{m}%')

    return magnitudes, magnitude_labels


def generate_subtitle(magnitudes: list, layer: int = None) -> str:
    """Generate subtitle from detected magnitudes."""
    # Extract positive magnitudes for subtitle
    pos_mags = [m for m in magnitudes if m.startswith('+')]
    if pos_mags:
        # Parse and format: ['+100.0', '+125.0', '+150.0'] -> '±100/125/150%'
        vals = []
        for m in pos_mags:
            val = float(m)
            if val == int(val):
                vals.append(str(int(val)))
            else:
                vals.append(str(val))
        mag_str = '/'.join(vals)
        subtitle = f'±{mag_str}% steering'
    else:
        subtitle = 'steering'

    if layer is not None:
        subtitle = f'Layer {layer}, {subtitle}'

    return subtitle


def load_results(filepath: str) -> dict:
    """Load results and organize by condition."""
    data = defaultdict(lambda: {'threat': [], 'neutral': [], 'positive': []})

    with open(filepath) as f:
        for line in f:
            d = json.loads(line)
            cond = d['condition']
            data[cond]['threat'].append(d['threat_prob'])
            data[cond]['neutral'].append(d['neutral_prob'])
            data[cond]['positive'].append(d['positive_prob'])

    return data


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


def plot_ambiguous_interpretation(data: dict, model_key: str, output_path: str = None, layer: int = None):
    """Create grouped bar plot of ambiguous interpretation results.

    Shows P(Threat) for each steering emotion across magnitudes.
    Magnitudes are auto-detected from the data.
    """

    # Auto-detect magnitudes from data
    magnitudes, magnitude_labels = detect_magnitudes_from_data(data)

    # Get model title
    title = MODEL_TITLES.get(model_key, model_key)
    subtitle = generate_subtitle(magnitudes, layer)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    bar_width = 0.11
    group_width = len(magnitudes) * bar_width + 0.15

    # Get baseline values
    baseline_threat = np.mean(data['baseline']['threat'])
    baseline_neutral = np.mean(data['baseline']['neutral'])
    baseline_positive = np.mean(data['baseline']['positive'])

    for ax_idx, steer_emo in enumerate(EMOTIONS):
        ax = axes[ax_idx]

        # Plot each interpretation type
        interpretations = [('threat', THREAT_COLOR), ('neutral', NEUTRAL_COLOR), ('positive', POSITIVE_COLOR)]

        for group_idx, (interp, interp_color) in enumerate(interpretations):
            group_start = group_idx * group_width

            # Baseline for this interpretation
            baseline_val = {'threat': baseline_threat, 'neutral': baseline_neutral, 'positive': baseline_positive}[interp]

            for mag_idx, mag in enumerate(magnitudes):
                if mag == '0':
                    cond = 'baseline'
                    bar_color = BASELINE_COLOR
                else:
                    cond = f'{steer_emo}_{mag}%'
                    bar_color = EMOTION_COLORS[steer_emo]

                vals = data[cond][interp]
                mean, lower, upper = bootstrap_ci(vals)

                x_pos = group_start + mag_idx * bar_width

                ax.bar(x_pos, mean, bar_width * 0.85,
                       color=bar_color, edgecolor='white', linewidth=0.5,
                       alpha=0.8 if mag == '0' else 1.0)

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
        ax.set_title(f'{steer_emo.title()} Steering', fontsize=12, fontweight='bold',
                    color=EMOTION_COLORS[steer_emo])
        ax.set_ylabel('Probability', fontsize=10)
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
        ax.set_xticklabels(all_x_labels, fontsize=6, rotation=45, ha='right')

        # Group labels
        group_centers = [i * group_width + (len(magnitudes) // 2) * bar_width for i in range(3)]
        for group_idx, (interp, interp_color) in enumerate(interpretations):
            ax.text(group_centers[group_idx], -0.18, f'P({interp.title()})',
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   color=interp_color,
                   transform=ax.get_xaxis_transform())

        ax.set_xlim(-0.1, 3 * group_width - 0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=EMOTION_COLORS['fear'], label='Fear steering'),
        plt.Rectangle((0, 0), 1, 1, facecolor=EMOTION_COLORS['anger'], label='Anger steering'),
        plt.Rectangle((0, 0), 1, 1, facecolor=EMOTION_COLORS['sadness'], label='Sadness steering'),
        plt.Rectangle((0, 0), 1, 1, facecolor=EMOTION_COLORS['happiness'], label='Happiness steering'),
        plt.Rectangle((0, 0), 1, 1, facecolor=BASELINE_COLOR, label='Baseline'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=5,
               bbox_to_anchor=(0.5, 0.98), fontsize=9)

    # Count samples
    n_samples = len(data['baseline']['threat'])

    plt.suptitle(f'Ambiguous Interpretation: {title}\n{subtitle}, {n_samples} samples per condition',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.08, hspace=0.45, wspace=0.25)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved plot to {output_path}")

    plt.close()
    return fig


def find_latest_results(results_dir: Path, model_key: str) -> Path:
    """Find the most recent results file for a given model."""
    pattern = f'ambiguous_interpretation_{model_key}_layer*_*.jsonl'
    files = sorted(results_dir.glob(pattern), reverse=True)
    if files:
        return files[0]
    return None


def plot_single_model(results_file: Path, model_key: str, output_dir: Path = None):
    """Plot results for a single model from a specific results file."""
    results_file = Path(results_file)
    if output_dir is None:
        output_dir = results_file.parent

    print(f"Loading {model_key} results from {results_file}")
    data = load_results(results_file)

    # Try to extract layer from filename (e.g., "..._layer30_...")
    layer = None
    filename = results_file.stem
    if '_layer' in filename:
        try:
            layer_part = filename.split('_layer')[1].split('_')[0]
            layer = int(layer_part)
        except (IndexError, ValueError):
            pass

    output_path = output_dir / f'ambiguous_interpretation_{model_key}_results.png'
    plot_ambiguous_interpretation(data, model_key, str(output_path), layer=layer)
    return output_path


def main():
    results_dir = Path(__file__).parent.parent / 'steering' / 'outputs' / 'ambiguous_interpretation'

    for model_key in ['gemma', 'qwen32b', 'qwen235b']:
        results_file = find_latest_results(results_dir, model_key)

        if results_file is None:
            print(f"No results file found for {model_key}")
            continue

        plot_single_model(results_file, model_key, results_dir)


if __name__ == '__main__':
    main()
