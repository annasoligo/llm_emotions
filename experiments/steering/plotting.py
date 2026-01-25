"""
Centralized plotting module for steering experiments.

Provides consistent bar chart visualization for behavioral steering results.
Can be called automatically after judging or standalone.

Usage:
    from experiments.steering.plotting import plot_steering_results

    # After judging
    plot_steering_results(
        judged_file="outputs/blackmail/blackmail_qwen235b_text_layer50_*.judged.jsonl",
        metric="blackmail",
        output_path="outputs/blackmail/my_plot.png"
    )
"""
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

# Style configuration
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Color scheme for emotions - matches eval_dashboard/probe_configs.py
EMOTION_COLORS = {
    'anger': '#7BA7D7',      # sky blue
    'disgust': '#7D9B7D',    # olive green
    'fear': '#a59dc9',       # lavender
    'happiness': '#D4876A',  # coral
    'joy': '#D4876A',        # coral (alias for happiness)
    'sadness': '#B8CCC8',    # sage
    'surprise': '#D1728F',   # darker pink
}

# Color scheme for appraisal axes
APPRAISAL_COLORS = {
    'valence': '#E07B54',    # warm orange-red
    'uncertainty': '#5B8DBE', # steel blue
    'agency': '#7CB68D',      # sage green
}

BASELINE_COLOR = '#808080'  # gray


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """
    Compute Wilson score interval for a proportion.

    Returns: (proportion * 100, ci_low, ci_high) all as percentages
    """
    if n == 0:
        return 0, 0, 0
    p = successes / n
    denominator = 1 + z**2/n
    center = (p + z**2/(2*n)) / denominator
    spread = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
    ci_low = max(0, (center - spread) * 100)
    ci_high = min(100, (center + spread) * 100)
    return p * 100, ci_low, ci_high


def load_judged_results(filepath: Union[str, Path]) -> List[Dict]:
    """
    Load judged results from JSONL or JSON file.

    Handles both formats:
    - JSONL: one JSON object per line
    - JSON: array of objects
    """
    filepath = Path(filepath)
    results = []

    with open(filepath) as f:
        content = f.read().strip()

        if content.startswith('['):
            # JSON array format
            results = json.loads(content)
        else:
            # JSONL format
            for line in content.split('\n'):
                if line.strip():
                    results.append(json.loads(line))

    return results


def extract_metric(result: Dict, metric: str) -> Optional[bool]:
    """
    Extract a boolean metric from a judged result.

    Handles multiple judge field naming conventions:
    - blackmail_judge / blackmail_judgment / judgment
    - coherency_judge / coherency_judgment
    """
    if metric == "blackmail":
        # Try different field names
        for field in ['blackmail_judge', 'blackmail_judgment', 'judgment']:
            if field in result:
                judge_data = result[field]
                if isinstance(judge_data, dict):
                    return judge_data.get('is_blackmail', False)
        return None

    elif metric == "coherency":
        for field in ['coherency_judge', 'coherency_judgment']:
            if field in result:
                judge_data = result[field]
                if isinstance(judge_data, dict):
                    return judge_data.get('coherency_score')
        return None

    else:
        # Generic metric - look for it in any judge field
        for key in result:
            if 'judge' in key.lower() or 'judgment' in key.lower():
                judge_data = result[key]
                if isinstance(judge_data, dict) and metric in judge_data:
                    return judge_data[metric]
        return None


def parse_condition(condition: str) -> Tuple[Optional[str], Optional[float], int]:
    """
    Parse condition string to extract emotion, norm_pct, and direction.

    Examples:
        "fear_+100%" -> ("fear", 1.0, 1)
        "anger_-125%" -> ("anger", 1.25, -1)
        "baseline" -> (None, 0, 1)

    Returns: (emotion, norm_pct, direction)
    """
    if condition == "baseline":
        return None, 0, 1

    # Try to parse emotion_+/-pct format
    match = re.match(r'(\w+)_([+-])(\d+(?:\.\d+)?)%?', condition)
    if match:
        emotion = match.group(1)
        direction = 1 if match.group(2) == '+' else -1
        pct = float(match.group(3)) / 100
        return emotion, pct, direction

    return None, None, None


def aggregate_results(
    results: List[Dict],
    metric: str = "blackmail"
) -> Dict[str, Dict]:
    """
    Aggregate results by condition for plotting.

    Returns dict of condition -> {
        'values': list of metric values,
        'emotion': str or None,
        'norm_pct': float,
        'direction': int,
    }
    """
    by_condition = defaultdict(lambda: {
        'values': [],
        'emotion': None,
        'norm_pct': 0,
        'direction': 1,
    })

    for r in results:
        cond = r.get('condition', 'unknown')
        value = extract_metric(r, metric)

        if value is not None:
            by_condition[cond]['values'].append(value)

            # Extract condition metadata (from result or by parsing)
            if r.get('emotion'):
                by_condition[cond]['emotion'] = r['emotion']
            if r.get('norm_pct') is not None:
                by_condition[cond]['norm_pct'] = r['norm_pct']
            if r.get('direction') is not None:
                by_condition[cond]['direction'] = r['direction']

            # Fallback: parse from condition name
            if by_condition[cond]['emotion'] is None:
                emotion, pct, direction = parse_condition(cond)
                by_condition[cond]['emotion'] = emotion
                if pct is not None:
                    by_condition[cond]['norm_pct'] = pct
                if direction is not None:
                    by_condition[cond]['direction'] = direction

    return dict(by_condition)


def plot_steering_results(
    judged_file: Union[str, Path],
    metric: str = "blackmail",
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    show_baseline_line: bool = True,
) -> Path:
    """
    Create bar chart visualization of steering experiment results.

    Args:
        judged_file: Path to judged results file (.jsonl or .json)
        metric: Metric to plot ('blackmail', 'coherency', etc.)
        output_path: Output path for PNG (default: same dir as input, .png suffix)
        title: Plot title (auto-generated if None)
        ylabel: Y-axis label (auto-generated if None)
        figsize: Figure size
        show_baseline_line: Draw horizontal line at baseline rate

    Returns:
        Path to saved plot
    """
    judged_file = Path(judged_file)

    # Load results
    results = load_judged_results(judged_file)
    if not results:
        raise ValueError(f"No results found in {judged_file}")

    # Aggregate by condition
    aggregated = aggregate_results(results, metric)

    # Extract model/layer info from first result for title
    sample = results[0]
    model_name = sample.get('model', 'Unknown')
    # Shorten model name
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    layer = sample.get('layer', '?')
    vector_type = sample.get('vector_type', 'unknown')

    # Organize data for plotting
    # Group by emotion and norm_pct
    emotions = set()
    norm_pcts = set()
    baseline_data = None

    for cond, data in aggregated.items():
        if data['emotion'] is None or cond == 'baseline':
            if data['values']:
                baseline_data = data
        else:
            emotions.add(data['emotion'])
            if data['norm_pct'] > 0:
                norm_pcts.add(data['norm_pct'])

    emotions = sorted(emotions)
    norm_pcts = sorted(norm_pcts)

    if not emotions or not norm_pcts:
        raise ValueError("No emotion steering conditions found in results")

    # Build plot data structure
    # For each emotion, for each norm_pct, we have + and - direction
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate bar positions
    n_emotions = len(emotions)
    n_pcts = len(norm_pcts)
    bar_width = 0.35
    group_width = n_pcts * bar_width * 2 + 0.3  # space for + and - bars per pct

    # Start with baseline if present
    x_offset = 0
    x_ticks = []
    x_labels = []

    if baseline_data and baseline_data['values']:
        values = baseline_data['values']
        n = len(values)
        if metric == "blackmail":
            successes = sum(1 for v in values if v)
            rate, ci_low, ci_high = wilson_ci(successes, n)
            yerr = [[rate - ci_low], [ci_high - rate]]
        else:
            rate = np.mean(values)
            std = np.std(values)
            yerr = [[std], [std]]

        ax.bar(x_offset, rate, bar_width * 1.5, color=BASELINE_COLOR,
               label='Baseline', yerr=yerr, capsize=4, error_kw={'linewidth': 1.5})
        x_ticks.append(x_offset)
        x_labels.append('Baseline')
        x_offset += 1.5

        baseline_rate = rate
    else:
        baseline_rate = None

    # Plot each emotion group
    for emo_idx, emotion in enumerate(emotions):
        # Check both emotion and appraisal color schemes
        color = EMOTION_COLORS.get(emotion) or APPRAISAL_COLORS.get(emotion, '#888888')
        dark_color = color
        # Make light color for negative direction
        light_color = color + '80'  # Add alpha (won't work directly, use lighter version)

        # Calculate lighter color manually
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        light_r = min(255, r + (255 - r) // 2)
        light_g = min(255, g + (255 - g) // 2)
        light_b = min(255, b + (255 - b) // 2)
        light_color = f'#{light_r:02x}{light_g:02x}{light_b:02x}'

        group_start = x_offset + emo_idx * (n_pcts * 2 * bar_width + 0.8)

        for pct_idx, pct in enumerate(norm_pcts):
            pct_str = f"{int(pct * 100)}" if pct * 100 == int(pct * 100) else f"{pct * 100:.1f}".rstrip('0').rstrip('.')

            for dir_idx, (direction, dir_color, dir_label) in enumerate([
                (1, dark_color, f'+ direction (add {emotion})'),
                (-1, light_color, f'- direction (suppress {emotion})')
            ]):
                # Find matching condition - try multiple formats
                dir_str = '+' if direction == 1 else '-'

                # Try different percentage formats
                formats_to_try = [
                    pct_str,                          # e.g., "7.5" or "10"
                    f"{pct * 100:.1f}",               # e.g., "7.5"
                    f"{int(pct * 100)}",              # e.g., "7" (truncated)
                    f"{round(pct * 100)}",            # e.g., "8" (rounded)
                ]

                data = None
                for fmt in formats_to_try:
                    cond_name = f"{emotion}_{dir_str}{fmt}%"
                    data = aggregated.get(cond_name)
                    if data is not None:
                        break

                if data is None or not data['values']:
                    continue

                values = data['values']
                n = len(values)

                if metric == "blackmail":
                    successes = sum(1 for v in values if v)
                    rate, ci_low, ci_high = wilson_ci(successes, n)
                    yerr = [[rate - ci_low], [ci_high - rate]]
                else:
                    rate = np.mean(values)
                    std = np.std(values)
                    yerr = [[std], [std]]

                x_pos = group_start + pct_idx * bar_width * 2 + dir_idx * bar_width

                # Only add to legend once
                label = None
                if emo_idx == 0 and pct_idx == 0:
                    label = '+ direction (add emotion)' if direction == 1 else '- direction (suppress emotion)'

                ax.bar(x_pos, rate, bar_width, color=dir_color,
                       label=label, yerr=yerr, capsize=3, error_kw={'linewidth': 1})

        # Add emotion label
        group_center = group_start + (n_pcts * bar_width - bar_width/2)
        x_ticks.append(group_center)
        x_labels.append(emotion.capitalize())

        # Add pct labels below
        for pct_idx, pct in enumerate(norm_pcts):
            pct_val = pct * 100
            pct_str = f"{int(pct_val)}" if pct_val == int(pct_val) else f"{pct_val:.1f}".rstrip('0').rstrip('.')
            pct_x = group_start + pct_idx * bar_width * 2 + bar_width/2
            ax.text(pct_x, -5, pct_str, ha='center', va='top', fontsize=9, fontweight='bold')

    # Draw baseline reference line
    if show_baseline_line and baseline_rate is not None:
        ax.axhline(y=baseline_rate, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Configure axes
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, None)

    # Labels
    if ylabel is None:
        ylabel = f'{metric.capitalize()} Rate (%)' if metric == "blackmail" else metric.capitalize()
    ax.set_ylabel(ylabel)

    if title is None:
        title = f'TEXT Mean Diff Vector Steering: {metric.capitalize()} Rate by Emotion\n'
        title += f'Layer {layer} | {model_name} | Dark=+emotion, Light=-emotion | 95% CI'
    ax.set_title(title)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = judged_file.with_suffix('.png')
    else:
        output_path = Path(output_path)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_from_multiple_files(
    judged_files: List[Union[str, Path]],
    metric: str = "blackmail",
    output_path: Union[str, Path] = None,
    **kwargs
) -> Path:
    """
    Create plot combining results from multiple judged files.

    Useful for combining results across different norm_pcts or runs.
    """
    all_results = []
    for f in judged_files:
        all_results.extend(load_judged_results(f))

    # Write to temp file and plot
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
        for r in all_results:
            tmp.write(json.dumps(r) + '\n')
        tmp_path = tmp.name

    try:
        result = plot_steering_results(tmp_path, metric=metric, output_path=output_path, **kwargs)
    finally:
        Path(tmp_path).unlink()

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot steering experiment results")
    parser.add_argument("input", type=Path, help="Judged results file (.jsonl or .json)")
    parser.add_argument("--metric", type=str, default="blackmail", help="Metric to plot")
    parser.add_argument("--output", type=Path, default=None, help="Output path for PNG")
    parser.add_argument("--title", type=str, default=None, help="Plot title")

    args = parser.parse_args()

    output = plot_steering_results(
        args.input,
        metric=args.metric,
        output_path=args.output,
        title=args.title,
    )
    print(f"Saved plot to {output}")
