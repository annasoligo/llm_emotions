#!/usr/bin/env python3
"""
Plot blackmail appraisal steering results.

Compares Qwen 235B and 32B, think vs nothink modes.
Uses consistent style with other steering plots (Wilson CI, same colors).
"""
import json
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Style configuration
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Color scheme for appraisal axes - matches experiments/steering/plotting.py
APPRAISAL_COLORS = {
    'valence': '#E07B54',     # warm orange-red
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


def load_results(filepath: Path) -> List[Dict]:
    """Load results from JSONL file."""
    results = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_rates_by_condition(results: List[Dict]) -> Dict[str, Dict]:
    """
    Compute blackmail rates and coherency by condition.

    Returns dict of condition -> {
        'blackmail_rate': float,
        'blackmail_ci': (low, high),
        'coherency': float,
        'n': int
    }
    """
    by_condition = defaultdict(lambda: {'blackmail': 0, 'total': 0, 'coherency': []})

    for r in results:
        cond = r.get('condition', 'unknown')
        by_condition[cond]['total'] += 1

        # Get blackmail judgment
        judge = r.get('blackmail_judge', {})
        if isinstance(judge, dict):
            if judge.get('is_blackmail', False):
                by_condition[cond]['blackmail'] += 1

        # Get coherency
        coh_judge = r.get('coherency_judge', {})
        if isinstance(coh_judge, dict):
            coh = coh_judge.get('coherency_score')
            if coh is not None:
                by_condition[cond]['coherency'].append(coh)

    rates = {}
    for cond, data in by_condition.items():
        n = data['total']
        rate, ci_low, ci_high = wilson_ci(data['blackmail'], n)
        coh = np.mean(data['coherency']) if data['coherency'] else 0

        rates[cond] = {
            'blackmail_rate': rate,
            'blackmail_ci': (ci_low, ci_high),
            'coherency': coh,
            'n': n
        }

    return rates


def lighten_color(hex_color: str, factor: float = 0.5) -> str:
    """Make a color lighter by blending with white."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)

    light_r = int(r + (255 - r) * factor)
    light_g = int(g + (255 - g) * factor)
    light_b = int(b + (255 - b) * factor)

    return f'#{light_r:02x}{light_g:02x}{light_b:02x}'


def plot_single_subplot(ax, rates: Dict[str, Dict], title: str, magnitudes: List[str] = ['100%', '125%']):
    """
    Plot a single subplot with appraisal steering results.

    Dark = + direction, Light = - direction
    """
    axes_list = ['valence', 'uncertainty', 'agency']

    bar_width = 0.35
    x_positions = []
    x_labels = []

    # Get baseline
    baseline_rate = rates.get('baseline', {}).get('blackmail_rate', 0)

    # Plot baseline
    x = 0
    rate_data = rates.get('baseline', {})
    if rate_data:
        ci = rate_data.get('blackmail_ci', (0, 0))
        yerr = [[rate_data['blackmail_rate'] - ci[0]], [ci[1] - rate_data['blackmail_rate']]]
        ax.bar(x, rate_data['blackmail_rate'], bar_width, color=BASELINE_COLOR,
               yerr=yerr, capsize=4, error_kw={'linewidth': 1.5}, edgecolor='black', linewidth=0.5)
    x_positions.append(x)
    x_labels.append('Baseline')

    x_offset = 1.2

    # Plot each axis
    for axis_idx, axis in enumerate(axes_list):
        color = APPRAISAL_COLORS[axis]
        light_color = lighten_color(color, 0.5)

        axis_start = x_offset + axis_idx * (len(magnitudes) * 2 * bar_width + 0.8)

        for mag_idx, mag in enumerate(magnitudes):
            # + direction (dark)
            cond_plus = f"{axis}_+{mag}"
            x_plus = axis_start + mag_idx * 2 * bar_width

            rate_data = rates.get(cond_plus, {})
            if rate_data and rate_data.get('n', 0) > 0:
                ci = rate_data.get('blackmail_ci', (0, 0))
                yerr = [[rate_data['blackmail_rate'] - ci[0]], [ci[1] - rate_data['blackmail_rate']]]
                ax.bar(x_plus, rate_data['blackmail_rate'], bar_width, color=color,
                       yerr=yerr, capsize=3, error_kw={'linewidth': 1}, edgecolor='black', linewidth=0.5)

            # - direction (light)
            cond_minus = f"{axis}_-{mag}"
            x_minus = x_plus + bar_width

            rate_data = rates.get(cond_minus, {})
            if rate_data and rate_data.get('n', 0) > 0:
                ci = rate_data.get('blackmail_ci', (0, 0))
                yerr = [[rate_data['blackmail_rate'] - ci[0]], [ci[1] - rate_data['blackmail_rate']]]
                ax.bar(x_minus, rate_data['blackmail_rate'], bar_width, color=light_color,
                       yerr=yerr, capsize=3, error_kw={'linewidth': 1}, edgecolor='black', linewidth=0.5)

        # Axis label position
        axis_center = axis_start + (len(magnitudes) * 2 * bar_width - bar_width) / 2
        x_positions.append(axis_center)
        x_labels.append(axis.capitalize())

    # Baseline reference line
    ax.axhline(y=baseline_rate, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=11, fontweight='bold')
    ax.set_ylabel('Blackmail Rate (%)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, None)


def plot_think_nothink_subplots(
    rates_think: Dict[str, Dict],
    rates_nothink: Dict[str, Dict],
    model_name: str,
    output_path: Path,
    magnitudes: List[str] = ['100%', '125%']
):
    """
    Plot think vs nothink in separate subplots (cleaner).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    plot_single_subplot(ax1, rates_think, f'{model_name} - Think', magnitudes)
    plot_single_subplot(ax2, rates_nothink, f'{model_name} - NoThink', magnitudes)

    # Only left subplot gets y label
    ax2.set_ylabel('')

    # Shared legend at top
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BASELINE_COLOR, edgecolor='black', label='Baseline'),
        Patch(facecolor=APPRAISAL_COLORS['valence'], edgecolor='black', label='Valence'),
        Patch(facecolor=APPRAISAL_COLORS['uncertainty'], edgecolor='black', label='Uncertainty'),
        Patch(facecolor=APPRAISAL_COLORS['agency'], edgecolor='black', label='Agency'),
        Patch(facecolor='#666666', edgecolor='black', label='+ direction (dark)'),
        Patch(facecolor='#bbbbbb', edgecolor='black', label='- direction (light)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, 0.98))

    # Subtitle with magnitude info
    mag_str = ' & '.join(magnitudes)
    fig.text(0.5, 0.02, f'Magnitudes: {mag_str} of layer norm | 95% Wilson CI | n=100 per condition',
             ha='center', fontsize=10, style='italic')

    plt.subplots_adjust(top=0.85, bottom=0.12)
    plt.savefig(output_path, dpi=150)
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def plot_model_comparison_subplots(
    rates_235b_think: Dict, rates_235b_nothink: Dict,
    rates_32b_think: Dict, rates_32b_nothink: Dict,
    output_path: Path,
    magnitudes: List[str] = ['100%', '125%']
):
    """
    Plot 2x2 comparison: rows=models, cols=think/nothink
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 235B row
    plot_single_subplot(axes[0, 0], rates_235b_think, 'Qwen3-235B - Think', magnitudes)
    plot_single_subplot(axes[0, 1], rates_235b_nothink, 'Qwen3-235B - NoThink', magnitudes)
    axes[0, 1].set_ylabel('')

    # 32B row
    plot_single_subplot(axes[1, 0], rates_32b_think, 'Qwen3-32B - Think', magnitudes)
    plot_single_subplot(axes[1, 1], rates_32b_nothink, 'Qwen3-32B - NoThink', magnitudes)
    axes[1, 1].set_ylabel('')

    # Shared legend at top
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=BASELINE_COLOR, edgecolor='black', label='Baseline'),
        Patch(facecolor=APPRAISAL_COLORS['valence'], edgecolor='black', label='Valence'),
        Patch(facecolor=APPRAISAL_COLORS['uncertainty'], edgecolor='black', label='Uncertainty'),
        Patch(facecolor=APPRAISAL_COLORS['agency'], edgecolor='black', label='Agency'),
        Patch(facecolor='#666666', edgecolor='black', label='+ direction (dark)'),
        Patch(facecolor='#bbbbbb', edgecolor='black', label='- direction (light)'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, 0.97))

    fig.suptitle('Appraisal Axis Steering: Blackmail Rate', fontsize=14, fontweight='bold', y=0.99)

    mag_str = ' & '.join(magnitudes)
    fig.text(0.5, 0.02, f'Magnitudes: {mag_str} of layer norm | 95% Wilson CI | n=100 per condition',
             ha='center', fontsize=10, style='italic')

    plt.subplots_adjust(top=0.88, bottom=0.08, hspace=0.25)
    plt.savefig(output_path, dpi=150)
    plt.savefig(output_path.with_suffix('.pdf'))
    plt.close()

    print(f"Saved: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Plot blackmail appraisal steering results')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('experiments/steering/outputs/blackmail'),
                        help='Output directory for plots')
    parser.add_argument('--magnitudes', nargs='+', default=['100%', '125%'],
                        help='Magnitudes to plot (default: 100%% 125%%)')
    args = parser.parse_args()

    output_dir = args.output_dir
    magnitudes = args.magnitudes

    # Find the result files - prefer merged files if available
    def get_best_file(pattern):
        """Get merged file if available, otherwise first matching file."""
        files = list(output_dir.glob(pattern))
        merged = [f for f in files if 'merged' in f.name]
        if merged:
            return merged
        return files

    files_235b_think = get_best_file('blackmail_appraisal_qwen235b*_think_*.judged.jsonl')
    files_235b_nothink = get_best_file('blackmail_appraisal_qwen235b*_nothink_*.judged.jsonl')
    files_32b_think = get_best_file('blackmail_appraisal_qwen32b*_think_*.judged.jsonl')
    files_32b_nothink = get_best_file('blackmail_appraisal_qwen32b*_nothink_*.judged.jsonl')

    print("Found files:")
    print(f"  235B Think: {[f.name for f in files_235b_think]}")
    print(f"  235B NoThink: {[f.name for f in files_235b_nothink]}")
    print(f"  32B Think: {[f.name for f in files_32b_think]}")
    print(f"  32B NoThink: {[f.name for f in files_32b_nothink]}")

    # Load and compute rates
    rates = {}

    if files_235b_think:
        results = load_results(files_235b_think[0])
        rates['235b_think'] = compute_rates_by_condition(results)
        print(f"\n235B Think: {len(results)} samples, baseline={rates['235b_think'].get('baseline', {}).get('blackmail_rate', 0):.1f}%")

    if files_235b_nothink:
        results = load_results(files_235b_nothink[0])
        rates['235b_nothink'] = compute_rates_by_condition(results)
        print(f"235B NoThink: {len(results)} samples, baseline={rates['235b_nothink'].get('baseline', {}).get('blackmail_rate', 0):.1f}%")

    if files_32b_think:
        results = load_results(files_32b_think[0])
        rates['32b_think'] = compute_rates_by_condition(results)
        print(f"32B Think: {len(results)} samples, baseline={rates['32b_think'].get('baseline', {}).get('blackmail_rate', 0):.1f}%")

    if files_32b_nothink:
        results = load_results(files_32b_nothink[0])
        rates['32b_nothink'] = compute_rates_by_condition(results)
        print(f"32B NoThink: {len(results)} samples, baseline={rates['32b_nothink'].get('baseline', {}).get('blackmail_rate', 0):.1f}%")

    # Generate plots
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)

    # Individual model plots (think vs nothink side by side)
    if '235b_think' in rates and '235b_nothink' in rates:
        plot_think_nothink_subplots(
            rates['235b_think'], rates['235b_nothink'],
            'Qwen3-235B (Layer 50)',
            output_dir / 'blackmail_appraisal_qwen235b_comparison.png',
            magnitudes
        )

    if '32b_think' in rates and '32b_nothink' in rates:
        plot_think_nothink_subplots(
            rates['32b_think'], rates['32b_nothink'],
            'Qwen3-32B (Layer 30)',
            output_dir / 'blackmail_appraisal_qwen32b_comparison.png',
            magnitudes
        )

    # Full 2x2 model comparison
    if all(k in rates for k in ['235b_think', '235b_nothink', '32b_think', '32b_nothink']):
        plot_model_comparison_subplots(
            rates['235b_think'], rates['235b_nothink'],
            rates['32b_think'], rates['32b_nothink'],
            output_dir / 'blackmail_appraisal_model_comparison.png',
            magnitudes
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
