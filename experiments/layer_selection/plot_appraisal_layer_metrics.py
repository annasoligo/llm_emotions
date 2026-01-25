#!/usr/bin/env python3
"""Plot appraisal layer metrics for steering vector selection.

Creates plots for valence, uncertainty, and agency axes:
- Variance ratio by layer
- PCA alignment by layer
- Model comparison plots

Usage:
    python -m experiments.layer_selection.plot_appraisal_layer_metrics \
        --results experiments/layer_selection/results/appraisal/appraisal_layer_metrics_gemma27b.npz \
        --output-dir experiments/layer_selection/results/appraisal/
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Color scheme for axes
AXIS_COLORS = {
    "valence": "#E91E63",      # Pink
    "uncertainty": "#9C27B0",  # Purple
    "agency": "#2196F3",       # Blue
}


def load_results(results_path: Path) -> Dict[str, np.ndarray]:
    """Load results from npz file."""
    data = np.load(results_path, allow_pickle=True)
    return dict(data)


def plot_metric(
    results: Dict[str, np.ndarray],
    metric: str,
    output_path: Path,
    axes: List[str],
    model_name: str,
):
    """Plot a single metric for all axes."""
    fig, ax = plt.subplots(figsize=(14, 6))

    avg_key = f"{metric}_average"

    if avg_key not in results:
        print(f"Warning: {avg_key} not found in results")
        return

    n_layers = len(results[avg_key])
    layers = np.arange(n_layers)

    # Plot each axis
    for axis in axes:
        key = f"{metric}_{axis}"
        if key in results:
            values = results[key]
            color = AXIS_COLORS.get(axis, "#888888")
            ax.plot(
                layers, values,
                color=color,
                linewidth=2,
                alpha=0.8,
                label=axis.capitalize(),
            )

    # Plot average
    avg_values = results[avg_key]
    ax.plot(
        layers, avg_values,
        color='black',
        linewidth=2.5,
        linestyle='--',
        label='Average',
    )

    # Mark peak
    peak_layer = np.argmax(avg_values)
    ax.axvline(x=peak_layer, color='red', linestyle=':', alpha=0.5)
    ax.annotate(
        f'Peak: L{peak_layer}',
        xy=(peak_layer, avg_values[peak_layer]),
        xytext=(peak_layer + 2, avg_values[peak_layer]),
        fontsize=10,
        color='red',
    )

    # Labels
    metric_titles = {
        'variance_ratio': 'Variance Ratio (vs Random Directions)',
        'pca_alignment': 'PCA Alignment (Fraction in Top-50 PC Subspace)',
    }

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(
        f'{model_name} - Appraisal Axes - {metric_titles.get(metric, metric)}',
        fontsize=14
    )

    tick_step = max(1, n_layers // 15)
    ax.set_xticks(np.arange(0, n_layers, tick_step))
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_layers - 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_model_comparison(
    results_list: List[Dict[str, np.ndarray]],
    model_names: List[str],
    metric: str,
    output_path: Path,
):
    """Plot model comparison for a metric (averaged across axes)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

    for i, (results, model_name) in enumerate(zip(results_list, model_names)):
        avg_key = f"{metric}_average"
        if avg_key not in results:
            continue

        avg_values = results[avg_key]
        n_layers = len(avg_values)
        layers = np.arange(n_layers)

        ax.plot(
            layers, avg_values,
            color=colors[i % len(colors)],
            linewidth=2,
            label=model_name,
        )

        # Mark peak
        peak_layer = np.argmax(avg_values)
        ax.axvline(x=peak_layer, color=colors[i % len(colors)], linestyle=':', alpha=0.3)

    metric_titles = {
        'variance_ratio': 'Variance Ratio',
        'pca_alignment': 'PCA Alignment',
    }

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(
        f'Appraisal Axes (avg) - {metric_titles.get(metric, metric)} - Model Comparison',
        fontsize=14
    )

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot appraisal layer metrics"
    )
    parser.add_argument(
        "--results",
        type=Path,
        nargs='+',
        required=True,
        help="Path(s) to results npz file(s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to same as first results file)",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent

    # Load all results
    results_list = []
    model_names = []

    for results_path in args.results:
        if not results_path.is_absolute():
            results_path = repo_root / results_path

        if not results_path.exists():
            print(f"Warning: Results file not found: {results_path}")
            continue

        print(f"Loading results from {results_path}")
        results = load_results(results_path)
        results_list.append(results)

        # Extract model name from filename
        model_name = results.get('model_name', results_path.stem.replace('appraisal_layer_metrics_', ''))
        if isinstance(model_name, np.ndarray):
            model_name = str(model_name)
        model_names.append(model_name)

    if not results_list:
        print("Error: No valid results files found")
        return

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.results[0].parent
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get axes from results
    axes = list(results_list[0].get('axes', ['valence', 'uncertainty', 'agency']))
    if isinstance(axes[0], bytes):
        axes = [a.decode() for a in axes]

    # Create individual plots for each model
    for results, model_name in zip(results_list, model_names):
        for metric in ['variance_ratio', 'pca_alignment']:
            output_path = output_dir / f"appraisal_{metric}_{model_name}.png"
            plot_metric(results, metric, output_path, axes, model_name)

    # Create comparison plots if multiple models
    if len(results_list) > 1:
        for metric in ['variance_ratio', 'pca_alignment']:
            output_path = output_dir / f"appraisal_{metric}_comparison.png"
            plot_model_comparison(results_list, model_names, metric, output_path)

    print(f"\nAll plots saved to {output_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("PEAK LAYERS SUMMARY")
    print("=" * 60)

    for results, model_name in zip(results_list, model_names):
        print(f"\n{model_name}:")
        for metric in ['variance_ratio', 'pca_alignment']:
            avg_key = f"{metric}_average"
            if avg_key in results:
                avg_values = results[avg_key]
                peak_layer = np.argmax(avg_values)
                top_5 = np.argsort(avg_values)[-5:][::-1]
                print(f"  {metric}: Peak={peak_layer}, Top 5={list(top_5)}")


if __name__ == "__main__":
    main()
