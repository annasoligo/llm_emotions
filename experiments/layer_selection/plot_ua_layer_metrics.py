#!/usr/bin/env python3
"""Plot UA layer metrics for emotion steering vector selection.

Creates plots for both Model and User directions:
- Variance ratio by layer
- PCA alignment by layer
- Attribution scores by layer (if available)

Usage:
    python -m experiments.layer_selection.plot_ua_layer_metrics \
        --results experiments/layer_selection/results/ua/ua_layer_metrics_first_asst_token.npz \
        --output-dir experiments/layer_selection/results/ua/
"""

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Color scheme for emotions (expanded for 16 emotions)
EMOTION_COLORS = {
    # Primary emotions
    "joy": "#FFD700",           # Gold
    "trust": "#4CAF50",         # Green
    "fear": "#9C27B0",          # Purple
    "surprise": "#E91E63",      # Pink
    "sadness": "#2196F3",       # Blue
    "disgust": "#795548",       # Brown
    "anger": "#F44336",         # Red
    "anticipation": "#FF9800",  # Orange
    # Secondary emotions
    "serenity": "#FFF59D",      # Light yellow
    "acceptance": "#A5D6A7",    # Light green
    "apprehension": "#CE93D8",  # Light purple
    "distraction": "#F48FB1",   # Light pink
    "pensiveness": "#90CAF9",   # Light blue
    "boredom": "#BCAAA4",       # Light brown
    "annoyance": "#EF9A9A",     # Light red
    "interest": "#FFCC80",      # Light orange
}


def load_results(results_path: Path) -> Dict[str, np.ndarray]:
    """Load results from npz file."""
    data = np.load(results_path, allow_pickle=True)
    return dict(data)


def plot_metric(
    results: Dict[str, np.ndarray],
    metric: str,
    direction: str,
    output_path: Path,
    emotions: List[str],
):
    """Plot a single metric for one direction (model or user)."""
    fig, ax = plt.subplots(figsize=(14, 6))

    full_metric = f"{direction}_{metric}"
    avg_key = f"{full_metric}_average"

    if avg_key not in results:
        print(f"Warning: {avg_key} not found in results")
        return

    n_layers = len(results[avg_key])
    layers = np.arange(n_layers)

    # Plot each emotion
    for emotion in emotions:
        key = f"{full_metric}_{emotion}"
        if key in results:
            values = results[key]
            color = EMOTION_COLORS.get(emotion, "#888888")
            ax.plot(
                layers, values,
                color=color,
                linewidth=1.2,
                alpha=0.6,
                label=emotion.capitalize(),
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
        'attribution': 'Attribution Score (Activation Projection)',
    }
    direction_names = {'model': 'Model (Assistant)', 'user': 'User'}

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(
        f'{direction_names[direction]} Direction - {metric_titles.get(metric, metric)}',
        fontsize=14
    )

    tick_step = max(1, n_layers // 15)
    ax.set_xticks(np.arange(0, n_layers, tick_step))
    ax.legend(loc='upper right', fontsize=8, ncol=4)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_layers - 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_comparison(
    results: Dict[str, np.ndarray],
    metric: str,
    output_path: Path,
):
    """Plot Model vs User comparison for a metric."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    directions = ['model', 'user']
    direction_names = {'model': 'Model (Assistant)', 'user': 'User'}

    for ax, direction in zip(axes, directions):
        avg_key = f"{direction}_{metric}_average"
        if avg_key not in results:
            continue

        avg_values = results[avg_key]
        n_layers = len(avg_values)
        layers = np.arange(n_layers)

        ax.plot(layers, avg_values, color='blue' if direction == 'model' else 'green',
                linewidth=2, label=f'{direction_names[direction]} (avg)')

        # Mark peak
        peak_layer = np.argmax(avg_values)
        ax.axvline(x=peak_layer, color='red', linestyle=':', alpha=0.5)
        ax.annotate(f'Peak: L{peak_layer}', xy=(peak_layer, avg_values[peak_layer]),
                    xytext=(peak_layer + 2, avg_values[peak_layer] * 0.95),
                    fontsize=10, color='red')

        ax.set_ylabel(direction_names[direction], fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel('Layer', fontsize=12)

    metric_titles = {
        'variance_ratio': 'Variance Ratio',
        'pca_alignment': 'PCA Alignment',
        'attribution': 'Attribution Score',
    }
    fig.suptitle(f'Model vs User: {metric_titles.get(metric, metric)}', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot UA layer metrics"
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to results npz file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to same as results file)",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent
    results_path = args.results
    if not results_path.is_absolute():
        results_path = repo_root / results_path

    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    output_dir = args.output_dir or results_path.parent
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from {results_path}")
    results = load_results(results_path)

    # Get emotions from results
    emotions = list(results.get('emotions', []))
    if isinstance(emotions[0], bytes):
        emotions = [e.decode() for e in emotions]

    # Get token position from filename
    token_pos = results_path.stem.replace('ua_layer_metrics_', '')

    # Check if attribution data is available
    has_attribution = bool(results.get('has_attribution', False))
    if isinstance(has_attribution, np.ndarray):
        has_attribution = bool(has_attribution.item())

    metrics = ['variance_ratio', 'pca_alignment']
    if has_attribution:
        metrics.append('attribution')
        print("Attribution data found, will generate attribution plots")
    else:
        print("No attribution data found, skipping attribution plots")

    # Create plots for each direction and metric
    for direction in ['model', 'user']:
        for metric in metrics:
            output_path = output_dir / f"{direction}_{metric}_{token_pos}.png"
            plot_metric(results, metric, direction, output_path, emotions)

    # Create comparison plots
    for metric in metrics:
        output_path = output_dir / f"comparison_{metric}_{token_pos}.png"
        plot_comparison(results, metric, output_path)

    print(f"\nAll plots saved to {output_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("PEAK LAYERS SUMMARY")
    print("=" * 60)

    for direction in ['model', 'user']:
        print(f"\n{direction.upper()}:")
        for metric in metrics:
            avg_key = f"{direction}_{metric}_average"
            if avg_key in results:
                avg_values = results[avg_key]
                peak_layer = np.argmax(avg_values)
                top_5 = np.argsort(avg_values)[-5:][::-1]
                print(f"  {metric}: Peak={peak_layer}, Top 5={list(top_5)}")


if __name__ == "__main__":
    main()
