#!/usr/bin/env python3
"""Plot layer metrics for emotion steering vector selection.

Creates 3 plots (one per metric):
- X-axis: Layer (0 to n_layers-1)
- Y-axis: Metric value
- 6 colored lines (one per emotion)
- 1 black dashed line (average across emotions)

Usage:
    python -m experiments.layer_selection.plot_layer_metrics \
        --results experiments/layer_selection/results/layer_metrics.npz \
        --output-dir experiments/layer_selection/results/
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# Color scheme for emotions (matching eval_dashboard)
EMOTION_COLORS = {
    "anger": "#7BA7D7",      # Sky blue
    "disgust": "#7D9B7D",    # Olive
    "fear": "#a59dc9",       # Lavender
    "happiness": "#D4876A",  # Coral
    "sadness": "#B8CCC8",    # Sage
    "surprise": "#D1728F",   # Darker pink
}

EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def smooth_values(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Apply running average smoothing."""
    if window <= 1:
        return values
    kernel = np.ones(window) / window
    # Use 'same' mode and handle edges by padding
    padded = np.pad(values, (window // 2, window // 2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(values)]


def load_results(results_path: Path) -> Dict[str, np.ndarray]:
    """Load results from npz file."""
    data = np.load(results_path, allow_pickle=True)
    return dict(data)


def plot_metric(
    results: Dict[str, np.ndarray],
    metric: str,
    output_path: Path,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
):
    """Plot a single metric across layers.

    Args:
        results: Dictionary with keys like 'metric_emotion' and 'metric_average'
        metric: One of 'attribution', 'variance_ratio', 'pca_alignment'
        output_path: Path to save the plot
        title: Optional custom title
        ylabel: Optional custom y-axis label
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Get number of layers from data
    avg_key = f"{metric}_average"
    if avg_key not in results:
        print(f"Warning: {avg_key} not found in results")
        return

    n_layers = len(results[avg_key])
    layers = np.arange(n_layers)

    # Plot each emotion (with smoothing)
    for emotion in EMOTIONS:
        key = f"{metric}_{emotion}"
        if key in results:
            values = smooth_values(results[key], window=3)
            ax.plot(
                layers, values,
                color=EMOTION_COLORS[emotion],
                linewidth=2.5,
                alpha=0.8,
                label=emotion.capitalize(),
            )

    # Plot average (solid grey)
    avg_values = smooth_values(results[avg_key], window=3)
    ax.plot(
        layers, avg_values,
        color='#555555',
        linewidth=3,
        linestyle='-',
        label='Average',
    )

    # Highlight top 5 layers for average
    top_5 = np.argsort(avg_values)[-5:][::-1]
    for layer in top_5:
        ax.axvline(x=layer, color='gray', linestyle=':', alpha=0.3)

    # Mark the peak layer
    peak_layer = np.argmax(avg_values)
    ax.axvline(x=peak_layer, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.annotate(
        f'Peak: L{peak_layer}',
        xy=(peak_layer, avg_values[peak_layer]),
        xytext=(peak_layer + 2, avg_values[peak_layer]),
        fontsize=10,
        color='red',
    )

    # Labels and styling
    metric_titles = {
        'attribution': 'Attribution Scores (Gradient-based, 3-layer running avg)',
        'variance_ratio': 'Variance Ratio vs Random Directions (3-layer running avg)',
        'pca_alignment': 'PCA Alignment in Top-50 PC Subspace (3-layer running avg)',
    }
    metric_ylabels = {
        'attribution': 'Attribution Score',
        'variance_ratio': 'Variance Ratio',
        'pca_alignment': 'Alignment (0-1)',
    }

    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel(ylabel or metric_ylabels.get(metric, 'Value'), fontsize=14)
    ax.set_title(title or metric_titles.get(metric, metric), fontsize=16)

    # Set x-axis ticks
    tick_step = max(1, n_layers // 15)
    ax.set_xticks(np.arange(0, n_layers, tick_step))
    ax.tick_params(axis='both', labelsize=12)

    # Legend
    ax.legend(loc='upper right', fontsize=12, ncol=2)

    # Grid
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_layers - 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_combined(
    results: Dict[str, np.ndarray],
    output_path: Path,
):
    """Create a combined plot with all 3 metrics (normalized for comparison)."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    metrics = ['attribution', 'variance_ratio', 'pca_alignment']
    titles = [
        'Attribution Scores',
        'Variance Ratio',
        'PCA Alignment',
    ]

    avg_key = f"{metrics[0]}_average"
    n_layers = len(results[avg_key])
    layers = np.arange(n_layers)

    for ax, metric, title in zip(axes, metrics, titles):
        # Plot each emotion (with smoothing)
        for emotion in EMOTIONS:
            key = f"{metric}_{emotion}"
            if key in results:
                values = smooth_values(results[key], window=3)
                ax.plot(
                    layers, values,
                    color=EMOTION_COLORS[emotion],
                    linewidth=2,
                    alpha=0.7,
                    label=emotion.capitalize() if ax == axes[0] else None,
                )

        # Plot average (solid grey, with smoothing)
        avg_values = smooth_values(results[f"{metric}_average"], window=3)
        ax.plot(
            layers, avg_values,
            color='#555555',
            linewidth=2.5,
            linestyle='-',
            label='Average' if ax == axes[0] else None,
        )

        # Mark peak
        peak_layer = np.argmax(avg_values)
        ax.axvline(x=peak_layer, color='red', linestyle=':', alpha=0.5)

        ax.set_ylabel(title, fontsize=13)
        ax.tick_params(axis='both', labelsize=11)
        ax.grid(True, alpha=0.3)

    # X-axis label only on bottom plot
    axes[-1].set_xlabel('Layer', fontsize=14)

    # Set x-axis ticks
    tick_step = max(1, n_layers // 15)
    axes[-1].set_xticks(np.arange(0, n_layers, tick_step))

    # Legend inside top plot
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='upper left', ncol=4, fontsize=13)

    fig.suptitle('Layer Selection Metrics for Emotion Steering (3-layer running avg)', fontsize=16, y=1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def plot_heatmap(
    results: Dict[str, np.ndarray],
    metric: str,
    output_path: Path,
):
    """Create a heatmap of metric values across layers and emotions."""
    avg_key = f"{metric}_average"
    n_layers = len(results[avg_key])

    # Build matrix [n_emotions, n_layers]
    matrix = []
    for emotion in EMOTIONS:
        key = f"{metric}_{emotion}"
        if key in results:
            matrix.append(results[key])

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(16, 4))

    im = ax.imshow(matrix, aspect='auto', cmap='viridis')
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels([e.capitalize() for e in EMOTIONS])

    # X-axis ticks
    tick_step = max(1, n_layers // 20)
    ax.set_xticks(np.arange(0, n_layers, tick_step))
    ax.set_xticklabels(np.arange(0, n_layers, tick_step))

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Emotion', fontsize=12)

    metric_titles = {
        'attribution': 'Attribution Scores',
        'variance_ratio': 'Variance Ratio',
        'pca_alignment': 'PCA Alignment',
    }
    ax.set_title(f'{metric_titles.get(metric, metric)} Heatmap', fontsize=14)

    plt.colorbar(im, ax=ax, label='Value')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot layer metrics for emotion steering"
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("experiments/layer_selection/results/layer_metrics.npz"),
        help="Path to results npz file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to same as results file)",
    )

    args = parser.parse_args()

    # Resolve paths
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

    # Check what metrics are available
    metrics = ['attribution', 'variance_ratio', 'pca_alignment']
    available_metrics = [m for m in metrics if f"{m}_average" in results]
    print(f"Available metrics: {available_metrics}")

    # Create individual plots for each metric
    for metric in available_metrics:
        output_path = output_dir / f"{metric}_scores_by_layer.png"
        plot_metric(results, metric, output_path)

        # Also create heatmap
        heatmap_path = output_dir / f"{metric}_heatmap.png"
        plot_heatmap(results, metric, heatmap_path)

    # Create combined plot
    if len(available_metrics) == 3:
        combined_path = output_dir / "combined_layer_metrics.png"
        plot_combined(results, combined_path)

    print(f"\nAll plots saved to {output_dir}")

    # Print summary of peak layers
    print("\n" + "=" * 60)
    print("PEAK LAYERS SUMMARY")
    print("=" * 60)

    for metric in available_metrics:
        avg_values = results[f"{metric}_average"]
        peak_layer = np.argmax(avg_values)
        top_5 = np.argsort(avg_values)[-5:][::-1]
        print(f"\n{metric}:")
        print(f"  Peak layer: {peak_layer} (value: {avg_values[peak_layer]:.4f})")
        print(f"  Top 5 layers: {list(top_5)}")


if __name__ == "__main__":
    main()
