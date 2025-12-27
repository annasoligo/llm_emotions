#!/usr/bin/env python3
"""Plotting functions for probe-based emotion detection results."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional


# Use emo lens color scheme
EMOTION_COLORS = {
    'anger': '#7BA7D7',      # sky_blue
    'disgust': '#7D9B7D',    # olive
    'fear': '#a59dc9',       # lavender
    'happiness': '#D4876A',  # coral
    'sadness': '#B8CCC8',    # sage
    'surprise': '#D1728F'    # darker pink
}


def plot_multilayer_results(
    results: Dict,
    output_dir: Path,
    experiment_name: str = "probe_experiment",
    layer_range: tuple = (20, 50)
):
    """Plot multilayer probe experiment results.

    Args:
        results: Results dictionary with 'results_by_layer' key
        output_dir: Directory to save plots
        experiment_name: Name prefix for plot files
        layer_range: (min_layer, max_layer) to restrict plot range (default: 20-50)
    """
    # Extract data
    all_layers = sorted([int(l) for l in results['results_by_layer'].keys()])

    # Filter layers to range
    layers = [l for l in all_layers if layer_range[0] <= l <= layer_range[1]]

    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

    # Build data matrix
    data = np.zeros((len(emotions), len(layers)))
    for i, layer in enumerate(layers):
        layer_results = results['results_by_layer'][str(layer)]
        for j, emotion in enumerate(emotions):
            data[j, i] = layer_results['mean'][emotion]

    # Plot 1: Heatmap
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions, fontsize=12)
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('Emotion', fontsize=13)
    ax.set_title(f'Case 4: {experiment_name} | [assistant_token]\nEmotion Scores by Layer (mean)', fontsize=14)
    plt.colorbar(im, ax=ax, label='Mean')

    # Set x-ticks every 5 layers
    tick_indices = range(0, len(layers), 5)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f'L{layers[i]}' for i in tick_indices])

    plt.tight_layout()
    heatmap_path = output_dir / f'{experiment_name}_case4_heatmap.png'
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved heatmap to {heatmap_path}")

    # Plot 2: Trajectories (emo lens style)
    fig, ax = plt.subplots(figsize=(12, 7))

    for j, emotion in enumerate(emotions):
        color = EMOTION_COLORS.get(emotion, '#808080')
        ax.plot(layers, data[j, :], marker='o', label=emotion.capitalize(),
                color=color, linewidth=2.5, markersize=5, alpha=0.8)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('Mean', fontsize=13)
    ax.set_title(f'Case 4: {experiment_name} | [assistant_token]\nEmotion Trajectories Across Layers (mean)', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    traj_path = output_dir / f'{experiment_name}_case4_trajectories_overlayed.png'
    plt.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved trajectories to {traj_path}")


def plot_singlelayer_results(results: Dict, output_dir: Path, experiment_name: str = "probe_experiment"):
    """Plot single-layer probe experiment results.

    Args:
        results: Results dictionary with 'results_by_layer' key
        output_dir: Directory to save plots
        experiment_name: Name prefix for plot files
    """
    # Get the single layer
    layer_key = list(results['results_by_layer'].keys())[0]
    layer_results = results['results_by_layer'][layer_key]

    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    means = [layer_results['mean'][e] for e in emotions]

    # Extract CI bounds
    ci_lower = [layer_results['bootstrap_ci'][e]['lower'] for e in emotions]
    ci_upper = [layer_results['bootstrap_ci'][e]['upper'] for e in emotions]
    errors_lower = [means[i] - ci_lower[i] for i in range(len(emotions))]
    errors_upper = [ci_upper[i] - means[i] for i in range(len(emotions))]

    # Plot bar chart with error bars (emo lens colors)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(emotions))
    colors = [EMOTION_COLORS.get(e, '#808080') for e in emotions]

    ax.bar(x, means, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax.errorbar(x, means, yerr=[errors_lower, errors_upper], fmt='none',
                ecolor='black', capsize=5, capthick=2)

    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in emotions], rotation=0, ha='center', fontsize=11)
    ax.set_ylabel('Mean Effect', fontsize=12)
    ax.set_title(f'Probe-Based Emotion Detection: {experiment_name} (Layer {layer_key})', fontsize=13)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    bar_path = output_dir / f'{experiment_name}_bars.png'
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved bar chart to {bar_path}")


def plot_probe_results(results: Dict, output_dir: Path, experiment_name: str = "probe_experiment"):
    """Plot probe experiment results (dispatches to multilayer or singlelayer).

    Args:
        results: Results dictionary
        output_dir: Directory to save plots
        experiment_name: Name prefix for plot files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if results.get('multilayer', False):
        plot_multilayer_results(results, output_dir, experiment_name)
    else:
        plot_singlelayer_results(results, output_dir, experiment_name)
