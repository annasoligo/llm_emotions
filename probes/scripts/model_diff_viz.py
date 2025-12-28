#!/usr/bin/env python3
"""
Visualization utilities for model diffing analysis.
"""

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import json


EMOTION_COLORS = {
    'anger': '#7BA7D7',
    'disgust': '#7D9B7D',
    'fear': '#a59dc9',
    'happiness': '#D4876A',
    'sadness': '#B8CCC8',
    'surprise': '#D1728F'
}


def plot_heatmap(
    results: Dict,
    layers: List[int],
    emotions: List[str],
    title: str = "Double-Diff Heatmap",
    question_module: str = None,
    adapter_name: str = None,
    output_path: Optional[Path] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    figsize: tuple = None
):
    """
    Plot heatmap of double-diff effects across layers.

    Args:
        results: Results dict from DoubleDiffExperiment
        layers: List of layer numbers
        emotions: List of emotion names
        title: Plot title (will be augmented with config details)
        question_module: Name of question module
        adapter_name: Name of adapter/finetuned model
        output_path: Optional path to save figure
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        figsize: Optional figure size (width, height)
    """
    # Extract data
    heatmap_data = np.array([
        [results['double_diff_results']['averaged'][layer]['mean_effect'][emotion]
         for emotion in emotions]
        for layer in layers
    ])

    # Build detailed title with config info
    config = results['config']
    title_parts = [title]

    # Add config details
    details = []
    if question_module:
        details.append(f"Q: {question_module}")
    if adapter_name:
        details.append(f"FT: {adapter_name}")

    probe_info = f"Probe: {config['probe_type']}"
    if config['probe_type'] == 'orthogonal':
        probe_info += f" (ortho={config['orthogonality_weight']}, repr={config['orthogonal_representation']})"
    details.append(probe_info)

    details.append(f"Act: {config['activation_strategy']}")

    if config.get('use_wildchat_normalization'):
        details.append("WildChat norm: ON")

    title_with_config = f"{title}\n{' | '.join(details)}"

    # Create figure
    if figsize is None:
        figsize = (10, len(layers) * 0.5 + 3)  # Extra space for longer title

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)

    # Set ticks
    ax.set_xticks(range(len(emotions)))
    ax.set_xticklabels([e.capitalize() for e in emotions], rotation=45, ha='right')
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"Layer {l}" for l in layers])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Double-Diff Effect', rotation=270, labelpad=20)

    # Add values as text
    for i in range(len(layers)):
        for j in range(len(emotions)):
            value = heatmap_data[i, j]
            color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:+.2f}', ha='center', va='center',
                    color=color, fontsize=9)

    ax.set_title(title_with_config, fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved heatmap to {output_path}")

    return fig, ax


def plot_trajectories(
    results: Dict,
    layers: List[int],
    emotions: List[str],
    probe_type: str = "orthogonal",
    title: str = "Emotion Trajectories",
    question_module: str = None,
    adapter_name: str = None,
    output_path: Optional[Path] = None,
    figsize: tuple = (20, 6)
):
    """
    Plot emotion trajectories across layers.

    Args:
        results: Results dict from DoubleDiffExperiment
        layers: List of layer numbers
        emotions: List of emotion names
        probe_type: "orthogonal" or "linear"
        title: Plot title prefix (will be augmented with config details)
        question_module: Name of question module
        adapter_name: Name of adapter/finetuned model
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    # Build detailed title with config info
    config = results['config']
    details = []
    if question_module:
        details.append(f"Q: {question_module}")
    if adapter_name:
        details.append(f"FT: {adapter_name}")

    probe_info = f"Probe: {config['probe_type']}"
    if config['probe_type'] == 'orthogonal':
        probe_info += f" (ortho={config['orthogonality_weight']}, repr={config['orthogonal_representation']})"
    details.append(probe_info)

    details.append(f"Act: {config['activation_strategy']}")

    if config.get('use_wildchat_normalization'):
        details.append("WildChat norm: ON")

    suptitle = f"{title}: Case 4 Double-Diff with 95% Bootstrap CI\n{' | '.join(details)}"

    if probe_type == "orthogonal":
        # 3-panel plot: user, assistant, averaged
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        for idx, (ax, result_key, title_suffix) in enumerate([
            (axes[0], 'user', "User Probe"),
            (axes[1], 'assistant', "Assistant Probe"),
            (axes[2], 'averaged', "Averaged")
        ]):
            _plot_single_trajectory(
                ax, results['double_diff_results'][result_key],
                layers, emotions, title_suffix
            )

            # Only show legend on last panel
            if idx == 2:
                ax.legend(loc='best', framealpha=0.9, fontsize=9, ncol=2)

        plt.suptitle(suptitle, fontsize=12, fontweight='bold', y=1.0)

    else:
        # Single plot for linear probes
        fig, ax = plt.subplots(figsize=(12, 7))
        _plot_single_trajectory(
            ax, results['double_diff_results']['averaged'],
            layers, emotions, title
        )
        ax.legend(loc='best', framealpha=0.9, fontsize=10, ncol=2)
        ax.set_title(suptitle, fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved trajectories to {output_path}")

    return fig, axes if probe_type == "orthogonal" else ax


def _plot_single_trajectory(ax, results_dict, layers, emotions, title):
    """Helper to plot a single trajectory panel."""
    # Prepare data
    trajectory_data = {
        emotion: {
            'layers': [],
            'means': [],
            'ci_lower': [],
            'ci_upper': []
        }
        for emotion in emotions
    }

    for layer in sorted(layers):
        result = results_dict[layer]
        for emotion in emotions:
            trajectory_data[emotion]['layers'].append(layer)
            trajectory_data[emotion]['means'].append(result['mean_effect'][emotion])
            trajectory_data[emotion]['ci_lower'].append(result['bootstrap_ci'][emotion]['lower'])
            trajectory_data[emotion]['ci_upper'].append(result['bootstrap_ci'][emotion]['upper'])

    # Plot each emotion
    for emotion in emotions:
        data = trajectory_data[emotion]
        color = EMOTION_COLORS[emotion]

        ax.plot(data['layers'], data['means'], marker='o', linewidth=2.5,
                markersize=6, color=color, label=emotion.capitalize(), alpha=0.9)

        ax.fill_between(data['layers'], data['ci_lower'], data['ci_upper'],
                         color=color, alpha=0.2, linewidth=0)

    # Formatting
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, zorder=1)
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Double-Diff Effect', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(sorted(layers))
    ax.set_xticklabels([str(l) for l in sorted(layers)],
                        rotation=45 if len(layers) > 10 else 0)
    ax.grid(axis='both', alpha=0.3)


def plot_comparison(
    results_list: List[Dict],
    labels: List[str],
    layers: List[int],
    emotions: List[str],
    title: str = "Probe Comparison",
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 8)
):
    """
    Plot comparison of multiple experiments (e.g., different probe types).

    Args:
        results_list: List of results dicts from different experiments
        labels: Labels for each experiment
        layers: List of layer numbers
        emotions: List of emotion names
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for emotion_idx, emotion in enumerate(emotions):
        ax = axes[emotion_idx]

        for result_idx, (results, label) in enumerate(zip(results_list, labels)):
            # Extract trajectory for this emotion
            means = []
            ci_lower = []
            ci_upper = []

            for layer in sorted(layers):
                result = results['double_diff_results']['averaged'][layer]
                means.append(result['mean_effect'][emotion])
                ci_lower.append(result['bootstrap_ci'][emotion]['lower'])
                ci_upper.append(result['bootstrap_ci'][emotion]['upper'])

            # Plot
            color = plt.cm.tab10(result_idx)
            ax.plot(sorted(layers), means, marker='o', linewidth=2,
                    label=label, color=color, alpha=0.8)
            ax.fill_between(sorted(layers), ci_lower, ci_upper,
                             color=color, alpha=0.15)

        # Formatting
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Effect', fontsize=10)
        ax.set_title(emotion.capitalize(), fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison to {output_path}")

    return fig, axes


def export_results(
    results: Dict,
    dataset_prompts: List[str],
    baseline_prompts: List[str],
    question_module: str,
    output_path: Path
):
    """
    Export results to JSON.

    Args:
        results: Results dict from DoubleDiffExperiment
        dataset_prompts: Dataset prompts used
        baseline_prompts: Baseline prompts used
        question_module: Name of question module
        output_path: Path to save JSON
    """
    config = results['config']
    double_diff = results['double_diff_results']

    export_data = {
        'config': {
            **config,
            'question_module': question_module,
            'n_dataset_prompts': len(dataset_prompts),
            'n_baseline_prompts': len(baseline_prompts),
        },
        'prompts': {
            'dataset_examples': dataset_prompts[:3],
            'baseline_examples': baseline_prompts[:3],
        },
        'results_by_layer': {}
    }

    for layer in config['layers']:
        layer_export = {
            'averaged': {
                'mean_effect': double_diff['averaged'][layer]['mean_effect'],
                'bootstrap_ci': double_diff['averaged'][layer]['bootstrap_ci'],
                'emotion_ranking': [
                    {'emotion': e, 'abs_value': float(v)}
                    for e, v in double_diff['averaged'][layer]['emotion_ranking']
                ],
                'per_pair_effects': double_diff['averaged'][layer]['per_pair_effects']
            }
        }

        # Add user/assistant if orthogonal
        if config['probe_type'] == "orthogonal":
            layer_export['user'] = {
                'mean_effect': double_diff['user'][layer]['mean_effect'],
                'bootstrap_ci': double_diff['user'][layer]['bootstrap_ci'],
                'emotion_ranking': [
                    {'emotion': e, 'abs_value': float(v)}
                    for e, v in double_diff['user'][layer]['emotion_ranking']
                ],
            }
            layer_export['assistant'] = {
                'mean_effect': double_diff['assistant'][layer]['mean_effect'],
                'bootstrap_ci': double_diff['assistant'][layer]['bootstrap_ci'],
                'emotion_ranking': [
                    {'emotion': e, 'abs_value': float(v)}
                    for e, v in double_diff['assistant'][layer]['emotion_ranking']
                ],
            }

        export_data['results_by_layer'][layer] = layer_export

    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"✓ Saved results to {output_path}")