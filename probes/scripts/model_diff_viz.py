#!/usr/bin/env python3
"""
Visualization utilities for model diffing analysis.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import json
import textwrap


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


def plot_token_trajectories(
    scores_dict: Dict[int, np.ndarray],
    emotions: List[str],
    token_strings: List[str],
    title: str = "Token-Level Emotion Trajectories",
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 7),
    window_size: Optional[int] = None
):
    """
    Plot token-level emotion trajectories.

    Args:
        scores_dict: Dict mapping token_pos -> emotion_scores [n_emotions]
        emotions: List of emotion names
        token_strings: List of token strings for labels
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size
        window_size: Optional window size for moving average smoothing
    """
    fig, ax = plt.subplots(figsize=figsize)

    token_positions = sorted(scores_dict.keys())

    # Plot each emotion
    for emotion in emotions:
        emotion_idx = emotions.index(emotion)
        scores = [scores_dict[pos][emotion_idx] for pos in token_positions]
        color = EMOTION_COLORS[emotion]

        if window_size:
            # Apply moving average
            smoothed_scores = _moving_average(scores, window_size)
            ax.plot(token_positions, smoothed_scores, linewidth=2.5,
                    color=color, label=emotion.capitalize(), alpha=0.9)
        else:
            ax.plot(token_positions, scores, marker='o', linewidth=2.5, markersize=4,
                    color=color, label=emotion.capitalize(), alpha=0.9)

    # Formatting
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Emotion Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    ax.grid(axis='both', alpha=0.3)
    ax.legend(loc='best', framealpha=0.9, fontsize=10, ncol=2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved trajectories to {output_path}")

    return fig, ax


def plot_token_trajectories_orthogonal(
    user_scores: Dict[int, np.ndarray],
    asst_scores: Dict[int, np.ndarray],
    averaged_scores: Dict[int, np.ndarray],
    emotions: List[str],
    token_strings: List[str],
    title: str = "Token-Level Emotion Trajectories: Orthogonal Probes",
    output_path: Optional[Path] = None,
    figsize: tuple = (20, 6),
    window_size: Optional[int] = None
):
    """
    Plot token-level emotion trajectories for orthogonal probes (3-panel).

    Args:
        user_scores: Dict mapping token_pos -> emotion_scores [n_emotions] for user probe
        asst_scores: Dict mapping token_pos -> emotion_scores [n_emotions] for assistant probe
        averaged_scores: Dict mapping token_pos -> emotion_scores [n_emotions] averaged
        emotions: List of emotion names
        token_strings: List of token strings for labels
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size
        window_size: Optional window size for moving average smoothing
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=12, fontweight='bold')

    token_positions = sorted(averaged_scores.keys())

    for idx, (ax, scores_dict, title_suffix) in enumerate([
        (axes[0], user_scores, "User Probe"),
        (axes[1], asst_scores, "Assistant Probe"),
        (axes[2], averaged_scores, "Averaged")
    ]):
        # Plot each emotion
        for emotion in emotions:
            emotion_idx = emotions.index(emotion)
            scores = [scores_dict[pos][emotion_idx] for pos in token_positions]
            color = EMOTION_COLORS[emotion]

            if window_size:
                # Apply moving average
                smoothed_scores = _moving_average(scores, window_size)
                ax.plot(token_positions, smoothed_scores, linewidth=2.5,
                        color=color, label=emotion.capitalize(), alpha=0.9)
            else:
                ax.plot(token_positions, scores, marker='o', linewidth=2.5, markersize=4,
                        color=color, label=emotion.capitalize(), alpha=0.9)

        # Formatting
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.set_xlabel('Token Position', fontsize=12, fontweight='bold')
        ax.set_ylabel('Emotion Score', fontsize=12, fontweight='bold')
        ax.set_title(title_suffix, fontsize=13, fontweight='bold')
        ax.grid(axis='both', alpha=0.3)

        # Only show legend on last panel
        if idx == 2:
            ax.legend(loc='best', framealpha=0.9, fontsize=9, ncol=2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved trajectories to {output_path}")

    return fig, axes


def _moving_average(data: List[float], window_size: int) -> np.ndarray:
    """Apply moving average smoothing."""
    data_array = np.array(data)
    if len(data_array) < window_size:
        return data_array

    cumsum = np.cumsum(np.insert(data_array, 0, 0))
    smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / window_size

    # Pad the beginning to maintain length
    pad_size = window_size - 1
    pad_values = np.full(pad_size, smoothed[0])
    return np.concatenate([pad_values, smoothed])


def plot_token_trajectories_with_labels(
    scores_dict: Dict[int, np.ndarray],
    emotions: List[str],
    token_strings: List[str],
    title: str = "Token-Level Emotion Trajectories",
    output_path: Optional[Path] = None,
    figsize: tuple = (16, 8),
    max_tokens: int = 30,
    window_size: Optional[int] = None
):
    """
    Plot token-level trajectories with token strings on x-axis.

    Args:
        scores_dict: Dict mapping token_pos -> emotion_scores [n_emotions]
        emotions: List of emotion names
        token_strings: List of token strings for labels
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size
        max_tokens: Maximum number of tokens to show
        window_size: Optional window size for moving average smoothing
    """
    fig, ax = plt.subplots(figsize=figsize)

    token_positions = sorted(scores_dict.keys())[:max_tokens]

    # Plot each emotion
    for emotion in emotions:
        emotion_idx = emotions.index(emotion)
        scores = [scores_dict[pos][emotion_idx] for pos in token_positions]
        color = EMOTION_COLORS[emotion]

        if window_size:
            smoothed_scores = _moving_average(scores, window_size)
            ax.plot(range(len(token_positions)), smoothed_scores, linewidth=2.5,
                    color=color, label=emotion.capitalize(), alpha=0.9)
        else:
            ax.plot(range(len(token_positions)), scores, marker='o', linewidth=2.5, markersize=5,
                    color=color, label=emotion.capitalize(), alpha=0.9)

    # Formatting
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Token', fontsize=12, fontweight='bold')
    ax.set_ylabel('Emotion Score', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='best', framealpha=0.9, fontsize=10, ncol=2)

    # Set x-axis labels to token strings
    ax.set_xticks(range(len(token_positions)))
    token_labels = [token_strings[pos] for pos in token_positions]
    ax.set_xticklabels(token_labels, rotation=45, ha='right', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved trajectories to {output_path}")

    return fig, ax


def split_into_sentences(token_strings: List[str], token_ids: List[int], chunk_size: int = 15) -> List[Tuple[int, int, str]]:
    """
    Split tokens into fixed-size chunks.

    Args:
        token_strings: List of decoded token strings
        token_ids: List of token IDs
        chunk_size: Number of tokens per chunk (default: 15)

    Returns:
        List of (start_idx, end_idx, chunk_text) tuples
    """
    chunks = []

    for start_idx in range(0, len(token_strings), chunk_size):
        end_idx = min(start_idx + chunk_size, len(token_strings))
        chunk_text = ''.join(token_strings[start_idx:end_idx])
        chunks.append((start_idx, end_idx, chunk_text.strip()))

    return chunks


def aggregate_scores_over_range(
    scores_dict: Dict[int, np.ndarray],
    start_idx: int,
    end_idx: int
) -> np.ndarray:
    """
    Average emotion scores over a range of token positions.

    Args:
        scores_dict: Dict mapping token_pos -> emotion_scores [6]
        start_idx: Start token position (inclusive)
        end_idx: End token position (exclusive)

    Returns:
        Averaged emotion scores [6]
    """
    scores_in_range = []
    for pos in range(start_idx, end_idx):
        if pos in scores_dict:
            scores_in_range.append(scores_dict[pos])

    if len(scores_in_range) == 0:
        return np.zeros(6)

    return np.mean(scores_in_range, axis=0)


def plot_sentence_level_bar_charts(
    scores_dict: Dict[int, np.ndarray],
    emotions: List[str],
    token_strings: List[str],
    token_ids: List[int],
    prompt_start_idx: int = 20,
    title: str = "Chunk-Level Emotion Analysis (15 tokens per chunk)",
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 10)
):
    """
    Create bar chart visualizations showing emotions for prompt and each 15-token chunk.

    Args:
        scores_dict: Dict mapping token_pos -> emotion_scores [6]
        emotions: List of emotion names
        token_strings: List of decoded token strings
        token_ids: List of token IDs
        prompt_start_idx: Token index where prompt emotions start being averaged
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size

    Returns:
        fig, axes
    """
    # Find where generation starts (look for common assistant markers)
    generation_start_idx = len(token_strings)
    for i in range(len(token_strings)):
        token = token_strings[i]
        # Look for common patterns that indicate generation start
        if i > prompt_start_idx and ('model' in token.lower() or '\n' == token):
            # Check if this looks like the assistant turn marker
            if i + 1 < len(token_strings):
                generation_start_idx = i + 1
                break

    # If we didn't find a clear marker, use a heuristic (80% through)
    if generation_start_idx == len(token_strings):
        generation_start_idx = int(len(token_strings) * 0.8)

    # Split generation into 15-token chunks
    generation_token_strings = token_strings[generation_start_idx:]
    generation_token_ids = token_ids[generation_start_idx:]
    chunks = split_into_sentences(generation_token_strings, generation_token_ids, chunk_size=15)

    # Compute emotion scores
    # 1. Prompt (from prompt_start_idx to generation_start_idx)
    prompt_scores = aggregate_scores_over_range(scores_dict, prompt_start_idx, generation_start_idx)

    # 2. Each chunk
    chunk_scores = []
    chunk_labels = []
    for chunk_idx, (start_offset, end_offset, chunk_text) in enumerate(chunks):
        # Convert offsets to absolute token positions
        abs_start = generation_start_idx + start_offset
        abs_end = generation_start_idx + end_offset

        scores = aggregate_scores_over_range(scores_dict, abs_start, abs_end)
        chunk_scores.append(scores)

        # Create label with full text (will be wrapped in title)
        chunk_labels.append((chunk_idx + 1, chunk_text))

    # Create subplots: 1 for prompt + N for chunks
    n_plots = 1 + len(chunks)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Auto-scale figure height based on number of rows
    # Use 4 inches per row for good spacing
    fig_width = figsize[0] if isinstance(figsize, tuple) else 14
    fig_height = max(4 * n_rows, 6)  # Minimum 6 inches, 4 inches per row

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot 1: Prompt emotions
    ax = axes[0]
    x_pos = np.arange(len(emotions))
    colors = [EMOTION_COLORS.get(e, '#808080') for e in emotions]
    bars = ax.bar(x_pos, prompt_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_title('Prompt\n(Tokens {}-{})'.format(prompt_start_idx, generation_start_idx-1),
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Emotion Score', fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([e.capitalize() for e in emotions], rotation=45, ha='right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if abs(height) > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8)

    # Plots 2+: Each chunk
    for plot_idx, (scores, (chunk_num, chunk_text)) in enumerate(zip(chunk_scores, chunk_labels)):
        ax = axes[plot_idx + 1]
        bars = ax.bar(x_pos, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Wrap text for title - break into lines of ~60 chars
        wrapped_lines = textwrap.wrap(chunk_text, width=60)
        title_text = f"Chunk {chunk_num}:\n" + "\n".join(wrapped_lines)
        ax.set_title(title_text, fontsize=9, fontweight='bold')

        ax.set_ylabel('Emotion Score', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([e.capitalize() for e in emotions], rotation=45, ha='right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=8)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved chunk-level bar charts to {output_path}")

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