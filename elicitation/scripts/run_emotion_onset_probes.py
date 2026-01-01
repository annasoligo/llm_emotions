#!/usr/bin/env python3
"""
Emotion Onset Probe Analysis

Applies emotion probes to window activations extracted around emotion onset positions.
Compares baseline vs onset windows to identify emotion signatures.

This script reuses the existing probe infrastructure from token_level_experiment_v2.py.

Usage:
    python run_emotion_onset_probes.py [--probe-type orthogonal|linear|centroid]
"""

import sys
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from transformers import AutoTokenizer

# Add paths
research_tools_path = Path("/workspace-vast/annas/git/research-tools")
believe_path = Path("/workspace-vast/annas/git/believe-it-or-not")
sys.path.insert(0, str(research_tools_path))
sys.path.insert(0, str(believe_path))

from probes.scripts.token_level_helpers import (
    TokenLevelExperiment,
    aggregate_scores_across_layers,
    get_token_strings
)
from probes.scripts.model_diff_viz import (
    EMOTION_COLORS,
    plot_token_trajectories,
    plot_token_trajectories_orthogonal,
)
from probes.scripts.probe_pipeline import ProbeInference

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ProbeConfig:
    """Configuration for probe analysis."""
    # Probe settings
    probe_type: str = "orthogonal"  # "orthogonal", "linear", or "centroid"

    # Paths for orthogonal probes
    probe_dir_orthogonal: Path = Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/")
    cpca_path_orthogonal: Path = Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/conversation_based/global/google/google/gemma-3-27b-it_cpca.npz")

    # Paths for linear probes
    probe_dir_linear: Path = Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/text_based/multiseed/")
    cpca_path_linear: Path = Path("/workspace-vast/annas/git/research-tools/probes/results/cpca_tier_data_high_alpha.tmp/google/gemma-3-27b-it_cpca.npz")
    probe_pattern_linear: str = "probe_layer{layer}_nc0_seed0.pkl"

    # Paths for centroid probes
    probe_dir_centroid: Path = Path("/workspace-vast/annas/git/research-tools/probes/emotion_probes/conversation/")
    k_value_centroid: int = 10

    # Baseline normalization
    use_baseline_normalization: bool = False  # Set to False - use probe normalization instead
    normalize_probe_scores: bool = True  # NEW: Z-score normalize probe outputs directly
    baseline_aggregation: str = "all_tokens"
    baseline_dir: Path = Path("/workspace-vast/annas/git/research-tools/data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it")
    center_probe_scores: bool = False

    # Probe-specific settings
    orthogonality_weight: float = 1000.0
    orthogonal_representation: str = "raw"  # Use raw activations (not cPCA)
    n_components: int = 10
    seed: int = 0

    # Analysis settings
    emotions: List[str] = None
    layers: List[int] = None  # If None, will default to 20-40

    def __post_init__(self):
        if self.emotions is None:
            self.emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']


# ============================================================================
# Helper Functions
# ============================================================================

def convert_window_to_token_level(window_data: Dict) -> Dict[int, Dict[int, np.ndarray]]:
    """
    Convert window activation format to token-level format expected by TokenLevelExperiment.

    Args:
        window_data: Window data with format:
            {
                'activations': {layer_idx: tensor[num_tokens, hidden_size]},
                'token_positions': [pos1, pos2, ...],
                'token_strings': ['tok1', 'tok2', ...]
            }

    Returns:
        activations_by_token: {token_pos: {layer_idx: array[hidden_size]}}
    """
    activations_by_token = {}

    for i, pos in enumerate(window_data['token_positions']):
        activations_by_token[pos] = {}
        for layer_idx, layer_acts in window_data['activations'].items():
            # Convert tensor to numpy if needed
            if torch.is_tensor(layer_acts):
                # Convert bfloat16 to float32 first for numpy compatibility
                if layer_acts.dtype == torch.bfloat16:
                    layer_acts = layer_acts.float()
                layer_acts = layer_acts.cpu().numpy()
            activations_by_token[pos][layer_idx] = layer_acts[i]

    return activations_by_token


def aggregate_window_scores(
    scores_by_token: Dict[int, Dict[int, np.ndarray]],
    layers: List[int],
    aggregation: str = 'mean'
) -> Dict[int, np.ndarray]:
    """
    Aggregate emotion scores across all tokens in a window.

    Args:
        scores_by_token: {token_pos: {layer: scores}}
        layers: List of layer indices
        aggregation: 'mean' or 'median'

    Returns:
        {layer: aggregated_scores} where scores are averaged across tokens
    """
    layer_scores = {layer: [] for layer in layers}

    # Collect scores for each layer
    for token_pos, layer_dict in scores_by_token.items():
        for layer in layers:
            score = layer_dict[layer]

            # Handle dict format (conversation-based probes with user/assistant)
            if isinstance(score, dict) and 'user' in score:
                # Average user and assistant scores
                avg_score = (score['user'] + score['assistant']) / 2
                layer_scores[layer].append(avg_score)
            else:
                # Simple array format
                layer_scores[layer].append(score)

    # Aggregate across tokens
    aggregated = {}
    for layer in layers:
        if len(layer_scores[layer]) == 0:
            continue

        scores_array = np.array(layer_scores[layer])
        if aggregation == 'mean':
            aggregated[layer] = np.mean(scores_array, axis=0)
        elif aggregation == 'median':
            aggregated[layer] = np.median(scores_array, axis=0)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    return aggregated


def aggregate_across_layers(
    layer_scores: Dict[int, np.ndarray],
    aggregation: str = 'mean'
) -> np.ndarray:
    """
    Aggregate scores across layers.

    Args:
        layer_scores: {layer: scores}
        aggregation: 'mean' or 'median'

    Returns:
        aggregated_scores: Single score vector
    """
    scores_list = list(layer_scores.values())
    scores_array = np.array(scores_list)

    if aggregation == 'mean':
        return np.mean(scores_array, axis=0)
    elif aggregation == 'median':
        return np.median(scores_array, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def compare_windows_statistical(
    baseline_scores: List[np.ndarray],
    onset_scores: List[np.ndarray],
    emotion_names: List[str]
) -> Dict:
    """
    Perform statistical comparison between baseline and onset windows.

    Args:
        baseline_scores: List of score vectors (one per sample)
        onset_scores: List of score vectors (one per sample)
        emotion_names: List of emotion names

    Returns:
        Dict with t-statistics, p-values, and effect sizes
    """
    n_emotions = len(emotion_names)
    n_samples = len(baseline_scores)

    # Organize scores by emotion
    baseline_by_emotion = {i: [] for i in range(n_emotions)}
    onset_by_emotion = {i: [] for i in range(n_emotions)}

    for baseline, onset in zip(baseline_scores, onset_scores):
        for i in range(n_emotions):
            baseline_by_emotion[i].append(baseline[i])
            onset_by_emotion[i].append(onset[i])

    # Paired t-test for each emotion
    results = {
        'emotion_names': emotion_names,
        't_statistics': [],
        'p_values': [],
        'effect_sizes': [],  # Cohen's d
        'baseline_mean': [],
        'onset_mean': [],
        'baseline_std': [],
        'onset_std': []
    }

    for i in range(n_emotions):
        baseline_vals = np.array(baseline_by_emotion[i])
        onset_vals = np.array(onset_by_emotion[i])

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(onset_vals, baseline_vals)

        # Cohen's d for paired samples
        diff = onset_vals - baseline_vals
        d = np.mean(diff) / np.std(diff)

        results['t_statistics'].append(t_stat)
        results['p_values'].append(p_val)
        results['effect_sizes'].append(d)
        results['baseline_mean'].append(np.mean(baseline_vals))
        results['onset_mean'].append(np.mean(onset_vals))
        results['baseline_std'].append(np.std(baseline_vals))
        results['onset_std'].append(np.std(onset_vals))

    return results


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_window_comparison(
    baseline_scores: List[np.ndarray],
    onset_scores: List[np.ndarray],
    emotions: List[str],
    stats_results: Dict,
    output_path: Path
):
    """
    Plot comparison of baseline vs onset emotion scores.

    Args:
        baseline_scores: List of baseline score vectors
        onset_scores: List of onset score vectors
        emotions: List of emotion names
        stats_results: Statistical comparison results
        output_path: Path to save plot
    """
    n_emotions = len(emotions)

    # Prepare data
    baseline_means = stats_results['baseline_mean']
    onset_means = stats_results['onset_mean']
    baseline_stds = stats_results['baseline_std']
    onset_stds = stats_results['onset_std']
    p_values = stats_results['p_values']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_emotions)
    width = 0.35

    # Plot bars
    bars1 = ax.bar(x - width/2, baseline_means, width,
                   yerr=baseline_stds, label='Baseline',
                   color='lightblue', capsize=5)
    bars2 = ax.bar(x + width/2, onset_means, width,
                   yerr=onset_stds, label='Onset',
                   color='salmon', capsize=5)

    # Add significance markers
    max_height = max(max(baseline_means), max(onset_means))
    for i, p in enumerate(p_values):
        if p < 0.001:
            marker = '***'
        elif p < 0.01:
            marker = '**'
        elif p < 0.05:
            marker = '*'
        else:
            continue

        y_pos = max_height * 1.1
        ax.text(i, y_pos, marker, ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Styling
    ax.set_xlabel('Emotion', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probe Score', fontsize=12, fontweight='bold')
    ax.set_title('Baseline vs Onset Window Comparison\n(* p<0.05, ** p<0.01, *** p<0.001)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in emotions])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_all_windows_comparison(
    window_scores: Dict[str, List[np.ndarray]],
    emotions: List[str],
    output_path: Path
):
    """
    Plot comparison across all windows (baseline, pre_onset, onset, post_onset).

    Args:
        window_scores: {'window_name': [score_vectors]}
        emotions: List of emotion names
        output_path: Path to save plot
    """
    n_emotions = len(emotions)
    window_names = list(window_scores.keys())
    n_windows = len(window_names)

    # Compute means and stds
    window_means = {}
    window_stds = {}
    for window_name, scores_list in window_scores.items():
        scores_array = np.array(scores_list)
        window_means[window_name] = np.mean(scores_array, axis=0)
        window_stds[window_name] = np.std(scores_array, axis=0)

    # Create figure with subplots for each emotion
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for emotion_idx, emotion in enumerate(emotions):
        ax = axes[emotion_idx]

        # Extract means and stds for this emotion
        means = [window_means[w][emotion_idx] for w in window_names]
        stds = [window_stds[w][emotion_idx] for w in window_names]

        # Plot line with error bars
        x = np.arange(n_windows)
        color = EMOTION_COLORS.get(emotion, 'gray')
        ax.plot(x, means, 'o-', color=color, linewidth=2, markersize=8, label=emotion.capitalize())
        ax.fill_between(x,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.3, color=color)

        # Styling
        ax.set_title(emotion.capitalize(), fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([w.replace('_', ' ').title() for w in window_names], rotation=45, ha='right')
        ax.set_ylabel('Probe Score', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.suptitle('Emotion Scores Across Windows', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# Main Experiment
# ============================================================================

def run_probe_analysis(
    window_activations_file: str,
    config: ProbeConfig,
    output_dir: str
):
    """
    Run emotion probe analysis on window activations.

    Args:
        window_activations_file: Path to window_activations.pkl
        config: Probe configuration
        output_dir: Output directory for results
    """
    print("="*80)
    print("EMOTION ONSET PROBE ANALYSIS")
    print("="*80)
    print(f"Input: {window_activations_file}")
    print(f"Probe type: {config.probe_type}")
    print(f"Output: {output_dir}")
    print()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load window activations
    print("\n" + "="*80)
    print("LOADING WINDOW ACTIVATIONS")
    print("="*80)

    with open(window_activations_file, 'rb') as f:
        results = pickle.load(f)

    print(f"Loaded {len(results)} samples")

    # Get layers from first sample
    first_sample = results[0]
    first_window = first_sample['windows']['baseline']
    available_layers = sorted(first_window['activations'].keys())

    if config.layers is None:
        # Default to layers 20-40 as specified
        config.layers = [l for l in range(20, 41) if l in available_layers]

    print(f"Available layers: {min(available_layers)}-{max(available_layers)}")
    print(f"Using layers: {min(config.layers)}-{max(config.layers)}")

    # Load tokenizer for token strings
    print("\n" + "="*80)
    print("LOADING TOKENIZER")
    print("="*80)
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-27b-it")
    print("✓ Tokenizer loaded")

    # Initialize probe experiment (no model needed - using cached activations)
    print("\n" + "="*80)
    print("INITIALIZING PROBE PIPELINE")
    print("="*80)

    # Select probe config based on type
    if config.probe_type == "orthogonal":
        exp_config = {
            'probe_type': 'orthogonal',
            'probe_dir': config.probe_dir_orthogonal,
            'cpca_path': config.cpca_path_orthogonal,
            'orthogonality_weight': config.orthogonality_weight,
            'orthogonal_representation': config.orthogonal_representation,
        }
    elif config.probe_type == "linear":
        exp_config = {
            'probe_type': 'linear',
            'probe_dir': config.probe_dir_linear,
            'cpca_path': config.cpca_path_linear,
            'probe_pattern': config.probe_pattern_linear,
            'n_components': config.n_components,
            'seed': config.seed,
        }
    elif config.probe_type == "centroid":
        exp_config = {
            'probe_type': 'centroid',
            'probe_dir': config.probe_dir_centroid,
            'k_value': config.k_value_centroid,
            'centroid_probe_format': 'conversation',
        }
    else:
        raise ValueError(f"Unknown probe type: {config.probe_type}")

    exp = TokenLevelExperiment(
        model=None,  # Not needed - using cached activations
        tokenizer=tokenizer,
        **exp_config,
        use_wildchat_normalization=config.use_baseline_normalization,
        normalize_probe_scores=config.normalize_probe_scores,
        wildchat_aggregation=config.baseline_aggregation,
        baseline_dir=config.baseline_dir,
        center_probe_scores=config.center_probe_scores,
        emotions=config.emotions
    )

    print("✓ Probe pipeline initialized")

    # Process each sample and window
    print("\n" + "="*80)
    print("APPLYING PROBES TO WINDOWS")
    print("="*80)

    window_names = ['baseline', 'pre_onset', 'onset', 'post_onset']

    # Store results per sample per window
    all_window_scores = {w: [] for w in window_names}

    for sample_idx, sample in enumerate(results):
        print(f"\n[Sample {sample_idx + 1}/{len(results)}]")

        sample_scores = {}

        for window_name in window_names:
            window_data = sample['windows'][window_name]

            # Convert to token-level format
            activations_by_token = convert_window_to_token_level(window_data)

            # Apply probes
            scores_by_token = exp._apply_probes(
                activations_by_token=activations_by_token,
                layers=config.layers,
                verbose=False
            )

            # Apply normalization if enabled (after probe application)
            if config.normalize_probe_scores and sample_idx == 0 and window_name == window_names[0]:
                # Compute baseline statistics once
                print("\n[Computing probe score baseline statistics...]")
                from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader

                baseline_loader = WildChatBaselineLoader(
                    aggregation_type=config.baseline_aggregation,
                    baseline_dir=config.baseline_dir
                )

                baseline_stats = baseline_loader.compute_probe_score_baselines(
                    probe_inference=exp.inference,
                    layers=config.layers,
                    probe_type=config.probe_type,
                    aggregation="mean",
                    return_std=True,
                    orthogonality_weight=config.orthogonality_weight,
                    orthogonal_representation=config.orthogonal_representation,
                    n_components=config.n_components,
                    seed=config.seed,
                    emotions=config.emotions
                )

                # Store for reuse
                global probe_mean, probe_std
                probe_mean = baseline_stats['mean'][-1]  # Layer-averaged
                probe_std = baseline_stats['std'][-1]
                print(f"✓ Baseline computed: mean={probe_mean[:3]}, std={probe_std[:3]}")

            if config.normalize_probe_scores:
                # Z-score normalize probe scores
                for token_pos in scores_by_token:
                    for layer in config.layers:
                        score = scores_by_token[token_pos][layer]
                        # Handle dict format (orthogonal probes)
                        if isinstance(score, dict) and 'user' in score:
                            scores_by_token[token_pos][layer] = {
                                'user': (score['user'] - probe_mean) / (probe_std + 1e-8),
                                'assistant': (score['assistant'] - probe_mean) / (probe_std + 1e-8)
                            }
                        else:
                            # Simple array format
                            scores_by_token[token_pos][layer] = (score - probe_mean) / (probe_std + 1e-8)

            # Aggregate within window
            layer_scores = aggregate_window_scores(
                scores_by_token=scores_by_token,
                layers=config.layers,
                aggregation='mean'
            )

            # Aggregate across layers
            aggregated_score = aggregate_across_layers(
                layer_scores=layer_scores,
                aggregation='mean'
            )

            sample_scores[window_name] = aggregated_score
            all_window_scores[window_name].append(aggregated_score)

            print(f"  {window_name:12s}: {len(window_data['token_positions'])} tokens processed")

    print("\n✓ Probe application complete")

    # Statistical comparison
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)

    stats_results = compare_windows_statistical(
        baseline_scores=all_window_scores['baseline'],
        onset_scores=all_window_scores['onset'],
        emotion_names=config.emotions
    )

    print("\nBaseline vs Onset Comparison:")
    print(f"{'Emotion':<12s} {'Baseline':<12s} {'Onset':<12s} {'Diff':<10s} {'p-value':<10s} {'Cohen_d':<10s}")
    print("-" * 76)

    for i, emotion in enumerate(config.emotions):
        baseline_mean = stats_results['baseline_mean'][i]
        onset_mean = stats_results['onset_mean'][i]
        diff = onset_mean - baseline_mean
        p_val = stats_results['p_values'][i]
        d = stats_results['effect_sizes'][i]

        sig = ""
        if p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"

        print(f"{emotion:<12s} {baseline_mean:>11.4f} {onset_mean:>11.4f} {diff:>9.4f} {p_val:>9.4f}{sig:<2s} {d:>9.4f}")

    # Save statistics
    stats_file = output_path / "statistical_comparison.json"
    with open(stats_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        stats_json = {
            'emotion_names': stats_results['emotion_names'],
            't_statistics': [float(x) for x in stats_results['t_statistics']],
            'p_values': [float(x) for x in stats_results['p_values']],
            'effect_sizes': [float(x) for x in stats_results['effect_sizes']],
            'baseline_mean': [float(x) for x in stats_results['baseline_mean']],
            'onset_mean': [float(x) for x in stats_results['onset_mean']],
            'baseline_std': [float(x) for x in stats_results['baseline_std']],
            'onset_std': [float(x) for x in stats_results['onset_std']],
        }
        json.dump(stats_json, f, indent=2)
    print(f"\n✓ Saved statistics: {stats_file}")

    # Visualization
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    plots_dir = output_path / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Plot 1: Baseline vs Onset comparison
    plot_window_comparison(
        baseline_scores=all_window_scores['baseline'],
        onset_scores=all_window_scores['onset'],
        emotions=config.emotions,
        stats_results=stats_results,
        output_path=plots_dir / "baseline_vs_onset_comparison.png"
    )

    # Plot 2: All windows comparison
    plot_all_windows_comparison(
        window_scores=all_window_scores,
        emotions=config.emotions,
        output_path=plots_dir / "all_windows_comparison.png"
    )

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_path}")
    print(f"  - Statistical comparison: statistical_comparison.json")
    print(f"  - Plots: plots/")


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run emotion probe analysis on window activations")
    parser.add_argument('--input', type=str,
                       default="/workspace-vast/annas/git/research-tools/elicitation/outputs/emotion_onset_analysis_gemma3/window_activations.pkl",
                       help='Path to window_activations.pkl')
    parser.add_argument('--output', type=str,
                       default="/workspace-vast/annas/git/research-tools/elicitation/outputs/emotion_onset_probes_gemma3_probe_norm",
                       help='Output directory')
    parser.add_argument('--probe-type', type=str, default='orthogonal',
                       choices=['orthogonal', 'linear', 'centroid'],
                       help='Probe type to use')

    args = parser.parse_args()

    config = ProbeConfig(probe_type=args.probe_type)

    run_probe_analysis(
        window_activations_file=args.input,
        config=config,
        output_dir=args.output
    )
