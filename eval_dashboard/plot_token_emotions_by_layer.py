"""
Plot token-level emotion scores BY LAYER (layers 20-40) for orthogonal_regularized probe.
Shows how emotion detection varies across layers.
"""

import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from probes.scripts.token_level_helpers import TokenLevelExperiment
from probes.scripts.probe_pipeline import normalize_probe_scores_zscore
from probe_configs import PROBE_CONFIGS, BASELINE_CONFIG, EMOTIONS, EMOTION_COLORS


# Use functions from the other script
import plot_token_emotions_at_onset as base_script


def plot_token_emotions_by_layer(
    conversations: List[Dict],
    model,
    tokenizer,
    probe_experiments_by_layer: Dict[int, TokenLevelExperiment],
    layers: List[int],
    window: int = 30,
    n_random_windows: int = 2,
    random_window_size: int = 60
):
    """Plot token-level emotions across multiple layers."""

    output_dir = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/token_onset_plots_by_layer")
    output_dir.mkdir(exist_ok=True)

    # Focus on negative emotions: anger, fear, sadness
    emotion_indices = [0, 2, 3]
    emotion_names = ['Anger', 'Fear', 'Sadness']

    for conv_idx, conversation in enumerate(conversations):
        sample_id = conversation['sample_id']
        onset_token = conversation['metadata']['onset_global_token']

        print(f"\n{'='*60}")
        print(f"Processing Sample #{sample_id}")
        print(f"Emotion onset at token: {onset_token}")
        print(f"{'='*60}")

        # Extract activations once for all layers
        print("Extracting activations for all layers...")
        activations_by_token, token_strings = base_script.extract_token_activations(
            conversation=conversation['conversation'],
            model=model,
            tokenizer=tokenizer,
            layers=layers
        )

        # Get scores for each layer
        print("Computing probe scores for all layers...")
        scores_by_layer = {}
        baseline_caches = {}  # Cache baseline stats per layer
        for layer in layers:
            print(f"  Layer {layer}...")
            probe_exp = probe_experiments_by_layer[layer]

            if layer not in baseline_caches:
                baseline_caches[layer] = {}

            scores_dict = base_script.apply_probes_to_activations(
                activations_by_token=activations_by_token,
                probe_experiment=probe_exp,
                layers=[layer],
                baseline_stats_cache=baseline_caches[layer]
            )

            scores_by_layer[layer] = scores_dict

        # Plot 1: Onset window
        print(f"\nPlotting onset window...")
        start_idx = max(0, onset_token - window)
        end_idx = min(len(token_strings), onset_token + window)
        token_positions = list(range(start_idx, end_idx))
        tokens_window = [token_strings[i] for i in token_positions]

        fig, axes = plt.subplots(len(layers), 1, figsize=(20, 3 * len(layers)), sharex=True)
        if len(layers) == 1:
            axes = [axes]

        for ax, layer in zip(axes, layers):
            scores_dict = scores_by_layer[layer]

            x = []
            scores_by_emotion = {emo_idx: [] for emo_idx in emotion_indices}

            for rel_idx, token_pos in enumerate(token_positions):
                if token_pos in scores_dict:
                    x.append(rel_idx)
                    token_score = scores_dict[token_pos]
                    if isinstance(token_score, dict):
                        scores = token_score['assistant']
                    else:
                        scores = token_score

                    for emo_idx in emotion_indices:
                        scores_by_emotion[emo_idx].append(scores[emo_idx])

            # Plot each emotion
            for emo_idx, emo_name in zip(emotion_indices, emotion_names):
                if scores_by_emotion[emo_idx]:
                    color = EMOTION_COLORS[EMOTIONS[emo_idx]]
                    ax.plot(x, scores_by_emotion[emo_idx],
                           label=emo_name,
                           color=color,
                           linewidth=2,
                           alpha=0.8)

            # Mark onset
            onset_rel = onset_token - start_idx
            if 0 <= onset_rel < len(tokens_window):
                ax.axvline(onset_rel, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Onset')

            # Styling
            ax.set_ylabel('Z-score', fontsize=10)
            ax.set_title(f"Layer {layer}", fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

        # Set x-axis labels
        axes[-1].set_xticks(range(len(tokens_window)))
        axes[-1].set_xticklabels(tokens_window, rotation=45, ha='right', fontsize=6)
        axes[-1].set_xlabel('Token', fontsize=10)

        fig.suptitle(f"Sample #{sample_id} - Onset Window by Layer (Ortho Reg λ=100)",
                    fontsize=14, fontweight='bold', y=0.995)

        plt.tight_layout()
        save_path = output_dir / f"sample_{sample_id}_onset_by_layer.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

        # Plot 2-3: Random windows
        max_start_token = max(0, len(token_strings) - random_window_size)
        if max_start_token > onset_token + window:
            import random
            random.seed(sample_id)

            possible_starts = list(range(onset_token + window, max_start_token))
            if len(possible_starts) >= n_random_windows:
                random_starts = random.sample(possible_starts, n_random_windows)

                for rand_idx, rand_start in enumerate(random_starts):
                    print(f"\nPlotting random window {rand_idx + 1} (starting at token {rand_start})...")

                    rand_end = min(len(token_strings), rand_start + random_window_size)
                    rand_positions = list(range(rand_start, rand_end))
                    rand_tokens = [token_strings[i] for i in rand_positions]

                    fig, axes = plt.subplots(len(layers), 1, figsize=(20, 3 * len(layers)), sharex=True)
                    if len(layers) == 1:
                        axes = [axes]

                    for ax, layer in zip(axes, layers):
                        scores_dict = scores_by_layer[layer]

                        x = []
                        scores_by_emotion = {emo_idx: [] for emo_idx in emotion_indices}

                        for rel_idx, token_pos in enumerate(rand_positions):
                            if token_pos in scores_dict:
                                x.append(rel_idx)
                                token_score = scores_dict[token_pos]
                                if isinstance(token_score, dict):
                                    scores = token_score['assistant']
                                else:
                                    scores = token_score

                                for emo_idx in emotion_indices:
                                    scores_by_emotion[emo_idx].append(scores[emo_idx])

                        # Plot each emotion
                        for emo_idx, emo_name in zip(emotion_indices, emotion_names):
                            if scores_by_emotion[emo_idx]:
                                color = EMOTION_COLORS[EMOTIONS[emo_idx]]
                                ax.plot(x, scores_by_emotion[emo_idx],
                                       label=emo_name,
                                       color=color,
                                       linewidth=2,
                                       alpha=0.8)

                        # Styling
                        ax.set_ylabel('Z-score', fontsize=10)
                        ax.set_title(f"Layer {layer}", fontsize=12, fontweight='bold')
                        ax.legend(loc='upper right', fontsize=8)
                        ax.grid(True, alpha=0.3)
                        ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

                    # Set x-axis labels
                    axes[-1].set_xticks(range(len(rand_tokens)))
                    axes[-1].set_xticklabels(rand_tokens, rotation=45, ha='right', fontsize=6)
                    axes[-1].set_xlabel('Token', fontsize=10)

                    fig.suptitle(f"Sample #{sample_id} - Random Window {rand_idx + 1} by Layer (Ortho Reg λ=100)",
                                fontsize=14, fontweight='bold', y=0.995)

                    plt.tight_layout()
                    save_path = output_dir / f"sample_{sample_id}_random_{rand_idx + 1}_by_layer.png"
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"Saved: {save_path}")
                    plt.close()

        torch.cuda.empty_cache()


def main():
    print("Loading high emotion samples...")
    conversations = base_script.load_high_emotion_samples(n_samples=3)

    print(f"\nLoaded {len(conversations)} conversations:")
    for i, conv in enumerate(conversations):
        print(f"  {i+1}. Sample #{conv['sample_id']}")

    # Layers to analyze
    layers = list(range(20, 41, 2))  # Layers 20, 22, 24, ..., 40 (11 layers)
    print(f"\nAnalyzing layers: {layers}")

    print("\nLoading model...")
    model_name = "google/gemma-3-27b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    model.eval()
    print("Model loaded")

    # Initialize probe experiments for each layer
    print("\nInitializing probe experiments for each layer...")
    probe_config = PROBE_CONFIGS['orthogonal_regularized_lambda100']
    probe_experiments_by_layer = {}

    for layer in layers:
        probe_experiments_by_layer[layer] = TokenLevelExperiment(
            model=None,
            tokenizer=tokenizer,
            probe_type=probe_config['type'],
            probe_dir=probe_config.get('probe_dir'),
            cpca_path=probe_config.get('cpca_path'),
            probe_pattern=probe_config.get('probe_pattern'),
            lambda_ortho=probe_config.get('lambda_ortho', 100.0),
            baseline_dir=BASELINE_CONFIG['baseline_dir'],
            emotions=EMOTIONS
        )

    print("\nGenerating plots...")
    plot_token_emotions_by_layer(
        conversations=conversations,
        model=model,
        tokenizer=tokenizer,
        probe_experiments_by_layer=probe_experiments_by_layer,
        layers=layers,
        window=30,
        n_random_windows=2,
        random_window_size=60
    )

    print(f"\n{'='*60}")
    print("All layer-wise plots generated!")
    print(f"Output: eval_dashboard/token_onset_plots_by_layer/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
