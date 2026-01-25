"""
Plot token-level emotion scores around emotion onset for high emotion samples.
Reuses existing token-level code pattern from data_preprocessing.py
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
from probe_configs import PROBE_CONFIGS, BASELINE_CONFIG, EMOTIONS, EMOTION_COLORS


def load_high_emotion_samples(n_samples: int = 3) -> List[Dict]:
    """Load high emotion conversations sorted by average emotion."""
    data_path = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus.pkl")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    conversations = data['conversations']

    # Sort by average anger score (first emotion)
    def get_avg_emotion(conv):
        if 'probe_scores' in conv and 'text_raw' in conv['probe_scores']:
            scores = list(conv['probe_scores']['text_raw'].values())
            if scores:
                return np.mean([s[0] for s in scores])  # anger
        return 0

    conversations_sorted = sorted(conversations, key=get_avg_emotion, reverse=True)
    return conversations_sorted[:n_samples]


def extract_token_activations(
    conversation: List[Dict],
    model,
    tokenizer,
    layers: List[int]
) -> tuple:
    """
    Extract token-level activations from model.
    Returns: (activations_by_token, token_strings)
    """
    # Reconstruct conversation text using chat template
    conversation_text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize
    inputs = tokenizer(conversation_text, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs['input_ids'][0]

    # Get token strings
    token_strings = [tokenizer.decode([tid], skip_special_tokens=False) for tid in input_ids]

    # Get activations
    activations_by_token = {}

    with torch.no_grad():
        outputs = model(
            input_ids.unsqueeze(0).to(model.device),
            output_hidden_states=True
        )

        for layer_idx in layers:
            layer_activations = outputs.hidden_states[layer_idx][0].float().cpu().numpy()  # [n_tokens, hidden_dim]

            for token_pos in range(len(token_strings)):
                if token_pos not in activations_by_token:
                    activations_by_token[token_pos] = {}
                activations_by_token[token_pos][layer_idx] = layer_activations[token_pos]

    return activations_by_token, token_strings


def apply_probes_to_activations(
    activations_by_token: Dict[int, Dict[int, np.ndarray]],
    probe_experiment: TokenLevelExperiment,
    layers: List[int],
    baseline_stats_cache: Dict = None
) -> Dict[int, np.ndarray]:
    """Apply probes to activations with proper z-score normalization. Returns token_position -> emotion_scores."""
    from probes.scripts.probe_pipeline import normalize_probe_scores_zscore
    from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader

    # Apply probes (without normalization)
    scores_by_token = probe_experiment._apply_probes(
        activations_by_token=activations_by_token,
        layers=layers,
        verbose=False
    )

    # Get baseline statistics for normalization (compute once and cache)
    if baseline_stats_cache is None or 'stats' not in baseline_stats_cache:
        baseline_loader = WildChatBaselineLoader(
            aggregation_type="all_tokens",
            baseline_dir=probe_experiment.baseline_dir
        )

        baseline_stats = baseline_loader.compute_probe_score_baselines(
            probe_inference=probe_experiment.inference,
            layers=layers,
            probe_type=probe_experiment.probe_type,
            aggregation="mean",
            return_std=True,
            orthogonality_weight=probe_experiment.orthogonality_weight,
            orthogonal_representation=probe_experiment.orthogonal_representation,
            n_components=probe_experiment.n_components,
            seed=probe_experiment.seed,
            emotions=probe_experiment.emotions,
            probe_pattern=probe_experiment.probe_pattern,
            lambda_ortho=probe_experiment.lambda_ortho
        )

        baseline_mean_vec = baseline_stats['mean'][-1]  # -1 = layer-averaged
        baseline_std_vec = baseline_stats['std'][-1]

        if baseline_stats_cache is not None:
            baseline_stats_cache['stats'] = (baseline_mean_vec, baseline_std_vec)
    else:
        baseline_mean_vec, baseline_std_vec = baseline_stats_cache['stats']

    # Extract scores for layer and apply z-score normalization
    result = {}
    for token_pos, layer_scores in scores_by_token.items():
        if layers[0] in layer_scores:
            raw_score = layer_scores[layers[0]]

            # Normalize: Handle both dict (orthogonal) and array (text) formats
            if isinstance(raw_score, dict):
                # Orthogonal probe - normalize user and assistant separately
                normalized = {}
                for key in raw_score:
                    normalized[key] = normalize_probe_scores_zscore(
                        raw_score[key], baseline_mean_vec, baseline_std_vec
                    )
                result[token_pos] = normalized
            else:
                # Text probe - normalize directly
                result[token_pos] = normalize_probe_scores_zscore(
                    raw_score, baseline_mean_vec, baseline_std_vec
                )

    return result


def plot_token_emotions(
    conversations: List[Dict],
    probe_keys: List[str],
    model,
    tokenizer,
    probe_experiments: Dict[str, TokenLevelExperiment],
    window: int = 30,
    n_random_windows: int = 2,
    random_window_size: int = 60
):
    """Plot token-level emotions around onset and in random windows after onset."""

    output_dir = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/token_onset_plots")
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

        # Extract activations once for all probes
        print("Extracting activations...")
        activations_by_token, token_strings = extract_token_activations(
            conversation=conversation['conversation'],
            model=model,
            tokenizer=tokenizer,
            layers=[30]
        )

        # Define window
        start_idx = max(0, onset_token - window)
        end_idx = min(len(token_strings), onset_token + window)
        token_positions = list(range(start_idx, end_idx))
        tokens_window = [token_strings[i] for i in token_positions]

        # Get scores for each probe
        all_scores = {}
        baseline_caches = {}  # Cache baseline stats per probe type
        for probe_key in probe_keys:
            print(f"Computing {probe_key}...")
            probe_exp = probe_experiments[probe_key]

            if probe_key not in baseline_caches:
                baseline_caches[probe_key] = {}

            scores_dict = apply_probes_to_activations(
                activations_by_token=activations_by_token,
                probe_experiment=probe_exp,
                layers=[30],
                baseline_stats_cache=baseline_caches[probe_key]
            )

            all_scores[probe_key] = scores_dict

        # Create figure
        n_probes = len(probe_keys)
        fig, axes = plt.subplots(n_probes, 1, figsize=(20, 4 * n_probes), sharex=True)

        if n_probes == 1:
            axes = [axes]

        # Plot each probe
        for ax, probe_key in zip(axes, probe_keys):
            scores_dict = all_scores[probe_key]

            # Extract scores for window
            x = []
            scores_by_emotion = {emo_idx: [] for emo_idx in emotion_indices}

            for rel_idx, token_pos in enumerate(token_positions):
                if token_pos in scores_dict:
                    x.append(rel_idx)

                    # Handle orthogonal probes (dict with 'user'/'assistant') vs regular probes (array)
                    token_score = scores_dict[token_pos]
                    if isinstance(token_score, dict):
                        # Orthogonal probe - use assistant scores
                        scores = token_score['assistant']
                    else:
                        # Regular probe - use scores directly
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
            ax.set_ylabel('Z-score', fontsize=12)
            ax.set_title(f"{PROBE_CONFIGS[probe_key]['display_name']}", fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

        # Set x-axis labels (token text) on bottom subplot
        axes[-1].set_xticks(range(len(tokens_window)))
        axes[-1].set_xticklabels(tokens_window, rotation=45, ha='right', fontsize=8)
        axes[-1].set_xlabel('Token', fontsize=12)

        # Overall title
        fig.suptitle(f"Sample #{sample_id} - Emotion Onset Window (±{window} tokens)",
                    fontsize=16, fontweight='bold', y=0.995)

        plt.tight_layout()
        save_path = output_dir / f"sample_{sample_id}_onset_window.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

        # Generate random window plots (60 tokens, starting after onset)
        max_start_token = max(0, len(token_strings) - random_window_size)
        if max_start_token > onset_token + window:
            import random
            random.seed(sample_id)  # Reproducible randomness

            # Generate n_random_windows starting after the onset window
            possible_starts = list(range(onset_token + window, max_start_token))
            if len(possible_starts) >= n_random_windows:
                random_starts = random.sample(possible_starts, n_random_windows)

                for rand_idx, rand_start in enumerate(random_starts):
                    print(f"\nGenerating random window {rand_idx + 1} (starting at token {rand_start})...")

                    rand_end = min(len(token_strings), rand_start + random_window_size)
                    rand_positions = list(range(rand_start, rand_end))
                    rand_tokens = [token_strings[i] for i in rand_positions]

                    # Create figure for random window
                    fig, axes = plt.subplots(n_probes, 1, figsize=(20, 4 * n_probes), sharex=True)
                    if n_probes == 1:
                        axes = [axes]

                    # Plot each probe
                    for ax, probe_key in zip(axes, probe_keys):
                        scores_dict = all_scores[probe_key]

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
                        ax.set_ylabel('Z-score', fontsize=12)
                        ax.set_title(f"{PROBE_CONFIGS[probe_key]['display_name']}", fontsize=14, fontweight='bold')
                        ax.legend(loc='upper right', fontsize=10)
                        ax.grid(True, alpha=0.3)
                        ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

                    # Set x-axis labels
                    axes[-1].set_xticks(range(len(rand_tokens)))
                    axes[-1].set_xticklabels(rand_tokens, rotation=45, ha='right', fontsize=8)
                    axes[-1].set_xlabel('Token', fontsize=12)

                    # Overall title
                    fig.suptitle(f"Sample #{sample_id} - Random Window {rand_idx + 1} (tokens {rand_start}-{rand_end}, post-onset)",
                                fontsize=16, fontweight='bold', y=0.995)

                    plt.tight_layout()
                    save_path = output_dir / f"sample_{sample_id}_random_window_{rand_idx + 1}.png"
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f"Saved: {save_path}")
                    plt.close()

        # Clear GPU cache
        torch.cuda.empty_cache()


def main():
    print("Loading high emotion samples...")
    conversations = load_high_emotion_samples(n_samples=3)

    print(f"\nLoaded {len(conversations)} conversations:")
    for i, conv in enumerate(conversations):
        print(f"  {i+1}. Sample #{conv['sample_id']}")

    # Select probe configurations (use only one for layer sweep to keep it manageable)
    probe_keys = ['orthogonal_regularized_lambda100']  # Focus on new probe type

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

    # Initialize probe experiments
    print("\nInitializing probe experiments...")
    probe_experiments = {}
    for probe_key in probe_keys:
        probe_config = PROBE_CONFIGS[probe_key]

        probe_experiments[probe_key] = TokenLevelExperiment(
            model=None,  # Not needed - using cached activations
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
    plot_token_emotions(
        conversations=conversations,
        probe_keys=probe_keys,
        model=model,
        tokenizer=tokenizer,
        probe_experiments=probe_experiments,
        window=30
    )

    print(f"\n{'='*60}")
    print("All plots generated!")
    print(f"Output: eval_dashboard/token_onset_plots/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
