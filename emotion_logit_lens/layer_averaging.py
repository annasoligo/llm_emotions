"""
Layer-averaged logit lens implementation - copied directly from emo_lens.

This module contains functions copied from believe-it-or-not/emotion_evals/emo_lens/token_trajectories.py
to ensure exact compatibility.
"""

import numpy as np
import torch
from typing import Dict, List


def compute_layer_averaged_baseline_stats(
    ref_stats: Dict,
    layers: List[int],
    emotion_token_ids: Dict[str, List[int]]
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Compute averaged baseline statistics across multiple layers.

    This function averages the mean and standard deviation statistics across
    a range of layers. Standard deviations are combined using root-mean-square (RMS).

    COPIED DIRECTLY FROM emo_lens/token_trajectories.py

    Args:
        ref_stats: Full baseline stats dict (from load_reference_stats_for_strategy)
        layers: List of layer indices to average over
        emotion_token_ids: Dict mapping emotion name -> list of token IDs

    Returns:
        Dict mapping {emotion: {token_id: {'mean': float, 'std': float}}}
        These are the AVERAGED statistics across all specified layers.
    """
    print(f"\nComputing layer-averaged baseline stats across {len(layers)} layers...")

    averaged_stats = {}

    for emotion, token_ids in emotion_token_ids.items():
        averaged_stats[emotion] = {}

        for token_id in token_ids:
            token_key = str(token_id)

            # Collect mean and std from each layer
            means = []
            stds = []

            for layer in layers:
                layer_key = str(layer)

                try:
                    layer_data = ref_stats['layers_data'][layer_key]
                    statistics = layer_data['statistics']

                    if token_key in statistics:
                        means.append(statistics[token_key]['mean'])
                        stds.append(statistics[token_key]['std'])
                except KeyError:
                    # Layer or token not found in baseline stats
                    continue

            if means and stds:
                # Average mean
                avg_mean = np.mean(means)

                # Average std using RMS (root-mean-square)
                # This is statistically correct for combining independent variance estimates
                avg_std = np.sqrt(np.mean(np.array(stds) ** 2))

                averaged_stats[emotion][token_id] = {
                    'mean': avg_mean,
                    'std': avg_std
                }

    # Print summary
    total_tokens = sum(len(tokens) for tokens in averaged_stats.values())
    print(f"✓ Averaged stats for {total_tokens} tokens across {len(averaged_stats)} emotions")

    # DEBUG: Sample some baseline stats to verify they loaded correctly
    print(f"[DEBUG] Sample baseline stats after averaging:")
    for emotion in list(emotion_token_ids.keys())[:2]:
        sample_tokens = list(emotion_token_ids[emotion])[:2]
        for token_id in sample_tokens:
            if token_id in averaged_stats.get(emotion, {}):
                stats = averaged_stats[emotion][token_id]
                print(f"  {emotion}/{token_id}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
            else:
                print(f"  {emotion}/{token_id}: MISSING!")

    return averaged_stats


def compute_token_emotion_scores(
    model,
    activations_by_token: Dict[int, Dict[int, np.ndarray]],
    layers: List[int],
    emotion_token_ids: Dict[str, List[int]],
    averaged_baseline_stats: Dict[str, Dict[int, Dict[str, float]]],
    emotions: List[str],
    aggregation: str = "mean",
    subtract_mean: bool = True
) -> Dict[int, np.ndarray]:
    """
    Compute emotion scores for each token position using layer-averaged activations.

    COPIED DIRECTLY FROM emo_lens/token_trajectories.py with minimal modifications
    to return np.ndarray instead of dict.

    This function averages activations across layers, projects to vocabulary space,
    and computes normalized emotion scores for each token position.

    Args:
        model: Model for vocabulary projection
        activations_by_token: Dict {token_pos: {layer: activation}}
        layers: Layers to average over
        emotion_token_ids: Emotion -> token IDs mapping
        averaged_baseline_stats: Output from compute_layer_averaged_baseline_stats()
        emotions: List of emotion names in order
        aggregation: How to aggregate multiple emotion tokens ("mean" or "max")
        subtract_mean: Whether to use (x-μ)/σ (True) vs x/σ (False) normalization

    Returns:
        Dict mapping {token_pos: scores_array} where scores_array is shape (n_emotions,)
    """
    print(f"\nComputing emotion scores for {len(activations_by_token)} token positions...")

    token_emotion_scores = {}

    for token_pos in sorted(activations_by_token.keys()):
        # Step 1: Average activations across all layers
        layer_activations = [
            activations_by_token[token_pos][layer]
            for layer in layers
        ]
        avg_activation = np.mean(layer_activations, axis=0)  # Shape: (hidden_dim,)

        # Step 2: Project to vocabulary space
        avg_activation_tensor = torch.from_numpy(avg_activation).to(model.device)
        # Match model dtype (usually bfloat16)
        if hasattr(model, 'dtype'):
            avg_activation_tensor = avg_activation_tensor.to(model.dtype)
        elif hasattr(model.lm_head, 'weight'):
            avg_activation_tensor = avg_activation_tensor.to(model.lm_head.weight.dtype)

        with torch.no_grad():
            # Try StandardizedTransformer method first (includes final LayerNorm)
            if hasattr(model, 'project_on_vocab'):
                logits = model.project_on_vocab(avg_activation_tensor)
            elif hasattr(model, 'lm_head'):
                # IMPORTANT: Apply final LayerNorm before projecting!
                # The baseline was computed with normalized activations.
                if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                    # For HuggingFace models (e.g., Gemma: model.model.norm)
                    normalized = model.model.norm(avg_activation_tensor)
                    logits = model.lm_head(normalized)
                elif hasattr(model, 'norm'):
                    # For some model architectures (e.g., model.norm)
                    normalized = model.norm(avg_activation_tensor)
                    logits = model.lm_head(normalized)
                else:
                    # Fallback: project without norm (may be incorrect!)
                    print("WARNING: Could not find final LayerNorm, projecting without normalization")
                    logits = model.lm_head(avg_activation_tensor)
            else:
                raise AttributeError("Model must have either 'project_on_vocab' or 'lm_head' attribute")

        logits_np = logits.cpu().float().numpy()

        # Step 3: Compute emotion scores
        scores = {}

        for emotion in emotions:
            token_ids = emotion_token_ids[emotion]

            if not token_ids:
                scores[emotion] = 0.0
                continue

            # Extract logits for this emotion's tokens
            emotion_logits = []

            for token_id in token_ids:
                if token_id not in averaged_baseline_stats.get(emotion, {}):
                    continue  # Skip tokens without baseline stats

                raw_logit = float(logits_np[token_id])
                mean = averaged_baseline_stats[emotion][token_id]['mean']
                std = averaged_baseline_stats[emotion][token_id]['std']

                # DEBUG: Print first normalization
                if token_pos == 0 and emotion == emotions[0] and len(emotion_logits) == 0:
                    print(f"[NORM DEBUG] tok_pos=0, emotion={emotion}, tok_id={token_id}: raw={raw_logit:.2f}, mean={mean:.2f}, std={std:.2f}")

                # Normalize
                if std > 1e-8:  # Avoid division by zero
                    if subtract_mean:
                        normalized = (raw_logit - mean) / std  # z-score
                    else:
                        normalized = raw_logit / std  # scaled
                else:
                    normalized = 0.0

                # DEBUG: Print normalized value
                if token_pos == 0 and emotion == emotions[0] and len(emotion_logits) == 0:
                    print(f"[NORM DEBUG] normalized={normalized:.2f}, subtract_mean={subtract_mean}")

                emotion_logits.append(normalized)

            # Aggregate across tokens
            if emotion_logits:
                if aggregation == "mean":
                    scores[emotion] = float(np.mean(emotion_logits))
                elif aggregation == "max":
                    scores[emotion] = float(np.max(emotion_logits))
                else:
                    raise ValueError(f"Unknown aggregation: {aggregation}")
            else:
                scores[emotion] = 0.0

        # Convert to array in emotion order
        token_emotion_scores[token_pos] = np.array([scores[e] for e in emotions])

    print(f"✓ Computed scores for {len(emotions)} emotions across {len(token_emotion_scores)} tokens")

    return token_emotion_scores
