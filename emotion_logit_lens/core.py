"""
Core logit projection and emotion scoring functions.

This module provides the fundamental operations for logit-based emotion detection:
1. Project hidden states to vocabulary logits
2. Extract logits for emotion tokens
3. Normalize using baseline statistics
4. Aggregate to emotion scores
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ============================================================================
# Logit Projection
# ============================================================================

def project_to_logits(
    model: Any,  # StandardizedTransformer or raw model
    hidden_state_vector: np.ndarray,
) -> torch.Tensor:
    """
    Project a hidden state vector directly to vocabulary logits.

    Works with either StandardizedTransformer or raw HuggingFace models.

    Args:
        model: StandardizedTransformer or raw model with lm_head
        hidden_state_vector: Vector of shape (hidden_size,)

    Returns:
        Logits tensor of shape (vocab_size,)
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    vector_tensor = torch.tensor(hidden_state_vector, dtype=torch.float32, device=device).to(dtype)

    with torch.no_grad():
        if len(vector_tensor.shape) == 1:
            vector_tensor = vector_tensor.unsqueeze(0)

        # Try StandardizedTransformer method first, fall back to direct lm_head
        if hasattr(model, 'project_on_vocab'):
            logits = model.project_on_vocab(vector_tensor)
        elif hasattr(model, 'lm_head'):
            # Direct projection for raw HuggingFace models
            # IMPORTANT: Must apply final layer norm before lm_head!
            # Try common locations for final norm
            final_norm = None
            if hasattr(model, 'model'):
                if hasattr(model.model, 'norm'):
                    final_norm = model.model.norm
                elif hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'norm'):
                    # Gemma-3 multimodal structure
                    final_norm = model.model.language_model.norm

            if final_norm is not None:
                vector_tensor = final_norm(vector_tensor)

            logits = model.lm_head(vector_tensor)
        else:
            raise AttributeError("Model must have either 'project_on_vocab' or 'lm_head' attribute")

        if logits.shape[0] == 1:
            logits = logits.squeeze(0)

    return logits  # (vocab_size,)


def project_to_logits_batched(
    model: Any,  # StandardizedTransformer or raw model
    hidden_state_vectors: np.ndarray,
) -> torch.Tensor:
    """
    Project multiple hidden state vectors to vocabulary logits in a single batch.

    Works with either StandardizedTransformer or raw HuggingFace models.

    IMPORTANT NOTE ON LAYER NORMALIZATION:
    Most transformer models apply a final LayerNorm before the LM head. LayerNorm
    normalizes across the feature dimension (hidden_size), NOT the batch dimension.

    When batching with shape (n_samples, hidden_size):
    - LayerNorm computes mean/std across hidden_size (dim=-1)
    - Each sample is normalized INDEPENDENTLY
    - This is CORRECT for batching

    Args:
        model: StandardizedTransformer or raw model with lm_head
        hidden_state_vectors: Array of shape (n_samples, hidden_size)

    Returns:
        Logits tensor of shape (n_samples, vocab_size)
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    vectors_tensor = torch.tensor(hidden_state_vectors, dtype=torch.float32, device=device).to(dtype)

    with torch.no_grad():
        # Try StandardizedTransformer method first, fall back to direct lm_head
        if hasattr(model, 'project_on_vocab'):
            logits = model.project_on_vocab(vectors_tensor)
        elif hasattr(model, 'lm_head'):
            # Direct projection for raw HuggingFace models
            # IMPORTANT: Must apply final layer norm before lm_head!
            final_norm = None
            if hasattr(model, 'model'):
                if hasattr(model.model, 'norm'):
                    final_norm = model.model.norm
                elif hasattr(model.model, 'language_model') and hasattr(model.model.language_model, 'norm'):
                    # Gemma-3 multimodal structure
                    final_norm = model.model.language_model.norm

            if final_norm is not None:
                vectors_tensor = final_norm(vectors_tensor)

            logits = model.lm_head(vectors_tensor)
        else:
            raise AttributeError("Model must have either 'project_on_vocab' or 'lm_head' attribute")

    return logits  # (n_samples, vocab_size)


# ============================================================================
# Emotion Scoring
# ============================================================================

def compute_emotion_scores_from_logits(
    logits: torch.Tensor,
    emotion_token_ids: Dict[str, List[int]],
    baseline_stats: Optional[Dict] = None,
    layer: Optional[int] = None,
    aggregation: str = "mean"
) -> Dict[str, float]:
    """
    Compute z-score normalized emotion scores from vocabulary logits.

    Normalization strategy:
    - With baseline: z = (logit - mean) / std  (z-score)
    - Without baseline: Use raw logits

    Args:
        logits: Vocabulary logits [vocab_size]
        emotion_token_ids: Dict mapping emotion -> list of token IDs
        baseline_stats: Baseline statistics for normalization
                       Format: {'layers_data': {'{layer}': {'statistics': {'{token_id}': {'mean': ..., 'std': ...}}}}}
        layer: Layer index (required if baseline_stats provided)
        aggregation: Aggregation method - "mean" or "max"

    Returns:
        Dict mapping emotion -> z-score
        Example: {'anger': 0.34, 'happiness': 1.82, 'sadness': -0.45, ...}

    Raises:
        ValueError: If aggregation method unknown or baseline stats invalid
    """
    if aggregation not in ["mean", "max"]:
        raise ValueError(f"Unknown aggregation: {aggregation}. Supported: 'mean', 'max'")

    # Convert to float32 to ensure compatibility with numpy and Python float()
    logits = logits.float()

    scores = {}

    for emotion, token_ids in emotion_token_ids.items():
        if not token_ids:
            scores[emotion] = 0.0
            continue

        if baseline_stats is None:
            # No normalization - use raw logits
            emotion_logits = logits[token_ids].float()
            if aggregation == "mean":
                scores[emotion] = float(emotion_logits.mean())
            elif aggregation == "max":
                scores[emotion] = float(emotion_logits.max())
            continue

        # Get baseline stats for this layer
        if layer is None:
            raise ValueError("layer must be specified when baseline_stats provided")

        layer_key = str(layer)
        if 'layers_data' not in baseline_stats:
            raise ValueError(
                "Invalid baseline stats format: missing 'layers_data' key. "
                "Expected format: {'layers_data': {'{layer}': {'statistics': {...}}}}"
            )

        layer_data = baseline_stats['layers_data'].get(layer_key, {})
        if not layer_data:
            raise ValueError(f"No baseline stats found for layer {layer}")

        layer_stats = layer_data.get('statistics', {})

        # Normalize each token
        normalized_scores = []
        missing_tokens = []

        for token_id in token_ids:
            token_key = str(token_id)
            if token_key in layer_stats:
                mean = layer_stats[token_key]['mean']
                std = layer_stats[token_key]['std']
                raw_logit = float(logits[token_id])

                # Z-score normalization: z = (x - μ) / σ
                if std > 1e-8:  # Avoid division by zero
                    z = (raw_logit - mean) / std
                else:
                    # std is zero - constant token
                    z = raw_logit - mean

                normalized_scores.append(z)
            else:
                missing_tokens.append((token_id, emotion))

        # Check for missing tokens
        if missing_tokens:
            missing_info = ", ".join([f"{tid} ({em})" for tid, em in missing_tokens[:5]])
            if len(missing_tokens) > 5:
                missing_info += f" ... and {len(missing_tokens) - 5} more"

            raise ValueError(
                f"Baseline stats missing for {len(missing_tokens)} emotion tokens at layer {layer}!\n"
                f"Missing tokens: {missing_info}\n"
                "Regenerate baseline stats using compute_logit_baselines.py"
            )

        # Aggregate
        if aggregation == "mean":
            scores[emotion] = float(np.mean(normalized_scores))
        elif aggregation == "max":
            scores[emotion] = float(np.max(normalized_scores))

    return scores


def compute_emotion_scores_batched(
    logits_batch: torch.Tensor,
    emotion_token_ids: Dict[str, List[int]],
    baseline_stats: Optional[Dict] = None,
    layer: Optional[int] = None,
    aggregation: str = "mean"
) -> List[Dict[str, float]]:
    """
    Compute emotion scores for a batch of logits (optimized vectorized version).

    Args:
        logits_batch: Batch of vocabulary logits [batch_size, vocab_size]
        emotion_token_ids: Dict mapping emotion -> list of token IDs
        baseline_stats: Baseline statistics for normalization
        layer: Layer index (required if baseline_stats provided)
        aggregation: Aggregation method - "mean" or "max"

    Returns:
        List of score dicts, one per sample
        Example: [{'anger': 0.34, ...}, {'anger': 1.21, ...}, ...]

    Raises:
        ValueError: If aggregation method unknown or baseline stats invalid
    """
    if aggregation not in ["mean", "max"]:
        raise ValueError(f"Unknown aggregation: {aggregation}. Supported: 'mean', 'max'")

    batch_size = logits_batch.shape[0]
    logits_np = logits_batch.cpu().float().numpy()

    # Initialize result list
    scores_batch = [{} for _ in range(batch_size)]

    # Get baseline stats if provided
    layer_stats = None
    if baseline_stats is not None:
        if layer is None:
            raise ValueError("layer must be specified when baseline_stats provided")

        layer_key = str(layer)
        if 'layers_data' not in baseline_stats:
            raise ValueError(
                "Invalid baseline stats format: missing 'layers_data' key. "
                "Expected format: {'layers_data': {'{layer}': {'statistics': {...}}}}"
            )

        layer_data = baseline_stats['layers_data'].get(layer_key, {})
        if not layer_data:
            raise ValueError(f"No baseline stats found for layer {layer}")

        layer_stats = layer_data.get('statistics', {})

    # Process each emotion
    for emotion, token_ids in emotion_token_ids.items():
        if not token_ids:
            for i in range(batch_size):
                scores_batch[i][emotion] = 0.0
            continue

        # Extract logits for all emotion tokens: shape (batch_size, n_tokens)
        token_ids_array = np.array(token_ids)
        emotion_logits_batch = logits_np[:, token_ids_array]  # (batch_size, n_tokens)

        if baseline_stats is None:
            # No normalization - use raw logits
            if aggregation == "mean":
                scores = emotion_logits_batch.mean(axis=1)  # (batch_size,)
            elif aggregation == "max":
                scores = emotion_logits_batch.max(axis=1)  # (batch_size,)

            for i in range(batch_size):
                scores_batch[i][emotion] = float(scores[i])
        else:
            # Normalize using baseline stats
            means = []
            stds = []
            missing_tokens = []

            for token_id in token_ids:
                token_key = str(token_id)
                if token_key in layer_stats:
                    means.append(layer_stats[token_key]['mean'])
                    stds.append(layer_stats[token_key]['std'])
                else:
                    missing_tokens.append((token_id, emotion))

            if missing_tokens:
                missing_info = ", ".join([f"{tid} ({em})" for tid, em in missing_tokens[:5]])
                if len(missing_tokens) > 5:
                    missing_info += f" ... and {len(missing_tokens) - 5} more"
                raise ValueError(
                    f"Baseline stats missing for {len(missing_tokens)} emotion tokens at layer {layer}!\n"
                    f"Missing tokens: {missing_info}\n"
                    "Regenerate baseline stats using compute_logit_baselines.py"
                )

            # Vectorized normalization
            means_array = np.array(means)  # (n_tokens,)
            stds_array = np.array(stds)    # (n_tokens,)

            # Z-score normalization: z = (x - μ) / σ
            # Shape: (batch_size, n_tokens)
            normalized = (emotion_logits_batch - means_array) / np.maximum(stds_array, 1e-8)

            # Handle zero std
            zero_std_mask = stds_array < 1e-8
            if np.any(zero_std_mask):
                normalized[:, zero_std_mask] = emotion_logits_batch[:, zero_std_mask] - means_array[zero_std_mask]

            # Aggregate
            if aggregation == "mean":
                scores = normalized.mean(axis=1)  # (batch_size,)
            elif aggregation == "max":
                scores = normalized.max(axis=1)  # (batch_size,)

            for i in range(batch_size):
                scores_batch[i][emotion] = float(scores[i])

    return scores_batch
