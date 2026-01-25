"""
OPTIMIZED VERSION with batched projection for 50-100x speedup.

Key optimization: Batch all layer activations together before projecting to vocab.
Instead of 42 layers × 1000 tokens = 42,000 forward passes,
we do ~10-20 batched passes with batch_size=2048.

Also supports computing per-layer probe scores (text_raw, text_cpca) during the same pass.
"""
import pickle
import numpy as np
from pathlib import Path
import sys
import torch
import argparse

# Add emo_lens to path
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not")
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import emo_lens functions
from emotion_evals.emo_lens.token_trajectories import (
    extract_token_level_activations,
    compute_layer_averaged_baseline_stats
)
from emotion_evals.emo_lens.model_utils import load_base_model
from emotion_evals.emo_lens.logit_lens_emotion_direct import (
    load_reference_stats_for_strategy,
    load_emotion_token_ids_from_json,
    normalize_logits_to_emotion_scores,
    load_axis_token_groups
)

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
AXES = ['valence', 'arousal', 'dominance', 'approach_avoidance']

# Probe configurations for per-layer probe scores
PROBE_DIR = Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/text_based/multiseed")
CPCA_PATH = Path("/workspace-vast/annas/git/research-tools/probes/results/cpca_tier_data_high_alpha.tmp/google/gemma-3-27b-it_cpca.npz")

PROBE_CONFIGS = {
    'text_raw': {
        'probe_pattern': 'probe_layer{layer}_nc0_seed0.pkl',
        'use_cpca': False,
        'n_components': 0,
    },
    'text_cpca': {
        'probe_pattern': 'probe_layer{layer}_nc10_seed0.pkl',
        'use_cpca': True,
        'n_components': 10,
    }
}

# Define all layer ranges to compute
LAYER_RANGE_CONFIGS = {
    'logit_lens_mean': {
        'layers': list(range(40, 51)),  # L40-50
        'display_name': 'Logit Lens L40-50'
    },
    'logit_lens_mean_l30_40': {
        'layers': list(range(30, 41)),  # L30-40
        'display_name': 'Logit Lens L30-40'
    },
    'logit_lens_mean_l20_30': {
        'layers': list(range(20, 31)),  # L20-30
        'display_name': 'Logit Lens L20-30'
    },
    # ALL LAYERS: For layerwise plotting across full model depth
    'logit_lens_all_layers': {
        'layers': list(range(0, 62)),  # All 62 layers (Gemma-2 27B)
        'display_name': 'Logit Lens All Layers'
    }
}

ALL_LAYERS_FOR_PLOTTING = list(range(0, 62))


def project_activations_batched(model, activations_dict, layers, batch_size=2048, verbose=True, emotion_token_ids=None, num_random_tokens=5000):
    """
    Project activations for multiple layers and tokens in batches.
    OPTIMIZED: Projects to emotion tokens + random subset instead of full 256k vocab.

    Uses subset projection: applies ln_final then projects only to ~6k tokens instead of 256k.
    This is ~40x faster than full vocab projection.

    Args:
        model: Model with project_on_vocab method and ln_final
        activations_dict: Dict[token_idx][layer] = activation array
        layers: List of layer indices to process
        batch_size: Number of activations to project at once
        verbose: Print progress
        emotion_token_ids: Optional dict of emotion -> token_ids for optimized extraction
        num_random_tokens: Number of random tokens to sample for mean estimation (default 5000)

    Returns:
        Dict[token_idx][layer] = logits array
    """
    import time
    device = next(model.parameters()).device
    dtype = model.lm_head.weight.dtype if hasattr(model, 'lm_head') else torch.float32
    vocab_size = model.lm_head.weight.shape[0]

    # Collect all (token, layer) pairs and their activations
    batch_items = []  # List of (token_idx, layer, activation)
    for token_idx, layer_dict in activations_dict.items():
        for layer in layers:
            if layer in layer_dict:
                batch_items.append((token_idx, layer, layer_dict[layer]))

    if not batch_items:
        return {}

    total_items = len(batch_items)
    num_batches = (total_items + batch_size - 1) // batch_size

    # Build emotion token indices and subset embeddings for optimized projection
    emotion_indices = None
    subset_embeddings = None
    n_emotion = 0

    if emotion_token_ids:
        all_emotion_ids = set()
        for emotion, ids in emotion_token_ids.items():
            all_emotion_ids.update(ids)
        emotion_indices = torch.tensor(sorted(all_emotion_ids), device=device, dtype=torch.long)
        n_emotion = len(emotion_indices)

        # Sample random tokens (excluding emotion tokens) for mean estimation
        all_indices = set(range(vocab_size))
        available_indices = list(all_indices - all_emotion_ids)
        np.random.seed(42)  # Fixed seed for reproducibility
        random_sample = np.random.choice(available_indices, size=min(num_random_tokens, len(available_indices)), replace=False)
        random_indices = torch.tensor(sorted(random_sample), device=device, dtype=torch.long)

        # Combine: [emotion_tokens, random_tokens]
        all_subset_indices = torch.cat([emotion_indices, random_indices])

        # Pre-extract subset embeddings from lm_head (only done once!)
        subset_embeddings = model.lm_head.weight[all_subset_indices].to(dtype)

        if verbose:
            print(f"      SUBSET PROJECTION: {n_emotion} emotion + {len(random_indices)} random = {len(all_subset_indices)} tokens (vs {vocab_size:,} full vocab)")

    if verbose:
        print(f"      Total (token, layer) pairs: {total_items:,}")
        print(f"      Processing in {num_batches} batches of {batch_size}...")

    # Process in batches
    logits_dict = {}
    mean_logits_dict = {}
    start_time = time.time()

    for batch_idx, batch_start in enumerate(range(0, len(batch_items), batch_size)):
        batch_end = min(batch_start + batch_size, len(batch_items))
        batch = batch_items[batch_start:batch_end]

        # Stack activations on CPU first, then single GPU transfer
        activations_np = np.stack([item[2] for item in batch])
        activations_batch = torch.from_numpy(activations_np).to(device).to(dtype)

        with torch.no_grad():
            if subset_embeddings is not None:
                # OPTIMIZED: Apply ln_final then project to subset only
                # This replicates project_on_vocab but only for ~6k tokens instead of 256k
                normalized = model.ln_final(activations_batch)  # Apply final layer norm
                logits_subset = normalized @ subset_embeddings.T  # Project to subset only

                # Split into emotion logits and random logits
                emotion_logits_batch = logits_subset[:, :n_emotion]
                random_logits_batch = logits_subset[:, n_emotion:]

                # Estimate mean from random tokens
                mean_logits_batch = random_logits_batch.mean(dim=-1)

                # Transfer only the small subset to CPU
                emotion_logits_np = emotion_logits_batch.cpu().float().numpy()
                mean_logits_np = mean_logits_batch.cpu().float().numpy()

                del normalized, logits_subset, emotion_logits_batch, random_logits_batch, mean_logits_batch
            else:
                # Full vocab projection (fallback)
                logits_batch = model.project_on_vocab(activations_batch)
                logits_batch_np = logits_batch.cpu().float().numpy()
                del logits_batch

        del activations_batch

        # Unpack results
        for idx, (token_idx, layer, _) in enumerate(batch):
            if token_idx not in logits_dict:
                logits_dict[token_idx] = {}

            if subset_embeddings is not None:
                logits_dict[token_idx][layer] = emotion_logits_np[idx]
                if token_idx not in mean_logits_dict:
                    mean_logits_dict[token_idx] = {}
                mean_logits_dict[token_idx][layer] = mean_logits_np[idx]
            else:
                logits_dict[token_idx][layer] = logits_batch_np[idx]

        # Progress logging
        if verbose and (batch_idx + 1) % max(1, num_batches // 10) == 0:
            elapsed = time.time() - start_time
            progress = (batch_idx + 1) / num_batches
            eta = elapsed / progress - elapsed if progress > 0 else 0
            print(f"      Batch {batch_idx+1}/{num_batches} ({progress*100:.1f}%) - Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")

    total_time = time.time() - start_time
    if verbose:
        print(f"      ✓ Completed {num_batches} batches in {total_time:.2f}s ({total_items/total_time:.0f} items/sec)")

    if emotion_indices is not None:
        return logits_dict, mean_logits_dict, emotion_indices
    else:
        return logits_dict


def preprocess_baseline_stats(baseline_stats_by_layer, emotion_token_ids, layers):
    """
    Pre-process baseline stats into arrays for fast vectorized lookups.

    Returns:
        baseline_arrays: Dict[layer][emotion] = {
            'token_id_to_idx': dict mapping token_id -> array index,
            'means': np array of means,
            'stds': np array of stds
        }
    """
    baseline_arrays = {}

    for layer in layers:
        if layer not in baseline_stats_by_layer:
            continue

        baseline_arrays[layer] = {}

        for emotion in EMOTIONS:
            emotion_token_list = emotion_token_ids[emotion]
            layer_baseline_stats = baseline_stats_by_layer[layer]  # Dict keyed by token ID (as strings)

            # Build arrays for tokens that have baseline stats
            token_ids = []
            means = []
            stds = []

            for tid in emotion_token_list:
                # Convert tid to string for lookup (baseline stats are keyed by string token IDs)
                tid_str = str(tid)
                if tid_str in layer_baseline_stats:
                    token_ids.append(tid)
                    means.append(layer_baseline_stats[tid_str]['mean'])
                    stds.append(layer_baseline_stats[tid_str]['std'])

            if token_ids:
                baseline_arrays[layer][emotion] = {
                    'token_id_to_idx': {tid: idx for idx, tid in enumerate(token_ids)},
                    'means': np.array(means),
                    'stds': np.array(stds)
                }

    return baseline_arrays


def compute_emotion_scores_vectorized(
    logits_dict,
    emotion_token_ids,
    layers,
    emotion_indices_map,
    baseline_arrays
):
    """
    FULLY VECTORIZED emotion score computation.

    Computes z-scores for all (tokens × layers × emotions) in batched numpy operations.
    ~10-50x faster than the loop-based version.

    Args:
        logits_dict: Dict[token_idx][layer] = numpy array of emotion logits
        emotion_token_ids: Dict[emotion] = list of token IDs
        layers: List of layer indices
        emotion_indices_map: Dict mapping token_id -> index in logits array
        baseline_arrays: Pre-processed baseline stats from preprocess_baseline_stats()

    Returns:
        Tuple of (scores_by_token, scores_by_token_by_layer)
    """
    # Build sorted token list for consistent indexing
    token_indices = sorted(logits_dict.keys())
    n_tokens = len(token_indices)
    n_layers = len(layers)

    if n_tokens == 0:
        return {}, {}

    # Get dimensions from sample
    sample_layer = next(iter(logits_dict[token_indices[0]].keys()))
    n_emotion_logits = len(logits_dict[token_indices[0]][sample_layer])

    # Build 3D logits array: (n_tokens, n_layers, n_emotion_logits)
    logits_arr = np.full((n_tokens, n_layers, n_emotion_logits), np.nan, dtype=np.float32)

    for t_idx, token_idx in enumerate(token_indices):
        layer_dict = logits_dict[token_idx]
        for l_idx, layer in enumerate(layers):
            if layer in layer_dict:
                logits_arr[t_idx, l_idx, :] = layer_dict[layer]

    # Pre-compute per-emotion data: indices into logits array and aligned baseline stats
    emotion_data = {}
    for emotion in EMOTIONS:
        tids = emotion_token_ids[emotion]
        # Get indices into the emotion logits array
        indices = [emotion_indices_map[tid] for tid in tids if tid in emotion_indices_map]
        if not indices:
            continue

        indices_arr = np.array(indices, dtype=np.int64)

        # Build aligned baseline arrays: (n_layers, n_emotion_tids)
        n_tids = len(indices)
        means = np.zeros((n_layers, n_tids), dtype=np.float32)
        stds = np.ones((n_layers, n_tids), dtype=np.float32)  # Default 1 to avoid div/0
        valid = np.zeros((n_layers, n_tids), dtype=bool)

        # Map from emotion_indices_map index -> position in our indices list
        idx_to_pos = {idx: pos for pos, idx in enumerate(indices)}

        for l_idx, layer in enumerate(layers):
            if layer not in baseline_arrays:
                continue
            baseline_info = baseline_arrays[layer].get(emotion)
            if not baseline_info:
                continue

            token_id_to_idx = baseline_info['token_id_to_idx']
            means_arr = baseline_info['means']
            stds_arr = baseline_info['stds']

            for tid in tids:
                if tid in token_id_to_idx and tid in emotion_indices_map:
                    b_idx = token_id_to_idx[tid]
                    emo_idx = emotion_indices_map[tid]
                    if emo_idx in idx_to_pos:
                        pos = idx_to_pos[emo_idx]
                        means[l_idx, pos] = means_arr[b_idx]
                        stds[l_idx, pos] = stds_arr[b_idx]
                        valid[l_idx, pos] = True

        emotion_data[emotion] = {
            'indices': indices_arr,
            'means': means,
            'stds': stds,
            'valid': valid
        }

    # Compute scores - VECTORIZED over all tokens at once
    # Result: (n_tokens, n_layers, n_emotions)
    scores_arr = np.full((n_tokens, n_layers, len(EMOTIONS)), np.nan, dtype=np.float32)

    for e_idx, emotion in enumerate(EMOTIONS):
        if emotion not in emotion_data:
            continue

        data = emotion_data[emotion]
        indices = data['indices']
        means = data['means']      # (n_layers, n_tids)
        stds = data['stds']
        valid = data['valid']

        # Extract emotion logits for all tokens: (n_tokens, n_layers, n_tids)
        emotion_logits = logits_arr[:, :, indices]

        # Broadcast baseline to match: (1, n_layers, n_tids)
        means_bc = means[np.newaxis, :, :]
        stds_bc = stds[np.newaxis, :, :]
        valid_bc = valid[np.newaxis, :, :]

        # VECTORIZED z-score computation: (n_tokens, n_layers, n_tids)
        z_scores = (emotion_logits - means_bc) / (stds_bc + 1e-8)

        # Mask invalid entries
        z_scores = np.where(valid_bc, z_scores, np.nan)

        # Mean over emotion tokens (axis=2): (n_tokens, n_layers)
        with np.errstate(invalid='ignore'):
            scores_arr[:, :, e_idx] = np.nanmean(z_scores, axis=2)

    # Convert to dict format for compatibility with rest of pipeline
    scores_by_token = {}
    scores_by_token_by_layer = {}

    # Compute layer-averaged scores: mean over layers (axis 1)
    with np.errstate(invalid='ignore'):
        layer_avg_scores = np.nanmean(scores_arr, axis=1)  # (n_tokens, n_emotions)

    for t_idx, token_idx in enumerate(token_indices):
        # Layer-averaged scores
        scores_by_token[token_idx] = {
            emotion: float(layer_avg_scores[t_idx, e_idx])
            if not np.isnan(layer_avg_scores[t_idx, e_idx]) else 0.0
            for e_idx, emotion in enumerate(EMOTIONS)
        }

        # Per-layer scores
        scores_by_token_by_layer[token_idx] = {}
        for l_idx, layer in enumerate(layers):
            scores_by_token_by_layer[token_idx][layer] = {
                emotion: float(scores_arr[t_idx, l_idx, e_idx])
                if not np.isnan(scores_arr[t_idx, l_idx, e_idx]) else 0.0
                for e_idx, emotion in enumerate(EMOTIONS)
            }

    return scores_by_token, scores_by_token_by_layer


def compute_emotion_scores_from_logits_batched(
    logits_dict,
    emotion_token_ids,
    baseline_stats_by_layer,
    layers,
    aggregation="mean",
    emotion_indices_map=None,
    baseline_arrays=None
):
    """
    Compute emotion scores from pre-computed logits.

    Args:
        logits_dict: Dict[token_idx][layer] = logits array (full vocab OR indexed subset if emotion_indices_map provided)
        emotion_token_ids: Dict[emotion] = list of token IDs
        baseline_stats_by_layer: Dict[layer][emotion][token_id] = {'mean', 'std'} (legacy, use baseline_arrays instead)
        layers: List of layers to process
        aggregation: "mean" or "max"
        emotion_indices_map: Optional dict mapping token_id -> index in logits array (for GPU-optimized mode)
        baseline_arrays: Pre-processed baseline stats (if None, will use baseline_stats_by_layer)

    Returns:
        Tuple of (scores_by_token, scores_by_token_by_layer)
        - scores_by_token: Dict[token_idx][emotion] = score (layer-averaged)
        - scores_by_token_by_layer: Dict[token_idx][layer][emotion] = score
    """
    scores_by_token = {}
    scores_by_token_by_layer = {}

    for token_idx, layer_logits in logits_dict.items():
        # Per-layer scores
        per_layer_scores = {}
        for layer in layers:
            if layer not in layer_logits:
                continue

            logits_np = layer_logits[layer]
            layer_scores = {}

            for emotion in EMOTIONS:
                token_ids = emotion_token_ids[emotion]

                # Extract emotion logits (indexed if using GPU optimization, direct if full vocab)
                if emotion_indices_map is not None:
                    # GPU-optimized: logits_np is indexed subset, use map to find positions
                    indices = [emotion_indices_map[tid] for tid in token_ids if tid in emotion_indices_map]
                    emotion_logits = logits_np[indices] if indices else np.array([])
                    # Align token_ids to match
                    token_ids = [tid for tid in token_ids if tid in emotion_indices_map]
                else:
                    # Legacy: logits_np is full vocab, index directly
                    emotion_logits = logits_np[token_ids]

                # Normalize using baseline stats for this layer
                normalized_scores = []

                if baseline_arrays is not None and layer in baseline_arrays:
                    # OPTIMIZED: Use pre-processed arrays for O(1) lookups
                    baseline_info = baseline_arrays[layer].get(emotion)
                    if baseline_info:
                        token_id_to_idx = baseline_info['token_id_to_idx']
                        means_arr = baseline_info['means']
                        stds_arr = baseline_info['stds']

                        # Find which token_ids have baseline stats and extract their indices
                        valid_indices = []
                        valid_logits = []
                        for tid, logit in zip(token_ids, emotion_logits):
                            if tid in token_id_to_idx:
                                valid_indices.append(token_id_to_idx[tid])
                                valid_logits.append(logit)

                        if valid_indices:
                            # Vectorized z-score: extract relevant means/stds using array indexing
                            valid_indices_arr = np.array(valid_indices)
                            logits_arr = np.array(valid_logits)
                            relevant_means = means_arr[valid_indices_arr]
                            relevant_stds = stds_arr[valid_indices_arr]
                            normalized_scores = ((logits_arr - relevant_means) / (relevant_stds + 1e-8)).tolist()

                elif layer in baseline_stats_by_layer:
                    # LEGACY: Dict-based lookups (slower)
                    baseline_dict = baseline_stats_by_layer[layer].get(emotion, {})
                    if baseline_dict:
                        means = []
                        stds = []
                        valid_logits = []
                        for token_id, logit in zip(token_ids, emotion_logits):
                            if token_id in baseline_dict:
                                stats = baseline_dict[token_id]
                                means.append(stats['mean'])
                                stds.append(stats['std'])
                                valid_logits.append(logit)

                        if valid_logits:
                            # Vectorized z-score computation
                            means_arr = np.array(means)
                            stds_arr = np.array(stds)
                            logits_arr = np.array(valid_logits)
                            normalized_scores = ((logits_arr - means_arr) / (stds_arr + 1e-8)).tolist()

                # Aggregate
                if normalized_scores:
                    if aggregation == "mean":
                        layer_scores[emotion] = np.mean(normalized_scores)
                    elif aggregation == "max":
                        layer_scores[emotion] = np.max(normalized_scores)
                    else:
                        layer_scores[emotion] = np.mean(normalized_scores)
                else:
                    layer_scores[emotion] = 0.0

            per_layer_scores[layer] = layer_scores

        # Layer-averaged scores: CRITICAL - Average LOGITS first, then compute scores
        # This matches the old working behavior: average activations/logits before scoring
        if per_layer_scores and token_idx in logits_dict:
            # Collect logits for each emotion across layers
            avg_logits_per_emotion = {}
            for emotion in EMOTIONS:
                # Get logits for this emotion's tokens across all layers
                token_ids_for_emotion = emotion_token_ids[emotion]
                layer_logits_for_emotion = []

                for layer in layers:
                    if layer in logits_dict[token_idx]:
                        logits_for_layer = logits_dict[token_idx][layer]
                        # Extract logits for this emotion's tokens
                        if emotion_indices_map:
                            # GPU-optimized mode: use indices
                            indices_for_emotion = [emotion_indices_map[tid] for tid in token_ids_for_emotion if tid in emotion_indices_map]
                            emotion_logits_this_layer = [logits_for_layer[idx] for idx in indices_for_emotion]
                        else:
                            # Full vocab mode: direct indexing
                            emotion_logits_this_layer = []
                            for tid in token_ids_for_emotion:
                                if tid < len(logits_for_layer):
                                    emotion_logits_this_layer.append(logits_for_layer[tid])

                        if emotion_logits_this_layer:
                            layer_logits_for_emotion.append(emotion_logits_this_layer)

                if layer_logits_for_emotion:
                    # Average across layers first (NOT across emotion tokens yet)
                    avg_logits_per_emotion[emotion] = np.mean(layer_logits_for_emotion, axis=0)

            # Now compute emotion scores from layer-averaged logits
            avg_scores = {}
            for emotion in EMOTIONS:
                if emotion in avg_logits_per_emotion:
                    averaged_logits = avg_logits_per_emotion[emotion]

                    # For layer-averaged scores, we need layer-averaged baseline stats
                    # Cannot use per-layer baseline_arrays here
                    token_ids_list = emotion_token_ids[emotion]
                    layer_stats_for_tokens = []
                    for layer in layers:
                        # baseline_stats_by_layer[layer] is a dict keyed by string token IDs
                        layer_baseline_stats = baseline_stats_by_layer.get(layer, {})
                        layer_means = []
                        layer_stds = []
                        for tid in token_ids_list:
                            tid_str = str(tid)
                            if tid_str in layer_baseline_stats:
                                layer_means.append(layer_baseline_stats[tid_str]['mean'])
                                layer_stds.append(layer_baseline_stats[tid_str]['std'])
                        if layer_means:
                            layer_stats_for_tokens.append((layer_means, layer_stds))
                    if layer_stats_for_tokens:
                        means = np.mean([s[0] for s in layer_stats_for_tokens], axis=0)
                        stds = np.mean([s[1] for s in layer_stats_for_tokens], axis=0)
                    else:
                        means = []
                        stds = []

                    # Compute z-scores
                    if len(means) > 0 and len(averaged_logits) == len(means):
                        means_arr = np.array(means)
                        stds_arr = np.array(stds)
                        normalized_scores = (averaged_logits - means_arr) / (stds_arr + 1e-8)

                        # Aggregate (mean or max)
                        if aggregation == "mean":
                            avg_scores[emotion] = np.mean(normalized_scores)
                        elif aggregation == "max":
                            avg_scores[emotion] = np.max(normalized_scores)
                        else:
                            avg_scores[emotion] = np.mean(normalized_scores)
                    else:
                        avg_scores[emotion] = 0.0
                else:
                    avg_scores[emotion] = 0.0

            scores_by_token[token_idx] = avg_scores

        scores_by_token_by_layer[token_idx] = per_layer_scores

    return scores_by_token, scores_by_token_by_layer


def compute_mean_logit_batched(logits_dict, layers):
    """Compute mean logit across vocabulary for each token."""
    mean_logits = {}
    for token_idx, layer_logits in logits_dict.items():
        # Average mean logit across layers
        layer_means = [layer_logits[layer].mean() for layer in layers if layer in layer_logits]
        mean_logits[token_idx] = np.mean(layer_means) if layer_means else 0.0
    return mean_logits


# ============================================================================
# PROBE SCORING FUNCTIONS
# ============================================================================

def load_probes_for_all_layers(probe_pattern: str, num_layers: int = 62):
    """
    Load all layer-specific probes into memory.

    Args:
        probe_pattern: Pattern like 'probe_layer{layer}_nc0_seed0.pkl'
        num_layers: Number of layers (default 62 for Gemma-2 27B)

    Returns:
        Dict[layer] = probe model (nn.Linear)
    """
    probes = {}
    for layer in range(num_layers):
        probe_file = PROBE_DIR / probe_pattern.format(layer=layer)
        if probe_file.exists():
            with open(probe_file, 'rb') as f:
                probe_data = pickle.load(f)
            probes[layer] = probe_data['model']
    return probes


def load_cpca_components(cpca_path: Path, n_components: int = 10):
    """
    Load cPCA transformation components.

    Returns:
        numpy array of shape (num_layers, n_components, hidden_dim)
    """
    cpca_data = np.load(cpca_path)
    components = cpca_data['components']  # Shape: (62, 50, 5376)
    return components[:, :n_components, :]


def apply_probes_batched(
    activations_by_token: dict,
    probes: dict,
    layers: list,
    cpca_components: np.ndarray = None,
    device: str = 'cuda',
    batch_size: int = 4096
):
    """
    Apply probes to activations in batches (GPU-accelerated).

    Args:
        activations_by_token: Dict[token_idx][layer] = activation array
        probes: Dict[layer] = probe model (nn.Linear)
        layers: List of layer indices to process
        cpca_components: Optional cPCA components for transformation
        device: 'cuda' or 'cpu'
        batch_size: Batch size for GPU processing

    Returns:
        Dict[token_idx][layer] = emotion scores array (6 emotions, excluding neutral)
    """
    # Collect all (token, layer) pairs
    batch_items = []
    for token_idx, layer_dict in activations_by_token.items():
        for layer in layers:
            if layer in layer_dict and layer in probes:
                batch_items.append((token_idx, layer, layer_dict[layer]))

    if not batch_items:
        return {}

    # Prepare probes on device
    probes_on_device = {}
    for layer in probes:
        probes_on_device[layer] = probes[layer].to(device).eval()

    # Prepare cPCA components on device if needed
    cpca_tensor = None
    if cpca_components is not None:
        cpca_tensor = torch.from_numpy(cpca_components).float().to(device)

    # Process in batches, grouped by layer
    scores_dict = {}
    total_items = len(batch_items)

    for batch_start in range(0, total_items, batch_size):
        batch_end = min(batch_start + batch_size, total_items)
        batch = batch_items[batch_start:batch_end]

        # Group by layer for efficient batch processing
        layer_groups = {}
        for token_idx, layer, activation in batch:
            if layer not in layer_groups:
                layer_groups[layer] = []
            layer_groups[layer].append((token_idx, activation))

        # Process each layer group
        for layer, items in layer_groups.items():
            if layer not in probes_on_device:
                continue

            probe = probes_on_device[layer]

            # Stack activations
            activations_np = np.stack([item[1] for item in items])
            activations_tensor = torch.from_numpy(activations_np).float().to(device)

            # Apply cPCA transform if needed
            if cpca_tensor is not None:
                # Transform: activations @ components.T
                activations_tensor = activations_tensor @ cpca_tensor[layer].T

            # Apply probe
            with torch.no_grad():
                logits = probe(activations_tensor)  # (batch, 7) - includes neutral
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)
                # Extract only 6 emotions (exclude neutral at index 6)
                emotion_probs = probs[:, :6].cpu().numpy()

            # Store results
            for idx, (token_idx, _) in enumerate(items):
                if token_idx not in scores_dict:
                    scores_dict[token_idx] = {}
                scores_dict[token_idx][layer] = emotion_probs[idx]

    return scores_dict


def compute_probe_scores_for_conversation(
    activations_by_token: dict,
    probes_raw: dict,
    probes_cpca: dict,
    cpca_components: np.ndarray,
    layers: list,
    sentences: list,
    device: str = 'cuda'
):
    """
    Compute per-layer probe scores for a conversation.

    Args:
        activations_by_token: Dict[token_idx][layer] = activation array
        probes_raw: Dict of raw probes by layer
        probes_cpca: Dict of cPCA probes by layer
        cpca_components: cPCA transformation components
        layers: List of layers to process
        sentences: List of sentence dicts with start_token, end_token, sentence_id
        device: 'cuda' or 'cpu'

    Returns:
        Tuple of (text_raw_by_layer, text_cpca_by_layer) dicts
        Each maps sentence_id -> layer -> scores array
    """
    # Apply raw probes
    scores_raw = apply_probes_batched(
        activations_by_token, probes_raw, layers,
        cpca_components=None, device=device
    )

    # Apply cPCA probes
    scores_cpca = apply_probes_batched(
        activations_by_token, probes_cpca, layers,
        cpca_components=cpca_components, device=device
    )

    # Aggregate to sentence level
    text_raw_by_layer = {}
    text_cpca_by_layer = {}

    for sent in sentences:
        sent_id = sent['sentence_id']
        start_tok = sent['start_token']
        end_tok = sent['end_token']

        # Initialize per-layer storage for this sentence
        text_raw_by_layer[sent_id] = {}
        text_cpca_by_layer[sent_id] = {}

        for layer in layers:
            # Collect raw scores for tokens in this sentence
            raw_scores_list = []
            cpca_scores_list = []

            for t in range(start_tok, end_tok):
                if t in scores_raw and layer in scores_raw[t]:
                    raw_scores_list.append(scores_raw[t][layer])
                if t in scores_cpca and layer in scores_cpca[t]:
                    cpca_scores_list.append(scores_cpca[t][layer])

            # Average within sentence
            if raw_scores_list:
                text_raw_by_layer[sent_id][layer] = np.mean(raw_scores_list, axis=0).tolist()
            if cpca_scores_list:
                text_cpca_by_layer[sent_id][layer] = np.mean(cpca_scores_list, axis=0).tolist()

    return text_raw_by_layer, text_cpca_by_layer


# ============================================================================
# AXIS SCORING FUNCTIONS
# ============================================================================

def compute_axis_scores_per_layer(
    model,
    activations_by_token: dict,
    axis_token_groups: dict,
    layers: list,
    sentences: list,
    batch_size: int = 4096,
    device: str = 'cuda'
):
    """
    Compute per-layer axis scores (valence, arousal, dominance, approach_avoidance).

    Axis score = mean(high_token_logits) - mean(low_token_logits), normalized.

    Args:
        model: Model with project_on_vocab method
        activations_by_token: Dict[token_idx][layer] = activation array
        axis_token_groups: Dict[axis]['high'/'low'] = list of token IDs
        layers: List of layer indices
        sentences: List of sentence dicts
        batch_size: Batch size for GPU processing
        device: 'cuda' or 'cpu'

    Returns:
        Dict[sentence_id][layer] = array of 4 axis scores
    """
    import time

    # Collect all axis token IDs (high + low for each axis)
    all_axis_token_ids = set()
    for axis, groups in axis_token_groups.items():
        all_axis_token_ids.update(groups['high'])
        all_axis_token_ids.update(groups['low'])
    all_axis_token_ids = sorted(all_axis_token_ids)

    # Build index map: token_id -> position in extracted logits
    axis_token_id_to_idx = {tid: idx for idx, tid in enumerate(all_axis_token_ids)}
    axis_indices_tensor = torch.tensor(all_axis_token_ids, device=device, dtype=torch.long)

    # Collect all (token, layer) pairs
    batch_items = []
    for token_idx, layer_dict in activations_by_token.items():
        for layer in layers:
            if layer in layer_dict:
                batch_items.append((token_idx, layer, layer_dict[layer]))

    if not batch_items:
        return {}

    dtype = model.lm_head.weight.dtype if hasattr(model, 'lm_head') else torch.float32

    # Extract axis token logits in batches
    axis_logits_dict = {}  # token_idx -> layer -> axis_logits array

    for batch_start in range(0, len(batch_items), batch_size):
        batch_end = min(batch_start + batch_size, len(batch_items))
        batch = batch_items[batch_start:batch_end]

        # Stack activations
        activations_np = np.stack([item[2] for item in batch])
        activations_tensor = torch.from_numpy(activations_np).to(device).to(dtype)

        # Project to vocab
        with torch.no_grad():
            logits_batch = model.project_on_vocab(activations_tensor)
            # Extract only axis token logits
            axis_logits_batch = logits_batch[:, axis_indices_tensor].cpu().float().numpy()
            del logits_batch

        del activations_tensor

        # Store results
        for idx, (token_idx, layer, _) in enumerate(batch):
            if token_idx not in axis_logits_dict:
                axis_logits_dict[token_idx] = {}
            axis_logits_dict[token_idx][layer] = axis_logits_batch[idx]

    # Compute axis scores per token per layer
    # axis_score = mean(high_logits) - mean(low_logits)
    token_axis_scores = {}  # token_idx -> layer -> [4 axis scores]

    for token_idx, layer_dict in axis_logits_dict.items():
        token_axis_scores[token_idx] = {}

        for layer, axis_logits in layer_dict.items():
            layer_scores = []

            for axis in AXES:
                high_ids = axis_token_groups[axis]['high']
                low_ids = axis_token_groups[axis]['low']

                # Get indices for high/low tokens
                high_indices = [axis_token_id_to_idx[tid] for tid in high_ids if tid in axis_token_id_to_idx]
                low_indices = [axis_token_id_to_idx[tid] for tid in low_ids if tid in axis_token_id_to_idx]

                # Compute mean logits
                high_mean = np.mean(axis_logits[high_indices]) if high_indices else 0.0
                low_mean = np.mean(axis_logits[low_indices]) if low_indices else 0.0

                # Axis score is the difference
                axis_score = high_mean - low_mean
                layer_scores.append(axis_score)

            token_axis_scores[token_idx][layer] = np.array(layer_scores)

    # Aggregate to sentence level
    axis_lens_by_layer = {}

    for sent in sentences:
        sent_id = sent['sentence_id']
        start_tok = sent['start_token']
        end_tok = sent['end_token']

        axis_lens_by_layer[sent_id] = {}

        for layer in layers:
            scores_list = []
            for t in range(start_tok, end_tok):
                if t in token_axis_scores and layer in token_axis_scores[t]:
                    scores_list.append(token_axis_scores[t][layer])

            if scores_list:
                axis_lens_by_layer[sent_id][layer] = np.mean(scores_list, axis=0).tolist()

    return axis_lens_by_layer


# Rest of the code (baseline correction functions) stays the same...
def compute_baseline_correction_alpha(all_emotion_scores, all_mean_logits):
    """Compute optimal alpha to remove correlation."""
    logit_mean = all_mean_logits.mean()
    logit_std = all_mean_logits.std()

    if logit_std < 1e-8:
        return {'alpha': 0.0, 'correlation_before': 0.0, 'correlation_after': 0.0,
                'logit_mean': logit_mean, 'logit_std': 1.0}

    normalized_logits = (all_mean_logits - logit_mean) / logit_std
    mean_emotion_scores = all_emotion_scores.mean(axis=1)
    correlation_before = np.corrcoef(mean_emotion_scores, all_mean_logits)[0, 1]

    # FIX: Don't divide by logit_std here since we apply to normalized_logits (already divided by logit_std)
    # Formula: corrected = y - r * σ_y * z_x where z_x = (x - μ_x) / σ_x
    alpha = correlation_before * mean_emotion_scores.std()

    corrected_scores = mean_emotion_scores - alpha * normalized_logits
    correlation_after = np.corrcoef(corrected_scores, all_mean_logits)[0, 1]

    return {
        'alpha': alpha,
        'correlation_before': correlation_before,
        'correlation_after': correlation_after,
        'logit_mean': logit_mean,
        'logit_std': logit_std
    }


def main():
    import time
    script_start = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="Add logit lens and optionally probe scores to pickle")
    parser.add_argument("--input", required=True, help="Input pickle file")
    parser.add_argument("--output", required=True, help="Output pickle file")
    parser.add_argument("--test", action="store_true", help="Test mode (5 conversations)")
    parser.add_argument("--add-probes", action="store_true", help="Also compute per-layer probe scores (text_raw, text_cpca)")
    parser.add_argument("--add-axes", action="store_true", help="Also compute per-layer axis scores (valence, arousal, dominance, approach_avoidance)")
    args = parser.parse_args()

    TEST_ONLY = args.test
    NUM_TEST = 5
    APPLY_BASELINE_CORRECTION = True
    STORE_PER_LAYER = True
    BATCH_SIZE = 16384  # Project 16384 (layer, token) pairs at once
    ADD_PROBE_SCORES = args.add_probes
    ADD_AXIS_SCORES = args.add_axes

    print("="*80)
    print("ADD LOGIT LENS TO EXISTING PICKLE (BATCHED VERSION - 50-100x FASTER)")
    if ADD_PROBE_SCORES:
        print("+ PER-LAYER PROBE SCORES (text_raw, text_cpca)")
    if ADD_AXIS_SCORES:
        print("+ PER-LAYER AXIS SCORES (valence, arousal, dominance, approach_avoidance)")
    print("="*80)

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    if TEST_ONLY:
        print(f"TEST MODE: Processing only first {NUM_TEST} conversations")

    # Load existing data
    print("\n[1/6] Loading existing data...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    total_convs = len(data['conversations'])
    print(f"  Total conversations: {total_convs}")

    # Add global_token_index field to all sentences (alias for start_token)
    print("  Adding global_token_index to sentences...")
    for conv in data['conversations']:
        for sent in conv['sentences']:
            if 'global_token_index' not in sent:
                sent['global_token_index'] = sent['start_token']
    print("  ✓ Added global_token_index")

    # Load model
    print("\n[2/6] Loading model...")
    BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
    model, tokenizer = load_base_model(BASE_MODEL_NAME)
    print("  ✓ Model loaded")

    # Load emotion token IDs
    print("\n[3/6] Loading emotion token IDs...")
    emotion_token_ids = load_emotion_token_ids_from_json(BASE_MODEL_NAME)
    print(f"  ✓ Loaded {len(emotion_token_ids)} emotions")

    # Load axis token groups if enabled
    axis_token_groups = None
    if ADD_AXIS_SCORES:
        print("  Loading axis token groups...")
        axis_token_groups = load_axis_token_groups(BASE_MODEL_NAME)
        for axis, groups in axis_token_groups.items():
            print(f"    {axis}: {len(groups['high'])} high, {len(groups['low'])} low")

    # Load reference statistics
    print("\n[4/6] Loading baseline statistics...")
    ACTIVATION_STRATEGY = "generated_tokens_avg"
    SCRIPT_DIR = Path("/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens")
    ref_stats = load_reference_stats_for_strategy(
        model_name=BASE_MODEL_NAME,
        activation_strategy=ACTIVATION_STRATEGY,
        script_dir=SCRIPT_DIR
    )

    # Detect actual layers from model
    try:
        num_model_layers = len(model.model.layers)
        actual_layers = list(range(num_model_layers))
        print(f"  Detected {num_model_layers} layers in model")
    except:
        actual_layers = list(range(42))
        print(f"  Using default layer range (0-41)")

    # Organize baseline stats by layer (using layers_data format from ref_stats)
    baseline_stats_by_layer = {}
    if 'layers_data' in ref_stats:
        for layer in actual_layers:
            layer_key = str(layer)
            if layer_key in ref_stats['layers_data']:
                baseline_stats_by_layer[layer] = ref_stats['layers_data'][layer_key]['statistics']

    print(f"  ✓ Loaded baseline stats for {len(baseline_stats_by_layer)} layers")

    # Pre-process baseline stats into arrays for fast lookups (OPTIMIZATION)
    print("  Pre-processing baseline stats into arrays...")
    baseline_arrays = preprocess_baseline_stats(baseline_stats_by_layer, emotion_token_ids, actual_layers)
    print(f"  ✓ Pre-processed baseline arrays for {len(baseline_arrays)} layers")

    # Load probes if enabled
    probes_raw = None
    probes_cpca = None
    cpca_components = None
    if ADD_PROBE_SCORES:
        print("\n  Loading probes for per-layer scoring...")
        print("    Loading text_raw probes (nc0)...")
        probes_raw = load_probes_for_all_layers(
            PROBE_CONFIGS['text_raw']['probe_pattern'],
            num_layers=len(actual_layers)
        )
        print(f"      ✓ Loaded {len(probes_raw)} layer probes")

        print("    Loading text_cpca probes (nc10)...")
        probes_cpca = load_probes_for_all_layers(
            PROBE_CONFIGS['text_cpca']['probe_pattern'],
            num_layers=len(actual_layers)
        )
        print(f"      ✓ Loaded {len(probes_cpca)} layer probes")

        print("    Loading cPCA components...")
        cpca_components = load_cpca_components(
            CPCA_PATH,
            n_components=PROBE_CONFIGS['text_cpca']['n_components']
        )
        print(f"      ✓ cPCA components shape: {cpca_components.shape}")

    # PASS 1: Extract activations and compute scores (BATCHED)
    print(f"\n[5/6] PASS 1 (BATCHED): Processing {total_convs if not TEST_ONLY else NUM_TEST} conversations...")

    num_to_process = NUM_TEST if TEST_ONLY else total_convs
    sentence_data_by_range = {probe_key: [] for probe_key in LAYER_RANGE_CONFIGS.keys()}

    import time
    overall_start = time.time()

    for i in range(num_to_process):
        conv_start = time.time()
        conv = data['conversations'][i]
        print(f"\n  [{i+1}/{num_to_process}] Sample ID: {conv['sample_id']}")

        # Build conversation text (EXACT format from working version)
        conversation_text = ""
        for turn in conv['conversation']:
            role = turn['role']
            content = turn['content']
            if role == 'user':
                conversation_text += f"User: {content}\n"
            else:
                conversation_text += f"Assistant: {content}\n"

        # Extract activations for ALL layers (with ALL required parameters)
        print(f"    Extracting activations...")
        extract_start = time.time()
        activations_by_token, token_ids = extract_token_level_activations(
            model=model,
            tokenizer=tokenizer,
            prompt=conversation_text,
            layers=actual_layers,
            start_token_idx=0,
            system_prompt=None,
            num_generated_tokens=0
        )
        extract_time = time.time() - extract_start

        print(f"    ✓ Extracted {len(activations_by_token)} tokens × {len(actual_layers)} layers ({extract_time:.2f}s)")

        # BATCH PROJECT ALL ACTIVATIONS AT ONCE (with GPU-side extraction optimization!)
        print(f"    Projecting to vocab (batched, batch_size={BATCH_SIZE})...")
        result = project_activations_batched(
            model, activations_by_token, actual_layers, batch_size=BATCH_SIZE, verbose=True, emotion_token_ids=emotion_token_ids
        )

        # Unpack optimized results
        if isinstance(result, tuple):
            logits_dict, mean_logits_dict, emotion_indices = result
            # Build index mapping for emotion scores
            emotion_indices_map = {int(tid): idx for idx, tid in enumerate(emotion_indices.cpu().tolist())}
        else:
            logits_dict = result
            mean_logits_dict = None
            emotion_indices_map = None

        # Compute emotion scores from logits
        print(f"    Computing emotion scores...")
        score_start = time.time()

        # Use VECTORIZED version when emotion_indices_map is available (GPU-optimized path)
        if emotion_indices_map is not None and baseline_arrays is not None:
            scores_by_token, scores_by_token_by_layer = compute_emotion_scores_vectorized(
                logits_dict, emotion_token_ids, actual_layers,
                emotion_indices_map, baseline_arrays
            )
        else:
            # Fall back to loop-based version for legacy path
            scores_by_token, scores_by_token_by_layer = compute_emotion_scores_from_logits_batched(
                logits_dict, emotion_token_ids, baseline_stats_by_layer, actual_layers,
                emotion_indices_map=emotion_indices_map, baseline_arrays=baseline_arrays
            )

        score_time = time.time() - score_start
        print(f"    ✓ Computed emotion scores ({score_time:.2f}s) [vectorized={emotion_indices_map is not None}]")

        # Compute probe scores if enabled
        text_raw_by_layer = None
        text_cpca_by_layer = None
        if ADD_PROBE_SCORES and probes_raw and probes_cpca:
            print(f"    Computing per-layer probe scores...")
            probe_start = time.time()
            device = str(next(model.parameters()).device)
            text_raw_by_layer, text_cpca_by_layer = compute_probe_scores_for_conversation(
                activations_by_token, probes_raw, probes_cpca, cpca_components,
                actual_layers, conv['sentences'], device=device
            )
            probe_time = time.time() - probe_start
            print(f"    ✓ Computed probe scores ({probe_time:.2f}s)")

            # Store probe scores in conversation
            conv['text_raw_by_layer'] = text_raw_by_layer
            conv['text_cpca_by_layer'] = text_cpca_by_layer

        # Compute axis scores if enabled
        if ADD_AXIS_SCORES and axis_token_groups:
            print(f"    Computing per-layer axis scores...")
            axis_start = time.time()
            device = str(next(model.parameters()).device)
            axis_lens_by_layer = compute_axis_scores_per_layer(
                model, activations_by_token, axis_token_groups,
                actual_layers, conv['sentences'], device=device
            )
            axis_time = time.time() - axis_start
            print(f"    ✓ Computed axis scores ({axis_time:.2f}s)")

            # Store axis scores in conversation
            conv['axis_lens_by_layer'] = axis_lens_by_layer

        # Process sentences for each layer range
        for probe_key, config in LAYER_RANGE_CONFIGS.items():
            layers = config['layers']

            # FIXED: Compute mean logits for THIS probe's layer range (not all layers)
            if mean_logits_dict is not None:
                mean_logits_by_token = {tok: np.mean([mean_logits_dict[tok][l] for l in layers if l in mean_logits_dict[tok]])
                                         for tok in mean_logits_dict.keys()}
            else:
                mean_logits_by_token = compute_mean_logit_batched(logits_dict, layers)

            for sent in conv['sentences']:
                sent_id = sent['sentence_id']
                start_tok = sent['start_token']
                end_tok = sent['end_token']

                # Aggregate scores for this sentence
                sent_scores_list = []
                sent_logits_list = []
                per_layer_sentence_scores = {}
                per_layer_sentence_logits = {}  # NEW: Store per-layer mean logits

                for t in range(start_tok, end_tok):
                    # Use correctly computed layer-averaged scores from scores_by_token
                    # (These were computed by averaging LOGITS first, then computing emotion scores)
                    if t in scores_by_token:
                        # Convert dict to array in correct emotion order
                        sent_scores_list.append(np.array([scores_by_token[t][e] for e in EMOTIONS]))

                        if t in mean_logits_by_token:
                            sent_logits_list.append(mean_logits_by_token[t])

                    # Per-layer scores and logits
                    if STORE_PER_LAYER:
                        if t in scores_by_token_by_layer:
                            for layer in layers:
                                if layer in scores_by_token_by_layer[t]:
                                    if layer not in per_layer_sentence_scores:
                                        per_layer_sentence_scores[layer] = []
                                    per_layer_sentence_scores[layer].append(
                                        np.array([scores_by_token_by_layer[t][layer][e] for e in EMOTIONS])
                                    )
                        # NEW: Also collect per-layer mean logits
                        if t in mean_logits_dict:
                            for layer in layers:
                                if layer in mean_logits_dict[t]:
                                    if layer not in per_layer_sentence_logits:
                                        per_layer_sentence_logits[layer] = []
                                    per_layer_sentence_logits[layer].append(mean_logits_dict[t][layer])

                # Average within sentence
                if sent_scores_list:
                    sentence_emotion_scores = np.mean(sent_scores_list, axis=0)
                    sentence_mean_logit = np.mean(sent_logits_list) if sent_logits_list else 0.0
                else:
                    sentence_emotion_scores = np.zeros(6)
                    sentence_mean_logit = 0.0

                # Average per-layer scores
                per_layer_avg = {}
                per_layer_mean_logits = {}  # NEW: Average per-layer mean logits
                if STORE_PER_LAYER:
                    for layer, scores_list in per_layer_sentence_scores.items():
                        per_layer_avg[layer] = np.mean(scores_list, axis=0) if scores_list else np.zeros(6)
                    # NEW: Average per-layer mean logits
                    for layer, logits_list in per_layer_sentence_logits.items():
                        per_layer_mean_logits[layer] = np.mean(logits_list) if logits_list else 0.0

                sentence_data_by_range[probe_key].append({
                    'conv_idx': i,
                    'sent_id': sent_id,
                    'emotion_scores': sentence_emotion_scores,
                    'mean_logit': sentence_mean_logit,
                    'per_layer_scores': per_layer_avg if STORE_PER_LAYER else None,
                    'per_layer_mean_logits': per_layer_mean_logits if STORE_PER_LAYER else None  # NEW
                })

        conv_time = time.time() - conv_start
        avg_time = (time.time() - overall_start) / (i + 1)
        eta = avg_time * (num_to_process - i - 1)
        print(f"    ✓ Processed {len(conv['sentences'])} sentences in {conv_time:.2f}s (avg: {avg_time:.1f}s/conv, ETA: {eta/60:.1f}m)")

        # EARLY SAVE: After first conversation, save checkpoint for testing
        if i == 0:
            print("\n  === EARLY CHECKPOINT: Validation & Saving ===")

            # VERIFICATION: Compare subset projection vs full projection for a sample
            print("\n  [VERIFICATION] Comparing subset vs full projection (with ln_final)...")
            verify_start = time.time()

            # Get device and dtype from model
            verify_device = next(model.parameters()).device
            verify_dtype = model.lm_head.weight.dtype

            # Get a small sample of activations to verify
            sample_tokens = list(activations_by_token.keys())[:5]
            sample_layers = actual_layers[:3]

            max_diff = 0.0
            for tok in sample_tokens:
                for layer in sample_layers:
                    if layer in activations_by_token[tok]:
                        act = activations_by_token[tok][layer]
                        act_tensor = torch.from_numpy(act).unsqueeze(0).to(verify_device).to(verify_dtype)

                        # Full projection
                        with torch.no_grad():
                            full_logits = model.project_on_vocab(act_tensor)
                            full_emotion = full_logits[:, emotion_indices].cpu().float().numpy()

                        # Get subset projection result
                        subset_emotion = logits_dict[tok][layer]

                        diff = np.abs(full_emotion.squeeze() - subset_emotion).max()
                        max_diff = max(max_diff, diff)

                        del act_tensor, full_logits

            verify_time = time.time() - verify_start
            print(f"    Emotion logits max diff: {max_diff:.6f} (should be ~0)")
            if max_diff > 0.01:
                print(f"    ⚠️  WARNING: Emotion logits differ! Check ln_final application.")
            else:
                print(f"    ✓ Emotion logits match perfectly!")
            print(f"    Verification completed in {verify_time:.2f}s")

            checkpoint_path = output_path.parent / "checkpoint_first_conv.pkl"

            # Mini baseline correction for first conversation only
            checkpoint_correction_info = {}
            for probe_key in LAYER_RANGE_CONFIGS.keys():
                probe_data = sentence_data_by_range[probe_key]
                if probe_data:
                    scores_arr = np.array([d['emotion_scores'] for d in probe_data])
                    logits_arr = np.array([d['mean_logit'] for d in probe_data])
                    corr_info = compute_baseline_correction_alpha(scores_arr, logits_arr)
                    checkpoint_correction_info[probe_key] = corr_info

                    # Apply correction to checkpoint data
                    for d in probe_data:
                        norm_logit = (d['mean_logit'] - corr_info['logit_mean']) / (corr_info['logit_std'] + 1e-8)
                        d['emotion_scores_corrected'] = d['emotion_scores'] - corr_info['alpha'] * norm_logit

                    print(f"    {probe_key}: alpha={corr_info['alpha']:.4f}, corr_before={corr_info['correlation_before']:.4f}, corr_after={corr_info['correlation_after']:.4f}")

            # Save checkpoint
            checkpoint_data = {
                'sentence_data_by_range': {k: v.copy() for k, v in sentence_data_by_range.items()},
                'correction_info': checkpoint_correction_info,
                'conv_0': data['conversations'][0]
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            print(f"  ✓ Checkpoint saved to {checkpoint_path}")
            print("  === END CHECKPOINT ===\n")

    pass1_time = time.time() - overall_start
    print(f"\n  ✓ Pass 1 completed in {pass1_time:.2f}s ({pass1_time/60:.1f} minutes)")
    print(f"    Average time per conversation: {pass1_time/num_to_process:.1f}s")

    # PASS 2 & 3: Baseline correction with per-layer alphas
    print("\n[6/6] PASS 2: Computing baseline correction...")

    correction_info_by_range = {}
    per_layer_correction_info_by_range = {}  # NEW: Per-layer alphas

    for probe_key in LAYER_RANGE_CONFIGS.keys():
        sentence_data = sentence_data_by_range[probe_key]
        if not sentence_data:
            continue

        # Aggregated alpha (for sentence-level scores)
        all_scores = np.array([d['emotion_scores'] for d in sentence_data])
        all_logits = np.array([d['mean_logit'] for d in sentence_data])

        correction_info = compute_baseline_correction_alpha(all_scores, all_logits)
        correction_info_by_range[probe_key] = correction_info

        print(f"  {probe_key}:")
        print(f"    Aggregated Alpha: {correction_info['alpha']:.4f}")
        print(f"    Correlation before: {correction_info['correlation_before']:.4f}")
        print(f"    Correlation after: {correction_info['correlation_after']:.4f}")

        # NEW: Per-layer alphas (for layerwise plots)
        if STORE_PER_LAYER:
            layers = LAYER_RANGE_CONFIGS[probe_key]['layers']
            per_layer_alphas = {}

            for layer in layers:
                # Collect all scores and logits for this specific layer
                layer_scores_list = []
                layer_logits_list = []

                for data_item in sentence_data:
                    if (data_item['per_layer_scores'] and layer in data_item['per_layer_scores'] and
                        data_item['per_layer_mean_logits'] and layer in data_item['per_layer_mean_logits']):
                        layer_scores_list.append(data_item['per_layer_scores'][layer])
                        layer_logits_list.append(data_item['per_layer_mean_logits'][layer])

                if layer_scores_list:
                    layer_all_scores = np.array(layer_scores_list)
                    layer_all_logits = np.array(layer_logits_list)

                    # Compute alpha for this specific layer
                    layer_correction_info = compute_baseline_correction_alpha(layer_all_scores, layer_all_logits)
                    per_layer_alphas[layer] = layer_correction_info

            per_layer_correction_info_by_range[probe_key] = per_layer_alphas

            # Print per-layer alpha statistics
            if per_layer_alphas:
                alphas_list = [info['alpha'] for info in per_layer_alphas.values()]
                print(f"    Per-layer alphas: min={min(alphas_list):.4f}, max={max(alphas_list):.4f}, mean={np.mean(alphas_list):.4f}")

    # Apply correction
    print("\n  Applying baseline correction...")
    for probe_key in LAYER_RANGE_CONFIGS.keys():
        if probe_key not in correction_info_by_range:
            continue

        correction_info = correction_info_by_range[probe_key]
        alpha = correction_info['alpha']
        logit_mean = correction_info['logit_mean']
        logit_std = correction_info['logit_std']

        for data_item in sentence_data_by_range[probe_key]:
            normalized_logit = (data_item['mean_logit'] - logit_mean) / (logit_std + 1e-8)
            data_item['emotion_scores'] = data_item['emotion_scores'] - alpha * normalized_logit

            # Also correct per-layer scores (using per-layer alphas and mean logits)
            if STORE_PER_LAYER and data_item['per_layer_scores'] and data_item['per_layer_mean_logits']:
                # Use per-layer alphas if available, otherwise fall back to aggregated alpha
                per_layer_alphas = per_layer_correction_info_by_range.get(probe_key, {})

                for layer, scores in data_item['per_layer_scores'].items():
                    # Use layer-specific correction info if available
                    if layer in per_layer_alphas:
                        layer_correction = per_layer_alphas[layer]
                        layer_alpha = layer_correction['alpha']
                        layer_logit_mean = layer_correction['logit_mean']
                        layer_logit_std = layer_correction['logit_std']
                    else:
                        # Fallback to aggregated correction (shouldn't happen in normal operation)
                        layer_alpha = alpha
                        layer_logit_mean = logit_mean
                        layer_logit_std = logit_std

                    # Use layer-specific mean logit for this sentence
                    layer_mean_logit = data_item['per_layer_mean_logits'].get(layer, data_item['mean_logit'])
                    normalized_layer_logit = (layer_mean_logit - layer_logit_mean) / (layer_logit_std + 1e-8)
                    data_item['per_layer_scores'][layer] = scores - layer_alpha * normalized_layer_logit

    # SANITY CHECK: Verify z-scores are in reasonable range before saving
    print("\n  [SANITY CHECK] Verifying z-score ranges...")
    for probe_key in LAYER_RANGE_CONFIGS.keys():
        probe_data = sentence_data_by_range[probe_key]
        if not probe_data:
            continue

        all_scores = np.concatenate([d['emotion_scores'] for d in probe_data])
        score_min, score_max = all_scores.min(), all_scores.max()
        score_mean, score_std = all_scores.mean(), all_scores.std()

        print(f"    {probe_key}: min={score_min:.2f}, max={score_max:.2f}, mean={score_mean:.2f}, std={score_std:.2f}")

        if abs(score_min) > 20 or abs(score_max) > 20:
            print(f"    ⚠️  WARNING: Z-scores outside expected range [-20, 20]!")
        if abs(score_mean) > 5:
            print(f"    ⚠️  WARNING: Mean z-score far from 0!")
        if score_std < 0.1 or score_std > 10:
            print(f"    ⚠️  WARNING: Z-score std unusual (expected ~1-3)!")

        # Also check per-layer scores
        if STORE_PER_LAYER:
            all_layer_scores = []
            for d in probe_data:
                if d['per_layer_scores']:
                    for layer, scores in d['per_layer_scores'].items():
                        all_layer_scores.extend(scores.tolist() if hasattr(scores, 'tolist') else scores)
            if all_layer_scores:
                layer_arr = np.array(all_layer_scores)
                layer_min, layer_max = layer_arr.min(), layer_arr.max()
                print(f"      Per-layer: min={layer_min:.2f}, max={layer_max:.2f}")
                if abs(layer_min) > 50 or abs(layer_max) > 50:
                    print(f"      ❌ CRITICAL: Per-layer z-scores OUTSIDE EXPECTED RANGE! Check optimization!")
                    raise ValueError(f"Per-layer z-scores out of range: min={layer_min:.2f}, max={layer_max:.2f}")

    # Save to conversations
    print("\n  Storing results in conversations...")
    for probe_key in LAYER_RANGE_CONFIGS.keys():
        data['probe_configs'][probe_key] = {'type': probe_key}

        for data_item in sentence_data_by_range[probe_key]:
            conv_idx = data_item['conv_idx']
            sent_id = data_item['sent_id']
            conv = data['conversations'][conv_idx]

            if 'probe_scores' not in conv:
                conv['probe_scores'] = {}
            if probe_key not in conv['probe_scores']:
                conv['probe_scores'][probe_key] = {}

            conv['probe_scores'][probe_key][sent_id] = data_item['emotion_scores'].tolist()

            # Store per-layer scores
            if STORE_PER_LAYER and data_item['per_layer_scores']:
                per_layer_key = f"{probe_key}_by_layer"
                if per_layer_key not in conv:
                    conv[per_layer_key] = {}
                if sent_id not in conv[per_layer_key]:
                    conv[per_layer_key][sent_id] = {}

                for layer, scores in data_item['per_layer_scores'].items():
                    conv[per_layer_key][sent_id][layer] = scores.tolist()

    # Save
    print(f"\n  Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved ({file_size_mb:.1f} MB)")

    total_time = time.time() - script_start
    print("\n" + "="*80)
    print("✓ BATCHED PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Processed {num_to_process} conversations with {len(actual_layers)} layers")
    print(f"Average: {total_time/num_to_process:.1f}s per conversation")


if __name__ == '__main__':
    main()
