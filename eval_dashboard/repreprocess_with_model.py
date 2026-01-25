#!/usr/bin/env python3
"""
Re-preprocess existing pickle data with a different model.

Takes an existing preprocessed pickle (with conversations and sentences) and
re-runs all probe and logit lens computations using a specified model.

Usage:
    python repreprocess_with_model.py \
        --input data/high_emotion_6plus.pkl \
        --output data/high_emotion_6plus_model_X.pkl \
        --model annasoli/gemma3-27b-dpo-calm-full-merged

    # Test mode (first 2 conversations)
    python repreprocess_with_model.py \
        --input data/high_emotion_6plus.pkl \
        --output test.pkl \
        --model annasoli/gemma3-27b-dpo-calm-full-merged \
        --test-only 2
"""

import sys
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import time

# Add paths
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')
sys.path.insert(0, '/workspace-vast/annas/git/believe-it-or-not')

import torch

# Probe imports
from probes.scripts.token_level_helpers import TokenLevelExperiment
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader
from probes.scripts.probe_pipeline import normalize_probe_scores_zscore

# Logit lens imports
from emotion_evals.emo_lens.token_trajectories import extract_token_level_activations
from emotion_evals.emo_lens.model_utils import load_base_model, StandardizedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from emotion_evals.emo_lens.logit_lens_emotion_direct import (
    load_reference_stats_for_strategy,
    load_emotion_token_ids_from_json,
)
import random

# Local imports
from probe_configs import PROBE_CONFIGS, BASELINE_CONFIG, MODEL_CONFIG, EMOTIONS
from sentence_aggregator import aggregate_scores_to_sentences
from data_preprocessing import (
    apply_probes_to_activations,
    aggregate_per_layer_scores_to_sentences
)

# =============================================================================
# Configuration
# =============================================================================

# Probe types to process
PROBE_KEYS = ['orthogonal_raw', 'text_raw', 'text_cpca']  # centroid_k10 only has layers 20-40

# Logit lens - compute all layers once, extract ranges as needed
LOGIT_LENS_ALL_LAYERS = list(range(0, 62))

# Per-layer baseline paths
PER_LAYER_BASELINE_DIR = Path('/workspace-vast/annas/git/research-tools/data/baselines/probe_baselines')


# =============================================================================
# Baseline Correction (regress out random correlation)
# =============================================================================

def compute_baseline_correction_alpha(all_emotion_scores, all_mean_logits):
    """
    Compute optimal alpha to remove correlation between emotion scores and mean logits.

    This regresses out the "random correlation" - the correlation that exists
    because all logits tend to move together regardless of emotion content.

    Args:
        all_emotion_scores: Array of shape (n_sentences, 6)
        all_mean_logits: Array of shape (n_sentences,)

    Returns:
        Dict with alpha and diagnostic info
    """
    # Normalize mean logits
    logit_mean = all_mean_logits.mean()
    logit_std = all_mean_logits.std()

    if logit_std < 1e-8:
        print("  Warning: Mean logits have zero variance, skipping correction")
        return {
            'alpha': 0.0,
            'correlation_before': 0.0,
            'logit_mean': logit_mean,
            'logit_std': 1.0
        }

    normalized_logits = (all_mean_logits - logit_mean) / logit_std

    # Compute correlation for mean emotion score
    mean_emotion_scores = all_emotion_scores.mean(axis=1)
    correlation_before = np.corrcoef(mean_emotion_scores, all_mean_logits)[0, 1]

    # Compute optimal alpha for regression: y_corrected = y - alpha * x_normalized
    # Since x_normalized has std=1, alpha = corr * std(y)
    alpha = correlation_before * mean_emotion_scores.std()

    # Verify correction effectiveness
    corrected_scores = mean_emotion_scores - alpha * normalized_logits
    correlation_after = np.corrcoef(corrected_scores, all_mean_logits)[0, 1]

    return {
        'alpha': alpha,
        'correlation_before': correlation_before,
        'correlation_after': correlation_after,
        'logit_mean': logit_mean,
        'logit_std': logit_std
    }


def compute_mean_logit_for_activation(model, activation):
    """Compute mean logit value for an activation vector."""
    activation_tensor = torch.from_numpy(activation).to(
        device=next(model.parameters()).device,
        dtype=next(model.parameters()).dtype
    )

    with torch.no_grad():
        logits = model.project_on_vocab(activation_tensor)

    return float(logits.mean().cpu())


# =============================================================================
# Random Token Baseline (for logit lens normalization)
# =============================================================================

def compute_random_token_baseline(model, tokenizer, layer_range, n_tokens=500, vocab_size=256000):
    """
    Compute baseline logit values from random tokens for logit lens normalization.

    This measures the "random correlation" - even random tokens will show some
    correlation with emotion tokens in logit space. We subtract this baseline
    to get more meaningful emotion scores.

    Args:
        model: The language model (StandardizedTransformer wrapper)
        tokenizer: Tokenizer
        layer_range: List of layer indices to compute baseline for
        n_tokens: Number of random tokens to sample
        vocab_size: Size of vocabulary

    Returns:
        Dict with:
            - 'per_layer': Dict mapping layer -> mean logit value across random tokens
            - 'global_mean': Overall mean across all layers
            - 'n_tokens': Number of tokens sampled
            - 'vocab_size': Vocabulary size used
    """
    print(f"\n  Computing random token baseline (n={n_tokens}, layers={layer_range[0]}-{layer_range[-1]})...")

    # Randomly sample token IDs
    random_token_ids = random.sample(range(min(vocab_size, tokenizer.vocab_size)), n_tokens)

    device = next(model.parameters()).device

    # Collect logit values per layer
    layer_logit_values = {layer: [] for layer in layer_range}

    # Process in batches of 50 tokens at a time
    batch_size = 50
    for i in range(0, len(random_token_ids), batch_size):
        batch_token_ids = random_token_ids[i:i+batch_size]
        token_ids_tensor = torch.tensor(batch_token_ids, device=device).unsqueeze(1)  # (batch, 1)

        with torch.no_grad():
            # Forward pass to get hidden states
            outputs = model.model(input_ids=token_ids_tensor, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_size)

            # For each layer, project to vocab and collect mean logits
            for layer_idx in layer_range:
                layer_hidden = hidden_states[layer_idx][:, 0, :]  # (batch, hidden_size)
                logits = model.project_on_vocab(layer_hidden)  # (batch, vocab_size)

                # Get mean logit value for each token in batch
                mean_logits_per_token = logits.mean(dim=1)  # (batch,)
                layer_logit_values[layer_idx].extend(mean_logits_per_token.cpu().float().numpy().tolist())

    # Compute per-layer means
    per_layer_means = {}
    all_values = []
    for layer_idx in layer_range:
        layer_mean = np.mean(layer_logit_values[layer_idx])
        per_layer_means[layer_idx] = layer_mean
        all_values.extend(layer_logit_values[layer_idx])

    global_mean = np.mean(all_values)

    print(f"  Random baseline: global_mean={global_mean:.4f}")

    return {
        'per_layer': per_layer_means,
        'global_mean': global_mean,
        'n_tokens': n_tokens,
        'vocab_size': vocab_size
    }


# =============================================================================
# Logit Lens Functions
# =============================================================================

def apply_logit_lens_to_activations(
    activations_by_token: Dict[int, Dict[int, np.ndarray]],
    model,
    tokenizer,
    ref_stats: Dict,
    emotion_token_ids: Dict[str, int],
    layers: List[int],
    emotions: List[str]
) -> Dict[int, Dict[int, Dict[str, float]]]:
    """
    Apply logit lens to activations for specified layers using batched operations.

    Returns:
        {token_pos: {layer: {emotion: z_score}}}
    """
    from emotion_evals.emo_lens.logit_lens_emotion_direct import (
        project_activations_to_logits,
        normalize_logits_to_emotion_scores_batched
    )

    # Get sorted token positions for consistent batching
    token_positions = sorted(activations_by_token.keys())
    if not token_positions:
        return {}

    results = {pos: {} for pos in token_positions}

    # Process each layer with batched operations
    for layer in layers:
        # Collect all activations for this layer
        batch_activations = []
        batch_positions = []

        for pos in token_positions:
            if layer in activations_by_token[pos]:
                batch_activations.append(activations_by_token[pos][layer])
                batch_positions.append(pos)

        if not batch_activations:
            continue

        # Stack into batch tensor (match model dtype for projection)
        activations_np = np.stack(batch_activations, axis=0)  # [batch, hidden]
        model_dtype = next(model.parameters()).dtype
        activations_tensor = torch.from_numpy(activations_np).to(
            device=next(model.parameters()).device,
            dtype=model_dtype
        )

        # Project to logits (batched) - uses model's lm_head and layer norm
        with torch.no_grad():
            logits_batch = project_activations_to_logits(model, activations_tensor)

        # Normalize and score (batched)
        scores_list = normalize_logits_to_emotion_scores_batched(
            logits_batch=logits_batch,
            emotion_token_ids=emotion_token_ids,
            layer=layer,
            ref_stats=ref_stats,
            subtract_mean=True,  # z-score normalization
            aggregation="mean"
        )

        # Store results
        for pos, scores in zip(batch_positions, scores_list):
            results[pos][layer] = scores

    return results


def aggregate_logit_lens_to_sentences(
    sentences: List,
    logit_scores: Dict[int, Dict[int, Dict[str, float]]],
    layers: List[int],
    emotions: List[str],
    aggregation: str = 'mean',
    activations_by_token: Dict = None,
    model = None
) -> tuple:
    """
    Aggregate token-level logit lens scores to sentence level.

    Returns:
        (aggregated_scores, per_layer_scores, mean_logits)
        mean_logits is a dict mapping sent_id -> mean logit value (for baseline correction)
    """
    aggregated = {}
    per_layer = {}
    mean_logits = {}

    for sent in sentences:
        sent_id = sent['sentence_id']
        start_tok = sent['start_token']
        end_tok = sent['end_token']

        # Collect scores for this sentence
        sentence_layer_scores = {layer: {emotion: [] for emotion in emotions} for layer in layers}

        for t in range(start_tok, end_tok):
            if t not in logit_scores:
                continue
            for layer in layers:
                if layer not in logit_scores[t]:
                    continue
                for emotion in emotions:
                    if emotion in logit_scores[t][layer]:
                        sentence_layer_scores[layer][emotion].append(logit_scores[t][layer][emotion])

        # Aggregate within sentence for each layer
        per_layer[sent_id] = {}
        layer_means = {emotion: [] for emotion in emotions}

        for layer in layers:
            layer_result = {}
            for emotion in emotions:
                scores = sentence_layer_scores[layer][emotion]
                if scores:
                    if aggregation == 'mean':
                        layer_result[emotion] = np.mean(scores)
                    elif aggregation == 'max':
                        layer_result[emotion] = np.max(scores)
                    layer_means[emotion].append(layer_result.get(emotion, 0))

            # Store as array for consistency with probes
            per_layer[sent_id][layer] = np.array([layer_result.get(e, 0) for e in emotions])

        # Aggregate across layers for final score
        aggregated[sent_id] = np.array([
            np.mean(layer_means[e]) if layer_means[e] else 0
            for e in emotions
        ])

        # Compute mean logit for this sentence (for baseline correction)
        if activations_by_token is not None and model is not None:
            sentence_mean_logits = []
            for t in range(start_tok, end_tok):
                if t in activations_by_token:
                    # Average activation across layers
                    layer_activations = [activations_by_token[t][l] for l in layers if l in activations_by_token[t]]
                    if layer_activations:
                        avg_activation = np.mean(layer_activations, axis=0)
                        mean_logit = compute_mean_logit_for_activation(model, avg_activation)
                        sentence_mean_logits.append(mean_logit)
            if sentence_mean_logits:
                mean_logits[sent_id] = np.mean(sentence_mean_logits)

    return aggregated, per_layer, mean_logits


# =============================================================================
# Main Processing
# =============================================================================

def process_conversation(
    conv: Dict,
    model,
    tokenizer,
    probe_experiments: Dict,
    probe_baselines: Dict,
    per_layer_baselines: Dict,
    ref_stats: Dict,
    emotion_token_ids: Dict[str, int],
    all_layers: List[int],
    emotions: List[str],
    device: str,
    model_name: str = None
) -> Dict:
    """Process a single conversation with all probes and logit lens."""

    # Build conversation text
    conversation_text = ""
    for turn in conv['conversation']:
        role = turn['role']
        content = turn['content']
        if role == 'user':
            conversation_text += f"User: {content}\n"
        else:
            conversation_text += f"Assistant: {content}\n"

    sentences = conv['sentences']

    # Extract activations for all needed layers
    activations_by_token, token_ids = extract_token_level_activations(
        model=model,
        tokenizer=tokenizer,
        prompt=conversation_text,
        layers=all_layers,
        start_token_idx=0,
        system_prompt=None,
        num_generated_tokens=0
    )

    # Initialize result
    result = {
        'sample_id': conv['sample_id'],
        'conversation': conv['conversation'],
        'rating': conv.get('rating'),
        'sentences': sentences,
        'probe_scores': {},
        'metadata': conv.get('metadata', {})
    }
    result['metadata']['model'] = model_name

    # Process each probe type
    probe_layers = MODEL_CONFIG.get('layers', list(range(20, 41)))

    for probe_key in PROBE_KEYS:
        if probe_key not in probe_experiments:
            continue

        probe_exp = probe_experiments[probe_key]

        # Apply probes - get aggregated, per-layer raw scores, and per-layer softmax
        token_scores, per_layer_token_scores, per_layer_softmax = apply_probes_to_activations(
            activations_by_token=activations_by_token,
            probe_experiment=probe_exp,
            layers=probe_layers,
            store_per_layer=True
        )

        if not token_scores:
            continue

        # Normalize aggregated scores
        if probe_key in probe_baselines:
            probe_mean = probe_baselines[probe_key]['mean']
            probe_std = probe_baselines[probe_key]['std']
            for token_pos in token_scores:
                token_scores[token_pos] = normalize_probe_scores_zscore(
                    token_scores[token_pos], probe_mean, probe_std
                )

        # Normalize per-layer raw scores with per-layer baselines
        if probe_key in per_layer_baselines and per_layer_token_scores:
            layer_baselines = per_layer_baselines[probe_key]
            for token_pos in per_layer_token_scores:
                for layer in per_layer_token_scores[token_pos]:
                    if layer in layer_baselines:
                        mean = np.array(layer_baselines[layer]['mean'])
                        std = np.array(layer_baselines[layer]['std'])
                        std_safe = np.maximum(std, 0.01)
                        score = per_layer_token_scores[token_pos][layer]
                        per_layer_token_scores[token_pos][layer] = np.clip(
                            (score - mean) / std_safe, -10, 10
                        )

        # Aggregate to sentences
        sentence_scores = aggregate_scores_to_sentences(sentences, token_scores)
        result['probe_scores'][probe_key] = sentence_scores

        # Aggregate per-layer raw scores to sentences (z-normalized)
        if per_layer_token_scores:
            per_layer_key = f'{probe_key}_by_layer'
            result[per_layer_key] = aggregate_per_layer_scores_to_sentences(
                sentences, per_layer_token_scores, probe_layers
            )

        # Aggregate per-layer softmax probabilities to sentences (no normalization needed)
        if per_layer_softmax:
            softmax_key = f'{probe_key}_softmax_by_layer'
            result[softmax_key] = aggregate_per_layer_scores_to_sentences(
                sentences, per_layer_softmax, probe_layers
            )

    # Process logit lens once for all layers (batched for efficiency)
    logit_scores = apply_logit_lens_to_activations(
        activations_by_token=activations_by_token,
        model=model,
        tokenizer=tokenizer,
        ref_stats=ref_stats,
        emotion_token_ids=emotion_token_ids,
        layers=LOGIT_LENS_ALL_LAYERS,
        emotions=emotions
    )

    # Store all-layer logit lens scores (can extract specific ranges later)
    # Also compute mean logits for baseline correction
    aggregated, per_layer, mean_logits = aggregate_logit_lens_to_sentences(
        sentences, logit_scores, LOGIT_LENS_ALL_LAYERS, emotions,
        activations_by_token=activations_by_token,
        model=model
    )

    result['probe_scores']['logit_lens'] = aggregated
    result['logit_lens_by_layer'] = per_layer
    result['sentence_mean_logits'] = mean_logits  # For baseline correction

    return result


def load_per_layer_baselines() -> Dict:
    """Load per-layer baselines for z-score normalization."""
    baselines = {}

    for probe_key in ['text_raw', 'text_cpca']:
        baseline_file = PER_LAYER_BASELINE_DIR / f'{probe_key}_baselines.json'
        if baseline_file.exists():
            import json
            with open(baseline_file, 'r') as f:
                data = json.load(f)
            baselines[probe_key] = {
                int(k): v for k, v in data['baselines'].items()
            }
            print(f"  Loaded per-layer baselines for {probe_key}: {len(baselines[probe_key])} layers")

    return baselines


def main():
    parser = argparse.ArgumentParser(description="Re-preprocess data with a different model")
    parser.add_argument("--input", required=True, help="Input pickle file")
    parser.add_argument("--output", required=True, help="Output pickle file")
    parser.add_argument("--model", required=True, help="Model name or path (HuggingFace)")
    parser.add_argument("--test-only", type=int, help="Only process first N conversations")
    args = parser.parse_args()

    start_time = time.time()

    print("=" * 80)
    print("RE-PREPROCESS WITH MODEL")
    print("=" * 80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Model:  {args.model}")

    # Load input data
    print("\n[1/6] Loading input data...")
    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    conversations = data['conversations']
    if args.test_only:
        conversations = conversations[:args.test_only]
        print(f"  TEST MODE: Processing only {len(conversations)} conversations")
    else:
        print(f"  Loaded {len(conversations)} conversations")

    # Load model
    print(f"\n[2/6] Loading model: {args.model}")
    model, tokenizer = load_base_model(args.model)
    device = next(model.parameters()).device
    print(f"  Model loaded on {device}")

    # Determine number of layers
    try:
        num_layers = len(model.model.layers)
        all_layers = list(range(num_layers))
        print(f"  Detected {num_layers} layers")
    except:
        all_layers = list(range(62))
        print(f"  Using default 62 layers")

    # Load probe experiments
    print("\n[3/6] Loading probe experiments...")
    probe_experiments = {}
    for probe_key in PROBE_KEYS:
        if probe_key in PROBE_CONFIGS:
            config = PROBE_CONFIGS[probe_key]
            try:
                probe_exp = TokenLevelExperiment(
                    model=None,  # Not needed for cached activations
                    tokenizer=tokenizer,
                    probe_type=config['type'],
                    probe_dir=config.get('probe_dir'),
                    cpca_path=config.get('cpca_path'),
                    probe_pattern=config.get('probe_pattern'),
                    orthogonality_weight=config.get('orthogonality_weight', 1000.0),
                    orthogonal_representation=config.get('orthogonal_representation', 'raw'),
                    n_components=config.get('n_components', 10),
                    seed=config.get('seed', 0),
                    k_value=config.get('k_value'),
                )
                probe_experiments[probe_key] = probe_exp
                print(f"  ✓ Loaded {probe_key}")
            except Exception as e:
                print(f"  ✗ Failed to load {probe_key}: {e}")

    # Load probe baselines (layer-averaged) using probe experiments
    print("\n[4/6] Computing baselines...")
    baseline_loader = WildChatBaselineLoader(BASELINE_CONFIG['baseline_dir'])
    probe_baselines = {}
    for probe_key in PROBE_KEYS:
        if probe_key in PROBE_CONFIGS and probe_key in probe_experiments:
            config = PROBE_CONFIGS[probe_key]
            try:
                baseline_stats = baseline_loader.compute_probe_score_baselines(
                    probe_inference=probe_experiments[probe_key].inference,
                    layers=MODEL_CONFIG.get('layers', list(range(20, 41))),
                    probe_type=config['type'],
                    aggregation="mean",
                    return_std=True,
                    orthogonality_weight=config.get('orthogonality_weight', 1000.0),
                    orthogonal_representation=config.get('orthogonal_representation', 'raw'),
                    n_components=config.get('n_components', 10),
                    seed=config.get('seed', 0),
                    emotions=EMOTIONS,
                    k_value=config.get('k_value')
                )
                probe_baselines[probe_key] = {
                    'mean': baseline_stats['mean'][-1],  # Averaged across layers
                    'std': baseline_stats['std'][-1]
                }
                print(f"  ✓ Computed baseline for {probe_key}")
            except Exception as e:
                print(f"  ✗ Failed to compute baseline for {probe_key}: {e}")

    # Load per-layer baselines
    per_layer_baselines = load_per_layer_baselines()

    # Load logit lens reference stats
    print("\n[5/7] Loading logit lens reference stats...")
    # Use the base model for reference stats (probes were trained on base model)
    BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
    SCRIPT_DIR = Path('/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens')
    ref_stats = load_reference_stats_for_strategy(
        model_name=BASE_MODEL_NAME,
        activation_strategy='layer_specific',
        script_dir=SCRIPT_DIR
    )

    # Load emotion token IDs using the base model name (tokenizer is same across finetuned models)
    emotion_token_ids = load_emotion_token_ids_from_json(BASE_MODEL_NAME)
    print(f"  Loaded reference stats and emotion token IDs")

    # Compute random token baseline for logit lens normalization
    print("\n[6/7] Computing random token baseline...")
    logit_lens_layers = list(range(40, 51))  # Layers typically used for logit lens
    random_baseline = compute_random_token_baseline(
        model=model,
        tokenizer=tokenizer,
        layer_range=logit_lens_layers,
        n_tokens=500
    )

    # Process conversations (Pass 1: compute scores and mean logits)
    print(f"\n[7/8] Processing {len(conversations)} conversations...")
    processed = []

    for i, conv in enumerate(tqdm(conversations, desc="Processing")):
        try:
            result = process_conversation(
                conv=conv,
                model=model,
                tokenizer=tokenizer,
                probe_experiments=probe_experiments,
                probe_baselines=probe_baselines,
                per_layer_baselines=per_layer_baselines,
                ref_stats=ref_stats,
                emotion_token_ids=emotion_token_ids,
                all_layers=all_layers,
                emotions=EMOTIONS,
                device=str(device),
                model_name=args.model
            )
            processed.append(result)
        except Exception as e:
            print(f"\n  ERROR processing conversation {i}: {e}")
            import traceback
            traceback.print_exc()

    # =========================================================================
    # Apply baseline correction to logit lens scores (regress out random correlation)
    # =========================================================================
    print("\n[8/8] Applying baseline correction to logit lens scores...")

    # Collect all emotion scores and mean logits
    all_emotion_scores = []
    all_mean_logits = []
    sentence_indices = []  # Track (conv_idx, sent_id) for applying correction

    for conv_idx, conv in enumerate(processed):
        if 'probe_scores' not in conv or 'logit_lens' not in conv['probe_scores']:
            continue
        if 'sentence_mean_logits' not in conv:
            continue

        for sent_id, scores in conv['probe_scores']['logit_lens'].items():
            if sent_id in conv['sentence_mean_logits']:
                all_emotion_scores.append(np.array(scores))
                all_mean_logits.append(conv['sentence_mean_logits'][sent_id])
                sentence_indices.append((conv_idx, sent_id))

    if len(all_emotion_scores) > 10:  # Need enough data points
        all_emotion_scores = np.array(all_emotion_scores)
        all_mean_logits = np.array(all_mean_logits)

        # Compute optimal alpha
        correction_info = compute_baseline_correction_alpha(all_emotion_scores, all_mean_logits)

        print(f"  Baseline Correction Results:")
        print(f"  {'='*60}")
        print(f"    Sentences:            {len(all_emotion_scores)}")
        print(f"    Correlation (before): {correction_info['correlation_before']:+.4f}")
        print(f"    Optimal alpha:        {correction_info['alpha']:+.4f}")
        print(f"    Correlation (after):  {correction_info['correlation_after']:+.4f}")
        print(f"  {'='*60}")

        # Apply correction to all logit lens scores
        for (conv_idx, sent_id), mean_logit in zip(sentence_indices, all_mean_logits):
            conv = processed[conv_idx]
            normalized_logit = (mean_logit - correction_info['logit_mean']) / correction_info['logit_std']

            # Correct aggregated scores
            scores = np.array(conv['probe_scores']['logit_lens'][sent_id])
            corrected = scores - correction_info['alpha'] * normalized_logit
            conv['probe_scores']['logit_lens'][sent_id] = corrected.tolist()

            # Correct per-layer scores
            if 'logit_lens_by_layer' in conv and sent_id in conv['logit_lens_by_layer']:
                for layer in conv['logit_lens_by_layer'][sent_id]:
                    layer_scores = np.array(conv['logit_lens_by_layer'][sent_id][layer])
                    corrected_layer = layer_scores - correction_info['alpha'] * normalized_logit
                    conv['logit_lens_by_layer'][sent_id][layer] = corrected_layer.tolist()

        print(f"  ✓ Applied correction to {len(sentence_indices)} sentences")
    else:
        print(f"  ⚠ Not enough data for baseline correction ({len(all_emotion_scores)} sentences)")
        correction_info = None

    # Save results
    print(f"\nSaving to {args.output}...")

    # Prepare probe_baselines dict (for z-score normalization)
    # Add logit_lens with identity transform (already z-scored by ref_stats)
    output_probe_baselines = dict(probe_baselines)
    output_probe_baselines['logit_lens_mean'] = {'mean': np.zeros(6), 'std': np.ones(6)}
    output_probe_baselines['logit_lens_max'] = {'mean': np.zeros(6), 'std': np.ones(6)}
    output_probe_baselines['random_token_baseline'] = random_baseline
    if correction_info:
        output_probe_baselines['baseline_correction'] = correction_info

    output_data = {
        'conversations': processed,
        'probe_baselines': output_probe_baselines,
        'metadata': {
            'model': args.model,
            'source_file': str(args.input),
            'num_conversations': len(processed),
            'baseline_correction_applied': correction_info is not None,
        }
    }

    with open(args.output, 'wb') as f:
        pickle.dump(output_data, f)

    file_size = Path(args.output).stat().st_size / (1024 * 1024)
    elapsed = time.time() - start_time

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"Processed: {len(processed)} conversations")
    print(f"Output: {args.output} ({file_size:.1f} MB)")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
