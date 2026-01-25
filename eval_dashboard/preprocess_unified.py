"""
Unified Preprocessing Pipeline

Single-pass preprocessing that loads the model ONCE and computes:
- All probe types (with per-layer scores using layer-averaged baseline)
- Logit lens (with per-layer baseline + baseline correction)
- Axis lens (with baseline correction)

This replaces the multi-script workflow:
  data_preprocessing.py → add_logit_lens_to_existing.py → add_axis_lens_to_existing.py

Usage:
    python preprocess_unified.py --input data.jsonl --output preprocessed.pkl
    python preprocess_unified.py --input data.jsonl --output preprocessed.pkl --model path/to/finetuned
    python preprocess_unified.py --input data.jsonl --output preprocessed.pkl --test-only 5
"""

import sys
import pickle
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add paths
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')
sys.path.insert(0, '/workspace-vast/annas/git/believe-it-or-not')

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Probe imports
from probes.scripts.token_level_helpers import TokenLevelExperiment
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader
from probes.scripts.probe_pipeline import normalize_probe_scores_zscore

# Logit lens imports
from emotion_evals.emo_lens.token_trajectories import (
    extract_token_level_activations,
    compute_layer_averaged_baseline_stats
)
from emotion_evals.emo_lens.model_utils import load_base_model
from emotion_evals.emo_lens.logit_lens_emotion_direct import (
    load_reference_stats_for_strategy,
    load_emotion_token_ids_from_json,
)

# Local imports
from probe_configs import PROBE_CONFIGS, BASELINE_CONFIG, MODEL_CONFIG, EMOTIONS
from sentence_aggregator import (
    split_conversation_into_sentences,
    split_conversation_into_sentences_simple,
    aggregate_scores_to_sentences,
    SentenceInfo
)
from data_preprocessing import (
    apply_probes_to_activations,
    aggregate_per_layer_scores_to_sentences
)


# =============================================================================
# Configuration
# =============================================================================

# Default probe types to include
DEFAULT_PROBE_KEYS = ['orthogonal_raw', 'text_raw', 'centroid_k10']

# Logit lens layer range configurations
LOGIT_LENS_CONFIGS = {
    'logit_lens_mean': {
        'layers': list(range(40, 51)),
        'display_name': 'Logit Lens L40-50'
    },
    'logit_lens_mean_l30_40': {
        'layers': list(range(30, 41)),
        'display_name': 'Logit Lens L30-40'
    },
    'logit_lens_mean_l20_30': {
        'layers': list(range(20, 31)),
        'display_name': 'Logit Lens L20-30'
    }
}


# =============================================================================
# Helper Functions
# =============================================================================

def load_emotion_onset_data(data_path: str) -> List[Dict]:
    """Load annotated emotion onset data from JSON/JSONL."""
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, 'r') as f:
        if data_path.suffix == '.jsonl':
            samples = [json.loads(line) for line in f]
        else:
            data = json.load(f)
            if isinstance(data, list):
                samples = data
            elif 'samples' in data:
                samples = data['samples']
            else:
                samples = [data]

    return samples


def build_conversation_text(conversation: List[Dict[str, str]]) -> str:
    """Build formatted conversation text for tokenization."""
    text = ""
    for turn in conversation:
        role = turn['role']
        content = turn['content']
        if role == 'user':
            text += f"User: {content}\n"
        else:
            text += f"Assistant: {content}\n"
    return text


def compute_emotion_scores_for_single_layer(
    model,
    activation: np.ndarray,
    emotion_token_ids: Dict[str, List[int]],
    baseline_stats_for_layer: Dict,
    aggregation: str = "mean"
) -> Dict[str, float]:
    """Compute emotion scores for a single layer's activation using logit lens."""
    device = next(model.parameters()).device

    with torch.no_grad():
        activation_tensor = torch.from_numpy(activation).to(device)
        if hasattr(model, 'dtype'):
            activation_tensor = activation_tensor.to(model.dtype)
        elif hasattr(model.lm_head, 'weight'):
            activation_tensor = activation_tensor.to(model.lm_head.weight.dtype)

        logits = model.project_on_vocab(activation_tensor.unsqueeze(0))
        logits_np = logits.cpu().float().numpy()[0]

    scores = {}
    for emotion in EMOTIONS:
        token_ids = emotion_token_ids[emotion]
        emotion_logits = logits_np[token_ids]

        normalized_scores = []
        for token_id, logit in zip(token_ids, emotion_logits):
            if token_id in baseline_stats_for_layer.get(emotion, {}):
                stats = baseline_stats_for_layer[emotion][token_id]
                z_score = (logit - stats['mean']) / (stats['std'] + 1e-8)
                normalized_scores.append(z_score)

        if normalized_scores:
            if aggregation == "mean":
                scores[emotion] = float(np.mean(normalized_scores))
            elif aggregation == "max":
                scores[emotion] = float(np.max(normalized_scores))
            else:
                scores[emotion] = float(np.mean(normalized_scores))
        else:
            scores[emotion] = 0.0

    return scores


def compute_mean_logit_for_activation(model, activation: np.ndarray) -> float:
    """Compute mean logit across vocabulary for baseline correction."""
    device = next(model.parameters()).device

    with torch.no_grad():
        activation_tensor = torch.from_numpy(activation).to(device)
        if hasattr(model, 'dtype'):
            activation_tensor = activation_tensor.to(model.dtype)
        elif hasattr(model.lm_head, 'weight'):
            activation_tensor = activation_tensor.to(model.lm_head.weight.dtype)

        logits = model.project_on_vocab(activation_tensor.unsqueeze(0))
        mean_logit = logits.mean().cpu().float().item()

    return mean_logit


def compute_baseline_correction_alpha(
    all_emotion_scores: np.ndarray,
    all_mean_logits: np.ndarray
) -> Dict:
    """Compute optimal alpha to remove correlation between emotion scores and mean logits."""
    logit_mean = all_mean_logits.mean()
    logit_std = all_mean_logits.std()

    if logit_std < 1e-8:
        return {
            'alpha': 0.0,
            'correlation_before': 0.0,
            'correlation_after': 0.0,
            'logit_mean': logit_mean,
            'logit_std': 1.0
        }

    normalized_logits = (all_mean_logits - logit_mean) / logit_std
    mean_emotion_scores = all_emotion_scores.mean(axis=1)
    correlation_before = np.corrcoef(mean_emotion_scores, all_mean_logits)[0, 1]

    alpha = correlation_before * mean_emotion_scores.std() / logit_std

    corrected_scores = mean_emotion_scores - alpha * normalized_logits
    correlation_after = np.corrcoef(corrected_scores, all_mean_logits)[0, 1]

    return {
        'alpha': alpha,
        'correlation_before': correlation_before,
        'correlation_after': correlation_after,
        'logit_mean': logit_mean,
        'logit_std': logit_std
    }


# =============================================================================
# Main Pipeline
# =============================================================================

def preprocess_unified(
    data_path: str,
    output_path: str,
    model_name: str = None,
    probe_keys: List[str] = None,
    include_logit_lens: bool = True,
    include_axis_lens: bool = False,  # TODO: implement if needed
    test_only: int = None,
    use_simple_splitter: bool = False
):
    """
    Unified preprocessing pipeline - loads model ONCE, computes everything.

    Args:
        data_path: Path to emotion onset data (JSON/JSONL)
        output_path: Where to save preprocessed pickle
        model_name: Model to use (default: gemma-3-27b-it). Can be path to finetuned model.
        probe_keys: Which probe types to apply (default: orthogonal_raw, text_raw, centroid_k10)
        include_logit_lens: Whether to compute logit lens scores
        include_axis_lens: Whether to compute axis lens scores (not yet implemented)
        test_only: If set, only process this many conversations
        use_simple_splitter: Use simple sentence splitter instead of Sentences library
    """
    print("=" * 80)
    print("UNIFIED PREPROCESSING PIPELINE")
    print("Single-pass: probes + logit lens (+ axis lens)")
    print("=" * 80)

    # Set defaults
    if model_name is None:
        model_name = MODEL_CONFIG['model_name']
    if probe_keys is None:
        probe_keys = DEFAULT_PROBE_KEYS

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Probes: {probe_keys}")
    print(f"  Logit lens: {include_logit_lens}")
    print(f"  Test mode: {f'{test_only} samples' if test_only else 'OFF'}")

    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[1/7] Loading data...")
    samples = load_emotion_onset_data(data_path)
    num_samples = test_only if test_only else len(samples)
    print(f"  Loaded {len(samples)} samples, will process {num_samples}")

    # =========================================================================
    # Step 2: Load model and tokenizer (ONCE!)
    # =========================================================================
    print("\n[2/7] Loading model and tokenizer...")

    # Use emo_lens loader for consistency (handles project_on_vocab method)
    model, tokenizer = load_base_model(model_name)

    # Detect number of layers
    try:
        num_model_layers = len(model.model.layers)
        print(f"  Detected {num_model_layers} layers")
    except:
        num_model_layers = 42  # Fallback for Gemma

    print(f"  ✓ Model loaded: {model_name}")

    # =========================================================================
    # Step 3: Initialize probe baselines and experiments
    # =========================================================================
    print("\n[3/7] Initializing probes...")

    baseline_loader = WildChatBaselineLoader(
        aggregation_type=BASELINE_CONFIG['aggregation_type'],
        baseline_dir=BASELINE_CONFIG['baseline_dir']
    )

    probe_baselines = {}
    probe_experiments = {}

    for probe_key in probe_keys:
        probe_config = PROBE_CONFIGS[probe_key]

        # Create experiment
        exp = TokenLevelExperiment(
            model=None,
            tokenizer=tokenizer,
            probe_type=probe_config['type'],
            probe_dir=probe_config.get('probe_dir'),
            cpca_path=probe_config.get('cpca_path'),
            probe_pattern=probe_config.get('probe_pattern'),
            orthogonality_weight=probe_config.get('orthogonality_weight', 1000.0),
            orthogonal_representation=probe_config.get('orthogonal_representation', 'raw'),
            n_components=probe_config.get('n_components', 10),
            seed=probe_config.get('seed', 0),
            k_value=probe_config.get('k_value'),
            centroid_probe_format=probe_config.get('centroid_probe_format', 'auto'),
            baseline_dir=BASELINE_CONFIG['baseline_dir'],
            emotions=EMOTIONS
        )

        # Compute baseline stats (layer-averaged)
        baseline_stats = baseline_loader.compute_probe_score_baselines(
            probe_inference=exp.inference,
            layers=MODEL_CONFIG['layers'],
            probe_type=probe_config['type'],
            aggregation="mean",
            return_std=True,
            orthogonality_weight=probe_config.get('orthogonality_weight', 1000.0),
            orthogonal_representation=probe_config.get('orthogonal_representation', 'raw'),
            n_components=probe_config.get('n_components', 10),
            seed=probe_config.get('seed', 0),
            emotions=EMOTIONS,
            k_value=probe_config.get('k_value')
        )

        probe_baselines[probe_key] = {
            'mean': baseline_stats['mean'][-1],
            'std': baseline_stats['std'][-1]
        }
        probe_experiments[probe_key] = exp
        print(f"  ✓ {probe_key}")

    # =========================================================================
    # Step 4: Initialize logit lens (if enabled)
    # =========================================================================
    logit_lens_baseline_stats = {}
    emotion_token_ids = None
    ref_stats = None

    if include_logit_lens:
        print("\n[4/7] Initializing logit lens...")

        # Load emotion token IDs
        # Use the base model name for token IDs (they're vocab-specific)
        base_model_for_tokens = "unsloth/gemma-3-27b-it"
        emotion_token_ids = load_emotion_token_ids_from_json(base_model_for_tokens)
        print(f"  Loaded emotion token IDs for {len(emotion_token_ids)} emotions")

        # Load reference statistics
        SCRIPT_DIR = Path("/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens")
        ref_stats = load_reference_stats_for_strategy(
            model_name=base_model_for_tokens,
            activation_strategy="generated_tokens_avg",
            script_dir=SCRIPT_DIR
        )
        print("  ✓ Loaded baseline stats")

        # Compute layer-averaged baseline stats for each range
        for ll_key, ll_config in LOGIT_LENS_CONFIGS.items():
            layers = ll_config['layers']
            logit_lens_baseline_stats[ll_key] = compute_layer_averaged_baseline_stats(
                ref_stats=ref_stats,
                layers=layers,
                emotion_token_ids=emotion_token_ids
            )
            print(f"  ✓ {ll_key}: layers {layers[0]}-{layers[-1]}")
    else:
        print("\n[4/7] Skipping logit lens (disabled)")

    # =========================================================================
    # Step 5: Determine all layers needed
    # =========================================================================
    print("\n[5/7] Computing layer requirements...")

    all_layers_needed = set(MODEL_CONFIG['layers'])  # Probe layers

    if include_logit_lens:
        for ll_config in LOGIT_LENS_CONFIGS.values():
            all_layers_needed.update(ll_config['layers'])

    # Filter to actual model layers
    all_layers_needed = sorted([l for l in all_layers_needed if l < num_model_layers])
    print(f"  Will extract layers: {all_layers_needed[0]}-{all_layers_needed[-1]} ({len(all_layers_needed)} total)")

    # =========================================================================
    # Step 6: Process conversations
    # =========================================================================
    print(f"\n[6/7] Processing {num_samples} conversations...")

    processed_conversations = []

    # For logit lens baseline correction (two-pass)
    logit_lens_sentence_data = {ll_key: [] for ll_key in LOGIT_LENS_CONFIGS.keys()}

    for sample_idx in tqdm(range(num_samples), desc="Processing"):
        sample = samples[sample_idx]
        conversation = sample['conversation']
        sample_id = sample.get('sample_id', sample_idx)
        rating = sample.get('rating', 0)

        # Build conversation text
        conversation_text = build_conversation_text(conversation)

        # ---------------------------------------------------------------------
        # Extract activations ONCE for all measurement types
        # ---------------------------------------------------------------------
        activations_by_token, token_ids = extract_token_level_activations(
            model=model,
            tokenizer=tokenizer,
            prompt=conversation_text,
            layers=all_layers_needed,
            start_token_idx=0,
            system_prompt=None,
            num_generated_tokens=0
        )

        # Decode tokens for sentence splitting
        token_strings = [tokenizer.decode([tid]) for tid in token_ids]

        # Split into sentences
        if use_simple_splitter:
            sentences = split_conversation_into_sentences_simple(
                conversation, token_strings, tokenizer
            )
        else:
            try:
                sentences = split_conversation_into_sentences(
                    conversation, token_strings, tokenizer
                )
            except ImportError:
                if sample_idx == 0:
                    print("  Warning: Sentences library not available, using simple splitter")
                sentences = split_conversation_into_sentences_simple(
                    conversation, token_strings, tokenizer
                )

        # ---------------------------------------------------------------------
        # Apply probes (with per-layer scores)
        # ---------------------------------------------------------------------
        probe_scores = {}
        per_layer_scores_by_probe = {}

        for probe_key in probe_keys:
            # Get both aggregated and per-layer scores
            token_scores, per_layer_token_scores = apply_probes_to_activations(
                activations_by_token=activations_by_token,
                probe_experiment=probe_experiments[probe_key],
                layers=MODEL_CONFIG['layers'],
                store_per_layer=True
            )

            # Normalize with layer-averaged baseline (SAME for all layers)
            probe_mean = probe_baselines[probe_key]['mean']
            probe_std = probe_baselines[probe_key]['std']

            for token_pos in token_scores:
                token_scores[token_pos] = normalize_probe_scores_zscore(
                    token_scores[token_pos], probe_mean, probe_std
                )

            if per_layer_token_scores:
                for token_pos in per_layer_token_scores:
                    for layer in per_layer_token_scores[token_pos]:
                        per_layer_token_scores[token_pos][layer] = normalize_probe_scores_zscore(
                            per_layer_token_scores[token_pos][layer], probe_mean, probe_std
                        )

            # Aggregate to sentence level
            sentence_scores = aggregate_scores_to_sentences(
                sentences=sentences,
                token_scores=token_scores,
                aggregation='mean'
            )
            probe_scores[probe_key] = sentence_scores

            if per_layer_token_scores:
                sentence_per_layer = aggregate_per_layer_scores_to_sentences(
                    sentences=sentences,
                    per_layer_scores=per_layer_token_scores,
                    layers=MODEL_CONFIG['layers'],
                    aggregation='mean'
                )
                per_layer_scores_by_probe[probe_key] = sentence_per_layer

        # ---------------------------------------------------------------------
        # Apply logit lens (if enabled)
        # ---------------------------------------------------------------------
        if include_logit_lens:
            for ll_key, ll_config in LOGIT_LENS_CONFIGS.items():
                layers = ll_config['layers']
                averaged_baseline = logit_lens_baseline_stats[ll_key]

                # Compute per-layer and aggregated scores
                token_emotion_scores = {}
                token_emotion_scores_by_layer = {}
                token_mean_logits = {}

                for token_pos in activations_by_token.keys():
                    # Per-layer scores
                    token_emotion_scores_by_layer[token_pos] = {}
                    for layer in layers:
                        if layer not in activations_by_token[token_pos]:
                            continue
                        activation = activations_by_token[token_pos][layer]

                        # Get per-layer baseline stats
                        layer_baseline_stats = {}
                        for emotion in EMOTIONS:
                            layer_baseline_stats[emotion] = {}
                            for token_id in emotion_token_ids[emotion]:
                                layer_key = str(layer)
                                if layer_key in ref_stats['layers_data']:
                                    if token_id in ref_stats['layers_data'][layer_key]['statistics']:
                                        layer_baseline_stats[emotion][token_id] = \
                                            ref_stats['layers_data'][layer_key]['statistics'][token_id]

                        scores = compute_emotion_scores_for_single_layer(
                            model=model,
                            activation=activation,
                            emotion_token_ids=emotion_token_ids,
                            baseline_stats_for_layer=layer_baseline_stats,
                            aggregation="mean"
                        )
                        token_emotion_scores_by_layer[token_pos][layer] = scores

                    # Aggregated scores (layer-averaged activation)
                    layer_activations = [activations_by_token[token_pos][l]
                                         for l in layers if l in activations_by_token[token_pos]]
                    if layer_activations:
                        avg_activation = np.mean(layer_activations, axis=0)
                        token_emotion_scores[token_pos] = compute_emotion_scores_for_single_layer(
                            model=model,
                            activation=avg_activation,
                            emotion_token_ids=emotion_token_ids,
                            baseline_stats_for_layer=averaged_baseline,
                            aggregation="mean"
                        )
                        token_mean_logits[token_pos] = compute_mean_logit_for_activation(
                            model, avg_activation
                        )

                # Aggregate to sentence level (store for baseline correction)
                for sent in sentences:
                    sent_id = sent.sentence_id if hasattr(sent, 'sentence_id') else sent['sentence_id']
                    start_tok = sent.start_token if hasattr(sent, 'start_token') else sent['start_token']
                    end_tok = sent.end_token if hasattr(sent, 'end_token') else sent['end_token']

                    sent_scores = []
                    sent_logits = []
                    per_layer_sent = {}

                    for t in range(start_tok, end_tok):
                        if t in token_emotion_scores:
                            scores_arr = np.array([token_emotion_scores[t][e] for e in EMOTIONS])
                            sent_scores.append(scores_arr)
                            if t in token_mean_logits:
                                sent_logits.append(token_mean_logits[t])

                        # Per-layer
                        for layer in layers:
                            if t in token_emotion_scores_by_layer and layer in token_emotion_scores_by_layer[t]:
                                if layer not in per_layer_sent:
                                    per_layer_sent[layer] = []
                                layer_scores = token_emotion_scores_by_layer[t][layer]
                                per_layer_sent[layer].append(
                                    np.array([layer_scores[e] for e in EMOTIONS])
                                )

                    if sent_scores:
                        emotion_scores = np.mean(sent_scores, axis=0)
                        mean_logit = np.mean(sent_logits) if sent_logits else 0.0

                        # Aggregate per-layer
                        per_layer_agg = {}
                        for layer, scores_list in per_layer_sent.items():
                            per_layer_agg[layer] = np.mean(scores_list, axis=0)

                        logit_lens_sentence_data[ll_key].append({
                            'conv_idx': sample_idx,
                            'sent_id': sent_id,
                            'emotion_scores': emotion_scores,
                            'mean_logit': mean_logit,
                            'per_layer_scores': per_layer_agg
                        })

        # ---------------------------------------------------------------------
        # Find onset sentence
        # ---------------------------------------------------------------------
        onset_sentence_id = None
        turn_number = sample.get('turn_number', 1)
        target_conv_turn = turn_number - 1

        for sent in sentences:
            sent_role = sent.turn_role if hasattr(sent, 'turn_role') else sent['turn_role']
            sent_turn = sent.turn_index if hasattr(sent, 'turn_index') else sent['turn_index']
            if sent_role == 'assistant' and sent_turn == target_conv_turn:
                onset_sentence_id = sent.sentence_id if hasattr(sent, 'sentence_id') else sent['sentence_id']
                break

        # Build conversation dict
        conv_dict = {
            'sample_id': sample_id,
            'conversation': conversation,
            'rating': rating,
            'sentences': [asdict(s) if hasattr(s, '__dataclass_fields__') else s for s in sentences],
            'probe_scores': probe_scores,
            'metadata': {
                'num_sentences': len(sentences),
                'num_tokens': len(token_strings),
                'num_turns': len(conversation),
                'onset_sentence_id': onset_sentence_id,
                'judge_evidence': sample.get('judge_evidence', ''),
                'judge_reasoning': sample.get('judge_reasoning', '')
            }
        }

        # Add per-layer probe scores
        for probe_key, per_layer_data in per_layer_scores_by_probe.items():
            conv_dict[f'{probe_key}_by_layer'] = per_layer_data

        processed_conversations.append(conv_dict)

    # =========================================================================
    # Step 6b: Apply logit lens baseline correction (two-pass)
    # =========================================================================
    logit_lens_correction_info = {}

    if include_logit_lens:
        print("\n  Applying logit lens baseline correction...")

        for ll_key in LOGIT_LENS_CONFIGS.keys():
            sentence_data = logit_lens_sentence_data[ll_key]
            if not sentence_data:
                continue

            all_emotion_scores = np.array([s['emotion_scores'] for s in sentence_data])
            all_mean_logits = np.array([s['mean_logit'] for s in sentence_data])

            correction_info = compute_baseline_correction_alpha(all_emotion_scores, all_mean_logits)
            logit_lens_correction_info[ll_key] = correction_info

            print(f"    {ll_key}: corr_before={correction_info['correlation_before']:.3f}, "
                  f"alpha={correction_info['alpha']:.3f}, "
                  f"corr_after={correction_info['correlation_after']:.3f}")

            # Apply correction
            for data_item in sentence_data:
                normalized_logit = (data_item['mean_logit'] - correction_info['logit_mean']) / correction_info['logit_std']
                data_item['emotion_scores'] = data_item['emotion_scores'] - correction_info['alpha'] * normalized_logit

                # Also correct per-layer scores
                if data_item['per_layer_scores']:
                    for layer in data_item['per_layer_scores'].keys():
                        data_item['per_layer_scores'][layer] = \
                            data_item['per_layer_scores'][layer] - correction_info['alpha'] * normalized_logit

            # Store in conversations
            for data_item in sentence_data:
                conv = processed_conversations[data_item['conv_idx']]
                if 'probe_scores' not in conv:
                    conv['probe_scores'] = {}
                conv['probe_scores'][ll_key] = conv['probe_scores'].get(ll_key, {})
                conv['probe_scores'][ll_key][data_item['sent_id']] = data_item['emotion_scores'].tolist()

                # Per-layer
                per_layer_key = f"{ll_key}_by_layer"
                if per_layer_key not in conv:
                    conv[per_layer_key] = {}
                conv[per_layer_key][data_item['sent_id']] = {
                    layer: scores.tolist() for layer, scores in data_item['per_layer_scores'].items()
                }

    # =========================================================================
    # Step 7: Save
    # =========================================================================
    print(f"\n[7/7] Saving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build probe configs for output
    output_probe_configs = {k: PROBE_CONFIGS[k] for k in probe_keys}
    if include_logit_lens:
        for ll_key, ll_config in LOGIT_LENS_CONFIGS.items():
            output_probe_configs[ll_key] = {
                'name': ll_config['display_name'],
                'type': 'logit_lens',
                'layers': ll_config['layers']
            }

    with open(output_path, 'wb') as f:
        pickle.dump({
            'conversations': processed_conversations,
            'probe_configs': output_probe_configs,
            'probe_baselines': probe_baselines,
            'metadata': {
                'num_conversations': len(processed_conversations),
                'emotions': EMOTIONS,
                'layers': MODEL_CONFIG['layers'],
                'model_name': model_name,
                'has_per_layer_scores': True,
                'has_logit_lens': include_logit_lens,
                'logit_lens_baseline_corrections': logit_lens_correction_info if include_logit_lens else None
            }
        }, f)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    print(f"\n{'=' * 80}")
    print("✓ PREPROCESSING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"  Processed {len(processed_conversations)} conversations")
    print(f"  Probes: {probe_keys}")
    if include_logit_lens:
        print(f"  Logit lens: {list(LOGIT_LENS_CONFIGS.keys())}")
    print(f"  Per-layer scores: YES (layers {MODEL_CONFIG['layers'][0]}-{MODEL_CONFIG['layers'][-1]})")
    print(f"  Normalization: Layer-averaged baseline (probes), per-layer + correction (logit lens)")
    print(f"  Output: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unified preprocessing: probes + logit lens in single pass'
    )
    parser.add_argument('--input', type=str, required=True,
                        help='Path to emotion onset data (JSON/JSONL)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for preprocessed pickle')
    parser.add_argument('--model', type=str, default=None,
                        help='Model to use (default: gemma-3-27b-it). Can be path to finetuned model.')
    parser.add_argument('--probes', type=str, nargs='+', default=None,
                        help='Probe types to include (default: orthogonal_raw, text_raw, centroid_k10)')
    parser.add_argument('--no-logit-lens', action='store_true',
                        help='Disable logit lens computation')
    parser.add_argument('--test-only', type=int, default=None,
                        help='Only process this many conversations (for testing)')
    parser.add_argument('--simple-splitter', action='store_true',
                        help='Use simple sentence splitter')

    args = parser.parse_args()

    preprocess_unified(
        data_path=args.input,
        output_path=args.output,
        model_name=args.model,
        probe_keys=args.probes,
        include_logit_lens=not args.no_logit_lens,
        test_only=args.test_only,
        use_simple_splitter=args.simple_splitter
    )
