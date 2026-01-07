"""
Add logit_lens scores to existing pickle file using emo_lens implementation.

NOW WITH:
- Integrated two-pass baseline correction
- All layer ranges computed in single pass (L40-50, L30-40, L20-30)
- Per-layer scores stored for layerwise plotting
"""
import pickle
import numpy as np
from pathlib import Path
import sys
import torch

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
    normalize_logits_to_emotion_scores
)

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

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
    }
}

def compute_mean_logit_for_activation(model, activation):
    """
    Project activation to vocabulary space and compute mean logit.

    Args:
        model: Language model with project_on_vocab method
        activation: Numpy array of shape (hidden_size,)

    Returns:
        Float: Mean logit value across vocabulary
    """
    device = next(model.parameters()).device

    with torch.no_grad():
        activation_tensor = torch.from_numpy(activation).to(device)
        if hasattr(model, 'dtype'):
            activation_tensor = activation_tensor.to(model.dtype)
        elif hasattr(model.lm_head, 'weight'):
            activation_tensor = activation_tensor.to(model.lm_head.weight.dtype)

        logits = model.project_on_vocab(activation_tensor.unsqueeze(0))  # (1, vocab_size)
        mean_logit = logits.mean().cpu().float().item()

    return mean_logit


def compute_emotion_scores_for_single_layer(
    model,
    activation,
    emotion_token_ids,
    baseline_stats_for_layer,
    aggregation="mean"
):
    """
    Compute emotion scores for a single layer's activation.

    Args:
        model: Language model
        activation: numpy array (hidden_size,)
        emotion_token_ids: Dict[str, List[int]]
        baseline_stats_for_layer: Dict[emotion][token_id] = {'mean': float, 'std': float}
        aggregation: "mean" or "max"

    Returns:
        Dict[emotion: float]
    """
    device = next(model.parameters()).device

    with torch.no_grad():
        activation_tensor = torch.from_numpy(activation).to(device)
        if hasattr(model, 'dtype'):
            activation_tensor = activation_tensor.to(model.dtype)
        elif hasattr(model.lm_head, 'weight'):
            activation_tensor = activation_tensor.to(model.lm_head.weight.dtype)

        logits = model.project_on_vocab(activation_tensor.unsqueeze(0))  # (1, vocab_size)
        logits_np = logits.cpu().float().numpy()[0]  # (vocab_size,)

    # Compute scores for each emotion
    scores = {}
    for emotion in EMOTIONS:
        token_ids = emotion_token_ids[emotion]
        emotion_logits = logits_np[token_ids]

        # Normalize using baseline stats
        normalized_scores = []
        for token_id, logit in zip(token_ids, emotion_logits):
            if token_id in baseline_stats_for_layer.get(emotion, {}):
                stats = baseline_stats_for_layer[emotion][token_id]
                z_score = (logit - stats['mean']) / (stats['std'] + 1e-8)
                normalized_scores.append(z_score)

        # Aggregate
        if normalized_scores:
            if aggregation == "mean":
                scores[emotion] = np.mean(normalized_scores)
            elif aggregation == "max":
                scores[emotion] = np.max(normalized_scores)
            else:
                scores[emotion] = np.mean(normalized_scores)
        else:
            scores[emotion] = 0.0

    return scores


def compute_baseline_correction_alpha(all_emotion_scores, all_mean_logits):
    """
    Compute optimal alpha to remove correlation between emotion scores and mean logits.

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
        print("  ⚠️  Warning: Mean logits have zero variance, skipping correction")
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

    # Compute optimal alpha
    alpha = correlation_before * mean_emotion_scores.std() / logit_std

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


def compute_correlation_stats(conversations, probe_key):
    """Compute correlation statistics for verification."""
    all_scores = []

    for conv in conversations:
        if 'probe_scores' not in conv or probe_key not in conv['probe_scores']:
            continue

        for sent_id, scores in conv['probe_scores'][probe_key].items():
            if isinstance(scores, (np.ndarray, list)) and len(scores) == 6:
                all_scores.append(scores)

    if not all_scores:
        return None

    scores_matrix = np.array(all_scores)
    corr_matrix = np.corrcoef(scores_matrix.T)

    off_diag_corrs = []
    for i in range(6):
        for j in range(i+1, 6):
            off_diag_corrs.append(abs(corr_matrix[i, j]))

    return {
        'n_scores': len(all_scores),
        'mean': scores_matrix.mean(),
        'std': scores_matrix.std(),
        'min': scores_matrix.min(),
        'max': scores_matrix.max(),
        'avg_correlation': np.mean(off_diag_corrs),
        'corr_matrix': corr_matrix
    }


def main():
    TEST_ONLY = False  # Set to False to run on all conversations
    NUM_TEST = 5
    APPLY_BASELINE_CORRECTION = True  # Enable integrated baseline correction
    STORE_PER_LAYER = True  # Store per-layer scores for layerwise plotting

    print("="*80)
    print("ADD LOGIT LENS TO EXISTING PICKLE (using emo_lens)")
    print("WITH INTEGRATED TWO-PASS BASELINE CORRECTION")
    print("AND ALL LAYER RANGES (L40-50, L30-40, L20-30)")
    if STORE_PER_LAYER:
        print("STORING PER-LAYER SCORES FOR LAYERWISE PLOTTING")
    print("="*80)

    input_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl')
    output_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl')

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
    print(f"  Existing probes: {list(data['probe_configs'].keys())}")

    # Load model using emo_lens loader
    print("\n[2/6] Loading model...")
    BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
    model, tokenizer = load_base_model(BASE_MODEL_NAME)
    print("  ✓ Model loaded")

    # Load emotion token IDs using emo_lens loader
    print("\n[3/6] Loading emotion token IDs...")
    emotion_token_ids = load_emotion_token_ids_from_json(BASE_MODEL_NAME)
    print(f"  ✓ Loaded {len(emotion_token_ids)} emotions")

    # Load reference statistics using emo_lens loader
    print("\n[4/6] Loading baseline statistics...")
    ACTIVATION_STRATEGY = "generated_tokens_avg"
    SCRIPT_DIR = Path("/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens")
    ref_stats = load_reference_stats_for_strategy(
        model_name=BASE_MODEL_NAME,
        activation_strategy=ACTIVATION_STRATEGY,
        script_dir=SCRIPT_DIR
    )
    print("  ✓ Loaded baseline stats")

    # Compute layer-averaged baseline stats for each layer range
    print(f"\n  Computing baseline stats for all layer ranges...")
    averaged_baseline_stats_by_range = {}
    for probe_key, config in LAYER_RANGE_CONFIGS.items():
        layers = config['layers']
        print(f"    {probe_key}: layers {layers[0]}-{layers[-1]}")
        averaged_baseline_stats_by_range[probe_key] = compute_layer_averaged_baseline_stats(
            ref_stats=ref_stats,
            layers=layers,
            emotion_token_ids=emotion_token_ids
        )
    print("  ✓ Computed all baseline statistics")

    # Extract ALL layers we need (min to max across all configs)
    all_layers_needed = set()
    for config in LAYER_RANGE_CONFIGS.values():
        all_layers_needed.update(config['layers'])
    all_layers_needed = sorted(list(all_layers_needed))
    print(f"\n  Will extract layers: {all_layers_needed[0]}-{all_layers_needed[-1]} ({len(all_layers_needed)} layers)")

    # =========================================================================
    # PASS 1: Extract activations, compute emotion scores AND mean logits
    # =========================================================================
    print(f"\n[5/6] PASS 1: Processing {total_convs if not TEST_ONLY else NUM_TEST} conversations...")
    print("  (Computing emotion scores for all layer ranges and mean logits)")

    num_to_process = NUM_TEST if TEST_ONLY else total_convs

    # Store intermediate results for baseline correction
    # Structure: {probe_key: [{conv_idx, sent_id, emotion_scores, mean_logit, per_layer_scores}]}
    sentence_data_by_range = {probe_key: [] for probe_key in LAYER_RANGE_CONFIGS.keys()}

    for i in range(num_to_process):
        conv = data['conversations'][i]

        print(f"\n  [{i+1}/{num_to_process}] Sample ID: {conv['sample_id']}")

        # Build conversation text
        conversation_text = ""
        for turn in conv['conversation']:
            role = turn['role']
            content = turn['content']
            if role == 'user':
                conversation_text += f"User: {content}\n"
            else:
                conversation_text += f"Assistant: {content}\n"

        # Extract token-level activations for ALL needed layers
        activations_by_token, token_ids = extract_token_level_activations(
            model=model,
            tokenizer=tokenizer,
            prompt=conversation_text,
            layers=all_layers_needed,
            start_token_idx=0,
            system_prompt=None,
            num_generated_tokens=0
        )

        # Process each layer range configuration
        for probe_key, range_config in LAYER_RANGE_CONFIGS.items():
            layers = range_config['layers']

            # Get baseline stats for this range
            averaged_baseline_stats = averaged_baseline_stats_by_range[probe_key]

            # Compute per-layer emotion scores for each token
            token_emotion_scores_by_layer = {}  # {token_pos: {layer: {emotion: score}}}

            if STORE_PER_LAYER:
                for token_pos in activations_by_token.keys():
                    token_emotion_scores_by_layer[token_pos] = {}
                    for layer in layers:
                        activation = activations_by_token[token_pos][layer]
                        # Get per-layer baseline stats
                        layer_baseline_stats = {}
                        for emotion in EMOTIONS:
                            layer_baseline_stats[emotion] = {}
                            for token_id in emotion_token_ids[emotion]:
                                if token_id in ref_stats['layers_data'][str(layer)]['statistics']:
                                    layer_baseline_stats[emotion][token_id] = ref_stats['layers_data'][str(layer)]['statistics'][token_id]

                        scores = compute_emotion_scores_for_single_layer(
                            model=model,
                            activation=activation,
                            emotion_token_ids=emotion_token_ids,
                            baseline_stats_for_layer=layer_baseline_stats,
                            aggregation="mean"
                        )
                        token_emotion_scores_by_layer[token_pos][layer] = scores

            # Compute layer-averaged emotion scores for each token (for aggregated probe score)
            token_emotion_scores = {}
            for token_pos in activations_by_token.keys():
                # Average activations across layers in this range
                layer_activations = [activations_by_token[token_pos][layer] for layer in layers]
                avg_activation = np.mean(layer_activations, axis=0)

                # Project and compute scores
                scores = compute_emotion_scores_for_single_layer(
                    model=model,
                    activation=avg_activation,
                    emotion_token_ids=emotion_token_ids,
                    baseline_stats_for_layer=averaged_baseline_stats,
                    aggregation="mean"
                )
                token_emotion_scores[token_pos] = scores

            # Compute mean logits for baseline correction (use layer-averaged activation)
            token_mean_logits = {}
            if APPLY_BASELINE_CORRECTION:
                for token_pos in activations_by_token.keys():
                    layer_activations = [activations_by_token[token_pos][layer] for layer in layers]
                    avg_activation = np.mean(layer_activations, axis=0)
                    mean_logit = compute_mean_logit_for_activation(model, avg_activation)
                    token_mean_logits[token_pos] = mean_logit

            # Aggregate to sentence level
            for sent in conv['sentences']:
                sent_id = sent['sentence_id']
                start_tok = sent['start_token']
                end_tok = sent['end_token']

                # Aggregate layer-averaged scores
                sent_score_list = []
                sent_logit_list = []

                for t in range(start_tok, end_tok):
                    if t in token_emotion_scores:
                        scores_array = np.array([token_emotion_scores[t][e] for e in EMOTIONS])
                        sent_score_list.append(scores_array)

                        if APPLY_BASELINE_CORRECTION and t in token_mean_logits:
                            sent_logit_list.append(token_mean_logits[t])

                # Aggregate per-layer scores
                per_layer_sentence_scores = {}
                if STORE_PER_LAYER:
                    for layer in layers:
                        layer_scores_list = []
                        for t in range(start_tok, end_tok):
                            if t in token_emotion_scores_by_layer and layer in token_emotion_scores_by_layer[t]:
                                scores_array = np.array([token_emotion_scores_by_layer[t][layer][e] for e in EMOTIONS])
                                layer_scores_list.append(scores_array)

                        if layer_scores_list:
                            per_layer_sentence_scores[layer] = np.mean(layer_scores_list, axis=0)
                        else:
                            per_layer_sentence_scores[layer] = np.zeros(6)

                if sent_score_list:
                    sentence_emotion_scores = np.mean(sent_score_list, axis=0)
                    sentence_mean_logit = np.mean(sent_logit_list) if sent_logit_list else 0.0

                    sentence_data_by_range[probe_key].append({
                        'conv_idx': i,
                        'sent_id': sent_id,
                        'emotion_scores': sentence_emotion_scores,
                        'mean_logit': sentence_mean_logit,
                        'per_layer_scores': per_layer_sentence_scores if STORE_PER_LAYER else None
                    })
                else:
                    sentence_data_by_range[probe_key].append({
                        'conv_idx': i,
                        'sent_id': sent_id,
                        'emotion_scores': np.zeros(6),
                        'mean_logit': 0.0,
                        'per_layer_scores': {layer: np.zeros(6) for layer in layers} if STORE_PER_LAYER else None
                    })

        print(f"    ✓ Processed {len([s for s in conv['sentences']])} sentences for {len(LAYER_RANGE_CONFIGS)} layer ranges")

    # =========================================================================
    # PASS 2: Compute baseline correction alpha for each range
    # =========================================================================
    correction_info_by_range = {}

    for probe_key in LAYER_RANGE_CONFIGS.keys():
        sentence_data = sentence_data_by_range[probe_key]

        if APPLY_BASELINE_CORRECTION and len(sentence_data) > 0:
            print(f"\n[6/6] PASS 2: Computing baseline correction for {probe_key}...")

            # Collect all scores and logits
            all_emotion_scores = np.array([s['emotion_scores'] for s in sentence_data])
            all_mean_logits = np.array([s['mean_logit'] for s in sentence_data])

            print(f"  Collected {len(all_emotion_scores)} sentences")

            # Compute optimal alpha
            correction_info = compute_baseline_correction_alpha(all_emotion_scores, all_mean_logits)
            correction_info_by_range[probe_key] = correction_info

            print(f"\n  Baseline Correction Results:")
            print(f"  {'='*76}")
            print(f"    Correlation (before): {correction_info['correlation_before']:+.4f}")
            print(f"    Optimal alpha:        {correction_info['alpha']:+.4f}")
            print(f"    Correlation (after):  {correction_info['correlation_after']:+.4f}")
            print(f"  {'='*76}")

            # Apply correction to all sentences
            print(f"\n  Applying correction to {len(sentence_data)} sentences...")

            for data_item in sentence_data:
                # Normalize logit
                normalized_logit = (data_item['mean_logit'] - correction_info['logit_mean']) / correction_info['logit_std']

                # Apply correction to aggregated scores
                corrected_scores = data_item['emotion_scores'] - correction_info['alpha'] * normalized_logit
                data_item['emotion_scores'] = corrected_scores

                # Also apply correction to per-layer scores
                if STORE_PER_LAYER and data_item['per_layer_scores']:
                    for layer in data_item['per_layer_scores'].keys():
                        corrected_layer_scores = data_item['per_layer_scores'][layer] - correction_info['alpha'] * normalized_logit
                        data_item['per_layer_scores'][layer] = corrected_layer_scores

            print(f"  ✓ Baseline correction applied to {probe_key}")

    # =========================================================================
    # PASS 3: Store corrected scores in conversations
    # =========================================================================
    print(f"\n  Storing scores in conversations...")

    for probe_key in LAYER_RANGE_CONFIGS.keys():
        for data_item in sentence_data_by_range[probe_key]:
            conv = data['conversations'][data_item['conv_idx']]

            # Initialize probe_scores dict if needed
            if 'probe_scores' not in conv:
                conv['probe_scores'] = {}
            if probe_key not in conv['probe_scores']:
                conv['probe_scores'][probe_key] = {}

            # Store aggregated scores
            conv['probe_scores'][probe_key][data_item['sent_id']] = data_item['emotion_scores'].tolist()

            # Store per-layer scores for layerwise plotting
            if STORE_PER_LAYER and data_item['per_layer_scores']:
                # Store in separate key for layerwise plotting
                per_layer_key = f"{probe_key}_by_layer"
                if per_layer_key not in conv:
                    conv[per_layer_key] = {}
                if data_item['sent_id'] not in conv[per_layer_key]:
                    conv[per_layer_key][data_item['sent_id']] = {}

                for layer, scores in data_item['per_layer_scores'].items():
                    conv[per_layer_key][data_item['sent_id']][layer] = scores.tolist()

    print(f"  ✓ Stored scores for {num_to_process} conversations across {len(LAYER_RANGE_CONFIGS)} layer ranges")

    # Verify results for each range
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}")

    for probe_key, config in LAYER_RANGE_CONFIGS.items():
        print(f"\n{config['display_name']}:")
        print("-" * 80)

        stats = compute_correlation_stats(data['conversations'][:num_to_process], probe_key)

        if stats:
            print(f"  Processed {stats['n_scores']} sentence scores")
            print(f"  Mean: {stats['mean']:+.3f}, Std: {stats['std']:.3f}")
            print(f"  Avg correlation: {stats['avg_correlation']:.3f}")

            # Validation
            if abs(stats['mean']) <= 5:
                print(f"  ✓ Mean in range")
            else:
                print(f"  ✗ Mean out of range: {stats['mean']:.1f}")

            if APPLY_BASELINE_CORRECTION and probe_key in correction_info_by_range:
                corr_after = correction_info_by_range[probe_key]['correlation_after']
                if abs(corr_after) < 0.1:
                    print(f"  ✓ Baseline correlation removed ({corr_after:+.3f})")
                else:
                    print(f"  ⚠ Baseline correlation present ({corr_after:+.3f})")

    # Save output
    print(f"\n{'='*80}")
    print("SAVING OUTPUT")
    print(f"{'='*80}")

    # Update probe configs
    from eval_dashboard.probe_configs import PROBE_CONFIGS
    for probe_key in LAYER_RANGE_CONFIGS.keys():
        if probe_key in PROBE_CONFIGS:
            data['probe_configs'][probe_key] = PROBE_CONFIGS[probe_key]

    # Store baseline correction info in metadata
    if correction_info_by_range:
        if 'metadata' not in data:
            data['metadata'] = {}
        data['metadata']['logit_lens_baseline_corrections'] = correction_info_by_range

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")
    print(f"\n  Added {len(LAYER_RANGE_CONFIGS)} logit lens configurations:")
    for probe_key, config in LAYER_RANGE_CONFIGS.items():
        print(f"    - {probe_key}: layers {config['layers'][0]}-{config['layers'][-1]}")
    if STORE_PER_LAYER:
        print(f"  Per-layer scores stored for layerwise plotting")

    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
