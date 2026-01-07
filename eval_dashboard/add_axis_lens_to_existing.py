"""
Add axis_lens scores to existing pickle file using emo_lens implementation.
Computes dimensional axis scores alongside existing emotion scores.
Tests on first 5 conversations, then can run on full dataset.

NOW WITH INTEGRATED TWO-PASS BASELINE CORRECTION:
- Pass 1: Extract activations, compute axis scores AND mean logits
- Pass 2: Compute optimal alpha to remove correlation
- Pass 3: Apply correction to all scores
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
    compute_layer_averaged_axis_baseline_stats,
    compute_token_axis_scores
)
from emotion_evals.emo_lens.model_utils import load_base_model
from emotion_evals.emo_lens.logit_lens_emotion_direct import (
    load_reference_stats_for_strategy,
    load_axis_token_groups
)

AXES = ['valence', 'arousal', 'dominance', 'approach_avoidance']

def compute_random_token_baseline(model, tokenizer, layer_range, n_tokens=500, vocab_size=256000):
    """
    Compute baseline logit values from random tokens.

    We sample random tokens, pass them through the model to get hidden states at each layer,
    then project those hidden states to vocabulary space and compute mean logit values.

    Args:
        model: The language model
        tokenizer: Tokenizer (for creating input)
        layer_range: List of layer indices to compute baseline for
        n_tokens: Number of random tokens to sample (sampled across multiple forward passes)
        vocab_size: Size of vocabulary

    Returns:
        Dict with:
            - 'per_layer': Dict mapping layer -> mean logit value across random tokens
            - 'global_mean': Overall mean across all layers
    """
    import random

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

    # Compute mean per layer
    layer_baselines = {}
    for layer_idx in layer_range:
        layer_baselines[layer_idx] = np.mean(layer_logit_values[layer_idx])

    # Compute global mean across all layers
    global_mean = np.mean(list(layer_baselines.values()))

    print(f"  ✓ Random baseline computed: global_mean={global_mean:.4f}")
    print(f"    Layer range: {min(layer_baselines.values()):.4f} to {max(layer_baselines.values()):.4f}")

    return {
        'per_layer': layer_baselines,
        'global_mean': global_mean,
        'n_tokens': n_tokens,
        'vocab_size': vocab_size
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


def compute_baseline_correction_alpha(all_axis_scores, all_mean_logits):
    """
    Compute optimal alpha to remove correlation between axis scores and mean logits.

    This is the core of baseline correction. We find the alpha such that:
        corrected_score = axis_score - alpha * normalized_logit
    has minimal correlation with mean_logit.

    Args:
        all_axis_scores: Array of shape (n_sentences, 4)
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

    # Compute correlation for mean axis score
    mean_axis_scores = all_axis_scores.mean(axis=1)
    correlation_before = np.corrcoef(mean_axis_scores, all_mean_logits)[0, 1]

    # Compute optimal alpha
    alpha = correlation_before * mean_axis_scores.std() / logit_std

    # Verify correction effectiveness
    corrected_scores = mean_axis_scores - alpha * normalized_logits
    correlation_after = np.corrcoef(corrected_scores, all_mean_logits)[0, 1]

    return {
        'alpha': alpha,
        'correlation_before': correlation_before,
        'correlation_after': correlation_after,
        'logit_mean': logit_mean,
        'logit_std': logit_std
    }


def compute_correlation_stats(conversations):
    """Compute correlation statistics for verification."""
    all_scores = []

    for conv in conversations:
        if 'probe_scores' not in conv or 'axis_lens_mean' not in conv['probe_scores']:
            continue

        for sent_id, scores in conv['probe_scores']['axis_lens_mean'].items():
            if isinstance(scores, (np.ndarray, list)) and len(scores) == 4:
                all_scores.append(scores)

    if not all_scores:
        return None

    scores_matrix = np.array(all_scores)
    corr_matrix = np.corrcoef(scores_matrix.T)

    off_diag_corrs = []
    for i in range(4):
        for j in range(i+1, 4):
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
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
        if len(sys.argv) > 2:
            output_path = Path(sys.argv[2])
        else:
            # Auto-generate output path
            output_path = input_path.parent / input_path.name.replace('.pkl', '_with_axes.pkl')
    else:
        # Default paths
        input_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus.pkl')
        output_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl')

    TEST_ONLY = False  # Set to False to run on all conversations
    NUM_TEST = 2  # Test on 2 conversations first
    APPLY_BASELINE_CORRECTION = True  # NEW: Enable integrated baseline correction

    print("="*80)
    print("ADD AXIS LENS TO EXISTING PICKLE (using emo_lens)")
    if APPLY_BASELINE_CORRECTION:
        print("WITH INTEGRATED TWO-PASS BASELINE CORRECTION")
    print("="*80)

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

    # Load axis token groups using emo_lens loader
    print("\n[3/6] Loading axis token groups...")
    axis_token_groups = load_axis_token_groups(BASE_MODEL_NAME)
    print(f"  ✓ Loaded {len(axis_token_groups)} axes:")
    for axis, groups in axis_token_groups.items():
        print(f"    - {axis}: {len(groups['high'])} high, {len(groups['low'])} low, {len(groups['neutral'])} neutral")

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

    # Compute layer-averaged baseline stats for axes
    LAYER_RANGE = list(range(40, 51))  # Layers 40-50
    print(f"\n  Computing layer-averaged axis baseline stats (layers {LAYER_RANGE[0]}-{LAYER_RANGE[-1]})...")
    averaged_axis_baseline_stats = compute_layer_averaged_axis_baseline_stats(
        ref_stats=ref_stats,
        layers=LAYER_RANGE,
        axis_token_groups=axis_token_groups
    )
    print("  ✓ Computed averaged axis baseline statistics")

    # Compute random token baseline for normalization
    random_baseline = compute_random_token_baseline(
        model=model,
        tokenizer=tokenizer,
        layer_range=LAYER_RANGE,
        n_tokens=500
    )

    # =========================================================================
    # PASS 1: Extract activations, compute axis scores AND mean logits
    # =========================================================================
    num_to_process = NUM_TEST if TEST_ONLY else total_convs
    print(f"\n[5/6] PASS 1: Processing {num_to_process} conversations...")
    print("  (Computing axis scores and mean logits)")

    # Store intermediate results for baseline correction
    sentence_data = []  # List of (conv_idx, sent_id, axis_scores, mean_logit)

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

        # Extract token-level activations using emo_lens function
        activations_by_token, token_ids = extract_token_level_activations(
            model=model,
            tokenizer=tokenizer,
            prompt=conversation_text,
            layers=LAYER_RANGE,
            start_token_idx=0,
            system_prompt=None,
            num_generated_tokens=0
        )

        # Compute axis scores using emo_lens function
        token_axis_scores = compute_token_axis_scores(
            model=model,
            activations_by_token=activations_by_token,
            layers=LAYER_RANGE,
            axis_token_groups=axis_token_groups,
            averaged_axis_baseline_stats=averaged_axis_baseline_stats,
            aggregation="mean",
            subtract_mean=True,  # Use z-score normalization
            global_mean_baseline=random_baseline['global_mean']
        )

        # Compute per-token mean logits
        token_mean_logits = {}
        for token_idx, activations in activations_by_token.items():
            # Average activations across layers
            layer_activations = []
            for layer_idx in sorted(activations.keys()):
                layer_activations.append(activations[layer_idx])

            avg_activation = np.mean(layer_activations, axis=0)  # (hidden_size,)
            mean_logit = compute_mean_logit_for_activation(model, avg_activation)
            token_mean_logits[token_idx] = mean_logit

        # Aggregate token scores to sentence level
        for sent in conv['sentences']:
            sent_id = sent['sentence_id']
            start_tok = sent['start_token']
            end_tok = sent['end_token']

            # Aggregate scores
            sent_score_list = []
            sent_logit_list = []

            for t in range(start_tok, end_tok):
                if t in token_axis_scores:
                    # Convert axis dict to array
                    scores_array = np.array([token_axis_scores[t][a] for a in AXES])
                    sent_score_list.append(scores_array)

                if t in token_mean_logits:
                    sent_logit_list.append(token_mean_logits[t])

            if sent_score_list:
                sentence_axis_scores = np.mean(sent_score_list, axis=0)
                sentence_mean_logit = np.mean(sent_logit_list) if sent_logit_list else 0.0

                # Store for later correction
                sentence_data.append({
                    'conv_idx': i,
                    'sent_id': sent_id,
                    'axis_scores': sentence_axis_scores,
                    'mean_logit': sentence_mean_logit
                })
            else:
                # If no tokens, use zeros
                sentence_data.append({
                    'conv_idx': i,
                    'sent_id': sent_id,
                    'axis_scores': np.zeros(4),
                    'mean_logit': 0.0
                })

        print(f"    ✓ Processed {len([s for s in conv['sentences']])} sentences")

    # =========================================================================
    # PASS 2: Compute baseline correction alpha
    # =========================================================================
    correction_info = None

    if APPLY_BASELINE_CORRECTION and len(sentence_data) > 0:
        print(f"\n[6/6] PASS 2: Computing baseline correction...")

        # Collect all scores and logits
        all_axis_scores = np.array([s['axis_scores'] for s in sentence_data])
        all_mean_logits = np.array([s['mean_logit'] for s in sentence_data])

        print(f"  Collected {len(all_axis_scores)} sentences")

        # Compute optimal alpha
        correction_info = compute_baseline_correction_alpha(all_axis_scores, all_mean_logits)

        print(f"\n  Baseline Correction Results:")
        print(f"  {'='*76}")
        print(f"    Correlation (before): {correction_info['correlation_before']:+.4f}")
        print(f"    Optimal alpha:        {correction_info['alpha']:+.4f}")
        print(f"    Correlation (after):  {correction_info['correlation_after']:+.4f}")
        print(f"    Logit mean:           {correction_info['logit_mean']:+.4f}")
        print(f"    Logit std:            {correction_info['logit_std']:+.4f}")
        print(f"  {'='*76}")

        # Apply correction to all sentences
        print(f"\n  Applying correction to {len(sentence_data)} sentences...")

        for data_item in sentence_data:
            # Normalize logit
            normalized_logit = (data_item['mean_logit'] - correction_info['logit_mean']) / correction_info['logit_std']

            # Apply correction: corrected = axis - alpha * normalized_logit
            corrected_scores = data_item['axis_scores'] - correction_info['alpha'] * normalized_logit

            # Update in place
            data_item['axis_scores'] = corrected_scores

        print(f"  ✓ Baseline correction applied")
    else:
        print(f"\n[6/6] Skipping baseline correction (disabled or no data)")

    # =========================================================================
    # PASS 3: Store corrected scores in conversations
    # =========================================================================
    print(f"\n  Storing scores in conversations...")

    for data_item in sentence_data:
        conv = data['conversations'][data_item['conv_idx']]

        # Initialize probe_scores dict if needed
        if 'probe_scores' not in conv:
            conv['probe_scores'] = {}
        if 'axis_lens_mean' not in conv['probe_scores']:
            conv['probe_scores']['axis_lens_mean'] = {}

        # Store as list (pickle-friendly)
        conv['probe_scores']['axis_lens_mean'][data_item['sent_id']] = data_item['axis_scores'].tolist()

    print(f"  ✓ Stored scores for {num_to_process} conversations")

    # Verify results
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print(f"{'='*80}")

    stats = compute_correlation_stats(data['conversations'][:num_to_process])

    if stats:
        print(f"\nProcessed {stats['n_scores']} sentence scores")
        print(f"\nScore statistics:")
        print(f"  Mean: {stats['mean']:+.3f}")
        print(f"  Std:  {stats['std']:.3f}")
        print(f"  Min:  {stats['min']:+.3f}")
        print(f"  Max:  {stats['max']:+.3f}")

        print(f"\nCorrelation matrix:")
        print("           ", " ".join(f"{a[:6]:>7s}" for a in AXES))
        for i, axis in enumerate(AXES):
            row = ' '.join(f'{stats["corr_matrix"][i, j]:7.3f}' for j in range(4))
            print(f"  {axis:18s} {row}")

        print(f"\nAverage absolute correlation: {stats['avg_correlation']:.3f}")

        # Validation
        print(f"\n{'='*80}")
        print("VALIDATION")
        print(f"{'='*80}")

        if abs(stats['mean']) > 5:
            print(f"✗ FAIL: Mean is {stats['mean']:.1f}, should be ~0")
        else:
            print(f"✓ PASS: Mean is {stats['mean']:.3f} (z-normalized)")

        if stats['avg_correlation'] < 0.7:
            print(f"✓ PASS: Correlation is {stats['avg_correlation']:.3f} (axes should be relatively independent)")
        else:
            print(f"⚠ WARNING: Correlation is {stats['avg_correlation']:.3f} (axes may be too correlated)")

        if APPLY_BASELINE_CORRECTION and correction_info:
            if abs(correction_info['correlation_after']) < 0.1:
                print(f"✓ PASS: Baseline correlation removed ({correction_info['correlation_after']:+.3f})")
            else:
                print(f"⚠ WARNING: Baseline correlation still present ({correction_info['correlation_after']:+.3f})")

    # Save output
    print(f"\n{'='*80}")
    print("SAVING OUTPUT")
    print(f"{'='*80}")

    # Update probe configs
    from eval_dashboard.probe_configs import PROBE_CONFIGS
    data['probe_configs']['axis_lens_mean'] = PROBE_CONFIGS['axis_lens_mean']

    # Store random baseline
    if 'probe_baselines' not in data:
        data['probe_baselines'] = {}
    data['probe_baselines']['random_token_baseline'] = random_baseline

    # Store baseline correction info in metadata
    if correction_info:
        if 'metadata' not in data:
            data['metadata'] = {}
        data['metadata']['axis_lens_baseline_correction'] = correction_info

    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ Saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")

    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
