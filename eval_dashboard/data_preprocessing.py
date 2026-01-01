"""
Data Preprocessing Pipeline

Load emotion onset data, apply probes, aggregate to sentences, and save for dashboard.
"""

import sys
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add research-tools to path
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

from transformers import AutoTokenizer
from probes.scripts.token_level_helpers import TokenLevelExperiment
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader
from probes.scripts.probe_pipeline import normalize_probe_scores_zscore

from probe_configs import PROBE_CONFIGS, BASELINE_CONFIG, MODEL_CONFIG, EMOTIONS
from sentence_aggregator import (
    split_conversation_into_sentences,
    split_conversation_into_sentences_simple,
    aggregate_scores_to_sentences,
    SentenceInfo
)


@dataclass
class ProcessedConversation:
    """Fully processed conversation with sentence-level probe scores."""
    sample_id: int
    conversation: List[Dict[str, str]]
    rating: float
    sentences: List[SentenceInfo]
    probe_scores: Dict[str, Dict[int, np.ndarray]]  # probe_key -> {sentence_id -> scores}
    metadata: Dict


def load_emotion_onset_data(data_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load the annotated emotion onset data.

    Returns:
        Tuple of (samples, annotations)
    """
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

    print(f"Loaded {len(samples)} samples from {data_path}")
    return samples


def extract_activations_for_conversation(
    conversation: List[Dict[str, str]],
    model,
    tokenizer,
    layers: List[int]
) -> Tuple[Dict[int, Dict[int, np.ndarray]], List[str]]:
    """
    Extract token-level activations for a full conversation.

    Returns:
        Tuple of (activations_by_token, token_strings)
        activations_by_token: {token_pos: {layer: activation}}
        token_strings: List of decoded token strings
    """
    import torch

    # Build full conversation text (same format as emotion onset)
    conversation_text = ""
    for turn in conversation:
        role = turn['role']
        content = turn['content']
        if role == 'user':
            conversation_text += f"User: {content}\n"
        else:
            conversation_text += f"Assistant: {content}\n"

    # Tokenize
    inputs = tokenizer(conversation_text, return_tensors="pt", add_special_tokens=False)
    token_ids = inputs['input_ids'][0].tolist()

    # Decode tokens
    token_strings = [tokenizer.decode([tid]) for tid in token_ids]

    # Extract activations
    with torch.no_grad():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)

    # Convert to token-level format
    activations_by_token = {}
    for token_pos in range(len(token_ids)):
        activations_by_token[token_pos] = {}
        for layer in layers:
            # hidden_states is tuple of (layer0, layer1, ..., layerN)
            # Each is shape [batch, seq_len, hidden_dim]
            act = outputs.hidden_states[layer][0, token_pos, :].float().cpu().numpy()
            activations_by_token[token_pos][layer] = act

    return activations_by_token, token_strings


def apply_probes_to_activations(
    activations_by_token: Dict[int, Dict[int, np.ndarray]],
    probe_experiment: TokenLevelExperiment,
    layers: List[int]
) -> Dict[int, np.ndarray]:
    """
    Apply a pre-initialized probe experiment to activations.

    Returns:
        Dict mapping token_position -> emotion_scores [n_emotions]
    """
    # Apply probes using pre-initialized experiment
    scores_by_token = probe_experiment._apply_probes(
        activations_by_token=activations_by_token,
        layers=layers,
        verbose=False
    )

    # Aggregate across layers (mean)
    aggregated_scores = {}
    for token_pos in scores_by_token:
        layer_scores = []
        user_scores = []
        asst_scores = []
        has_orthogonal = False

        for layer in layers:
            score = scores_by_token[token_pos][layer]
            if isinstance(score, dict) and 'user' in score:
                # Orthogonal probes - keep user and assistant separate
                has_orthogonal = True
                user_scores.append(score['user'])
                asst_scores.append(score['assistant'])
            else:
                layer_scores.append(score)

        if has_orthogonal:
            # Return dict with separate user/assistant scores
            aggregated_scores[token_pos] = {
                'user': np.mean(user_scores, axis=0),
                'assistant': np.mean(asst_scores, axis=0)
            }
        else:
            aggregated_scores[token_pos] = np.mean(layer_scores, axis=0)

    return aggregated_scores


def preprocess_all_conversations(
    data_path: str,
    probe_keys: List[str],
    output_path: str,
    use_simple_splitter: bool = False
):
    """
    Main preprocessing pipeline.

    Args:
        data_path: Path to emotion onset data
        probe_keys: List of probe configuration keys to apply
        output_path: Where to save preprocessed data
        use_simple_splitter: Use simple sentence splitter instead of Sentences library
    """
    print("="*80)
    print("EMOTION ONSET DASHBOARD - DATA PREPROCESSING")
    print("="*80)

    # Load data
    print("\n[1/6] Loading data...")
    samples = load_emotion_onset_data(data_path)

    # Load model and tokenizer
    print("\n[2/6] Loading model and tokenizer...")
    from transformers import AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print(f"✓ Model loaded on {model.device}")

    # Initialize baseline normalization (compute once)
    print("\n[3/6] Computing probe baseline statistics...")
    from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader

    baseline_loader = WildChatBaselineLoader(
        aggregation_type=BASELINE_CONFIG['aggregation_type'],
        baseline_dir=BASELINE_CONFIG['baseline_dir']
    )

    # We'll compute baseline stats for each probe type
    probe_baselines = {}
    for probe_key in probe_keys:
        probe_config = PROBE_CONFIGS[probe_key]

        # Create temporary experiment to get inference object
        exp_temp = TokenLevelExperiment(
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
            baseline_dir=BASELINE_CONFIG['baseline_dir'],  # Scores automatically normalized
            emotions=EMOTIONS
        )

        print(f"  Computing baseline for {probe_key}...")
        baseline_stats = baseline_loader.compute_probe_score_baselines(
            probe_inference=exp_temp.inference,
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
            'mean': baseline_stats['mean'][-1],  # Averaged across layers
            'std': baseline_stats['std'][-1]
        }

    print("✓ Baseline statistics computed")

    # Initialize probe experiments (load probes once)
    print("\n[3.5/6] Initializing probe experiments...")
    probe_experiments = {}
    for probe_key in probe_keys:
        probe_config = PROBE_CONFIGS[probe_key]

        print(f"  Loading probes for {probe_key}...")
        probe_experiments[probe_key] = TokenLevelExperiment(
            model=None,  # Not needed for cached activations
            tokenizer=tokenizer,
            probe_type=probe_config['type'],
            probe_dir=probe_config.get('probe_dir'),
            cpca_path=probe_config.get('cpca_path'),
            probe_pattern=probe_config.get('probe_pattern'),
            orthogonality_weight=probe_config.get('orthogonality_weight', 1000.0),
            orthogonal_representation=probe_config.get('orthogonal_representation', 'raw'),
            n_components=probe_config.get('n_components', 10),
            seed=probe_config.get('seed', 0),
            baseline_dir=BASELINE_CONFIG['baseline_dir'],  # Scores automatically normalized
            emotions=EMOTIONS,
            k_value=probe_config.get('k_value'),
            centroid_probe_format=probe_config.get('centroid_probe_format', 'auto')
        )

    print("✓ Probe experiments initialized")

    # Process each conversation
    print(f"\n[4/6] Processing {len(samples)} conversations...")
    processed_conversations = []

    for sample_idx, sample in enumerate(tqdm(samples, desc="Processing")):
        conversation = sample['conversation']
        sample_id = sample.get('sample_id', sample_idx)
        rating = sample.get('rating', 0)

        # Extract activations
        activations_by_token, token_strings = extract_activations_for_conversation(
            conversation=conversation,
            model=model,
            tokenizer=tokenizer,
            layers=MODEL_CONFIG['layers']
        )

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
                    print("Warning: Sentences library not available, using simple splitter")
                sentences = split_conversation_into_sentences_simple(
                    conversation, token_strings, tokenizer
                )

        # Apply all probes
        probe_scores = {}
        for probe_key in probe_keys:
            # Apply probe to get token-level scores (using pre-initialized experiment)
            token_scores = apply_probes_to_activations(
                activations_by_token=activations_by_token,
                probe_experiment=probe_experiments[probe_key],
                layers=MODEL_CONFIG['layers']
            )

            # Apply z-score normalization (always applied for consistency)
            # Note: We call _apply_probes directly for efficiency, so we normalize manually here
            probe_mean = probe_baselines[probe_key]['mean']
            probe_std = probe_baselines[probe_key]['std']

            for token_pos in token_scores:
                score = token_scores[token_pos]
                token_scores[token_pos] = normalize_probe_scores_zscore(
                    score, probe_mean, probe_std
                )

            # Aggregate to sentence level
            sentence_scores = aggregate_scores_to_sentences(
                sentences=sentences,
                token_scores=token_scores,
                aggregation='mean'
            )

            probe_scores[probe_key] = sentence_scores

        # Find onset sentence based on turn_number and judge evidence
        # turn_number indicates which assistant turn was judged
        # We want to find the first sentence of that assistant turn
        onset_sentence_id = None
        turn_number = sample.get('turn_number', 1)

        # Assistant turns in conversation: turn 1 = conv index 1, turn 2 = conv index 3, etc.
        # So assistant turn N is at conversation index (N * 2 - 1)
        target_conv_turn = turn_number - 1  # turn_index is 0-indexed

        # Find first assistant sentence in that turn
        for sent in sentences:
            if sent.turn_role == 'assistant' and sent.turn_index == target_conv_turn:
                onset_sentence_id = sent.sentence_id
                break

        processed_conv = ProcessedConversation(
            sample_id=sample_id,
            conversation=conversation,
            rating=rating,
            sentences=[asdict(s) for s in sentences],
            probe_scores=probe_scores,
            metadata={
                'num_sentences': len(sentences),
                'num_tokens': len(token_strings),
                'num_turns': len(conversation),
                'onset_sentence_id': onset_sentence_id,
                'judge_evidence': sample.get('judge_evidence', ''),
                'judge_reasoning': sample.get('judge_reasoning', '')
            }
        )

        processed_conversations.append(processed_conv)

    # Save preprocessed data
    print(f"\n[5/6] Saving to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump({
            'conversations': [asdict(c) for c in processed_conversations],
            'probe_configs': {k: PROBE_CONFIGS[k] for k in probe_keys},
            'probe_baselines': probe_baselines,
            'metadata': {
                'num_conversations': len(processed_conversations),
                'emotions': EMOTIONS,
                'layers': MODEL_CONFIG['layers']
            }
        }, f)

    print(f"\n[6/6] Cleanup...")
    # Free GPU memory
    del model
    import torch
    torch.cuda.empty_cache()

    print(f"\n✓ Preprocessing complete!")
    print(f"  Processed {len(processed_conversations)} conversations")
    print(f"  Applied {len(probe_keys)} probe types")
    print(f"  Output: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess emotion onset data for dashboard')
    parser.add_argument('--input', type=str,
                       default='/workspace-vast/annas/git/research-tools/elicitation/outputs/annotated_emotion_onset_gemma3.jsonl',
                       help='Path to emotion onset data')
    parser.add_argument('--output', type=str,
                       default='/workspace-vast/annas/git/research-tools/eval_dashboard/data/preprocessed_conversations.pkl',
                       help='Output path for preprocessed data')
    parser.add_argument('--probes', type=str, nargs='+',
                       default=['orthogonal_raw', 'text_raw', 'centroid_k10'],
                       help='Probe types to include')
    parser.add_argument('--simple-splitter', action='store_true',
                       help='Use simple sentence splitter')

    args = parser.parse_args()

    preprocess_all_conversations(
        data_path=args.input,
        probe_keys=args.probes,
        output_path=args.output,
        use_simple_splitter=args.simple_splitter
    )
