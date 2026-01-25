"""
Add diverse isolation probe scores to existing preprocessed data.

Loads both user and assistant probes (λ=10) for layers 30-40,
applies them to activations, and aggregates across layers.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

from eval_dashboard.probe_configs import PROBE_CONFIGS, BASELINE_CONFIG, MODEL_CONFIG, EMOTIONS
from eval_dashboard.data_preprocessing import extract_activations_for_conversation
from eval_dashboard.sentence_aggregator import aggregate_scores_to_sentences, SentenceInfo
from probes.scripts.probe_pipeline import normalize_probe_scores_zscore


class SourceSpecificEmotionProbe(nn.Module):
    """Linear probe with orthogonality regularization."""

    def __init__(self, hidden_dim: int, num_emotions: int, orthogonal_pcs: np.ndarray = None):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_emotions)

        # Register orthogonal PCs as buffer (not trained)
        if orthogonal_pcs is not None:
            self.register_buffer(
                'orthogonal_pcs',
                torch.from_numpy(orthogonal_pcs).float()
            )
        else:
            self.orthogonal_pcs = None

    def forward(self, x):
        return self.linear(x)


def load_diverse_isolation_probe(probe_dir: Path, isolation_type: str, layer: int,
                                  lambda_reg: float, device: str = 'cuda') -> nn.Module:
    """
    Load a trained diverse isolation probe.

    Args:
        probe_dir: Base directory containing probes
        isolation_type: 'user' or 'assistant'
        layer: Layer number
        lambda_reg: Lambda regularization value
        device: Device to load model on

    Returns:
        Loaded probe model
    """
    model_path = probe_dir / f"{isolation_type}_layer{layer}_lambda{lambda_reg}" / "model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Probe not found: {model_path}")

    # Load checkpoint to CPU first
    checkpoint = torch.load(model_path, map_location='cpu')

    # Get model config from saved state dict
    hidden_dim = checkpoint['linear.weight'].shape[1]
    num_classes = checkpoint['linear.weight'].shape[0]

    # Get orthogonal PCs if saved
    orthogonal_pcs = checkpoint.get('orthogonal_pcs', None)
    if orthogonal_pcs is not None:
        orthogonal_pcs = orthogonal_pcs.cpu().numpy()

    # Create and load model
    model = SourceSpecificEmotionProbe(hidden_dim, num_classes, orthogonal_pcs)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    return model


def apply_diverse_isolation_probes(
    activations_by_token: dict,
    user_probes: dict,
    asst_probes: dict,
    layers: list,
    device: str = 'cuda'
) -> dict:
    """
    Apply both user and assistant probes to activations and aggregate across layers.

    Args:
        activations_by_token: {token_pos: {layer: activation}}
        user_probes: {layer: model}
        asst_probes: {layer: model}
        layers: List of layers to use
        device: Device for computation

    Returns:
        Dict mapping token_pos -> {'user': scores, 'assistant': scores}
    """
    token_scores = {}

    with torch.no_grad():
        for token_pos in activations_by_token:
            # Collect scores across layers
            user_layer_scores = []
            asst_layer_scores = []

            for layer in layers:
                # Get activation for this layer
                activation = activations_by_token[token_pos][layer]
                activation_tensor = torch.FloatTensor(activation).unsqueeze(0).to(device)

                # Apply user probe
                user_logits = user_probes[layer](activation_tensor)
                user_layer_scores.append(user_logits.cpu().numpy()[0])

                # Apply assistant probe
                asst_logits = asst_probes[layer](activation_tensor)
                asst_layer_scores.append(asst_logits.cpu().numpy()[0])

            # Average across layers
            token_scores[token_pos] = {
                'user': np.mean(user_layer_scores, axis=0),
                'assistant': np.mean(asst_layer_scores, axis=0)
            }

    return token_scores


def compute_baseline_statistics_from_data(
    user_probes: dict,
    asst_probes: dict,
    layers: list,
    baseline_dir: Path,
    device: str = 'cuda',
    num_samples: int = 500
) -> dict:
    """
    Compute baseline statistics from actual baseline data.

    Args:
        user_probes: Dict of user probes by layer
        asst_probes: Dict of assistant probes by layer
        layers: Layers to use
        baseline_dir: Directory with per-layer baseline files
        device: Device for computation
        num_samples: Number of token samples to use per layer

    Returns:
        Dict with 'user' and 'assistant' baseline stats
    """
    import h5py

    print("  Computing baseline statistics from Alpaca data...")

    user_all_scores = []
    asst_all_scores = []

    # Sample from a few layers to get diverse baseline
    sample_layers = [layers[0], layers[len(layers)//2], layers[-1]]

    for layer in sample_layers:
        baseline_file = baseline_dir / f"layer{layer}_activations.h5"

        if not baseline_file.exists():
            print(f"    Warning: {baseline_file.name} not found, skipping...")
            continue

        print(f"    Processing layer {layer} baseline...")

        with h5py.File(baseline_file, 'r') as f:
            # Each file has keys like "all_tokens" with shape (num_tokens, hidden_dim)
            dataset_keys = list(f.keys())

            for key in dataset_keys:
                activations = f[key][:]  # Shape: [num_tokens, hidden_dim]

                # Sample random tokens
                num_tokens = activations.shape[0]
                sample_size = min(num_samples, num_tokens)
                indices = np.random.choice(num_tokens, sample_size, replace=False)

                for idx in indices:
                    activation = activations[idx]  # Shape: [hidden_dim]

                    # Apply probes
                    with torch.no_grad():
                        act_tensor = torch.FloatTensor(activation).unsqueeze(0).to(device)
                        user_logits = user_probes[layer](act_tensor)
                        asst_logits = asst_probes[layer](act_tensor)

                        user_all_scores.append(user_logits.cpu().numpy()[0])
                        asst_all_scores.append(asst_logits.cpu().numpy()[0])

    # Compute statistics
    user_scores_array = np.array(user_all_scores)
    asst_scores_array = np.array(asst_all_scores)

    baseline_stats = {
        'user': {
            'mean': np.mean(user_scores_array, axis=0),
            'std': np.std(user_scores_array, axis=0)
        },
        'assistant': {
            'mean': np.mean(asst_scores_array, axis=0),
            'std': np.std(asst_scores_array, axis=0)
        }
    }

    print(f"    ✓ Collected {len(user_all_scores)} baseline samples")
    print(f"    ✓ User baseline - mean: {baseline_stats['user']['mean'][:3]}... std: {baseline_stats['user']['std'][:3]}...")
    print(f"    ✓ Asst baseline - mean: {baseline_stats['assistant']['mean'][:3]}... std: {baseline_stats['assistant']['std'][:3]}...")

    return baseline_stats


def process_subset_file(data_path: str, probe_key: str, probe_config: dict,
                        user_probes: dict, asst_probes: dict, baseline_stats: dict,
                        model, tokenizer, device: str, layers: list):
    """Process a single subset file."""
    print(f"\n{'='*80}")
    print(f"Processing: {Path(data_path).name}")
    print(f"{'='*80}")

    # Load existing data
    print(f"Loading data...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    conversations = data['conversations']
    print(f"✓ Loaded {len(conversations)} conversations")
    print(f"  Existing probe types: {list(conversations[0]['probe_scores'].keys())}")

    # Check if already exists
    if probe_key in conversations[0]['probe_scores']:
        print(f"⚠ {probe_key} scores already exist - overwriting...")

    # Process each conversation
    print(f"Adding {probe_key} scores to {len(conversations)} conversations...")

    for i, conv in enumerate(tqdm(conversations, desc="Processing")):
        conversation = conv['conversation']
        sentences = [SentenceInfo(**s) for s in conv['sentences']]

        # Re-extract activations (need layers 30-40)
        activations_by_token, token_strings = extract_activations_for_conversation(
            conversation=conversation,
            model=model,
            tokenizer=tokenizer,
            layers=layers
        )

        # Apply diverse isolation probes
        token_scores = apply_diverse_isolation_probes(
            activations_by_token=activations_by_token,
            user_probes=user_probes,
            asst_probes=asst_probes,
            layers=layers,
            device=device
        )

        # Normalize with baseline (z-score)
        for token_pos in token_scores:
            user_score = token_scores[token_pos]['user']
            asst_score = token_scores[token_pos]['assistant']

            token_scores[token_pos] = {
                'user': normalize_probe_scores_zscore(
                    user_score,
                    baseline_stats['user']['mean'],
                    baseline_stats['user']['std']
                ),
                'assistant': normalize_probe_scores_zscore(
                    asst_score,
                    baseline_stats['assistant']['mean'],
                    baseline_stats['assistant']['std']
                )
            }

        # Aggregate to sentences
        sentence_scores = aggregate_scores_to_sentences(
            sentences=sentences,
            token_scores=token_scores,
            aggregation='mean'
        )

        # Add to conversation
        conv['probe_scores'][probe_key] = sentence_scores

    print(f"✓ Added {probe_key} scores to all conversations")

    # Update probe configs in data
    data['probe_configs'][probe_key] = probe_config
    data['probe_baselines'][probe_key] = baseline_stats

    # Save updated data
    print("Saving updated data...")
    with open(data_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✓ Saved {Path(data_path).name}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Add diverse isolation probes to dashboard data')
    parser.add_argument('--subset', type=str, default='all',
                       help='Which subset to process (all, high, mid, low, low_shutdown, low_no_shutdown, baseline)')
    args = parser.parse_args()

    print("="*80)
    print("ADDING DIVERSE ISOLATION PROBES TO PREPROCESSED DATA")
    print("="*80)

    # Get probe config
    probe_key = 'diverse_isolation_lambda10'
    probe_config = PROBE_CONFIGS[probe_key]
    probe_dir = probe_config['probe_dir']
    lambda_reg = probe_config['lambda_reg']
    layers = probe_config['layers']

    print(f"\nProbe configuration:")
    print(f"  Type: {probe_config['type']}")
    print(f"  Dir: {probe_dir}")
    print(f"  Lambda: {lambda_reg}")
    print(f"  Layers: {layers[0]}-{layers[-1]} ({len(layers)} layers)")

    # Define subset files
    data_dir = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data')
    subsets = {
        'high': 'high_emotion_6plus.pkl',
        'mid': 'mid_emotion_3to5.pkl',
        'low': 'low_emotion_0to2.pkl',
        'low_shutdown': 'low_emotion_with_shutdown.pkl',
        'low_no_shutdown': 'low_emotion_no_shutdown.pkl',
        'baseline': 'baseline_v12_solvable.pkl'
    }

    # Select which files to process
    if args.subset == 'all':
        files_to_process = list(subsets.values())
    elif args.subset in subsets:
        files_to_process = [subsets[args.subset]]
    else:
        print(f"Error: Unknown subset '{args.subset}'")
        print(f"Available: all, {', '.join(subsets.keys())}")
        return

    print(f"\nProcessing {len(files_to_process)} subset(s)...")

    # Load model and tokenizer (shared across all subsets)
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIG['model_name'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    device = str(model.device)
    print(f"✓ Model loaded on {device}")

    # Load probes for all layers (shared across all subsets)
    print(f"\nLoading diverse isolation probes...")
    user_probes = {}
    asst_probes = {}

    for layer in tqdm(layers, desc="Loading probes"):
        user_probes[layer] = load_diverse_isolation_probe(
            probe_dir, 'user', layer, lambda_reg, device
        )
        asst_probes[layer] = load_diverse_isolation_probe(
            probe_dir, 'assistant', layer, lambda_reg, device
        )

    print(f"✓ Loaded {len(layers)} layers for both user and assistant")

    # Compute baseline statistics from actual data (shared across all subsets)
    print("\nComputing baseline statistics...")
    baseline_stats = compute_baseline_statistics_from_data(
        user_probes=user_probes,
        asst_probes=asst_probes,
        layers=layers,
        baseline_dir=BASELINE_CONFIG['baseline_dir'],
        device=device,
        num_samples=500
    )
    print("✓ Baseline statistics computed")

    # Process each subset file
    for filename in files_to_process:
        file_path = data_dir / filename
        if not file_path.exists():
            print(f"\n⚠ Skipping {filename} (not found)")
            continue

        process_subset_file(
            data_path=str(file_path),
            probe_key=probe_key,
            probe_config=probe_config,
            user_probes=user_probes,
            asst_probes=asst_probes,
            baseline_stats=baseline_stats,
            model=model,
            tokenizer=tokenizer,
            device=device,
            layers=layers
        )

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    print(f"\nDiverse isolation probes (λ={lambda_reg}, layers {layers[0]}-{layers[-1]}) have been added.")
    print(f"You can now view them in the dashboard by selecting '{probe_config['display_name']}'.")
    print(f"\nTo launch the dashboard:")
    print(f"  cd eval_dashboard")
    print(f"  streamlit run app.py")


if __name__ == '__main__':
    main()
