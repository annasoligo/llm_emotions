"""
Add per-layer probe scores to existing pickle files.

Applies trained emotion probes (text_raw and text_cpca) to each layer,
storing per-layer emotion scores for layerwise visualization.

This complements the logit lens preprocessing by adding:
- text_raw_by_layer: Per-layer scores from raw activation probes
- text_cpca_by_layer: Per-layer scores from cPCA-transformed probes
"""
import pickle
import numpy as np
from pathlib import Path
import sys
import torch
import argparse
import time

# Add paths
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not")
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import activation extraction
from emotion_evals.emo_lens.token_trajectories import extract_token_level_activations
from emotion_evals.emo_lens.model_utils import load_base_model

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
EMOTIONS_WITH_NEUTRAL = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']

# Probe configurations
PROBE_CONFIGS = {
    'text_raw': {
        'probe_pattern': 'probe_layer{layer}_nc0_seed0.pkl',
        'use_cpca': False,
        'n_components': 0,
        'display_name': 'Text Raw'
    },
    'text_cpca': {
        'probe_pattern': 'probe_layer{layer}_nc10_seed0.pkl',
        'use_cpca': True,
        'n_components': 10,
        'display_name': 'Text cPCA-10'
    }
}

PROBE_DIR = Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/text_based/multiseed")
CPCA_PATH = Path("/workspace-vast/annas/git/research-tools/probes/results/cpca_tier_data_high_alpha.tmp/google/gemma-3-27b-it_cpca.npz")


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
        else:
            print(f"    Warning: Probe not found for layer {layer}: {probe_file}")
    return probes


def load_cpca_components(cpca_path: Path, n_components: int = 10):
    """
    Load cPCA transformation components.

    Args:
        cpca_path: Path to cPCA .npz file
        n_components: Number of components to use

    Returns:
        numpy array of shape (num_layers, n_components, hidden_dim)
    """
    cpca_data = np.load(cpca_path)
    components = cpca_data['components']  # Shape: (62, 50, 5376)
    return components[:, :n_components, :]  # Return only needed components


def apply_probes_batched(
    activations_by_token: dict,
    probes: dict,
    layers: list,
    cpca_components: np.ndarray = None,
    device: str = 'cuda',
    batch_size: int = 2048
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
        Tuple of:
        - raw_scores: Dict[token_idx][layer] = raw probe scores (6 emotions)
        - softmax_scores: Dict[token_idx][layer] = softmax probabilities (6 emotions)
    """
    # Collect all (token, layer) pairs
    batch_items = []
    for token_idx, layer_dict in activations_by_token.items():
        for layer in layers:
            if layer in layer_dict and layer in probes:
                batch_items.append((token_idx, layer, layer_dict[layer]))

    if not batch_items:
        return {}, {}

    # Prepare probes on device
    probes_on_device = {}
    for layer, probe in probes.items():
        probes_on_device[layer] = probe.to(device).eval()

    # Prepare cPCA components on device if needed
    cpca_tensor = None
    if cpca_components is not None:
        cpca_tensor = torch.from_numpy(cpca_components).float().to(device)

    # Process in batches - store both raw and softmax
    raw_scores_dict = {}
    softmax_scores_dict = {}
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
                # cpca_tensor[layer] shape: (n_components, hidden_dim)
                # activations_tensor shape: (batch, hidden_dim)
                # Result: (batch, n_components)
                activations_tensor = activations_tensor @ cpca_tensor[layer].T

            # Apply probe
            with torch.no_grad():
                logits = probe(activations_tensor)  # (batch, 7) - includes neutral
                # Extract only 6 emotions (exclude neutral at index 6)
                raw_scores = logits[:, :6].cpu().numpy()
                # Also compute softmax probabilities
                probs = torch.softmax(logits, dim=-1)
                softmax_probs = probs[:, :6].cpu().numpy()

            # Store results (both raw and softmax)
            for idx, (token_idx, _) in enumerate(items):
                if token_idx not in raw_scores_dict:
                    raw_scores_dict[token_idx] = {}
                    softmax_scores_dict[token_idx] = {}
                raw_scores_dict[token_idx][layer] = raw_scores[idx]
                softmax_scores_dict[token_idx][layer] = softmax_probs[idx]

    return raw_scores_dict, softmax_scores_dict


def process_conversation(
    conv: dict,
    model,
    tokenizer,
    probes_raw: dict,
    probes_cpca: dict,
    cpca_components: np.ndarray,
    layers: list,
    device: str = 'cuda'
):
    """
    Process a single conversation to get per-layer probe scores.

    Args:
        conv: Conversation dict with 'conversation' and 'sentences' keys
        model: Loaded model for activation extraction
        tokenizer: Tokenizer
        probes_raw: Dict of raw probes by layer
        probes_cpca: Dict of cPCA probes by layer
        cpca_components: cPCA transformation components
        layers: List of layers to process
        device: 'cuda' or 'cpu'

    Returns:
        Tuple of (text_raw_by_layer, text_raw_softmax_by_layer,
                  text_cpca_by_layer, text_cpca_softmax_by_layer) dicts
    """
    # Build conversation text
    conversation_text = ""
    for turn in conv['conversation']:
        role = turn['role']
        content = turn['content']
        if role == 'user':
            conversation_text += f"User: {content}\n"
        else:
            conversation_text += f"Assistant: {content}\n"

    # Extract activations for all layers
    activations_by_token, token_ids = extract_token_level_activations(
        model=model,
        tokenizer=tokenizer,
        prompt=conversation_text,
        layers=layers,
        start_token_idx=0,
        system_prompt=None,
        num_generated_tokens=0
    )

    # Apply raw probes - get both raw scores and softmax probabilities
    scores_raw, softmax_raw = apply_probes_batched(
        activations_by_token, probes_raw, layers,
        cpca_components=None, device=device
    )

    # Apply cPCA probes - get both raw scores and softmax probabilities
    scores_cpca, softmax_cpca = apply_probes_batched(
        activations_by_token, probes_cpca, layers,
        cpca_components=cpca_components, device=device
    )

    # Aggregate to sentence level (all four variants)
    text_raw_by_layer = {}
    text_raw_softmax_by_layer = {}
    text_cpca_by_layer = {}
    text_cpca_softmax_by_layer = {}

    for sent in conv['sentences']:
        sent_id = sent['sentence_id']
        start_tok = sent['start_token']
        end_tok = sent['end_token']

        # Initialize per-layer storage for this sentence
        text_raw_by_layer[sent_id] = {}
        text_raw_softmax_by_layer[sent_id] = {}
        text_cpca_by_layer[sent_id] = {}
        text_cpca_softmax_by_layer[sent_id] = {}

        for layer in layers:
            # Collect scores for tokens in this sentence
            raw_scores_list = []
            raw_softmax_list = []
            cpca_scores_list = []
            cpca_softmax_list = []

            for t in range(start_tok, end_tok):
                if t in scores_raw and layer in scores_raw[t]:
                    raw_scores_list.append(scores_raw[t][layer])
                if t in softmax_raw and layer in softmax_raw[t]:
                    raw_softmax_list.append(softmax_raw[t][layer])
                if t in scores_cpca and layer in scores_cpca[t]:
                    cpca_scores_list.append(scores_cpca[t][layer])
                if t in softmax_cpca and layer in softmax_cpca[t]:
                    cpca_softmax_list.append(softmax_cpca[t][layer])

            # Average within sentence
            if raw_scores_list:
                text_raw_by_layer[sent_id][layer] = np.mean(raw_scores_list, axis=0).tolist()
            if raw_softmax_list:
                text_raw_softmax_by_layer[sent_id][layer] = np.mean(raw_softmax_list, axis=0).tolist()
            if cpca_scores_list:
                text_cpca_by_layer[sent_id][layer] = np.mean(cpca_scores_list, axis=0).tolist()
            if cpca_softmax_list:
                text_cpca_softmax_by_layer[sent_id][layer] = np.mean(cpca_softmax_list, axis=0).tolist()

    return text_raw_by_layer, text_raw_softmax_by_layer, text_cpca_by_layer, text_cpca_softmax_by_layer


def main():
    parser = argparse.ArgumentParser(description="Add per-layer probe scores to pickle")
    parser.add_argument("--input", required=True, help="Input pickle file")
    parser.add_argument("--output", required=True, help="Output pickle file")
    parser.add_argument("--test", action="store_true", help="Test mode (5 conversations)")
    args = parser.parse_args()

    script_start = time.time()

    print("=" * 80)
    print("ADD PER-LAYER PROBE SCORES")
    print("=" * 80)

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")

    # Load existing data
    print("\n[1/5] Loading existing data...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    total_convs = len(data['conversations'])
    num_to_process = 5 if args.test else total_convs
    print(f"  Total conversations: {total_convs}")
    if args.test:
        print(f"  TEST MODE: Processing only first 5 conversations")

    # Load model
    print("\n[2/5] Loading model...")
    BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
    model, tokenizer = load_base_model(BASE_MODEL_NAME)
    device = next(model.parameters()).device
    print(f"  ✓ Model loaded on {device}")

    # Detect layers
    try:
        num_layers = len(model.model.layers)
        layers = list(range(num_layers))
        print(f"  Detected {num_layers} layers")
    except:
        layers = list(range(62))
        print(f"  Using default 62 layers")

    # Load probes
    print("\n[3/5] Loading probes...")

    print("  Loading text_raw probes (nc0)...")
    probes_raw = load_probes_for_all_layers(
        PROBE_CONFIGS['text_raw']['probe_pattern'],
        num_layers=len(layers)
    )
    print(f"    ✓ Loaded {len(probes_raw)} layer probes")

    print("  Loading text_cpca probes (nc10)...")
    probes_cpca = load_probes_for_all_layers(
        PROBE_CONFIGS['text_cpca']['probe_pattern'],
        num_layers=len(layers)
    )
    print(f"    ✓ Loaded {len(probes_cpca)} layer probes")

    print("  Loading cPCA components...")
    cpca_components = load_cpca_components(
        CPCA_PATH,
        n_components=PROBE_CONFIGS['text_cpca']['n_components']
    )
    print(f"    ✓ cPCA components shape: {cpca_components.shape}")

    # Process conversations
    print(f"\n[4/5] Processing {num_to_process} conversations...")
    overall_start = time.time()

    for i in range(num_to_process):
        conv_start = time.time()
        conv = data['conversations'][i]
        print(f"\n  [{i+1}/{num_to_process}] Sample ID: {conv['sample_id']}")

        # Process conversation - get both raw scores and softmax probabilities
        text_raw_by_layer, text_raw_softmax_by_layer, text_cpca_by_layer, text_cpca_softmax_by_layer = process_conversation(
            conv, model, tokenizer,
            probes_raw, probes_cpca, cpca_components,
            layers, device=str(device)
        )

        # Store all four variants in conversation
        conv['text_raw_by_layer'] = text_raw_by_layer
        conv['text_raw_softmax_by_layer'] = text_raw_softmax_by_layer
        conv['text_cpca_by_layer'] = text_cpca_by_layer
        conv['text_cpca_softmax_by_layer'] = text_cpca_softmax_by_layer

        # Progress
        conv_time = time.time() - conv_start
        avg_time = (time.time() - overall_start) / (i + 1)
        eta = avg_time * (num_to_process - i - 1)
        print(f"    ✓ Processed {len(conv['sentences'])} sentences in {conv_time:.2f}s")
        print(f"      (avg: {avg_time:.1f}s/conv, ETA: {eta/60:.1f}m)")

    # Save
    print(f"\n[5/5] Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved ({file_size_mb:.1f} MB)")

    total_time = time.time() - script_start
    print("\n" + "=" * 80)
    print("✓ PER-LAYER PROBE PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Processed {num_to_process} conversations with {len(layers)} layers")
    print(f"Added: text_raw_by_layer, text_raw_softmax_by_layer, text_cpca_by_layer, text_cpca_softmax_by_layer")


if __name__ == '__main__':
    main()
