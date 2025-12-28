#!/usr/bin/env python3
"""
Compute baseline activations from WildChat dataset.

This script extracts activations from neutral conversational data (WildChat)
and saves them in multiple aggregation formats for fast probe baseline computation.

Usage:
    python compute_wildchat_baseline_activations.py --layer 31 --num-samples 100
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from nnterp import StandardizedTransformer


def load_wildchat_cache(cache_path: Path, num_samples: int = None) -> List[Dict[str, str]]:
    """Load conversations from wildchat cache.

    Args:
        cache_path: Path to wildchat_cache.jsonl
        num_samples: Number of samples to load (None = all)

    Returns:
        List of conversations with 'user' and 'assistant' keys
    """
    conversations = []
    with open(cache_path, 'r') as f:
        for i, line in enumerate(f):
            if num_samples and i >= num_samples:
                break
            conversations.append(json.loads(line))

    print(f"✓ Loaded {len(conversations)} conversations from WildChat cache")
    return conversations


def extract_conversation_activations(
    model: StandardizedTransformer,
    tokenizer,
    conversation: Dict[str, str],
    layers: List[int],
    system_prompt: str = None
) -> Dict[int, Dict[str, np.ndarray]]:
    """Extract activations for a single conversation at ALL layers in ONE forward pass.

    Args:
        model: Model wrapped in StandardizedTransformer
        tokenizer: Tokenizer
        conversation: Dict with 'user' and 'assistant' keys
        layers: List of layer indices to extract
        system_prompt: Optional system prompt

    Returns:
        Dict mapping layer -> aggregation type -> activation
        {
            layer_20: {
                'all_tokens': ndarray,
                'user_turn': ndarray,
                ...
            },
            layer_31: {...},
            ...
        }
    """
    # Build chat messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": conversation["user"]})
    messages.append({"role": "assistant", "content": conversation["assistant"]})

    # Format with chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    # Extract activations for ALL layers in ONE forward pass
    layer_outputs = {}
    with torch.no_grad():
        with model.trace(input_ids, scan=False):
            for layer in layers:
                layer_outputs[layer] = model.layers_output[layer].save()

    # Process each layer's activations
    all_layer_results = {}

    # Find user turn end (before assistant response) - computed once for all layers
    formatted_user_only = tokenizer.apply_chat_template(
        messages[:-1],  # Without assistant message
        tokenize=False,
        add_generation_prompt=True
    )
    user_turn_end_pos = len(tokenizer(formatted_user_only)["input_ids"]) - 1

    # Skip first 20 tokens (formatting)
    SKIP_FIRST_N = 20

    for layer, layer_output in layer_outputs.items():
        # layer_output shape: [batch=1, seq_len, hidden_dim]
        activations = layer_output[0].detach().cpu().float().numpy()  # [seq_len, hidden_dim]

        start_pos = min(SKIP_FIRST_N, activations.shape[0])

        # Compute aggregations for this layer
        results = {}

        # All tokens (from position 20 onward)
        if start_pos < activations.shape[0]:
            results['all_tokens'] = activations[start_pos:].mean(axis=0)

        # User turn (from position 20 to user_turn_end)
        user_start = start_pos
        user_end = min(user_turn_end_pos + 1, activations.shape[0])
        if user_start < user_end:
            results['user_turn'] = activations[user_start:user_end].mean(axis=0)

        # Assistant turn (from user_turn_end onward)
        asst_start = user_turn_end_pos + 1
        if asst_start < activations.shape[0]:
            results['assistant_turn'] = activations[asst_start:].mean(axis=0)

        # Special positions
        if user_turn_end_pos < activations.shape[0]:
            results['last_user_token'] = activations[user_turn_end_pos]

        if user_turn_end_pos + 1 < activations.shape[0]:
            results['first_assistant_token'] = activations[user_turn_end_pos + 1]

        # Between turns average
        if 'last_user_token' in results and 'first_assistant_token' in results:
            results['between_turns'] = (results['last_user_token'] + results['first_assistant_token']) / 2

        all_layer_results[layer] = results

    return all_layer_results


def compute_baseline_activations(
    conversations: List[Dict[str, str]],
    model: StandardizedTransformer,
    tokenizer,
    layers: List[int],
    output_dir: Path
):
    """Extract activations for all conversations at ALL layers and save to HDF5.

    Saves one file per layer with multiple aggregation formats:
    - all_tokens: [n_samples, hidden_dim]
    - user_turn: [n_samples, hidden_dim]
    - assistant_turn: [n_samples, hidden_dim]
    - last_user_token: [n_samples, hidden_dim]
    - first_assistant_token: [n_samples, hidden_dim]
    - between_turns: [n_samples, hidden_dim]
    """
    print(f"\nExtracting activations from {len(conversations)} conversations...")
    print(f"  Layers: {layers}")
    print(f"  Model: {getattr(model, 'model_id', 'unknown')}")
    print(f"  Using SINGLE forward pass per conversation (efficient!)")

    # Collect activations per layer
    layer_results = {layer: {
        'all_tokens': [],
        'user_turn': [],
        'assistant_turn': [],
        'last_user_token': [],
        'first_assistant_token': [],
        'between_turns': []
    } for layer in layers}

    for conv in tqdm(conversations, desc="Extracting activations"):
        # Extract ALL layers in ONE forward pass
        all_layer_acts = extract_conversation_activations(
            model=model,
            tokenizer=tokenizer,
            conversation=conv,
            layers=layers
        )

        # Organize by layer
        for layer, results in all_layer_acts.items():
            for key in layer_results[layer].keys():
                if key in results:
                    layer_results[layer][key].append(results[key])

        # Clear CUDA cache to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save one file per layer
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        output_path = output_dir / f"layer{layer}_activations.h5"
        print(f"\nSaving layer {layer} to {output_path}")

        with h5py.File(output_path, 'w') as f:
            # Save metadata
            f.attrs['model_name'] = getattr(model, 'model_id', 'google/gemma-3-27b-it')
            f.attrs['layer'] = layer
            f.attrs['num_samples'] = len(conversations)
            f.attrs['skip_first_n_tokens'] = 20
            f.attrs['source'] = 'wildchat'

            # Save each aggregation type
            for key, acts in layer_results[layer].items():
                if len(acts) > 0:
                    acts_array = np.stack(acts, axis=0)  # [n_samples, hidden_dim]
                    f.create_dataset(key, data=acts_array, compression='gzip')
                    print(f"  ✓ Saved '{key}': shape {acts_array.shape}")
                else:
                    print(f"  ⚠ Skipped '{key}': no data")

    print(f"\n✓ All {len(layers)} layers saved to {output_dir}")


def compute_summary_statistics(h5_path: Path, output_json_path: Path):
    """Compute mean/std statistics from saved activations and save to JSON.

    This creates a lightweight summary file with just statistics,
    useful for quick baseline comparisons.
    """
    print(f"\nComputing summary statistics...")

    with h5py.File(h5_path, 'r') as f:
        stats = {
            'model_name': f.attrs['model_name'],
            'layer': int(f.attrs['layer']),
            'num_samples': int(f.attrs['num_samples']),
            'source': f.attrs['source'],
            'aggregations': {}
        }

        for key in f.keys():
            acts = f[key][:]  # [n_samples, hidden_dim]
            stats['aggregations'][key] = {
                'mean': float(acts.mean()),
                'std': float(acts.std()),
                'shape': list(acts.shape)
            }
            print(f"  {key}: mean={acts.mean():.3f}, std={acts.std():.3f}")

    # Save to JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ Summary statistics saved to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute WildChat baseline activations for ALL layers")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it",
                       help="Model name or path")
    parser.add_argument("--layers", type=str, default=None,
                       help="Comma-separated layer indices (e.g., '20,31,39,50,60') or 'all' for all layers")
    parser.add_argument("--num-samples", type=int, default=None,
                       help="Number of samples to process (default: all 512)")
    parser.add_argument("--wildchat-cache", type=str,
                       default="/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens/wildchat_cache.jsonl",
                       help="Path to wildchat cache file")
    parser.add_argument("--output-dir", type=str,
                       default="/workspace-vast/annas/git/research-tools/data/baselines/wildchat",
                       help="Output directory for baseline activations")

    args = parser.parse_args()

    # Load wildchat conversations
    wildchat_cache = Path(args.wildchat_cache)
    if not wildchat_cache.exists():
        raise FileNotFoundError(f"WildChat cache not found: {wildchat_cache}")

    conversations = load_wildchat_cache(wildchat_cache, args.num_samples)

    # Load model
    print(f"\nLoading model: {args.model}")
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("  Loading raw model with device_map='auto'...")
    model_raw = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("  Wrapping in StandardizedTransformer...")
    model = StandardizedTransformer(
        model_raw,
        tokenizer=tokenizer,
        trust_remote_code=True,
        check_renaming=False,
        allow_dispatch=True
    )

    # Handle tuple return
    if isinstance(model, tuple):
        model = model[0]

    print(f"✓ Model loaded: {model.num_layers} layers")

    # Parse layers argument
    if args.layers is None or args.layers.lower() == 'all':
        layers = list(range(model.num_layers))
        print(f"  Using ALL {len(layers)} layers")
    else:
        layers = [int(x.strip()) for x in args.layers.split(',')]
        print(f"  Using {len(layers)} layers: {layers}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    model_safe_name = args.model.replace("/", "_").replace("-", "_")
    output_dir = output_dir / model_safe_name

    # Extract and save activations
    compute_baseline_activations(
        conversations=conversations,
        model=model,
        tokenizer=tokenizer,
        layers=layers,
        output_dir=output_dir
    )

    # Compute summary statistics for each layer
    print("\n" + "="*80)
    print("COMPUTING SUMMARY STATISTICS")
    print("="*80)

    for layer in layers:
        h5_path = output_dir / f"layer{layer}_activations.h5"
        json_path = output_dir / f"layer{layer}_stats.json"
        compute_summary_statistics(h5_path, json_path)

    print("\n" + "="*80)
    print("BASELINE ACTIVATIONS COMPLETE")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Layers processed: {len(layers)}")
    print(f"\nFiles created:")
    print(f"  - layer{{N}}_activations.h5  (activation arrays)")
    print(f"  - layer{{N}}_stats.json      (summary statistics)")
    print(f"\nTo use these baselines with probes:")
    print(f"  python probes/scripts/compute_probe_baselines_from_wildchat.py \\")
    print(f"      --layer <layer> --ortho-weight <weight>")


if __name__ == "__main__":
    main()
