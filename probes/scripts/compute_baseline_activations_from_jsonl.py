#!/usr/bin/env python3
"""
Compute baseline activations from a JSONL file of conversations.

This script extracts activations from your local model using conversations
generated via OpenRouter or other sources.

Usage:
    python compute_baseline_activations_from_jsonl.py \\
        --input data/alpaca_responses.jsonl \\
        --output data/baselines/alpaca_gemma27b/ \\
        --model-name unsloth/gemma-3-27b-it \\
        --layers 0-62
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import h5py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from nnterp import StandardizedTransformer


def load_conversations(jsonl_path: Path) -> List[Dict[str, str]]:
    """Load conversations from JSONL file."""
    conversations = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            conversations.append({
                "user": data["user"],
                "assistant": data["assistant"]
            })
    return conversations


def extract_conversation_activations(
    model: StandardizedTransformer,
    tokenizer,
    conversation: Dict[str, str],
    layers: List[int],
    system_prompt: str = None
) -> Dict[int, Dict[str, np.ndarray]]:
    """Extract activations for a single conversation at ALL layers in ONE forward pass.

    Returns:
        Dict mapping layer -> aggregation type -> activation
        {
            layer_20: {
                'all_tokens': ndarray [hidden_dim],
                'user_turn': ndarray [hidden_dim],
                'assistant_turn': ndarray [hidden_dim],
                ...
            },
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

    # Find token boundaries for user and assistant turns
    # This is approximate - we'll look for role markers in the formatted text
    tokens_str = tokenizer.batch_decode(input_ids[0])

    # Simple heuristic: find where assistant turn starts
    # (This works for most chat templates)
    user_end_idx = len(input_ids[0]) // 2  # Rough midpoint
    asst_start_idx = user_end_idx

    # Try to find a better boundary by looking for common markers
    for i in range(1, len(input_ids[0]) - 1):
        token = tokens_str[i]
        if 'model' in token.lower() or 'assistant' in token.lower():
            asst_start_idx = i
            break

    # Extract activations for ALL layers in ONE forward pass
    layer_outputs = {}
    with torch.no_grad():
        with model.trace(input_ids, scan=False):
            for layer in layers:
                layer_output = model.layers_output[layer].save()  # [1, seq_len, hidden_dim]

                # Compute aggregations
                activations = layer_output[0].cpu().numpy()  # [seq_len, hidden_dim]

                # Different aggregation types
                aggregations = {}

                # 1. All tokens (mean over all positions)
                aggregations['all_tokens'] = np.mean(activations, axis=0)

                # 2. User turn (mean over user tokens)
                user_tokens = activations[1:asst_start_idx]  # Skip BOS
                if len(user_tokens) > 0:
                    aggregations['user_turn'] = np.mean(user_tokens, axis=0)
                else:
                    aggregations['user_turn'] = aggregations['all_tokens']

                # 3. Assistant turn (mean over assistant tokens)
                asst_tokens = activations[asst_start_idx:]
                if len(asst_tokens) > 0:
                    aggregations['assistant_turn'] = np.mean(asst_tokens, axis=0)
                else:
                    aggregations['assistant_turn'] = aggregations['all_tokens']

                # 4. Last user token
                if asst_start_idx > 1:
                    aggregations['last_user_token'] = activations[asst_start_idx - 1]
                else:
                    aggregations['last_user_token'] = aggregations['user_turn']

                # 5. First assistant token
                if asst_start_idx < len(activations):
                    aggregations['first_assistant_token'] = activations[asst_start_idx]
                else:
                    aggregations['first_assistant_token'] = aggregations['assistant_turn']

                # 6. Between turns (average of last user and first assistant)
                aggregations['between_turns'] = (
                    aggregations['last_user_token'] + aggregations['first_assistant_token']
                ) / 2

                layer_outputs[layer] = aggregations

    return layer_outputs


def main():
    parser = argparse.ArgumentParser(description="Compute baseline activations from JSONL")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--model-name", type=str, default="unsloth/gemma-3-27b-it",
                        help="Model name or path")
    parser.add_argument("--layers", type=str, default="0-62",
                        help="Layer range (e.g., '0-62' or '20,30,40')")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Optional system prompt to add")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    # Parse layers
    if '-' in args.layers:
        start, end = map(int, args.layers.split('-'))
        layers = list(range(start, end + 1))
    else:
        layers = [int(x) for x in args.layers.split(',')]

    print("="*80)
    print("COMPUTING BASELINE ACTIVATIONS FROM JSONL")
    print("="*80)
    print(f"  Input: {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Model: {args.model_name}")
    print(f"  Layers: {len(layers)} layers ({min(layers)}-{max(layers)})")
    print(f"  Device: {args.device}")
    print()

    # Load conversations
    print("Loading conversations...")
    input_path = Path(args.input)
    conversations = load_conversations(input_path)
    print(f"✓ Loaded {len(conversations)} conversations")
    print()

    # Load model
    print("Loading model...")
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print("  Loading raw model with device_map='auto'...")
    model_raw = AutoModelForCausalLM.from_pretrained(
        args.model_name,
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

    model.eval()
    print("✓ Model loaded")
    print()

    # Initialize storage for activations
    # Structure: {layer: {aggregation_type: [list of activations]}}
    all_activations = {
        layer: {
            'all_tokens': [],
            'user_turn': [],
            'assistant_turn': [],
            'last_user_token': [],
            'first_assistant_token': [],
            'between_turns': []
        }
        for layer in layers
    }

    # Extract activations
    print("Extracting activations...")
    for conversation in tqdm(conversations, desc="Processing conversations"):
        try:
            layer_activations = extract_conversation_activations(
                model=model,
                tokenizer=tokenizer,
                conversation=conversation,
                layers=layers,
                system_prompt=args.system_prompt
            )

            # Accumulate activations
            for layer in layers:
                for agg_type, activation in layer_activations[layer].items():
                    all_activations[layer][agg_type].append(activation)

        except Exception as e:
            print(f"Error processing conversation: {e}")
            continue

    print("✓ Activation extraction complete")
    print()

    # Save to HDF5 files (one per layer)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving activations to HDF5...")
    for layer in tqdm(layers, desc="Saving layers"):
        h5_file = output_dir / f"layer{layer}_activations.h5"

        with h5py.File(h5_file, 'w') as f:
            for agg_type in all_activations[layer].keys():
                activations_list = all_activations[layer][agg_type]

                if len(activations_list) == 0:
                    continue

                # Stack into array [n_samples, hidden_dim]
                activations_array = np.stack(activations_list, axis=0)

                # Save to HDF5
                f.create_dataset(
                    agg_type,
                    data=activations_array,
                    compression="gzip",
                    compression_opts=4
                )

        # Save metadata
        stats = {}
        with h5py.File(h5_file, 'r') as f:
            aggregations_data = {}
            for key in f.keys():
                data = f[key][:]
                aggregations_data[key] = {
                    "mean": float(np.mean(data)),
                    "std": float(np.std(data)),
                    "shape": list(data.shape)
                }

            stats = {
                "model_name": args.model_name,
                "layer": layer,
                "num_samples": len(conversations),
                "source": "alpaca_gemma27b",
                "aggregations": aggregations_data
            }

        json_file = output_dir / f"layer{layer}_stats.json"
        with open(json_file, 'w') as f:
            json.dump(stats, f, indent=2)

    print(f"✓ Saved activations to {output_dir}")
    print()
    print("="*80)
    print("BASELINE ACTIVATIONS READY!")
    print("="*80)
    print(f"You can now use this baseline with:")
    print(f"  WildChatBaselineLoader(")
    print(f"      aggregation_type='assistant_turn',")
    print(f"      baseline_dir=Path('{output_dir}')")
    print(f"  )")
    print()


if __name__ == "__main__":
    main()
