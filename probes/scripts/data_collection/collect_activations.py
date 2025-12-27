#!/usr/bin/env python3
"""Collect activations from model using nnterp - supports texts and conversations.

For TEXTS:
- DO NOT chat tokenize
- Collect activations averaged over all tokens after the 20th

For CONVERSATIONS:
- MUST chat tokenize
- Collect activations in 3 ways:
  1. Regional: Averaged over each turn and section of special tokens separately
  2. Global: Averaged over all tokens after the 20th
  3. Special tokens: User/assistant text with each special token activation collected separately

Usage:
    # For text pairs
    python scripts/collect_activations_v2.py \\
        --input data/pairs.jsonl \\
        --output activations/pairs.h5 \\
        --model google/gemma-2-9b-it \\
        --data_type texts

    # For conversations
    python scripts/collect_activations_v2.py \\
        --input data/conversations.jsonl \\
        --output activations/conversations.h5 \\
        --model google/gemma-2-9b-it \\
        --data_type conversations
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

from probes.core import multiturn, load_jsonl

# ============================================================================
# Text Collection (NO chat tokenization)
# ============================================================================

def collect_text_activations(
    model,
    tokenizer,
    text: str,
    device: str,
    start_token: int = 20,
) -> Optional[np.ndarray]:
    """Collect activations for single text, averaged after start_token.

    Args:
        model: nnterp StandardizedTransformer
        tokenizer: HuggingFace tokenizer
        text: Text string
        device: Device model is on
        start_token: Start averaging from this token index

    Returns:
        [num_layers, hidden_size] averaged activations, or None if text too short

    Raises:
        RuntimeError: If collection fails
    """
    # Tokenize WITHOUT chat template
    inputs = tokenizer(text, return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].shape[1]

    # Skip if too short - use min() to handle gracefully
    if seq_len <= start_token:
        return None

    # Use flexible start position
    actual_start = min(start_token, seq_len)

    # Collect activations using nnterp trace
    activations = []

    with torch.no_grad():
        with model.trace(inputs):
            for layer_idx in range(model.num_layers):
                layer_output = model.layers_output[layer_idx].save()
                activations.append(layer_output)

    # Extract and average from actual_start onwards for each layer
    layer_activations = []
    for layer_act in activations:
        # layer_act shape: [1, seq_len, hidden_size]
        # Average over tokens from actual_start onwards
        averaged = layer_act[0, actual_start:, :].mean(dim=0)
        layer_activations.append(averaged)

    # Stack and convert: [num_layers, hidden_size]
    all_layers = torch.stack(layer_activations)
    return all_layers.detach().cpu().to(torch.float32).numpy()


# ============================================================================
# Conversation Collection (WITH chat tokenization + regional extraction)
# ============================================================================

def collect_conversation_activations(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    device: str,
) -> Dict[str, np.ndarray]:
    """Collect activations for conversation with regional extraction.

    Returns 3 types:
    1. Regional: Per-turn and special tokens (averaged separately)
    2. Global: All tokens after 20th (averaged)
    3. Special tokens: Each special token separately

    Args:
        model: nnterp StandardizedTransformer
        tokenizer: HuggingFace tokenizer
        messages: List of message dicts with 'role' and 'content'
        device: Device model is on

    Returns:
        Dict with keys:
        - 'regional': Dict[region_name, [num_layers, hidden_size]]
        - 'global': [num_layers, hidden_size]
        - 'special_tokens': Dict[token_name, [num_layers, hidden_size]]

    Raises:
        RuntimeError: If collection fails
        ValueError: If conversation format invalid
    """
    try:
        # Apply chat template
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].cpu().numpy()
        seq_len = len(token_ids)

        # Collect all token activations
        all_activations = []

        with torch.no_grad():
            with model.trace(inputs, scan=False):
                for layer_idx in range(model.num_layers):
                    # [1, seq_len, hidden_size]
                    layer_output = model.layers_output[layer_idx].save()
                    # [seq_len, hidden_size]
                    all_activations.append(layer_output[0])

        # Stack and convert OUTSIDE the with blocks: [num_layers, seq_len, hidden_size]
        all_acts = torch.stack(all_activations).float().cpu().numpy()

        # 1. Regional extraction
        regional_acts = extract_regional_activations(
            all_acts, token_ids, tokenizer
        )

        # 2. Global (average after token 20)
        global_start = min(20, seq_len - 1)
        # [num_layers, hidden_size]
        global_acts = all_acts[:, global_start:, :].mean(axis=1)

        # 3. Special tokens
        special_token_acts = extract_special_token_activations(
            all_acts, token_ids, tokenizer
        )

        return {
            "regional": regional_acts,
            "global": global_acts,
            "special_tokens": special_token_acts,
        }

    except Exception as e:
        raise RuntimeError(f"Conversation collection failed: {e}")


def extract_regional_activations(
    activations: np.ndarray,
    token_ids: np.ndarray,
    tokenizer,
) -> Dict[str, np.ndarray]:
    """Extract regional activations (per-turn averaged).

    Args:
        activations: [num_layers, seq_len, hidden_size]
        token_ids: [seq_len]
        tokenizer: Tokenizer (for special token IDs)

    Returns:
        Dict mapping region_name -> [num_layers, hidden_size]
    """
    num_layers = activations.shape[0]

    # Get special token IDs
    special_ids = {}
    if hasattr(tokenizer, 'convert_tokens_to_ids'):
        special_ids['start_of_turn'] = tokenizer.convert_tokens_to_ids('<start_of_turn>')
        special_ids['end_of_turn'] = tokenizer.convert_tokens_to_ids('<end_of_turn>')
    else:
        # Fallback: search token vocab
        for token, token_id in tokenizer.get_vocab().items():
            if 'start_of_turn' in token.lower():
                special_ids['start_of_turn'] = token_id
            elif 'end_of_turn' in token.lower():
                special_ids['end_of_turn'] = token_id

    if 'start_of_turn' not in special_ids or 'end_of_turn' not in special_ids:
        # No special tokens - treat as single region
        return {
            "full_conversation": activations.mean(axis=1)  # [num_layers, hidden_size]
        }

    # Process each layer independently
    regional_by_layer = {}

    for layer_idx in range(num_layers):
        # [seq_len, hidden_size]
        layer_acts = activations[layer_idx]

        # Split into regions
        try:
            regions = multiturn.split_regional_activations(
                layer_acts,
                token_ids,
                special_ids,
            )

            # Pool each region
            pooled_regions = multiturn.pool_regional_activations(
                regions,
                pooling="mean",
            )

            # Store
            for region_name, pooled_act in pooled_regions.items():
                if region_name not in regional_by_layer:
                    regional_by_layer[region_name] = []
                regional_by_layer[region_name].append(pooled_act)

        except (ValueError, KeyError) as e:
            # Fallback: use full sequence
            if "full_conversation" not in regional_by_layer:
                regional_by_layer["full_conversation"] = []
            regional_by_layer["full_conversation"].append(layer_acts.mean(axis=0))

    # Stack layers: {region_name: [num_layers, hidden_size]}
    regional_final = {}
    for region_name, layer_list in regional_by_layer.items():
        regional_final[region_name] = np.stack(layer_list)

    return regional_final


def extract_special_token_activations(
    activations: np.ndarray,
    token_ids: np.ndarray,
    tokenizer,
) -> Dict[str, np.ndarray]:
    """Extract activations for each special token.

    Args:
        activations: [num_layers, seq_len, hidden_size]
        token_ids: [seq_len]
        tokenizer: Tokenizer

    Returns:
        Dict mapping token_name -> [num_layers, hidden_size]
    """
    special_tokens = {}

    # Common special tokens to extract
    token_patterns = {
        'start_of_turn': ['<start_of_turn>', '<|start_of_turn|>'],
        'end_of_turn': ['<end_of_turn>', '<|end_of_turn|>'],
        'bos': ['<bos>', '<s>', '<|begin_of_text|>'],
        'eos': ['<eos>', '</s>', '<|end_of_text|>'],
    }

    token_id_map = {}
    for name, patterns in token_patterns.items():
        for pattern in patterns:
            try:
                if hasattr(tokenizer, 'convert_tokens_to_ids'):
                    token_id = tokenizer.convert_tokens_to_ids(pattern)
                    if token_id != tokenizer.unk_token_id:
                        token_id_map[name] = token_id
                        break
            except:
                continue

    # Extract activations for each special token (first occurrence)
    for token_name, token_id in token_id_map.items():
        indices = np.where(token_ids == token_id)[0]
        if len(indices) > 0:
            # Take first occurrence: [num_layers, hidden_size]
            special_tokens[token_name] = activations[:, indices[0], :]

    return special_tokens


# ============================================================================
# HDF5 Save/Load
# ============================================================================

def save_text_activations_hdf5(
    activations: Dict[str, Dict[str, np.ndarray]],
    metadata: List[Dict],
    output_path: Path,
    model_name: str,
):
    """Save text pair activations to HDF5.

    Args:
        activations: {pair_id: {'neutral': [layers, hidden], 'emotional': [layers, hidden]}}
        metadata: List of pair metadata dicts
        output_path: Output HDF5 path
        model_name: Model identifier

    Raises:
        OSError: If cannot write file
        ValueError: If data invalid
    """
    if not activations:
        raise ValueError("activations dict is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(output_path, 'w') as f:
            # Store activations
            acts_group = f.create_group('activations')

            for pair_id, pair_acts in activations.items():
                pair_group = acts_group.create_group(str(pair_id))
                pair_group.create_dataset('neutral', data=pair_acts['neutral'])
                pair_group.create_dataset('emotional', data=pair_acts['emotional'])

            # Store metadata
            f.create_dataset('metadata', data=json.dumps(metadata))

            # Store attributes
            f.attrs['model_name'] = model_name
            f.attrs['data_type'] = 'texts'
            f.attrs['position'] = 'averaged_after_20'
            f.attrs['num_pairs'] = len(activations)

    except OSError as e:
        raise OSError(f"Failed to write HDF5 file: {e}")


def save_conversation_activations_hdf5(
    activations: Dict[str, Dict[str, Dict]],
    metadata: List[Dict],
    output_path: Path,
    model_name: str,
):
    """Save conversation activations to HDF5 with regional structure.

    Args:
        activations: {conv_id: {'regional': {...}, 'global': [...], 'special_tokens': {...}}}
        metadata: List of conversation metadata dicts
        output_path: Output HDF5 path
        model_name: Model identifier

    Raises:
        OSError: If cannot write file
        ValueError: If data invalid
    """
    if not activations:
        raise ValueError("activations dict is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(output_path, 'w') as f:
            acts_group = f.create_group('activations')

            for conv_id, conv_acts in activations.items():
                conv_group = acts_group.create_group(str(conv_id))

                # Regional
                if 'regional' in conv_acts:
                    regional_group = conv_group.create_group('regional')
                    for region_name, region_acts in conv_acts['regional'].items():
                        regional_group.create_dataset(region_name, data=region_acts)

                # Global
                if 'global' in conv_acts:
                    conv_group.create_dataset('global', data=conv_acts['global'])

                # Special tokens
                if 'special_tokens' in conv_acts:
                    special_group = conv_group.create_group('special_tokens')
                    for token_name, token_acts in conv_acts['special_tokens'].items():
                        special_group.create_dataset(token_name, data=token_acts)

            # Metadata
            f.create_dataset('metadata', data=json.dumps(metadata))

            # Attributes
            f.attrs['model_name'] = model_name
            f.attrs['data_type'] = 'conversations'
            f.attrs['num_conversations'] = len(activations)

    except OSError as e:
        raise OSError(f"Failed to write HDF5 file: {e}")


# ============================================================================
# Main
# ============================================================================

def load_data_jsonl(path: Path) -> List[Dict]:
    """Load data from JSONL file.

    DEPRECATED: Use probes.core.load_jsonl() directly instead.
    This wrapper is kept for backward compatibility.
    """
    # Use centralized loading function with require_id=False for backward compatibility
    return load_jsonl(path, require_id=False)


def main():
    parser = argparse.ArgumentParser(
        description="Collect activations - supports texts and conversations"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output HDF5 file",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (HuggingFace identifier)",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        choices=["texts", "conversations"],
        help="Data type: 'texts' (NO chat tokenization) or 'conversations' (WITH chat tokenization + regional)",
    )
    parser.add_argument(
        "--start_token",
        type=int,
        default=20,
        help="For texts: start averaging from this token (default: 20)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=100,
        help="Save checkpoint every N items",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ACTIVATION COLLECTION")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"Data type: {args.data_type}")
    if args.data_type == "texts":
        print(f"Start token: {args.start_token}")
    print()

    # Load nnterp
    try:
        from nnterp import StandardizedTransformer
    except ImportError:
        print("Error: nnterp not installed")
        print("Install with: pip install nnterp")
        sys.exit(1)

    # Load data
    print("Loading data...")
    data = load_data_jsonl(args.input)
    print(f"Loaded {len(data)} items")

    # Validate format and detect variant
    if args.data_type == "texts":
        # Check if using emotional_variants format (tier data) or emotional_text format (simple pairs)
        sample = data[0]
        has_variants = "emotional_variants" in sample

        for i, item in enumerate(data[:3]):
            if "neutral_text" not in item:
                raise ValueError(f"Item {i} missing 'neutral_text'")
            if has_variants:
                if "emotional_variants" not in item:
                    raise ValueError(f"Item {i} missing 'emotional_variants'")
            else:
                if "emotional_text" not in item:
                    raise ValueError(f"Item {i} missing 'emotional_text'")
            if "id" not in item:
                item["id"] = str(i)
    else:  # conversations
        for i, item in enumerate(data[:3]):
            if "messages" not in item:
                raise ValueError(f"Item {i} missing 'messages' key")
            if "id" not in item:
                item["id"] = str(i)

    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Pre-load with AutoModelForCausalLM first (avoids meta device issue)
        print("Loading with AutoModelForCausalLM...")
        model_raw = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        # Wrap in StandardizedTransformer
        print("Wrapping in StandardizedTransformer...")
        model = StandardizedTransformer(
            model_raw,
            trust_remote_code=True,
            check_renaming=False,
            allow_dispatch=True
        )

        device = next(model.model.parameters()).device

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print(f"Model loaded: {model.num_layers} layers, {model.hidden_size} hidden dim")
    print(f"Device: {device}")

    # Collect activations
    print("\nCollecting activations...")
    activations = {}
    metadata = []

    if args.data_type == "texts":
        # Text pairs - handle both formats
        sample = data[0]
        has_variants = "emotional_variants" in sample

        if has_variants:
            # emotional_variants format: expand into multiple pairs
            for item in tqdm(data, desc="Processing text pairs (variants)"):
                try:
                    neutral_acts = collect_text_activations(
                        model, tokenizer, item["neutral_text"], device, args.start_token
                    )

                    if neutral_acts is None:
                        tqdm.write(f"Skipping item {item['id']}: neutral text too short")
                        continue

                    # Process each emotion variant
                    for emotion, emotional_text in item["emotional_variants"].items():
                        emotional_acts = collect_text_activations(
                            model, tokenizer, emotional_text, device, args.start_token
                        )

                        if emotional_acts is None:
                            tqdm.write(f"Skipping {item['id']}/{emotion}: emotional text too short")
                            continue

                        pair_id = f"{item['id']}_{emotion}"
                        activations[pair_id] = {
                            "neutral": neutral_acts,
                            "emotional": emotional_acts,
                        }

                        metadata.append({
                            "id": pair_id,
                            "emotion": emotion,
                            "topic": item.get("topic", "unknown"),
                            "tier": item.get("tier", "unknown"),
                        })

                except Exception as e:
                    tqdm.write(f"Error processing item {item['id']}: {e}")
                    continue
        else:
            # Simple format: neutral_text and emotional_text
            for item in tqdm(data, desc="Processing text pairs"):
                try:
                    neutral_acts = collect_text_activations(
                        model, tokenizer, item["neutral_text"], device, args.start_token
                    )
                    emotional_acts = collect_text_activations(
                        model, tokenizer, item["emotional_text"], device, args.start_token
                    )

                    # Skip if either text is too short
                    if neutral_acts is None or emotional_acts is None:
                        tqdm.write(f"Skipping item {item['id']}: text too short (<{args.start_token} tokens)")
                        continue

                    item_id = item["id"]
                    activations[item_id] = {
                        "neutral": neutral_acts,
                        "emotional": emotional_acts,
                    }

                    metadata.append({
                        "id": item_id,
                        "emotion": item.get("emotion", "unknown"),
                        "topic": item.get("topic", "unknown"),
                        "tier": item.get("tier", "unknown"),
                    })

                except Exception as e:
                    tqdm.write(f"Error processing item {item['id']}: {e}")
                    continue

        # Save
        print(f"\nSaving to {args.output}")
        save_text_activations_hdf5(activations, metadata, args.output, args.model)

    else:  # conversations
        for idx, item in enumerate(tqdm(data, desc="Processing conversations")):
            try:
                conv_acts = collect_conversation_activations(
                    model, tokenizer, item["messages"], device
                )

                item_id = item.get("id", str(idx))
                activations[item_id] = conv_acts

                metadata.append({
                    "id": item_id,
                    "user_emotion": item.get("user_emotion", "unknown"),
                    "asst_emotion": item.get("asst_emotion", "unknown"),
                    "topic": item.get("topic", "unknown"),
                })

            except Exception as e:
                item_id = item.get("id", str(idx))
                tqdm.write(f"Error processing item {item_id}: {e}")
                continue

        # Save
        print(f"\nSaving to {args.output}")
        save_conversation_activations_hdf5(activations, metadata, args.output, args.model)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Collected {len(activations)} items")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
