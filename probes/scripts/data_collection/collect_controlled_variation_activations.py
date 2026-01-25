#!/usr/bin/env python3
"""Collect activations from controlled variation data and create paired HDF5 files.

This script:
1. Loads controlled variation conversations (with isolation_type and anchor_set)
2. Groups by anchor_set and matches emotional with baseline conversations
3. Extracts activations using nnterp
4. Creates TWO paired HDF5 files:
   - user_isolation.h5: (baseline, user=emotion) pairs
   - assistant_isolation.h5: (baseline, asst=emotion) pairs

Usage:
    python scripts/data_collection/collect_controlled_variation_activations.py \\
        --input data/controlled_variation/controlled_variation_all.jsonl \\
        --output-dir activations/controlled_variation/ \\
        --model google/gemma-3-27b-it
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import h5py
import numpy as np
import torch
from tqdm import tqdm

from probes.core import load_jsonl
from probes.scripts.data_collection.collect_activations import collect_conversation_activations


def group_by_anchor_and_type(data: List[Dict]) -> Dict[str, Dict[str, List[Dict]]]:
    """Group conversations by anchor_set and isolation_type.

    Returns:
        {anchor_set: {isolation_type: [conversations]}}
    """
    grouped = defaultdict(lambda: defaultdict(list))

    for item in data:
        anchor_set = item.get("anchor_set", "unknown")
        isolation_type = item.get("isolation_type", "unknown")
        grouped[anchor_set][isolation_type].append(item)

    return grouped


def create_pairs(
    grouped_data: Dict[str, Dict[str, List[Dict]]],
    isolation_type: str,
) -> List[Tuple[Dict, Dict]]:
    """Create (baseline, emotional) pairs for specified isolation type.

    Args:
        grouped_data: Conversations grouped by anchor_set and isolation_type
        isolation_type: 'user' or 'assistant'

    Returns:
        List of (baseline_conv, emotional_conv) tuples
    """
    pairs = []

    for anchor_set, types_dict in grouped_data.items():
        baseline_convs = types_dict.get("baseline", [])
        emotional_convs = types_dict.get(isolation_type, [])

        if not baseline_convs:
            print(f"Warning: No baseline for anchor_set {anchor_set}")
            continue

        # Use first baseline (should typically be only one)
        baseline = baseline_convs[0]

        # Pair baseline with each emotional conversation
        for emotional in emotional_convs:
            pairs.append((baseline, emotional))

    return pairs


def save_paired_activations_hdf5(
    paired_activations: Dict[str, Dict[str, np.ndarray]],
    metadata: List[Dict],
    output_path: Path,
    model_name: str,
    isolation_type: str,
):
    """Save paired activations to HDF5.

    Args:
        paired_activations: {pair_id: {'neutral': [...], 'emotional': [...]}}
        metadata: List of pair metadata dicts
        output_path: Output HDF5 path
        model_name: Model identifier
        isolation_type: 'user' or 'assistant'
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Store activations
        acts_group = f.create_group('activations')

        for pair_id, pair_acts in paired_activations.items():
            pair_group = acts_group.create_group(str(pair_id))
            pair_group.create_dataset('neutral', data=pair_acts['neutral'])
            pair_group.create_dataset('emotional', data=pair_acts['emotional'])

        # Store metadata
        f.create_dataset('metadata', data=json.dumps(metadata))

        # Store attributes
        f.attrs['model_name'] = model_name
        f.attrs['data_type'] = 'controlled_variation'
        f.attrs['isolation_type'] = isolation_type
        f.attrs['num_pairs'] = len(paired_activations)
        f.attrs['extraction_type'] = 'global'  # Using global activations


def main():
    parser = argparse.ArgumentParser(
        description="Collect activations from controlled variation data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file (controlled variation data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (HuggingFace identifier)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CONTROLLED VARIATION ACTIVATION COLLECTION")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model: {args.model}")
    print()

    # Load nnterp
    try:
        from nnterp import StandardizedTransformer
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as e:
        print(f"Error: Missing dependencies - {e}")
        sys.exit(1)

    # Load data
    print("Loading data...")
    data = load_jsonl(args.input, require_id=True)
    print(f"Loaded {len(data)} conversations")

    # Group by anchor_set and isolation_type
    print("\nGrouping conversations by anchor_set and isolation_type...")
    grouped_data = group_by_anchor_and_type(data)
    print(f"Found {len(grouped_data)} anchor sets")

    # Create pairs for each isolation type
    user_pairs = create_pairs(grouped_data, "user")
    asst_pairs = create_pairs(grouped_data, "assistant")

    print(f"\nCreated {len(user_pairs)} user isolation pairs")
    print(f"Created {len(asst_pairs)} assistant isolation pairs")

    # Load model
    print(f"\nLoading model: {args.model}")
    try:
        model_raw = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

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

    # Process each isolation type
    for isolation_type, pairs in [("user", user_pairs), ("assistant", asst_pairs)]:
        print("\n" + "=" * 80)
        print(f"PROCESSING {isolation_type.upper()} ISOLATION")
        print("=" * 80)

        paired_activations = {}
        metadata = []

        for idx, (baseline_conv, emotional_conv) in enumerate(tqdm(pairs, desc=f"{isolation_type} pairs")):
            try:
                # Extract activations for baseline
                baseline_acts = collect_conversation_activations(
                    model, tokenizer, baseline_conv["messages"], device
                )

                # Extract activations for emotional
                emotional_acts = collect_conversation_activations(
                    model, tokenizer, emotional_conv["messages"], device
                )

                # Use global activations
                baseline_global = baseline_acts["global"]
                emotional_global = emotional_acts["global"]

                # Create pair
                pair_id = f"pair_{idx:04d}"
                paired_activations[pair_id] = {
                    "neutral": baseline_global,
                    "emotional": emotional_global,
                }

                # Store metadata
                if isolation_type == "user":
                    emotion = emotional_conv["user_emotion"]
                else:
                    emotion = emotional_conv["asst_emotion"]

                metadata.append({
                    "id": pair_id,
                    "emotion": emotion,
                    "topic": emotional_conv.get("topic", "unknown"),
                    "anchor_set": emotional_conv.get("anchor_set", "unknown"),
                    "baseline_id": baseline_conv.get("id", "unknown"),
                    "emotional_id": emotional_conv.get("id", "unknown"),
                })

            except Exception as e:
                tqdm.write(f"Error processing pair {idx}: {e}")
                continue

        # Save
        output_path = args.output_dir / f"{isolation_type}_isolation.h5"
        print(f"\nSaving to {output_path}")
        save_paired_activations_hdf5(
            paired_activations,
            metadata,
            output_path,
            args.model,
            isolation_type,
        )

        print(f"Saved {len(paired_activations)} pairs")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"User isolation: {args.output_dir / 'user_isolation.h5'}")
    print(f"Assistant isolation: {args.output_dir / 'assistant_isolation.h5'}")


if __name__ == "__main__":
    main()
