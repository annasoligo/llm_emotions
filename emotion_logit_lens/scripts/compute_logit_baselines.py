#!/usr/bin/env python3
"""
Compute logit baseline statistics from Alpaca activations.

This script:
1. Loads activation baselines from existing h5 files (Alpaca dataset)
2. Projects activations to vocabulary logits using model
3. Extracts logits for emotion token IDs
4. Computes mean/std per token per layer
5. Saves as JSON files (one per layer)

Usage:
    python compute_logit_baselines.py \\
        --model google/gemma-3-27b-it \\
        --activation-baseline-dir /path/to/alpaca_gemma27b_v2/google_gemma_3_27b_it \\
        --output-dir /path/to/logit_emotion_alpaca/google_gemma_3_27b_it \\
        --layers 0-61
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import h5py
import torch
from tqdm import tqdm

# Add research-tools to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from nnterp import StandardizedTransformer

from emotion_logit_lens import (
    project_to_logits_batched,
    EmotionTokenManager,
    EKMAN6_EMOTIONS
)


def parse_layers(layers_str: str) -> List[int]:
    """
    Parse layer specification string.

    Args:
        layers_str: Layer range (e.g., "0-61") or comma-separated list (e.g., "20,30,40")

    Returns:
        List of layer integers
    """
    if '-' in layers_str:
        start, end = map(int, layers_str.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(x.strip()) for x in layers_str.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="Compute logit baseline statistics for emotion detection"
    )
    parser.add_argument(
        '--model',
        default='google/gemma-3-27b-it',
        help='Model name (e.g., google/gemma-3-27b-it)'
    )
    parser.add_argument(
        '--activation-baseline-dir',
        required=True,
        help='Directory containing activation baseline h5 files'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for logit baseline statistics'
    )
    parser.add_argument(
        '--layers',
        default='0-61',
        help='Layer range (e.g., "0-61") or comma-separated list (e.g., "20,30,40")'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for logit projection'
    )
    args = parser.parse_args()

    # Parse layers
    layers = parse_layers(args.layers)

    print("=" * 80)
    print("COMPUTING LOGIT BASELINES FOR EMOTION DETECTION")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Activation baseline dir: {args.activation_baseline_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Layers: {len(layers)} layers ({min(layers)}-{max(layers)})")
    print(f"Batch size: {args.batch_size}")
    print()

    # Load model
    print("[1/5] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_raw = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    model = StandardizedTransformer(
        model_raw,
        check_renaming=False,
        allow_dispatch=True
    )
    model.eval()
    print(f"  ✓ Model loaded: {model.num_layers} layers")
    print()

    # Load emotion token IDs
    print("[2/5] Loading emotion token IDs...")
    model_name_safe = args.model.replace("/", "_").replace("-", "_")
    emotion_mgr = EmotionTokenManager(model_name=model_name_safe)
    emotion_token_ids = emotion_mgr.load_emotion_token_ids()

    # Collect all unique token IDs
    all_token_ids = set()
    for emotion, token_list in emotion_token_ids.items():
        all_token_ids.update(token_list)
        print(f"  {emotion}: {len(token_list)} tokens")
    print(f"  Total unique tokens: {len(all_token_ids)}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each layer
    print("[3/5] Loading activations and projecting to logits...")
    baseline_dir = Path(args.activation_baseline_dir)

    n_samples = None  # Will be set from first layer

    for layer in tqdm(layers, desc="Processing layers"):
        # Load activations from h5
        h5_file = baseline_dir / f"layer{layer}_activations.h5"
        if not h5_file.exists():
            print(f"  WARNING: Skipping layer {layer} - file not found: {h5_file}")
            continue

        with h5py.File(h5_file, 'r') as f:
            activations = f['all_tokens'][:]  # [n_samples, hidden_dim]

        if n_samples is None:
            n_samples = activations.shape[0]

        # Project to logits in batches
        all_logits = []
        for batch_start in range(0, n_samples, args.batch_size):
            batch_end = min(batch_start + args.batch_size, n_samples)
            batch_acts = activations[batch_start:batch_end]

            batch_logits = project_to_logits_batched(model, batch_acts)
            all_logits.append(batch_logits.cpu().float().numpy())

        all_logits = np.concatenate(all_logits, axis=0)  # [n_samples, vocab_size]

        # Compute statistics for emotion tokens only
        layer_stats = {}
        for token_id in all_token_ids:
            token_logits = all_logits[:, token_id]  # [n_samples]
            layer_stats[str(token_id)] = {
                'mean': float(np.mean(token_logits)),
                'std': float(np.std(token_logits))
            }

        # Save layer statistics
        layer_file = output_dir / f"layer_{layer}.json"
        with open(layer_file, 'w') as f:
            json.dump({
                'layer': layer,
                'num_samples': n_samples,
                'statistics': layer_stats
            }, f, indent=2)

    # Save metadata
    print("\n[4/5] Saving metadata...")
    metadata = {
        'model_name': args.model,
        'format_version': '1.0',
        'num_samples': n_samples,
        'layers': layers,
        'num_emotion_tokens': len(all_token_ids),
        'emotions': EKMAN6_EMOTIONS,
        'source': 'alpaca_gemma27b_v2',
        'aggregation_type': 'all_tokens'
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[5/5] ✓ Done! Baselines saved to {output_dir}")
    print(f"  Layers: {len(layers)}")
    print(f"  Tokens per layer: {len(all_token_ids)}")
    print(f"  Samples: {n_samples}")


if __name__ == '__main__':
    main()
