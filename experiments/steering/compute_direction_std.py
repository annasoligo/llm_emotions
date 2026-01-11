#!/usr/bin/env python3
"""
Compute the standard deviation of activations projected onto steering directions.

For a fair comparison between probe-based and logit-based vectors, we need to know
how much the model naturally varies along each direction.
"""

import argparse
import json
import logging
import numpy as np
import torch
from pathlib import Path
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Simple neutral prompts for baseline
BASELINE_PROMPTS = [
    "The weather today is",
    "I went to the store and",
    "The book on the table",
    "Yesterday I learned that",
    "The best way to cook",
    "In the morning I usually",
    "My favorite color is",
    "The city has many",
    "During the summer we",
    "The computer program",
    "She walked down the",
    "The meeting was scheduled",
    "I think the answer is",
    "The restaurant serves",
    "He decided to go",
    "The project requires",
    "We should consider",
    "The report shows that",
    "After the movie we",
    "The garden has beautiful",
    "They arrived at the",
    "The music was playing",
    "I remember when the",
    "The train departed at",
    "She mentioned that the",
    "The building stands",
    "We visited the museum",
    "The coffee was hot",
    "He looked at the",
    "The plan involves",
]

# Global storage for collected activations (will be pickled to worker)
_collected_activations = []


def _find_target_layer(model, layer_idx: int):
    """Find the transformer layer to hook in various model architectures."""
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        return model.language_model.model.layers[layer_idx]
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx]
    else:
        raise ValueError(f"Could not find layer {layer_idx} in model architecture")


def _collection_hook(module, inputs, outputs):
    """Forward hook that collects activations."""
    if isinstance(outputs, tuple):
        hidden_states = outputs[0]
    else:
        hidden_states = outputs

    # vLLM may flatten batch and sequence dimensions
    # Shape could be [batch, seq, hidden] or [batch*seq, hidden]
    # Convert bfloat16 to float32 for numpy
    act = hidden_states.detach().float().cpu().numpy()

    if len(act.shape) == 3:
        # [batch, seq, hidden] - take last token
        last_token = act[:, -1, :]
    elif len(act.shape) == 2:
        # [batch*seq, hidden] - take last row (assumes single request)
        last_token = act[-1:, :]  # Keep 2D shape
    else:
        return outputs  # Unknown shape, skip

    # Store on module
    if not hasattr(module, '_collected'):
        module._collected = []
    module._collected.append(last_token)

    return outputs


def _setup_collection_hook(model, layer_idx: int):
    """Register hook to collect activations."""
    layer = _find_target_layer(model, layer_idx)
    layer._collected = []
    layer._collection_handle = layer.register_forward_hook(_collection_hook)
    return f"Collection hook registered on layer {layer_idx}"


def _get_collected_activations(model, layer_idx: int):
    """Get collected activations from model."""
    layer = _find_target_layer(model, layer_idx)
    if hasattr(layer, '_collected'):
        collected = layer._collected.copy()
        layer._collected = []  # Clear for next batch
        return collected
    return []


def _remove_collection_hook(model, layer_idx: int):
    """Remove collection hook."""
    layer = _find_target_layer(model, layer_idx)
    if hasattr(layer, '_collection_handle'):
        layer._collection_handle.remove()
        del layer._collection_handle
    if hasattr(layer, '_collected'):
        del layer._collected
    return "Collection hook removed"


def compute_direction_stats(activations: np.ndarray, vectors: dict) -> dict:
    """Compute projection statistics for each direction."""
    stats = {}

    for name, vector in vectors.items():
        # Normalize vector
        vector = vector / np.linalg.norm(vector)

        # Project activations onto direction
        projections = activations @ vector

        stats[name] = {
            'mean': float(np.mean(projections)),
            'std': float(np.std(projections)),
            'min': float(np.min(projections)),
            'max': float(np.max(projections)),
        }

        logger.info(f"{name}: mean={stats[name]['mean']:.2f}, std={stats[name]['std']:.2f}")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='google/gemma-3-27b-it')
    parser.add_argument('--layer', type=int, default=30)
    parser.add_argument('--vector-dir', default='experiments/steering/vectors')
    parser.add_argument('--output', default='experiments/steering/direction_stats.json')
    parser.add_argument('--n-prompts', type=int, default=30)
    args = parser.parse_args()

    # Load vectors
    vector_dir = Path(args.vector_dir)
    vectors = {}

    for npz_file in vector_dir.glob(f'*_layer{args.layer}.npz'):
        name = npz_file.stem.replace(f'_layer{args.layer}', '')
        if name in ['all_emotions']:
            continue
        data = np.load(npz_file)
        vectors[name] = data['vector']
        logger.info(f"Loaded {name}: shape={vectors[name].shape}")

    # Initialize vLLM
    logger.info(f"Loading model {args.model}...")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype='bfloat16',
        enforce_eager=True,
        disable_log_stats=True,
    )

    # Setup collection hook (capture layer in closure)
    layer_idx = args.layer
    logger.info(f"Setting up collection hook on layer {layer_idx}...")

    def setup_fn(model, _layer=layer_idx):
        return _setup_collection_hook(model, _layer)
    result = llm.apply_model(setup_fn)
    logger.info(f"Setup result: {result}")

    # Generate with baseline prompts
    prompts = BASELINE_PROMPTS[:args.n_prompts]
    logger.info(f"Generating for {len(prompts)} prompts...")

    sampling_params = SamplingParams(max_tokens=1, temperature=0)
    _ = llm.generate(prompts, sampling_params)

    # Get collected activations
    def get_fn(model, _layer=layer_idx):
        return _get_collected_activations(model, _layer)
    collected = llm.apply_model(get_fn)
    logger.info(f"Collected {len(collected)} activation batches")

    # Clean up
    def cleanup_fn(model, _layer=layer_idx):
        return _remove_collection_hook(model, _layer)
    llm.apply_model(cleanup_fn)

    if not collected:
        logger.error("No activations collected!")
        return

    # Stack activations
    all_acts = np.vstack(collected)
    logger.info(f"Total activations shape: {all_acts.shape}")

    # Compute global stats
    global_std = np.std(all_acts)
    global_mean = np.mean(np.linalg.norm(all_acts, axis=1))
    logger.info(f"Global activation std: {global_std:.2f}")
    logger.info(f"Mean activation norm: {global_mean:.2f}")

    # Compute per-direction stats
    logger.info("\nPer-direction statistics:")
    direction_stats = compute_direction_stats(all_acts, vectors)

    # Save results
    results = {
        'global_std': float(global_std),
        'mean_norm': float(global_mean),
        'n_samples': len(prompts),
        'layer': args.layer,
        'directions': direction_stats,
    }

    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {output_path}")

    # Print comparison table
    print("\n" + "="*70)
    print("Direction STD Comparison (for scaling steering vectors)")
    print("="*70)
    print(f"{'Direction':<25} {'STD':>10} {'vs Global':>12} {'vs BASELINE_STD':>15}")
    print("-"*70)
    BASELINE_STD = 576.98
    for name, stats in sorted(direction_stats.items()):
        ratio_global = stats['std'] / global_std
        ratio_baseline = stats['std'] / BASELINE_STD
        print(f"{name:<25} {stats['std']:>10.2f} {ratio_global:>11.2f}x {ratio_baseline:>14.4f}x")
    print("-"*70)
    print(f"{'Global STD:':<25} {global_std:>10.2f}")
    print(f"{'Current BASELINE_STD:':<25} {BASELINE_STD:>10.2f}")


if __name__ == '__main__':
    main()
