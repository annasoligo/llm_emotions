#!/usr/bin/env python3
"""
Compute layer activation norms for Qwen3-32B.

Used for calibrating steering strength in experiments.
"""

import argparse
import json
import logging
import numpy as np
import torch
from pathlib import Path
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-32B"

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


def _find_target_layer(model, layer_idx: int):
    """Find the transformer layer in Qwen architecture."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers[layer_idx]
    else:
        raise ValueError(f"Could not find layer {layer_idx} in model architecture")


def _collection_hook(module, inputs, outputs):
    """Forward hook that collects activations."""
    if isinstance(outputs, tuple):
        hidden_states = outputs[0]
    else:
        hidden_states = outputs

    act = hidden_states.detach().float().cpu().numpy()

    if len(act.shape) == 3:
        last_token = act[:, -1, :]
    elif len(act.shape) == 2:
        last_token = act[-1:, :]
    else:
        return outputs

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
        layer._collected = []
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


def compute_layer_norm(llm: LLM, tokenizer, layer_idx: int, n_prompts: int = 30) -> dict:
    """Compute activation statistics for a layer."""

    # Setup collection hook
    def setup_fn(model, _layer=layer_idx):
        return _setup_collection_hook(model, _layer)
    result = llm.apply_model(setup_fn)
    logger.info(f"Setup result: {result}")

    # Format prompts with chat template
    prompts = BASELINE_PROMPTS[:n_prompts]
    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    # Generate (just need to run forward pass)
    sampling_params = SamplingParams(max_tokens=1, temperature=0)
    _ = llm.generate(formatted_prompts, sampling_params)

    # Get collected activations
    def get_fn(model, _layer=layer_idx):
        return _get_collected_activations(model, _layer)
    collected = llm.apply_model(get_fn)

    # Clean up
    def cleanup_fn(model, _layer=layer_idx):
        return _remove_collection_hook(model, _layer)
    llm.apply_model(cleanup_fn)

    if not collected:
        logger.error(f"No activations collected for layer {layer_idx}!")
        return None

    # Stack and compute stats
    all_acts = np.vstack(collected)
    norms = np.linalg.norm(all_acts, axis=1)

    stats = {
        'layer': layer_idx,
        'mean_norm': float(np.mean(norms)),
        'std_norm': float(np.std(norms)),
        'global_std': float(np.std(all_acts)),
        'n_samples': len(prompts),
    }

    logger.info(f"Layer {layer_idx}: mean_norm={stats['mean_norm']:.2f}, std_norm={stats['std_norm']:.2f}")
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, nargs='+', default=[20, 30, 40])
    parser.add_argument('--n-prompts', type=int, default=30)
    parser.add_argument('--output', default='experiments/steering/qwen_layer_norms.json')
    parser.add_argument('--gpu-memory', type=float, default=0.9)
    args = parser.parse_args()

    # Load tokenizer
    logger.info(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Initialize vLLM
    logger.info(f"Loading model {MODEL_NAME}...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype='bfloat16',
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=args.gpu_memory,
    )

    # Compute norms for each layer
    results = {
        'model': MODEL_NAME,
        'layers': {},
    }

    for layer_idx in args.layers:
        logger.info(f"\n{'='*50}")
        logger.info(f"Computing norms for layer {layer_idx}")
        logger.info(f"{'='*50}")

        stats = compute_layer_norm(llm, tokenizer, layer_idx, args.n_prompts)
        if stats:
            results['layers'][str(layer_idx)] = stats

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {output_path}")

    # Print summary
    print("\n" + "="*60)
    print("QWEN3-32B LAYER NORMS (for steering calibration)")
    print("="*60)
    print(f"{'Layer':<10} {'Mean Norm':>15} {'Std Norm':>15}")
    print("-"*60)
    for layer_str, stats in results['layers'].items():
        print(f"{layer_str:<10} {stats['mean_norm']:>15.2f} {stats['std_norm']:>15.2f}")
    print("-"*60)
    print("\nCopy these values to LAYER_NORMS in sandbagging_steering_qwen.py:")
    print("LAYER_NORMS = {")
    for layer_str, stats in results['layers'].items():
        print(f"    {layer_str}: {stats['mean_norm']:.2f},")
    print("}")


if __name__ == '__main__':
    main()
