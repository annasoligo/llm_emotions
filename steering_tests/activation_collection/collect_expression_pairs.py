#!/usr/bin/env python3
"""
Collect activations from expression suppression pairs and extract direction vectors.

These pairs have the same facts/intentions but differ only in emotional expression style:
- "expressive": openly emotional language
- "composed": same content but neutral/professional tone

The extracted vector (expressive - composed) should capture "expression style"
rather than "emotional state", allowing suppression of emotional expression
while potentially preserving emotion-driven behavior.

Usage:
    python -m steering_tests.activation_collection.collect_expression_pairs \
        --model Qwen/Qwen3-235B-A22B \
        --layers 35-45 \
        --output steering_tests/vectors/expression_suppression
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class ActivationCapture:
    """Capture activations using PyTorch hooks."""

    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.activations = {}
        self.hooks = []

        layers_module = self._get_layers_module(model)

        for layer_idx in layers:
            layer = layers_module[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

    def _get_layers_module(self, model):
        """Find the layers module in the model architecture."""
        # Multimodal models (Gemma-3, etc) - check language_model.layers first
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
            return model.language_model.layers
        # Also try language_model.model.layers
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
            return model.language_model.model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h
        elif hasattr(model, 'layers'):
            return model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            return model.model.decoder.layers
        else:
            raise ValueError(f"Could not find layers in model. Available: {dir(model)}")

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations[layer_idx] = hidden_states.detach().cpu()
        return hook

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def parse_layers(layer_spec: str, num_layers: int) -> List[int]:
    """Parse layer specification like '0-61', '20,30,40', or 'all'."""
    if layer_spec == 'all':
        return list(range(num_layers))
    elif '-' in layer_spec:
        start, end = map(int, layer_spec.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(x.strip()) for x in layer_spec.split(',')]


def get_last_token_activation(
    capture: ActivationCapture,
    tokenizer,
    model,
    text: str,
) -> Dict[int, np.ndarray]:
    """Get activation at last token position for given text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    capture.clear()
    with torch.no_grad():
        _ = model(**inputs)

    result = {}
    for layer_idx in capture.layers:
        hidden = capture.activations[layer_idx][0]  # [seq_len, hidden_dim]
        result[layer_idx] = hidden[-1].float().numpy()  # Last token

    return result


def get_mean_activation(
    capture: ActivationCapture,
    tokenizer,
    model,
    text: str,
    start_token: int = 5,
) -> Dict[int, np.ndarray]:
    """Get mean activation over tokens for given text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]

    capture.clear()
    with torch.no_grad():
        _ = model(**inputs)

    result = {}
    for layer_idx in capture.layers:
        hidden = capture.activations[layer_idx][0]  # [seq_len, hidden_dim]
        # Mean over tokens from start_token onwards
        start = min(start_token, seq_len - 1)
        result[layer_idx] = hidden[start:].mean(dim=0).float().numpy()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Collect activations from expression suppression pairs"
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--layers", type=str, default="all", help="Layers to collect: 'all', '20-40', or '20,30,40'")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for vectors")
    parser.add_argument("--input", type=Path,
                       default=Path("steering_tests/activation_collection/data/expression_suppression_pairs.json"),
                       help="Input JSON file with pairs")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--method", type=str, default="last_token", choices=["last_token", "mean"],
                       help="How to extract activations: last_token or mean over sequence")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel degree (for device_map)")

    args = parser.parse_args()

    print("=" * 70)
    print("EXPRESSION SUPPRESSION VECTOR EXTRACTION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Method: {args.method}")
    print()

    # Load pairs
    with open(args.input) as f:
        data = json.load(f)

    pairs = data["pairs"]
    print(f"Loaded {len(pairs)} pairs")
    print(f"Emotions: {', '.join(p['emotion'] for p in pairs)}")
    print()

    # Load model
    print(f"Loading model: {args.model}")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Get model info
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config

    num_layers = getattr(config, 'num_hidden_layers', None) or \
                 getattr(config, 'n_layers', None) or \
                 getattr(config, 'num_layers', None)
    hidden_dim = config.hidden_size
    layers = parse_layers(args.layers, num_layers)

    print(f"Model loaded: {num_layers} layers, {hidden_dim} hidden dim")
    print(f"Collecting from layers: {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print()

    # Setup activation capture
    capture = ActivationCapture(model, layers)

    # Collect activations for each pair
    print("Collecting activations...")

    pair_diffs = {layer: [] for layer in layers}
    pair_metadata = []

    get_activation = get_last_token_activation if args.method == "last_token" else get_mean_activation

    for pair in tqdm(pairs, desc="Processing pairs"):
        try:
            # Get activations for expressive version
            expressive_acts = get_activation(capture, tokenizer, model, pair["expressive"])

            # Get activations for composed version
            composed_acts = get_activation(capture, tokenizer, model, pair["composed"])

            # Compute difference: expressive - composed
            # This gives us the "expression" direction
            for layer_idx in layers:
                diff = expressive_acts[layer_idx] - composed_acts[layer_idx]
                pair_diffs[layer_idx].append(diff)

            pair_metadata.append({
                "id": pair["id"],
                "emotion": pair["emotion"],
                "scenario": pair["scenario"],
            })

        except Exception as e:
            tqdm.write(f"Error processing pair {pair['id']}: {e}")
            continue

    # Clean up
    capture.remove_hooks()

    print(f"\nSuccessfully processed {len(pair_metadata)} pairs")

    # Compute mean direction across all pairs
    print("Computing mean direction vectors...")

    expression_vectors = {}
    for layer_idx in layers:
        diffs = np.stack(pair_diffs[layer_idx])  # [num_pairs, hidden_dim]
        mean_diff = diffs.mean(axis=0)
        expression_vectors[layer_idx] = mean_diff

        # Also compute std for reference
        std_diff = diffs.std(axis=0).mean()
        norm = np.linalg.norm(mean_diff)
        print(f"  Layer {layer_idx:2d}: norm={norm:.2f}, mean_std={std_diff:.4f}")

    # Save vectors
    args.output.mkdir(parents=True, exist_ok=True)

    # Get model short name
    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    # Save as pickle (compatible with existing steering infrastructure)
    vector_file = args.output / f"expression_suppression_{model_short}.pkl"
    with open(vector_file, "wb") as f:
        pickle.dump({
            "vectors": expression_vectors,
            "layers": layers,
            "hidden_dim": hidden_dim,
            "model": args.model,
            "method": args.method,
            "num_pairs": len(pair_metadata),
            "pair_metadata": pair_metadata,
            "description": "Expression style vector (expressive - composed). Subtract to reduce emotional expression while preserving content.",
        }, f)
    print(f"\nSaved vectors to: {vector_file}")

    # Also save per-layer .npy files for compatibility
    for layer_idx, vec in expression_vectors.items():
        layer_file = args.output / f"expression_suppression_{model_short}_layer{layer_idx}.npy"
        np.save(layer_file, vec)
    print(f"Saved {len(layers)} per-layer .npy files")

    # Save metadata
    metadata = {
        "model": args.model,
        "layers": layers,
        "hidden_dim": hidden_dim,
        "num_pairs": len(pair_metadata),
        "method": args.method,
        "input_file": str(args.input),
        "timestamp": datetime.now().isoformat(),
        "pair_metadata": pair_metadata,
        "description": "Expression suppression vectors. Computed as mean(expressive - composed) across pairs with same facts but different expression styles.",
        "usage": "Subtract vector to reduce emotional expression. Add vector to increase emotional expression.",
    }

    meta_file = args.output / f"expression_suppression_{model_short}_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {meta_file}")

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Vectors saved to: {args.output}")
    print(f"To suppress expression: SUBTRACT the vector")
    print(f"To enhance expression: ADD the vector")


if __name__ == "__main__":
    main()
