#!/usr/bin/env python3
"""
Collect activations from emotion denial pairs and extract direction vectors.

These pairs contrast:
- "emotional": genuine emotional expression
- "denial": acknowledges situation but explicitly denies having emotions

The extracted vector (emotional - denial) captures "emotional expression"
direction. NEGATE before use to suppress emotional expression.

Usage:
    python -m steering_tests.activation_collection.collect_emotion_denial \
        --model Qwen/Qwen3-235B-A22B \
        --layers 55-65 \
        --output steering_tests/vectors/emotion_denial
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict
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
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
            return model.language_model.layers
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
            return model.language_model.model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h
        elif hasattr(model, 'layers'):
            return model.layers
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
        hidden = capture.activations[layer_idx][0]
        result[layer_idx] = hidden[-1].float().numpy()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Collect activations from emotion denial pairs"
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--layers", type=str, default="all", help="Layers to collect")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--input", type=Path,
                       default=Path("steering_tests/activation_collection/data/emotion_denial_pairs.json"))
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--tp", type=int, default=1)

    args = parser.parse_args()

    print("=" * 70)
    print("EMOTION DENIAL VECTOR EXTRACTION")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
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
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

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

    print(f"Model: {num_layers} layers, {hidden_dim} hidden dim")
    print(f"Collecting from layers: {layers[0]}-{layers[-1]} ({len(layers)} layers)")
    print()

    # Setup activation capture
    capture = ActivationCapture(model, layers)

    # Collect activations
    print("Collecting activations...")
    pair_diffs = {layer: [] for layer in layers}
    pair_metadata = []

    for pair in tqdm(pairs, desc="Processing pairs"):
        try:
            emotional_acts = get_last_token_activation(capture, tokenizer, model, pair["emotional"])
            denial_acts = get_last_token_activation(capture, tokenizer, model, pair["denial"])

            # Compute: emotional - denial (pointing toward emotional expression)
            for layer_idx in layers:
                diff = emotional_acts[layer_idx] - denial_acts[layer_idx]
                pair_diffs[layer_idx].append(diff)

            pair_metadata.append({
                "id": pair["id"],
                "emotion": pair["emotion"],
            })

        except Exception as e:
            tqdm.write(f"Error processing pair {pair['id']}: {e}")
            continue

    capture.remove_hooks()
    print(f"\nProcessed {len(pair_metadata)} pairs")

    # Compute mean direction (then NEGATE for suppression use)
    print("Computing mean direction vectors (NEGATED for suppression)...")

    # Store NEGATED vectors so adding them suppresses emotion
    suppression_vectors = {}
    for layer_idx in layers:
        diffs = np.stack(pair_diffs[layer_idx])
        mean_diff = diffs.mean(axis=0)
        # NEGATE: so adding pushes toward denial/suppression
        suppression_vectors[layer_idx] = -mean_diff
        norm = np.linalg.norm(mean_diff)
        print(f"  Layer {layer_idx:2d}: norm={norm:.2f}")

    # Save
    args.output.mkdir(parents=True, exist_ok=True)
    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    # Save vectors
    vector_file = args.output / f"emotion_denial_{model_short}.pkl"
    with open(vector_file, "wb") as f:
        pickle.dump({
            "vectors": suppression_vectors,
            "layers": layers,
            "hidden_dim": hidden_dim,
            "model": args.model,
            "num_pairs": len(pair_metadata),
            "pair_metadata": pair_metadata,
            "description": "Emotion denial vectors (NEGATED: denial - emotional). ADD to suppress emotional expression.",
        }, f)
    print(f"\nSaved vectors to: {vector_file}")

    # Save metadata
    metadata = {
        "model": args.model,
        "layers": layers,
        "hidden_dim": hidden_dim,
        "num_pairs": len(pair_metadata),
        "timestamp": datetime.now().isoformat(),
        "description": "Emotion denial suppression vectors. NEGATED so adding suppresses emotion.",
        "usage": "ADD vector to suppress emotional expression",
    }

    meta_file = args.output / f"emotion_denial_{model_short}_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 70}")
    print("EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"Vectors saved to: {args.output}")
    print("Vectors are PRE-NEGATED: ADD them to suppress emotional expression")


if __name__ == "__main__":
    main()
