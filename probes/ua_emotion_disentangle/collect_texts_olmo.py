#!/usr/bin/env python3
"""
Collect activations from texts_combined dataset for OLMo-3-32B.
Uses the raw texts (no chat template) as in the original Gemma extraction.
"""

import argparse
import json
import torch
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_layers(layer_spec: str) -> List[int]:
    """Parse layer specification like '0-61' or '20,30,40'."""
    if '-' in layer_spec:
        start, end = map(int, layer_spec.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(x.strip()) for x in layer_spec.split(',')]


class ActivationCapture:
    """Capture activations from specified layers using hooks."""

    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.activations = {}
        self.hooks = []

        for layer_idx in layers:
            layer = model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations[layer_idx] = hidden_states.detach()
        return hook

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def extract_text_activations(
    text: str,
    model,
    tokenizer,
    layers: List[int],
    capture: ActivationCapture,
    hidden_dim: int,
    min_token_idx: int = 20,
) -> np.ndarray:
    """
    Extract activations from text (no chat template, like original).
    Averages activations over tokens from position min_token_idx onwards.
    Returns: [num_layers, hidden_dim]
    """
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    seq_len = inputs['input_ids'].shape[1]

    capture.clear()

    with torch.no_grad():
        _ = model(**inputs)

    # Extract and average activations
    activations = np.zeros((len(layers), hidden_dim), dtype=np.float32)

    for i, layer in enumerate(layers):
        hidden = capture.activations[layer][0]  # [seq_len, hidden_dim]

        # Average over tokens from min_token_idx onwards
        start_idx = min(min_token_idx, seq_len - 1)
        avg_hidden = hidden[start_idx:].mean(dim=0)
        activations[i] = avg_hidden.float().cpu().numpy()

    return activations


def load_text_pairs(jsonl_path: Path) -> List[Dict]:
    """Load text pairs from JSONL file."""
    pairs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def collect_texts_streaming(
    model,
    tokenizer,
    text_pairs: List[Dict],
    layers: List[int],
    output_path: Path,
    hidden_dim: int,
    model_name: str,
):
    """Collect activations from all text pairs with streaming writes."""

    num_pairs = len(text_pairs)
    num_layers = len(layers)

    print(f"\n{'='*80}")
    print(f"TEXT ACTIVATION COLLECTION")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Text pairs: {num_pairs}")
    print(f"Layers: {num_layers} ({min(layers)} to {max(layers)})")
    print(f"Hidden dim: {hidden_dim}")
    print(f"{'='*80}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = ActivationCapture(model, layers)

    with h5py.File(output_path, 'w') as f:
        f.attrs['model_name'] = model_name
        f.attrs['hidden_dim'] = hidden_dim
        f.attrs['num_layers'] = num_layers
        f.attrs['layers'] = layers

        # Create activations group
        act_grp = f.create_group('activations')

        print("Collecting activations...")
        for idx, pair in enumerate(tqdm(text_pairs, desc="Processing")):
            pair_id = pair['id']

            # Create group for this pair
            pair_grp = act_grp.create_group(pair_id)

            # Store metadata
            pair_grp.attrs['emotion'] = pair['emotion']
            pair_grp.attrs['tier'] = pair['tier']
            pair_grp.attrs['topic'] = pair['topic']

            # Extract neutral activations
            neutral_acts = extract_text_activations(
                pair['neutral_text'],
                model, tokenizer, layers, capture, hidden_dim
            )

            # Extract emotional activations
            emotional_acts = extract_text_activations(
                pair['emotional_text'],
                model, tokenizer, layers, capture, hidden_dim
            )

            # Store
            pair_grp.create_dataset('neutral', data=neutral_acts, compression='gzip')
            pair_grp.create_dataset('emotional', data=emotional_acts, compression='gzip')

            # Periodic cache clearing
            if idx % 100 == 0:
                torch.cuda.empty_cache()
                if idx % 500 == 0:
                    f.flush()

        capture.remove_hooks()

    print(f"\n Saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description='Collect text activations for OLMo')
    parser.add_argument('--model', type=str,
                        default='allenai/OLMo-2-1124-32B',
                        help='Model name')
    parser.add_argument('--input', type=str,
                        default='/workspace-vast/annas/git/research-tools/outputs/data/texts_combined_pairs.jsonl',
                        help='Input JSONL file with text pairs')
    parser.add_argument('--layers', type=str, default='22-42',
                        help='Layer specification')
    parser.add_argument('--output', type=str, required=True,
                        help='Output HDF5 file path')
    args = parser.parse_args()

    layers = parse_layers(args.layers)

    print(f"{'='*80}")
    print(f"TEXT ACTIVATION COLLECTION - {args.model}")
    print(f"{'='*80}")

    # Load text pairs
    print(f"\nLoading text pairs from: {args.input}")
    text_pairs = load_text_pairs(Path(args.input))
    print(f"  Loaded {len(text_pairs)} pairs")

    # Show sample
    sample = text_pairs[0]
    print(f"\nSample entry:")
    print(f"  ID: {sample['id']}")
    print(f"  Emotion: {sample['emotion']}")
    print(f"  Neutral: {sample['neutral_text'][:80]}...")
    print(f"  Emotional: {sample['emotional_text'][:80]}...")

    print(f"\nLoading model: {args.model}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )

    hidden_dim = model.config.hidden_size
    print(f" Model loaded (hidden_dim={hidden_dim})")

    collect_texts_streaming(
        model=model,
        tokenizer=tokenizer,
        text_pairs=text_pairs,
        layers=layers,
        output_path=Path(args.output),
        hidden_dim=hidden_dim,
        model_name=args.model,
    )

    print(f"\n{'='*80}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
