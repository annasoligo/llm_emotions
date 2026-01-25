#!/usr/bin/env python3
"""
Collect activations from texts_combined dataset for Qwen3-235B-A22B.
Uses HuggingFace with device_map='auto' across 4 GPUs.

Key optimizations:
- Collects ALL layers in a single forward pass
- Batches multiple texts together for efficiency
- Uses proper padding and attention masks
"""

import argparse
import json
import torch
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-235B-A22B"


class AllLayerActivationCapture:
    """Capture activations from ALL layers using hooks."""

    def __init__(self, model, num_layers: int):
        self.model = model
        self.num_layers = num_layers
        self.activations = {}
        self.hooks = []

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

        logger.info(f"Registered hooks on all {num_layers} layers")

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Store on CPU immediately to save GPU memory
            self.activations[layer_idx] = hidden_states.detach().cpu()
        return hook

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def extract_batch_activations(
    texts: List[str],
    model,
    tokenizer,
    capture: AllLayerActivationCapture,
    num_layers: int,
    hidden_dim: int,
    position: str = "last",
    max_length: int = 512,
) -> np.ndarray:
    """
    Extract activations from a batch of texts.

    Returns: [batch_size, num_layers, hidden_dim]
    """
    # Tokenize with padding
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    batch_size = inputs['input_ids'].shape[0]
    seq_lens = inputs['attention_mask'].sum(dim=1)  # actual lengths

    capture.clear()

    with torch.no_grad():
        _ = model(**inputs)

    # Extract activations for each item in batch
    activations = np.zeros((batch_size, num_layers, hidden_dim), dtype=np.float32)

    for layer_idx in range(num_layers):
        hidden = capture.activations[layer_idx]  # [batch, seq_len, hidden_dim]

        for batch_idx in range(batch_size):
            seq_len = seq_lens[batch_idx].item()

            if position == "last":
                # Get last non-padding token
                act = hidden[batch_idx, seq_len - 1]
            elif position == "mean":
                # Mean over non-padding tokens
                act = hidden[batch_idx, :seq_len].mean(dim=0)
            elif position == "first":
                act = hidden[batch_idx, 0]
            else:
                raise ValueError(f"Unknown position: {position}")

            activations[batch_idx, layer_idx] = act.float().numpy()

    return activations


def load_text_pairs(jsonl_path: Path) -> List[Dict]:
    """Load text pairs from JSONL file."""
    pairs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs


def collect_texts_batched(
    model,
    tokenizer,
    text_pairs: List[Dict],
    num_layers: int,
    output_path: Path,
    hidden_dim: int,
    position: str = "last",
    batch_size: int = 8,
):
    """Collect activations from all text pairs with batched processing."""

    num_pairs = len(text_pairs)

    logger.info(f"\n{'='*80}")
    logger.info(f"TEXT ACTIVATION COLLECTION - QWEN 235B (BATCHED)")
    logger.info(f"{'='*80}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Text pairs: {num_pairs}")
    logger.info(f"Layers: ALL {num_layers}")
    logger.info(f"Hidden dim: {hidden_dim}")
    logger.info(f"Position: {position}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"{'='*80}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = AllLayerActivationCapture(model, num_layers)

    # Pre-allocate arrays for all data
    all_neutral = np.zeros((num_pairs, num_layers, hidden_dim), dtype=np.float32)
    all_emotional = np.zeros((num_pairs, num_layers, hidden_dim), dtype=np.float32)
    emotions = []
    tiers = []
    topics = []
    pair_ids = []

    # Process in batches
    logger.info("Collecting activations in batches...")

    for start_idx in tqdm(range(0, num_pairs, batch_size), desc="Batches"):
        end_idx = min(start_idx + batch_size, num_pairs)
        batch_pairs = text_pairs[start_idx:end_idx]

        # Extract neutral texts
        neutral_texts = [p['neutral_text'] for p in batch_pairs]
        neutral_acts = extract_batch_activations(
            neutral_texts, model, tokenizer, capture,
            num_layers, hidden_dim, position
        )
        all_neutral[start_idx:end_idx] = neutral_acts

        # Extract emotional texts
        emotional_texts = [p['emotional_text'] for p in batch_pairs]
        emotional_acts = extract_batch_activations(
            emotional_texts, model, tokenizer, capture,
            num_layers, hidden_dim, position
        )
        all_emotional[start_idx:end_idx] = emotional_acts

        # Store metadata
        for p in batch_pairs:
            emotions.append(p['emotion'])
            tiers.append(p['tier'])
            topics.append(p['topic'])
            pair_ids.append(p['id'])

        # Periodic cache clearing
        if (start_idx // batch_size) % 20 == 0:
            torch.cuda.empty_cache()

    capture.remove_hooks()

    # Save to HDF5
    logger.info(f"\nSaving to {output_path}...")

    with h5py.File(output_path, 'w') as f:
        f.attrs['model_name'] = MODEL_NAME
        f.attrs['hidden_dim'] = hidden_dim
        f.attrs['num_layers'] = num_layers
        f.attrs['num_pairs'] = num_pairs
        f.attrs['position'] = position

        # Save activations as large arrays (much more efficient)
        f.create_dataset('neutral', data=all_neutral, compression='gzip', compression_opts=4)
        f.create_dataset('emotional', data=all_emotional, compression='gzip', compression_opts=4)

        # Save metadata as arrays
        f.create_dataset('emotions', data=np.array(emotions, dtype='S20'))
        f.create_dataset('tiers', data=np.array(tiers, dtype='S20'))
        f.create_dataset('topics', data=np.array(topics, dtype='S100'))
        f.create_dataset('pair_ids', data=np.array(pair_ids, dtype='S100'))

    logger.info(f"✓ Saved to {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1e9:.2f} GB")
    logger.info(f"  Shape: neutral={all_neutral.shape}, emotional={all_emotional.shape}")


def main():
    parser = argparse.ArgumentParser(description='Collect text activations for Qwen 235B')
    parser.add_argument('--input', type=str,
                        default='/workspace-vast/annas/git/research-tools/outputs/data/texts_combined_pairs.jsonl',
                        help='Input JSONL file with text pairs')
    parser.add_argument('--output', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--position', type=str, default='last',
                        choices=['last', 'mean', 'first'],
                        help='Token position to extract (default: last)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for processing (default: 8)')
    parser.add_argument('--max-pairs', type=int, default=None,
                        help='Limit to first N pairs (for testing)')
    args = parser.parse_args()

    logger.info(f"{'='*80}")
    logger.info(f"TEXT ACTIVATION COLLECTION - QWEN3-235B-A22B")
    logger.info(f"{'='*80}")

    # Load text pairs
    logger.info(f"\nLoading text pairs from: {args.input}")
    text_pairs = load_text_pairs(Path(args.input))
    logger.info(f"  Loaded {len(text_pairs)} pairs")

    if args.max_pairs:
        text_pairs = text_pairs[:args.max_pairs]
        logger.info(f"  Limited to {len(text_pairs)} pairs")

    # Show sample
    sample = text_pairs[0]
    logger.info(f"\nSample entry:")
    logger.info(f"  ID: {sample['id']}")
    logger.info(f"  Emotion: {sample['emotion']}")
    logger.info(f"  Neutral: {sample['neutral_text'][:80]}...")
    logger.info(f"  Emotional: {sample['emotional_text'][:80]}...")

    logger.info(f"\nLoading model: {MODEL_NAME}...")
    logger.info("This may take several minutes...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    logger.info(f"✓ Model loaded (hidden_dim={hidden_dim}, num_layers={num_layers})")

    # Show device placement
    if hasattr(model, 'hf_device_map'):
        devices = set(str(v) for v in model.hf_device_map.values())
        logger.info(f"  Devices used: {devices}")

    collect_texts_batched(
        model=model,
        tokenizer=tokenizer,
        text_pairs=text_pairs,
        num_layers=num_layers,
        output_path=Path(args.output),
        hidden_dim=hidden_dim,
        position=args.position,
        batch_size=args.batch_size,
    )

    logger.info(f"\n{'='*80}")
    logger.info(f"COLLECTION COMPLETE")
    logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
