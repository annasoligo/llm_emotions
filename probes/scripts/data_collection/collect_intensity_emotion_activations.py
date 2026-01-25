#!/usr/bin/env python3
"""Collect activations from intensity emotion responses.

Collects activations on ASSISTANT RESPONSES ONLY, starting from token 20.
Uses efficient batching and saves to HDF5 format.

Usage:
    python collect_intensity_emotion_activations.py \
        --input data/intensity_emotions/intensity_medium_full_1000samples_*.json \
        --output activations/intensity_medium_1000.h5 \
        --model google/gemma-2-27b-it \
        --batch-size 8
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.action_prompts import EMOTIONS


# ============================================================================
# Activation Collection
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
        text: Text string (assistant response)
        device: Device model is on
        start_token: Start averaging from this token index

    Returns:
        [num_layers, hidden_size] averaged activations, or None if text too short
    """
    # Tokenize WITHOUT chat template (just raw text)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].shape[1]

    # Skip if too short
    if seq_len <= start_token:
        return None

    # Collect activations using nnterp trace
    activations = []

    with torch.no_grad():
        with model.trace(inputs):
            for layer_idx in range(model.num_layers):
                layer_output = model.layers_output[layer_idx].save()
                activations.append(layer_output)

    # Extract and average from start_token onwards for each layer
    layer_activations = []
    for layer_act in activations:
        # layer_act shape: [1, seq_len, hidden_size]
        averaged = layer_act[0, start_token:, :].mean(dim=0)
        layer_activations.append(averaged)

    # Stack and convert: [num_layers, hidden_size]
    all_layers = torch.stack(layer_activations)
    return all_layers.detach().cpu().to(torch.float32).numpy()


def collect_batch_activations(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    start_token: int = 20,
    max_length: int = 512,
) -> List[Optional[np.ndarray]]:
    """Collect activations for a batch of texts efficiently.

    Uses padding for batched inference, then extracts per-sample activations.

    Args:
        model: nnterp StandardizedTransformer
        tokenizer: HuggingFace tokenizer
        texts: List of text strings
        device: Device model is on
        start_token: Start averaging from this token index
        max_length: Maximum sequence length

    Returns:
        List of [num_layers, hidden_size] arrays, None for texts too short
    """
    if not texts:
        return []

    # Tokenize batch with padding
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    batch_size = inputs["input_ids"].shape[0]
    seq_len = inputs["input_ids"].shape[1]

    # Get attention mask to know actual lengths
    attention_mask = inputs["attention_mask"]

    # Collect activations
    activations = []

    with torch.no_grad():
        with model.trace(inputs):
            for layer_idx in range(model.num_layers):
                layer_output = model.layers_output[layer_idx].save()
                activations.append(layer_output)

    # Process each sample in batch
    results = []
    for batch_idx in range(batch_size):
        # Get actual sequence length for this sample
        sample_len = attention_mask[batch_idx].sum().item()

        if sample_len <= start_token:
            results.append(None)
            continue

        # Extract and average for each layer
        layer_acts = []
        for layer_act in activations:
            # layer_act shape: [batch_size, seq_len, hidden_size]
            # Only average over actual tokens (not padding) after start_token
            sample_act = layer_act[batch_idx, start_token:sample_len, :]
            averaged = sample_act.mean(dim=0)
            layer_acts.append(averaged)

        # Stack: [num_layers, hidden_size]
        stacked = torch.stack(layer_acts)
        results.append(stacked.detach().cpu().to(torch.float32).numpy())

    return results


# ============================================================================
# Data Processing
# ============================================================================

def load_intensity_data(input_path: Path) -> Tuple[List[Dict], Dict]:
    """Load intensity emotion generation results.

    Args:
        input_path: Path to JSON file from generate_intensity_emotions_batch.py

    Returns:
        results: List of sample dicts
        info: Dict with batch metadata
    """
    with open(input_path) as f:
        data = json.load(f)

    results = data.get("results", [])
    info = {
        "batch_id": data.get("batch_id", "unknown"),
        "summary": data.get("summary", {}),
        "status": data.get("status", {}),
    }

    return results, info


def extract_responses(sample: Dict) -> Dict[str, str]:
    """Extract response texts from a sample.

    Args:
        sample: Sample dict with 'responses' containing emotion -> {internal, response}

    Returns:
        Dict mapping emotion -> response text
    """
    responses = {}
    for emotion in EMOTIONS + ["neutral"]:
        if emotion in sample.get("responses", {}):
            resp = sample["responses"][emotion]
            if isinstance(resp, dict) and "response" in resp:
                responses[emotion] = resp["response"]
            elif isinstance(resp, str):
                responses[emotion] = resp
    return responses


# ============================================================================
# HDF5 Saving
# ============================================================================

def save_activations_hdf5(
    activations: Dict[str, Dict[str, np.ndarray]],
    metadata: List[Dict],
    output_path: Path,
    model_name: str,
    intensity: str,
    start_token: int,
):
    """Save activations to HDF5 file.

    Structure:
        activations/
            sample_0000/
                anger: [num_layers, hidden_size]
                fear: [num_layers, hidden_size]
                ...
                neutral: [num_layers, hidden_size]
            sample_0001/
                ...
        metadata: JSON string
        attrs: model_name, intensity, start_token, num_samples, etc.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Store activations
        acts_group = f.create_group('activations')

        for sample_id, sample_acts in activations.items():
            sample_group = acts_group.create_group(sample_id)
            for emotion, acts in sample_acts.items():
                sample_group.create_dataset(emotion, data=acts, compression='gzip')

        # Store metadata
        f.create_dataset('metadata', data=json.dumps(metadata))

        # Store attributes
        f.attrs['model_name'] = model_name
        f.attrs['intensity'] = intensity
        f.attrs['start_token'] = start_token
        f.attrs['num_samples'] = len(activations)
        f.attrs['emotions'] = json.dumps(EMOTIONS + ["neutral"])

    print(f"✓ Saved to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Collect activations from intensity emotion responses"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input JSON file from generate_intensity_emotions_batch.py"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output HDF5 file path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-27b-it",
        help="Model to collect activations from"
    )
    parser.add_argument(
        "--start-token",
        type=int,
        default=20,
        help="Start averaging from this token index (default: 20)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)"
    )

    args = parser.parse_args()

    # Load data
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    print(f"Input: {args.input}")

    results, info = load_intensity_data(args.input)
    print(f"Loaded {len(results)} samples")
    print(f"Batch ID: {info['batch_id']}")
    print(f"Summary: {info['summary']}")

    if args.max_samples:
        results = results[:args.max_samples]
        print(f"Limited to {len(results)} samples")

    # Detect intensity from first sample
    intensity = results[0].get("intensity", "unknown") if results else "unknown"
    print(f"Intensity: {intensity}")

    # Load model
    print()
    print("=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    print(f"Model: {args.model}")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from nnterp import StandardizedTransformer

        # Load model
        print("Loading with AutoModelForCausalLM...")
        model_raw = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

        # Wrap in StandardizedTransformer
        print("Wrapping in StandardizedTransformer...")
        model = StandardizedTransformer(
            model_raw,
            trust_remote_code=True,
            check_renaming=False,
            allow_dispatch=True,
        )

        device = next(model.model.parameters()).device

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"✓ Model loaded: {model.num_layers} layers, {model.hidden_size} hidden dim")
    print(f"✓ Device: {device}")

    # Collect activations
    print()
    print("=" * 80)
    print("COLLECTING ACTIVATIONS")
    print("=" * 80)
    print(f"Start token: {args.start_token}")
    print(f"Batch size: {args.batch_size}")
    print(f"Collecting on: ASSISTANT RESPONSES ONLY")
    print()

    all_activations = {}
    metadata = []
    skipped = 0
    emotions_list = EMOTIONS + ["neutral"]

    # Process in batches for efficiency
    # Flatten all responses first, then batch process
    print("Preparing batches...")
    batch_items = []  # List of (sample_idx, emotion, response_text)

    for sample_idx, sample in enumerate(results):
        responses = extract_responses(sample)
        for emotion in emotions_list:
            if emotion in responses:
                batch_items.append((sample_idx, emotion, responses[emotion]))

    print(f"Total responses to process: {len(batch_items)}")

    # Process in batches
    processed = 0
    batch_results = {}  # sample_idx -> {emotion -> activations}

    for batch_start in tqdm(range(0, len(batch_items), args.batch_size), desc="Processing batches"):
        batch_end = min(batch_start + args.batch_size, len(batch_items))
        batch = batch_items[batch_start:batch_end]

        # Extract texts for this batch
        texts = [item[2] for item in batch]

        # Collect activations for batch
        try:
            acts_list = collect_batch_activations(
                model, tokenizer, texts, device,
                start_token=args.start_token,
            )

            # Store results
            for (sample_idx, emotion, _), acts in zip(batch, acts_list):
                if acts is not None:
                    if sample_idx not in batch_results:
                        batch_results[sample_idx] = {}
                    batch_results[sample_idx][emotion] = acts
                    processed += 1
                else:
                    skipped += 1

        except Exception as e:
            tqdm.write(f"Error processing batch at {batch_start}: {e}")
            continue

    # Organize results by sample
    print()
    print("Organizing results...")

    for sample_idx, sample in enumerate(results):
        if sample_idx not in batch_results:
            continue

        sample_id = f"sample_{sample_idx:04d}"
        sample_acts = batch_results[sample_idx]

        # Only include samples with all emotions
        if len(sample_acts) == len(emotions_list):
            all_activations[sample_id] = sample_acts
            metadata.append({
                "id": sample_id,
                "sample_idx": sample_idx,
                "topic": sample.get("topic", "unknown"),
                "intensity": sample.get("intensity", intensity),
                "user": sample.get("user", "")[:200],  # Truncate for storage
                "emotions": list(sample_acts.keys()),
            })

    print(f"✓ Collected: {len(all_activations)} complete samples")
    print(f"✓ Processed: {processed} responses")
    print(f"✗ Skipped (too short): {skipped} responses")

    # Validate activations
    print()
    print("Validating activations...")
    valid_count = 0
    for sample_id, sample_acts in all_activations.items():
        for emotion, acts in sample_acts.items():
            if np.all(np.isfinite(acts)):
                valid_count += 1
            else:
                print(f"  Warning: {sample_id}/{emotion} has NaN/Inf values")

    print(f"✓ Valid activations: {valid_count}/{len(all_activations) * len(emotions_list)}")

    # Save
    print()
    print("=" * 80)
    print("SAVING")
    print("=" * 80)

    save_activations_hdf5(
        activations=all_activations,
        metadata=metadata,
        output_path=args.output,
        model_name=args.model,
        intensity=intensity,
        start_token=args.start_token,
    )

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Input samples: {len(results)}")
    print(f"Output samples: {len(all_activations)}")
    print(f"Emotions per sample: {len(emotions_list)}")
    print(f"Total activations: {len(all_activations) * len(emotions_list)}")
    print(f"Model: {args.model}")
    print(f"Layers: {model.num_layers}")
    print(f"Hidden dim: {model.hidden_size}")
    print(f"Output file: {args.output}")


if __name__ == "__main__":
    main()
