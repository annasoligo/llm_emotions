#!/usr/bin/env python3
"""Extract activations from axis paraphrase generation results.

Loads generated paraphrases and extracts model activations for probe training.
Saves to HDF5 format with metadata.
"""

import argparse
import json
import sys
import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

# Add paths for model utilities
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not/emotion_evals")
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data"))

from emo_lens.model_utils import load_base_model
from emo_lens.token_trajectories import extract_token_level_activations
from constants import NEUTRAL_TEXT_TEMPLATES


def parse_custom_id(custom_id: str) -> Dict[str, str]:
    """Parse custom_id to extract axis labels.

    Format: neutral_{idx}_combo_{idx}_{valence}_{arousal}_{dominance}_{trust}

    Returns:
        Dict with keys: neutral_idx, combo_idx, valence, arousal, dominance, trust
    """
    parts = custom_id.split('_')

    return {
        'neutral_idx': int(parts[1]),
        'combo_idx': int(parts[3]),
        'valence': parts[4],
        'arousal': parts[5],
        'dominance': parts[6],
        'trust': parts[7],
    }


def extract_activations_for_text(
    model,
    tokenizer,
    text: str,
    layers: List[int],
    skip_first_n_tokens: int = 20,
) -> np.ndarray:
    """Extract activations for a single text.

    Args:
        model: Loaded model
        tokenizer: Tokenizer
        text: Input text
        layers: List of layer indices
        skip_first_n_tokens: Number of initial tokens to skip when averaging (default: 20)

    Returns:
        activations: [n_layers, hidden_dim] averaged over token positions
    """
    # Extract token-level activations
    activations_by_token, token_ids = extract_token_level_activations(
        model=model,
        tokenizer=tokenizer,
        prompt=text,
        layers=layers,
    )

    # Average over token positions for each layer
    layer_activations = []
    for layer in layers:
        # Extract dict mapping token positions to activations
        token_acts_dict = activations_by_token[layer]  # Dict[int, np.ndarray]

        # Stack token activations into array [n_tokens, hidden_dim]
        token_positions = sorted(token_acts_dict.keys())
        token_acts = np.stack([token_acts_dict[pos] for pos in token_positions], axis=0)

        # Convert to tensor
        token_acts = torch.from_numpy(token_acts)

        # Skip first N tokens (prompt/formatting tokens)
        n_tokens = token_acts.shape[0]
        if n_tokens > skip_first_n_tokens:
            token_acts = token_acts[skip_first_n_tokens:]  # [n_tokens - skip_first_n_tokens, hidden_dim]

        # Average over tokens
        avg_act = token_acts.mean(dim=0).cpu().numpy()  # [hidden_dim]
        layer_activations.append(avg_act)

    # Stack: [n_layers, hidden_dim]
    return np.stack(layer_activations, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Extract activations from axis paraphrases")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON file with batch results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output HDF5 file for activations",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/gemma-3-27b-it",
        help="Model name",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="40-50",
        help="Layer range (e.g., '40-50')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--include-neutral",
        action="store_true",
        help="Include original neutral texts as additional samples",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--skip-first-n-tokens",
        type=int,
        default=20,
        help="Number of initial tokens to skip when averaging (default: 20)",
    )

    args = parser.parse_args()

    # Parse layer range
    if '-' in args.layers:
        start, end = map(int, args.layers.split('-'))
        layers = list(range(start, end + 1))
    else:
        layers = [int(args.layers)]

    print("=" * 80)
    print("EXTRACTING AXIS PARAPHRASE ACTIVATIONS")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"Layers: {layers}")
    print(f"Device: {args.device}")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load_base_model(args.model, device_map=args.device)
    print(f"✓ Model loaded: {model.num_layers} layers, "
          f"hidden_size={model.hidden_size}")
    print()

    # Load results
    print("Loading batch results...")
    with open(args.input) as f:
        data = json.load(f)

    results = data['results']
    print(f"✓ Loaded {len(results)} results")
    print()

    # Filter to succeeded results only
    succeeded = {
        k: v for k, v in results.items()
        if v['result']['type'] == 'succeeded'
    }

    if len(succeeded) < len(results):
        print(f"⚠ Filtered to {len(succeeded)} succeeded results "
              f"({len(results) - len(succeeded)} errors)")

    # Limit samples if requested
    if args.max_samples:
        succeeded = dict(list(succeeded.items())[:args.max_samples])
        print(f"✓ Limited to {len(succeeded)} samples for testing")

    print()

    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Process all samples
    print("Extracting activations...")
    print()

    metadata = {}

    with h5py.File(args.output, 'w') as f:
        # Create groups
        acts_group = f.create_group('activations')

        # Process each result
        for custom_id, result in tqdm(succeeded.items(), desc="Processing"):
            # Parse metadata
            meta = parse_custom_id(custom_id)
            text = result['result']['message']['content'][0]['text']

            # Extract activations
            try:
                activations = extract_activations_for_text(
                    model=model,
                    tokenizer=tokenizer,
                    text=text,
                    layers=layers,
                    skip_first_n_tokens=args.skip_first_n_tokens,
                )

                # Save to HDF5
                acts_group.create_dataset(
                    custom_id,
                    data=activations,
                    dtype='float32',
                )

                # Save metadata
                metadata[custom_id] = {
                    **meta,
                    'text': text,
                    'model': result['result']['message']['model'],
                }

            except Exception as e:
                print(f"\n⚠ Error processing {custom_id}: {e}")
                continue

        # Add neutral texts if requested
        if args.include_neutral:
            print()
            print("Adding original neutral texts...")

            for idx, neutral_text in enumerate(tqdm(NEUTRAL_TEXT_TEMPLATES, desc="Neutral texts")):
                custom_id = f"neutral_original_{idx}"

                try:
                    activations = extract_activations_for_text(
                        model=model,
                        tokenizer=tokenizer,
                        text=neutral_text,
                        layers=layers,
                        skip_first_n_tokens=args.skip_first_n_tokens,
                    )

                    acts_group.create_dataset(
                        custom_id,
                        data=activations,
                        dtype='float32',
                    )

                    metadata[custom_id] = {
                        'neutral_idx': idx,
                        'combo_idx': -1,
                        'valence': 'neutral',
                        'arousal': 'neutral',
                        'dominance': 'neutral',
                        'trust': 'neutral',
                        'text': neutral_text,
                        'model': args.model,
                        'is_original_neutral': True,
                    }

                except Exception as e:
                    print(f"\n⚠ Error processing neutral {idx}: {e}")
                    continue

        # Save metadata as JSON string
        f.attrs['metadata'] = json.dumps(metadata)
        f.attrs['layers'] = json.dumps(layers)
        f.attrs['model'] = args.model
        f.attrs['n_samples'] = len(metadata)

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"✓ Extracted activations for {len(metadata)} samples")
    print(f"✓ Saved to: {args.output}")
    print(f"  Layers: {len(layers)}")
    print(f"  Hidden dim: {model.hidden_size}")
    print()


if __name__ == "__main__":
    main()
