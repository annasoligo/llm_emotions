#!/usr/bin/env python3
"""
Compute mean-diff direction vectors from text activations for Qwen 235B.

For each emotion and layer, computes:
    direction = mean(emotional_activations) - mean(neutral_activations)

This gives steering vectors that represent the emotional direction.
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

EMOTIONS = ['anger', 'fear', 'happiness', 'surprise', 'disgust', 'sadness']


def compute_directions(h5_path: Path, output_dir: Path, target_layers: list = None):
    """Compute mean-diff directions for each emotion and layer."""

    logger.info(f"Loading activations from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        num_layers = f.attrs['num_layers']
        hidden_dim = f.attrs['hidden_dim']
        model_name = f.attrs['model_name']
        num_pairs = f.attrs['num_pairs']

        logger.info(f"Model: {model_name}")
        logger.info(f"Total layers: {num_layers}")
        logger.info(f"Hidden dim: {hidden_dim}")
        logger.info(f"Num pairs: {num_pairs}")

        # Load all data
        neutral_all = f['neutral'][:]  # [num_pairs, num_layers, hidden_dim]
        emotional_all = f['emotional'][:]
        emotions = [e.decode() for e in f['emotions'][:]]

        logger.info(f"Loaded shapes: neutral={neutral_all.shape}, emotional={emotional_all.shape}")

        # Group by emotion
        emotion_indices = defaultdict(list)
        for idx, emotion in enumerate(emotions):
            emotion_indices[emotion].append(idx)

        logger.info(f"Emotions found: {list(emotion_indices.keys())}")
        for emo, indices in emotion_indices.items():
            logger.info(f"  {emo}: {len(indices)} pairs")

        # Which layers to process
        if target_layers is None:
            target_layers = list(range(num_layers))

        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute directions for each layer
        for layer_idx in target_layers:
            logger.info(f"\nLayer {layer_idx}:")

            vectors = {}

            for emotion in EMOTIONS:
                if emotion not in emotion_indices:
                    logger.warning(f"  No data for emotion: {emotion}")
                    continue

                indices = emotion_indices[emotion]

                # Get activations for this emotion and layer
                neutral_acts = neutral_all[indices, layer_idx, :]  # [n, hidden_dim]
                emotional_acts = emotional_all[indices, layer_idx, :]

                neutral_mean = neutral_acts.mean(axis=0)
                emotional_mean = emotional_acts.mean(axis=0)

                # Mean-diff direction
                direction = emotional_mean - neutral_mean
                direction_norm = float(np.linalg.norm(direction))

                # Normalize to unit vector
                direction_unit = direction / direction_norm

                vectors[emotion] = direction_unit.astype(np.float32)
                vectors[f'{emotion}_norm'] = direction_norm
                vectors[f'{emotion}_raw'] = direction.astype(np.float32)

                logger.info(f"  {emotion}: norm={direction_norm:.2f}, n_pairs={len(indices)}")

            # Also compute overall neutral mean for reference
            all_neutral_layer = neutral_all[:, layer_idx, :]
            vectors['neutral_mean'] = all_neutral_layer.mean(axis=0).astype(np.float32)

            # Save
            output_file = output_dir / f"qwen235b_text_directions_layer{layer_idx}.npz"
            np.savez(output_file, **vectors)
            logger.info(f"  Saved: {output_file}")

    logger.info(f"\n{'='*60}")
    logger.info(f"DIRECTION COMPUTATION COMPLETE")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Compute text-based steering directions for Qwen 235B')
    parser.add_argument('--input', type=str, required=True,
                        help='Input HDF5 file with text activations')
    parser.add_argument('--output-dir', type=str,
                        default='probes/ua_emotion_disentangle/vectors',
                        help='Output directory for direction vectors')
    parser.add_argument('--layers', type=str, default=None,
                        help='Specific layers to process (comma-separated, default: all)')
    args = parser.parse_args()

    target_layers = None
    if args.layers:
        target_layers = [int(x.strip()) for x in args.layers.split(',')]

    compute_directions(Path(args.input), Path(args.output_dir), target_layers)


if __name__ == '__main__':
    main()
