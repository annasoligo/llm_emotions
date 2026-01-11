#!/usr/bin/env python3
"""Extract steering vectors from trained emotion probes for specific layers."""

import argparse
import pickle
import numpy as np
from pathlib import Path

EMOTION_NAMES = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']


def extract_vectors(probe_path: Path, output_dir: Path):
    """Extract per-emotion steering vectors from a multi-probe file."""

    with open(probe_path, 'rb') as f:
        probe = pickle.load(f)

    layer = probe['layer']
    label_names = probe['label_names']
    probe_sets = probe['model_state_dict']['probe_sets']

    # Move to CPU if needed
    if hasattr(probe_sets, 'cpu'):
        probe_sets = probe_sets.cpu().numpy()

    print(f"Loaded probe: layer={layer}, shape={probe_sets.shape}")
    print(f"Labels: {label_names}")

    # probe_sets shape: [n_sets, n_emotions, hidden_dim]
    # Average across sets to get mean direction per emotion
    mean_vectors = probe_sets.mean(axis=0)  # [n_emotions, hidden_dim]

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, emotion in enumerate(label_names):
        vector = mean_vectors[i]
        # Normalize
        vector = vector / np.linalg.norm(vector)

        output_path = output_dir / f"{emotion}_layer{layer}.npz"
        np.savez(output_path,
                vector=vector,
                layer=layer,
                emotion=emotion,
                normalized=True)
        print(f"Saved: {output_path}")

    # Also save neutral as negative mean of all emotions
    all_mean = mean_vectors.mean(axis=0)
    neutral = -all_mean / np.linalg.norm(all_mean)
    output_path = output_dir / f"neutral_layer{layer}.npz"
    np.savez(output_path,
            vector=neutral,
            layer=layer,
            emotion='neutral',
            normalized=True)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('experiments/steering/vectors'))
    args = parser.parse_args()

    extract_vectors(args.probe_path, args.output_dir)


if __name__ == '__main__':
    main()
