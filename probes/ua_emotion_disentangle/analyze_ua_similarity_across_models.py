#!/usr/bin/env python3
"""
Compare User vs Model emotion direction similarity across models.

For each model and layer:
1. Compute mean direction for each emotion for U (user) and M (model/assistant)
2. Compute cosine similarity between paired U and M emotions
3. Average across all emotion pairs
4. Plot layer vs avg cosine similarity with one line per model
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def get_plutchik_emotion_pairs():
    """Get 16 Plutchik emotion pairs (opposites)."""
    return [
        ('joy', 'sadness'),
        ('trust', 'disgust'),
        ('fear', 'anger'),
        ('surprise', 'anticipation'),
        ('ecstasy', 'grief'),
        ('admiration', 'loathing'),
        ('terror', 'rage'),
        ('amazement', 'vigilance'),
        ('serenity', 'pensiveness'),
        ('acceptance', 'boredom'),
        ('apprehension', 'annoyance'),
        ('distraction', 'interest'),
        ('awe', 'contempt'),
        ('submission', 'aggressiveness'),
        ('love', 'remorse'),
        ('optimism', 'disappointment'),
    ]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def load_and_compute_emotion_directions(
    h5_path: Path,
    position: str = 'first_asst_token'
) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Load activations and compute mean direction for each emotion.

    Returns:
        Dict mapping layer -> emotion -> (M_direction, U_direction)
    """
    print(f"Loading {h5_path}...")

    results = {}

    with h5py.File(h5_path, 'r') as f:
        layers = f.attrs['layers']

        for layer in layers:
            layer_group = f[f'layer_{layer}']

            # Load activations and metadata
            activations = layer_group[f'{position}_activations'][:]
            M_labels = [x.decode() for x in layer_group[f'{position}_M'][:]]
            U_labels = [x.decode() for x in layer_group[f'{position}_U'][:]]

            # Group activations by M emotion and U emotion
            M_by_emotion = defaultdict(list)
            U_by_emotion = defaultdict(list)

            for i, (m_emo, u_emo) in enumerate(zip(M_labels, U_labels)):
                M_by_emotion[m_emo].append(activations[i])
                U_by_emotion[u_emo].append(activations[i])

            # Compute mean directions
            layer_directions = {}
            all_emotions = set(M_by_emotion.keys()) | set(U_by_emotion.keys())

            for emotion in all_emotions:
                M_acts = M_by_emotion.get(emotion, [])
                U_acts = U_by_emotion.get(emotion, [])

                M_dir = np.mean(M_acts, axis=0) if M_acts else np.zeros(activations.shape[1])
                U_dir = np.mean(U_acts, axis=0) if U_acts else np.zeros(activations.shape[1])

                layer_directions[emotion] = (M_dir, U_dir)

            results[layer] = layer_directions

    print(f"  Loaded {len(results)} layers")
    return results


def compute_pairwise_um_similarity(
    directions: Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]
) -> Dict[int, float]:
    """
    Compute average cosine similarity between U and M directions for paired emotions.

    Note: Due to data structure, M and U have disjoint emotion sets (e.g., M=joy always
    pairs with U=sadness). So we compare:
    - M_direction for emotion e1 (samples where M=e1, U=e2)
    - U_direction for emotion e2 (same samples, but labeled by U)

    For paired emotions (e1, e2), compare M_dir[e1] to U_dir[e2].
    High similarity means the model represents the whole scenario similarly.
    Low similarity means M and U emotions have distinct representations.
    """
    emotion_pairs = get_plutchik_emotion_pairs()

    results = {}

    for layer, layer_dirs in sorted(directions.items()):
        similarities = []

        for e1, e2 in emotion_pairs:
            # e1 is always M emotion, e2 is always U emotion in this pair
            if e1 in layer_dirs and e2 in layer_dirs:
                M_dir_e1, _ = layer_dirs[e1]  # M direction when M=e1
                _, U_dir_e2 = layer_dirs[e2]  # U direction when U=e2
                sim = cosine_similarity(M_dir_e1, U_dir_e2)
                similarities.append(sim)

        if similarities:
            results[layer] = np.mean(similarities)
        else:
            results[layer] = 0.0

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare U-M emotion similarity across models')
    parser.add_argument('--qwen', type=Path, default=None,
                        help='Path to Qwen UA emotions HDF5')
    parser.add_argument('--gemma', type=Path, default=None,
                        help='Path to Gemma UA emotions HDF5')
    parser.add_argument('--olmo', type=Path, default=None,
                        help='Path to OLMo UA emotions HDF5')
    parser.add_argument('--position', type=str, default='first_asst_token',
                        choices=['last_user_token', 'first_asst_token', 'final_token'],
                        help='Token position to use')
    parser.add_argument('--output', type=Path, default=None,
                        help='Output plot path')
    args = parser.parse_args()

    # Default paths if not provided
    data_dir = Path(__file__).parent / 'data'
    if args.qwen is None:
        args.qwen = data_dir / 'qwen3_32b_ua_emotions_corrected.h5'
    if args.gemma is None:
        args.gemma = data_dir / 'gemma_27b_it_ua_emotions.h5'
    if args.olmo is None:
        args.olmo = data_dir / 'olmo_32b_think_ua_emotions.h5'
    if args.output is None:
        args.output = Path(__file__).parent / 'outputs' / 'ua_similarity_across_models.png'

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load and process each model
    model_results = {}
    model_configs = {
        'Qwen3-32B': (args.qwen, '#e74c3c'),      # red
        'Gemma-3-27B': (args.gemma, '#3498db'),   # blue
        'OLMo-3-32B': (args.olmo, '#2ecc71'),     # green
    }

    for model_name, (path, color) in model_configs.items():
        if path.exists():
            print(f"\nProcessing {model_name}...")
            directions = load_and_compute_emotion_directions(path, args.position)
            similarities = compute_pairwise_um_similarity(directions)
            model_results[model_name] = (similarities, color)
        else:
            print(f"\nSkipping {model_name} - file not found: {path}")

    if not model_results:
        print("No data files found!")
        return

    # Normalize layers to relative position (0-1) for comparison
    # Since models have different numbers of layers
    print("\n" + "="*60)
    print("RESULTS: Average U-M Cosine Similarity by Layer")
    print("="*60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Absolute layer numbers
    for model_name, (similarities, color) in model_results.items():
        layers = sorted(similarities.keys())
        sims = [similarities[l] for l in layers]
        ax1.plot(layers, sims, 'o-', color=color, label=model_name, linewidth=2, markersize=4)

        print(f"\n{model_name}:")
        print(f"  Layers: {min(layers)} - {max(layers)}")
        print(f"  Avg similarity: {np.mean(sims):.3f}")
        print(f"  Min: {np.min(sims):.3f} (layer {layers[np.argmin(sims)]})")
        print(f"  Max: {np.max(sims):.3f} (layer {layers[np.argmax(sims)]})")

    ax1.set_xlabel('Layer', fontsize=11)
    ax1.set_ylabel('Avg Cosine Similarity (U vs M)', fontsize=11)
    ax1.set_title('U-M Emotion Direction Similarity\n(Absolute Layer Numbers)', fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Plot 2: Normalized layer position (0-1)
    for model_name, (similarities, color) in model_results.items():
        layers = sorted(similarities.keys())
        sims = [similarities[l] for l in layers]

        # Normalize to 0-1 range
        min_layer = min(layers)
        max_layer = max(layers)
        if max_layer > min_layer:
            norm_layers = [(l - min_layer) / (max_layer - min_layer) for l in layers]
        else:
            norm_layers = [0.5] * len(layers)

        ax2.plot(norm_layers, sims, 'o-', color=color, label=model_name, linewidth=2, markersize=4)

    ax2.set_xlabel('Relative Layer Position (0=earliest, 1=latest)', fontsize=11)
    ax2.set_ylabel('Avg Cosine Similarity (U vs M)', fontsize=11)
    ax2.set_title('U-M Emotion Direction Similarity\n(Normalized Layer Position)', fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.suptitle(f'User vs Model Emotion Direction Similarity Across Models\n(Position: {args.position})',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plot to {args.output}")


if __name__ == '__main__':
    main()
