#!/usr/bin/env python3
"""
Compute M and U direction vectors from UA emotion disentanglement data.

For each emotion pair (e.g., joy/sadness):
- M direction: mean(activations where M=joy) - mean(activations where M=sadness)
- U direction: mean(activations where U=joy) - mean(activations where U=sadness)

This creates vectors that capture the model's representation of its OWN emotional
state (M) separately from the user's emotional state (U).

Usage:
    python compute_ua_direction_vectors.py --data qwen3_32b_ua_emotions_corrected.h5 --layer 30
    python compute_ua_direction_vectors.py --data olmo_32b_think_ua_emotions.h5 --layer 30
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict


# Plutchik emotion pairs (opposites)
EMOTION_PAIRS = [
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

# Create lookup for opposites
OPPOSITES = {}
for e1, e2 in EMOTION_PAIRS:
    OPPOSITES[e1] = e2
    OPPOSITES[e2] = e1


def load_ua_data(h5_path: Path, layer: int, position: str = 'first_asst_token') -> Tuple[np.ndarray, list, list]:
    """
    Load activations and metadata from UA emotions HDF5 file.

    Returns:
        activations: [n_samples, hidden_dim]
        M_values: list of M emotion labels
        U_values: list of U emotion labels
    """
    with h5py.File(h5_path, 'r') as f:
        layer_group = f[f'layer_{layer}']

        activations = layer_group[f'{position}_activations'][:]
        M_values = [m.decode() for m in layer_group[f'{position}_M'][:]]
        U_values = [u.decode() for u in layer_group[f'{position}_U'][:]]

        print(f"Loaded {len(activations)} samples from layer {layer}, position {position}")
        print(f"  Activations shape: {activations.shape}")
        print(f"  Unique M emotions: {len(set(M_values))}")
        print(f"  Unique U emotions: {len(set(U_values))}")

    return activations, M_values, U_values


def compute_emotion_means(
    activations: np.ndarray,
    M_values: list,
    U_values: list,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute mean activation for each M and U emotion.

    Returns:
        M_means: {emotion: mean_activation}
        U_means: {emotion: mean_activation}
    """
    # Group activations by M and U values
    M_groups = defaultdict(list)
    U_groups = defaultdict(list)

    for i, (m, u) in enumerate(zip(M_values, U_values)):
        M_groups[m].append(activations[i])
        U_groups[u].append(activations[i])

    # Compute means
    M_means = {emotion: np.mean(acts, axis=0) for emotion, acts in M_groups.items()}
    U_means = {emotion: np.mean(acts, axis=0) for emotion, acts in U_groups.items()}

    print(f"\nM emotion means computed for: {sorted(M_means.keys())}")
    print(f"U emotion means computed for: {sorted(U_means.keys())}")

    return M_means, U_means


def compute_direction_vectors(
    M_means: Dict[str, np.ndarray],
    U_means: Dict[str, np.ndarray],
    normalize: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute direction vectors as: emotion - opposite(emotion).

    Returns:
        M_directions: {emotion: direction_vector}
        U_directions: {emotion: direction_vector}
    """
    M_directions = {}
    U_directions = {}

    print("\nComputing direction vectors...")

    for emotion in M_means.keys():
        if emotion not in OPPOSITES:
            print(f"  Skipping {emotion} (no opposite defined)")
            continue

        opposite = OPPOSITES[emotion]
        if opposite not in M_means:
            print(f"  Skipping {emotion} (opposite {opposite} not in data)")
            continue

        # M direction
        m_dir = M_means[emotion] - M_means[opposite]
        m_norm = np.linalg.norm(m_dir)
        if normalize and m_norm > 1e-8:
            m_dir = m_dir / m_norm
        M_directions[emotion] = m_dir

        # U direction
        if emotion in U_means and opposite in U_means:
            u_dir = U_means[emotion] - U_means[opposite]
            u_norm = np.linalg.norm(u_dir)
            if normalize and u_norm > 1e-8:
                u_dir = u_dir / u_norm
            U_directions[emotion] = u_dir

        print(f"  {emotion}: M_norm={m_norm:.2f}, U_norm={np.linalg.norm(U_means[emotion] - U_means[opposite]):.2f}")

    return M_directions, U_directions


def compute_projection_stats(
    activations: np.ndarray,
    M_values: list,
    directions: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Compute projection statistics for each direction."""
    stats = {}

    for emotion, direction in directions.items():
        projections = activations @ direction
        stats[emotion] = {
            'mean': float(np.mean(projections)),
            'std': float(np.std(projections)),
            'min': float(np.min(projections)),
            'max': float(np.max(projections)),
        }

    return stats


def save_vectors(
    M_directions: Dict[str, np.ndarray],
    U_directions: Dict[str, np.ndarray],
    M_stats: Dict[str, Dict[str, float]],
    U_stats: Dict[str, Dict[str, float]],
    layer: int,
    model_name: str,
    output_dir: Path,
):
    """Save direction vectors to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine model prefix
    if 'qwen' in model_name.lower():
        prefix = 'qwen'
    elif 'olmo' in model_name.lower():
        prefix = 'olmo'
    elif 'gemma' in model_name.lower():
        prefix = 'gemma'
    else:
        prefix = 'model'

    # Save individual M vectors
    for emotion, vector in M_directions.items():
        output_path = output_dir / f"M_{emotion}_{prefix}_ua_layer{layer}.npz"
        np.savez(
            output_path,
            vector=vector.astype(np.float32),
            layer=layer,
            emotion=emotion,
            role='M',  # Model's emotion
            source=f'{prefix}_ua_meandiff',
            normalized=True,
            projection_std=M_stats[emotion]['std'],
        )

    # Save individual U vectors
    for emotion, vector in U_directions.items():
        output_path = output_dir / f"U_{emotion}_{prefix}_ua_layer{layer}.npz"
        np.savez(
            output_path,
            vector=vector.astype(np.float32),
            layer=layer,
            emotion=emotion,
            role='U',  # User's emotion
            source=f'{prefix}_ua_meandiff',
            normalized=True,
            projection_std=U_stats[emotion]['std'],
        )

    # Save combined file with all vectors
    combined_path = output_dir / f"ua_directions_{prefix}_layer{layer}.npz"
    save_dict = {
        'layer': layer,
        'model': model_name,
        'emotions': list(M_directions.keys()),
    }
    for emotion in M_directions:
        save_dict[f'M_{emotion}'] = M_directions[emotion]
        save_dict[f'M_{emotion}_std'] = M_stats[emotion]['std']
    for emotion in U_directions:
        save_dict[f'U_{emotion}'] = U_directions[emotion]
        save_dict[f'U_{emotion}_std'] = U_stats[emotion]['std']

    np.savez(combined_path, **save_dict)

    print(f"\nSaved {len(M_directions)} M vectors and {len(U_directions)} U vectors")
    print(f"Combined file: {combined_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute UA direction vectors')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to UA emotions HDF5 file')
    parser.add_argument('--layer', type=int, default=30,
                        help='Layer to extract')
    parser.add_argument('--position', type=str, default='first_asst_token',
                        choices=['last_user_token', 'first_asst_token', 'final_token'],
                        help='Token position to use')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('vectors'),
                        help='Output directory')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Skip L2 normalization')
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return

    print("=" * 60)
    print("UA DIRECTION VECTOR COMPUTATION")
    print("=" * 60)
    print(f"Data: {data_path.name}")
    print(f"Layer: {args.layer}")
    print(f"Position: {args.position}")
    print("=" * 60)

    # Load data
    activations, M_values, U_values = load_ua_data(
        data_path, args.layer, args.position
    )

    # Get model name from file
    with h5py.File(data_path, 'r') as f:
        model_name = str(f.attrs.get('model_name', 'unknown'))
    print(f"Model: {model_name}")

    # Compute means
    M_means, U_means = compute_emotion_means(activations, M_values, U_values)

    # Compute directions
    M_directions, U_directions = compute_direction_vectors(
        M_means, U_means, normalize=not args.no_normalize
    )

    # Compute projection stats
    print("\nComputing projection statistics...")
    M_stats = compute_projection_stats(activations, M_values, M_directions)
    U_stats = compute_projection_stats(activations, U_values, U_directions)

    # Save
    save_vectors(
        M_directions, U_directions,
        M_stats, U_stats,
        args.layer, model_name, args.output_dir
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
