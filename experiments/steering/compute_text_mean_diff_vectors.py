#!/usr/bin/env python3
"""Compute mean-difference steering vectors directly from text activations.

This creates steering vectors by computing:
    vector[emotion] = mean(emotional_activations) - mean(neutral_activations)

This is simpler than trained probes and may yield more consistent behavioral effects.

Usage:
    python compute_text_mean_diff_vectors.py --layer 30 --output-dir vectors/text_mean_diff
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import h5py
import numpy as np


EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def load_text_activations(
    h5_path: Path, layer: int, template_filter: str | None = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load emotional and neutral activations per emotion.

    Args:
        h5_path: Path to texts_combined.h5
        layer: Which layer to extract (0-indexed)
        template_filter: Optional filter for template type (e.g., 'direct_address')

    Returns:
        Dict[emotion] -> (emotional_acts, neutral_acts) where each is [n_samples, hidden_dim]
    """
    print(f"Loading text activations from {h5_path}")
    print(f"Layer: {layer}")
    if template_filter:
        print(f"Template filter: {template_filter}")

    emotion_data: Dict[str, Tuple[List, List]] = {e: ([], []) for e in EMOTIONS}

    with h5py.File(h5_path, "r") as f:
        activations_group = f["activations"]

        for key in activations_group.keys():
            # Keys are like: batch1_set_0_direct_address_anger
            parts = key.split('_')
            if len(parts) < 4:
                continue

            # Extract emotion from the end of the key
            emotion = parts[-1]
            if emotion not in EMOTIONS:
                continue

            # Check template filter
            if template_filter:
                # Template is in the middle of the key
                # e.g., batch1_set_0_direct_address_anger -> direct_address
                template = '_'.join(parts[3:-1])
                if template != template_filter:
                    continue

            # Load activations for this text
            text_group = activations_group[key]

            if "emotional" not in text_group or "neutral" not in text_group:
                continue

            # Shape is [n_layers, hidden_dim]
            emotional_data = text_group["emotional"][:]
            neutral_data = text_group["neutral"][:]

            if layer >= len(emotional_data):
                continue

            # Get activation for the specified layer
            emotional_act = emotional_data[layer]  # [hidden_dim]
            neutral_act = neutral_data[layer]

            emotion_data[emotion][0].append(emotional_act)
            emotion_data[emotion][1].append(neutral_act)

    # Convert to arrays
    result = {}
    for emotion in EMOTIONS:
        emotional_list, neutral_list = emotion_data[emotion]
        if len(emotional_list) > 0:
            result[emotion] = (
                np.stack(emotional_list).astype(np.float32),
                np.stack(neutral_list).astype(np.float32)
            )
            print(f"  {emotion}: {len(emotional_list)} samples")
        else:
            print(f"  {emotion}: NO SAMPLES FOUND")

    return result


def compute_mean_diff_vectors(
    emotion_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """Compute mean difference vectors for each emotion.

    Args:
        emotion_data: Dict[emotion] -> (emotional_acts, neutral_acts)
        normalize: Whether to normalize vectors to unit length

    Returns:
        Dict[emotion] -> steering_vector [hidden_dim]
    """
    vectors = {}

    for emotion, (emotional_acts, neutral_acts) in emotion_data.items():
        # Compute means
        mean_emotional = emotional_acts.mean(axis=0)
        mean_neutral = neutral_acts.mean(axis=0)

        # Mean difference vector
        diff = mean_emotional - mean_neutral

        if normalize:
            norm = np.linalg.norm(diff)
            if norm > 1e-8:
                diff = diff / norm

        vectors[emotion] = diff

        # Print stats
        diff_norm = np.linalg.norm(diff)
        print(f"  {emotion}: diff_norm={diff_norm:.4f}")

    return vectors


def compute_std_per_vector(
    emotion_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    vectors: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute standard deviation of projections onto each vector.

    This is useful for scaling steering magnitude.
    """
    stds = {}

    for emotion, vector in vectors.items():
        emotional_acts, neutral_acts = emotion_data[emotion]

        # Project all samples onto the vector
        all_acts = np.vstack([emotional_acts, neutral_acts])
        projections = all_acts @ vector

        stds[emotion] = float(np.std(projections))
        print(f"  {emotion}: std={stds[emotion]:.4f}")

    return stds


def save_vectors(
    vectors: Dict[str, np.ndarray],
    stds: Dict[str, float],
    layer: int,
    output_dir: Path,
    suffix: str = "textmeandiff",
):
    """Save steering vectors to npz files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for emotion, vector in vectors.items():
        output_path = output_dir / f"{emotion}_{suffix}_layer{layer}.npz"
        np.savez(
            output_path,
            vector=vector,
            layer=layer,
            emotion=emotion,
            source=f"text_mean_diff_{suffix}",
            normalized=True,
            std=stds.get(emotion, 0.0),
        )
        print(f"Saved: {output_path}")

    # Save all vectors in one file too
    all_vectors = np.stack([vectors[e] for e in EMOTIONS])
    all_path = output_dir / f"all_emotions_{suffix}_layer{layer}.npz"
    np.savez(
        all_path,
        vectors=all_vectors,
        emotions=EMOTIONS,
        layer=layer,
        stds=[stds.get(e, 0.0) for e in EMOTIONS],
    )
    print(f"Saved all: {all_path}")


def compare_with_existing(
    vectors: Dict[str, np.ndarray],
    existing_dir: Path,
    layer: int,
    existing_suffix: str = "textraw",
):
    """Compare new vectors with existing textraw vectors."""
    print(f"\nComparing with existing {existing_suffix} vectors...")

    for emotion, new_vec in vectors.items():
        existing_path = existing_dir / f"{emotion}_{existing_suffix}_layer{layer}.npz"
        if existing_path.exists():
            data = np.load(existing_path)
            existing_vec = data["vector"]

            # Cosine similarity
            cosine_sim = np.dot(new_vec, existing_vec) / (
                np.linalg.norm(new_vec) * np.linalg.norm(existing_vec)
            )
            print(f"  {emotion}: cosine_sim with {existing_suffix} = {cosine_sim:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="outputs/data/activations/texts_combined.h5",
        help="Path to text activations HDF5",
    )
    parser.add_argument("--layer", type=int, required=True, help="Layer to extract")
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        choices=["direct_address", "second_person_eliciting", "third_person"],
        help="Optional filter for template type",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/steering/vectors"),
        help="Output directory",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with existing textraw vectors",
    )

    args = parser.parse_args()

    # Resolve data path
    data_path = Path(args.data)
    if not data_path.is_absolute():
        # Try relative to repo root
        repo_root = Path(__file__).parent.parent.parent
        data_path = repo_root / args.data

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return

    # Load data
    emotion_data = load_text_activations(data_path, args.layer, args.template)

    if not emotion_data:
        print("ERROR: No data loaded!")
        return

    # Compute vectors
    print("\nComputing mean difference vectors...")
    vectors = compute_mean_diff_vectors(emotion_data, normalize=True)

    # Compute std for each direction
    print("\nComputing projection standard deviations...")
    stds = compute_std_per_vector(emotion_data, vectors)

    # Determine suffix based on template filter
    suffix = "textmeandiff"
    if args.template:
        suffix = f"textmeandiff_{args.template}"

    # Save
    print(f"\nSaving vectors to {args.output_dir}...")
    save_vectors(vectors, stds, args.layer, args.output_dir, suffix)

    # Compare with existing
    if args.compare:
        compare_with_existing(vectors, args.output_dir, args.layer, "textraw")

    # Print summary for config.py
    print("\n" + "="*60)
    print("DIRECTION_STD values for config.py:")
    print("="*60)
    for emotion in EMOTIONS:
        std = stds.get(emotion, 0.0)
        print(f'    "{emotion}_{suffix}": {std:.2f},')


if __name__ == "__main__":
    main()
