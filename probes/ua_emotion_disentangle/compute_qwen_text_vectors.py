#!/usr/bin/env python3
"""Compute mean-difference steering vectors from Qwen text activations.

Creates steering vectors by computing:
    vector[emotion] = mean(emotional_activations) - mean(neutral_activations)

Usage:
    python compute_qwen_text_vectors.py --layer 30 --output-dir vectors
"""

import argparse
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
        h5_path: Path to qwen texts h5 file
        layer: Which layer to extract (actual layer number like 30)
        template_filter: Optional filter for template type (e.g., 'direct_address')

    Returns:
        Dict[emotion] -> (emotional_acts, neutral_acts) where each is [n_samples, hidden_dim]
    """
    print(f"Loading text activations from {h5_path}")
    print(f"Target layer: {layer}")
    if template_filter:
        print(f"Template filter: {template_filter}")

    emotion_data: Dict[str, Tuple[List, List]] = {e: ([], []) for e in EMOTIONS}

    with h5py.File(h5_path, "r") as f:
        # Get layer list to find index
        layers = list(f.attrs['layers'])
        print(f"Available layers: {min(layers)} to {max(layers)}")

        if layer not in layers:
            raise ValueError(f"Layer {layer} not in available layers: {layers}")

        layer_idx = layers.index(layer)
        print(f"Layer {layer} is at index {layer_idx}")

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

            # Get activation for the specified layer
            emotional_act = emotional_data[layer_idx]  # [hidden_dim]
            neutral_act = neutral_data[layer_idx]

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
    """Compute mean difference vectors for each emotion."""
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
        print(f"  {emotion}: norm={diff_norm:.4f}")

    return vectors


def compute_projection_std(
    emotion_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    vectors: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute standard deviation of projections onto each vector."""
    stds = {}

    for emotion, vector in vectors.items():
        emotional_acts, neutral_acts = emotion_data[emotion]

        # Project all samples onto the vector
        all_acts = np.vstack([emotional_acts, neutral_acts])
        projections = all_acts @ vector

        stds[emotion] = float(np.std(projections))
        print(f"  {emotion}: projection_std={stds[emotion]:.2f}")

    return stds


def compute_baseline_std(
    emotion_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> float:
    """Compute baseline activation std across all samples."""
    all_acts = []
    for emotion, (emotional_acts, neutral_acts) in emotion_data.items():
        all_acts.append(emotional_acts)
        all_acts.append(neutral_acts)

    combined = np.vstack(all_acts)
    # Compute std per dimension then average, or overall
    overall_std = float(np.std(combined))
    per_dim_std = float(np.std(combined, axis=0).mean())
    l2_norm_mean = float(np.linalg.norm(combined, axis=1).mean())

    print(f"\nBaseline statistics:")
    print(f"  Overall std: {overall_std:.2f}")
    print(f"  Per-dim std (mean): {per_dim_std:.2f}")
    print(f"  L2 norm (mean): {l2_norm_mean:.2f}")

    return per_dim_std


def save_vectors(
    vectors: Dict[str, np.ndarray],
    stds: Dict[str, float],
    layer: int,
    output_dir: Path,
    baseline_std: float,
):
    """Save steering vectors to npz files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for emotion, vector in vectors.items():
        output_path = output_dir / f"{emotion}_qwen_textmeandiff_layer{layer}.npz"
        np.savez(
            output_path,
            vector=vector,
            layer=layer,
            emotion=emotion,
            source="qwen_text_mean_diff",
            normalized=True,
            projection_std=stds.get(emotion, 0.0),
            baseline_std=baseline_std,
        )
        print(f"Saved: {output_path}")

    # Save all vectors in one file too
    all_vectors = np.stack([vectors[e] for e in EMOTIONS if e in vectors])
    emotions_saved = [e for e in EMOTIONS if e in vectors]
    all_path = output_dir / f"all_emotions_qwen_textmeandiff_layer{layer}.npz"
    np.savez(
        all_path,
        vectors=all_vectors,
        emotions=emotions_saved,
        layer=layer,
        projection_stds=[stds.get(e, 0.0) for e in emotions_saved],
        baseline_std=baseline_std,
    )
    print(f"Saved all: {all_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="/workspace-vast/annas/git/research-tools/probes/ua_emotion_disentangle/data/qwen3_32b_texts_combined.h5",
        help="Path to Qwen text activations HDF5",
    )
    parser.add_argument("--layer", type=int, default=30, help="Layer to extract")
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Optional filter for template type",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace-vast/annas/git/research-tools/probes/ua_emotion_disentangle/vectors"),
        help="Output directory",
    )

    args = parser.parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return

    print("=" * 60)
    print("QWEN TEXT MEAN-DIFF VECTOR COMPUTATION")
    print("=" * 60)

    # Load data
    emotion_data = load_text_activations(data_path, args.layer, args.template)

    if not emotion_data:
        print("ERROR: No data loaded!")
        return

    # Compute baseline std
    baseline_std = compute_baseline_std(emotion_data)

    # Compute vectors
    print("\nComputing mean difference vectors...")
    vectors = compute_mean_diff_vectors(emotion_data, normalize=True)

    # Compute std for each direction
    print("\nComputing projection standard deviations...")
    stds = compute_projection_std(emotion_data, vectors)

    # Save
    print(f"\nSaving vectors to {args.output_dir}...")
    save_vectors(vectors, stds, args.layer, args.output_dir, baseline_std)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Layer: {args.layer}")
    print(f"Baseline std: {baseline_std:.2f}")
    print("\nProjection stds for config:")
    for emotion in EMOTIONS:
        if emotion in stds:
            print(f'    "{emotion}_qwen_textmeandiff": {stds[emotion]:.2f},')


if __name__ == "__main__":
    main()
