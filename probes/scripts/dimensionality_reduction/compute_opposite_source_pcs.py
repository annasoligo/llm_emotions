#!/usr/bin/env python3
"""
Compute opposite-source emotion PCs for diverse isolation data.

For user emotion probe: compute PCs based on ASSISTANT emotion variance
For assistant emotion probe: compute PCs based on USER emotion variance

This penalizes using features that encode the OTHER person's emotions.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


def load_diverse_isolation_data(
    data_path: str,
    layer: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load diverse isolation data for a specific layer.

    Returns:
        activations: (n_samples, hidden_dim)
        target_emotions: emotion labels for target person
        other_emotions: emotion labels for other person
    """
    with h5py.File(data_path, 'r') as f:
        # Load activations for specific layer
        activations = f[f'layer_{layer}'][:]

        # Load metadata
        metadata_json = f['metadata'][()]
        if isinstance(metadata_json, bytes):
            metadata_json = metadata_json.decode('utf-8')
        metadata = json.loads(metadata_json)

        target_emotions = [entry['emotion'] for entry in metadata]
        other_emotions = [entry['other_emotion'] for entry in metadata]

    return activations, np.array(target_emotions), np.array(other_emotions)


def compute_neutral_pcs(
    emotional_acts: np.ndarray,
    baseline_acts: np.ndarray,
    k: int = 20,
    n_components: int = 100,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute neutral PCs using ratio-based method.

    Args:
        emotional_acts: Activations from emotional samples
        baseline_acts: Activations from neutral baseline
        k: Number of neutral PCs to extract
        n_components: Total PCs to compute before selecting top-k
    """
    # Compute differences
    diffs = emotional_acts - baseline_acts[:len(emotional_acts)]

    # Combine and compute PCA
    combined = np.vstack([baseline_acts, emotional_acts])
    pca = PCA(n_components=n_components)
    pca.fit(combined)

    all_pcs = pca.components_  # (n_components, hidden_dim)

    # Project onto PCs
    baseline_proj = baseline_acts @ all_pcs.T
    diff_proj = diffs @ all_pcs.T

    # Compute variance ratios
    baseline_var = baseline_proj.var(axis=0)
    diff_var = diff_proj.var(axis=0)
    ratio = baseline_var / (diff_var + 1e-8)

    # Select top-k by ratio (highest ratio = most neutral-dominated)
    top_k_indices = np.argsort(ratio)[-k:][::-1]
    neutral_pcs = all_pcs[top_k_indices].T  # (hidden_dim, k)

    info = {
        'n_components': n_components,
        'k': k,
        'mean_ratio': float(ratio[top_k_indices].mean()),
        'explained_variance': pca.explained_variance_ratio_[top_k_indices].tolist(),
    }

    return neutral_pcs, info


def compute_opposite_source_emotion_pcs(
    activations: np.ndarray,
    other_emotions: np.ndarray,
    k: int = 10,
    n_components: int = 100,
) -> Tuple[np.ndarray, Dict]:
    """
    Compute PCs that capture variance in the OTHER person's emotions.

    These PCs encode how the activations vary with the other person's emotion.
    Penalizing these forces the probe to avoid using opposite-source features.

    Args:
        activations: Global activations (n_samples, hidden_dim)
        other_emotions: Emotion labels for the OTHER person
        k: Number of PCs to extract
        n_components: Total PCs to compute
    """
    # Encode emotions as integers
    unique_emotions = sorted(set(other_emotions))
    emotion_to_idx = {e: i for i, e in enumerate(unique_emotions)}
    emotion_indices = np.array([emotion_to_idx[e] for e in other_emotions])

    # Compute PCA on activations
    pca = PCA(n_components=min(n_components, min(activations.shape)))
    pca.fit(activations)

    all_pcs = pca.components_  # (n_components, hidden_dim)

    # Project activations onto PCs
    projected = activations @ all_pcs.T  # (n_samples, n_components)

    # For each PC, compute how much variance it captures in OTHER person's emotions
    # Use ANOVA F-statistic: between-group variance / within-group variance
    pc_scores = []

    for pc_idx in range(all_pcs.shape[0]):
        pc_values = projected[:, pc_idx]

        # Compute between-group variance
        group_means = []
        for emotion_idx in range(len(unique_emotions)):
            mask = emotion_indices == emotion_idx
            if mask.sum() > 0:
                group_means.append(pc_values[mask].mean())

        overall_mean = pc_values.mean()
        between_var = sum(
            ((mean - overall_mean) ** 2) * (emotion_indices == i).sum()
            for i, mean in enumerate(group_means)
        ) / len(unique_emotions)

        # Compute within-group variance
        within_var = sum(
            ((pc_values[emotion_indices == i] - group_means[i]) ** 2).sum()
            for i in range(len(unique_emotions))
        ) / (len(pc_values) - len(unique_emotions))

        # F-statistic (higher = more variance explained by other emotion)
        f_stat = between_var / (within_var + 1e-8)
        pc_scores.append(f_stat)

    pc_scores = np.array(pc_scores)

    # Select top-k PCs by F-statistic
    top_k_indices = np.argsort(pc_scores)[-k:][::-1]
    opposite_source_pcs = all_pcs[top_k_indices].T  # (hidden_dim, k)

    info = {
        'n_components': min(n_components, min(activations.shape)),
        'k': k,
        'mean_f_statistic': float(pc_scores[top_k_indices].mean()),
        'explained_variance': pca.explained_variance_ratio_[top_k_indices].tolist(),
        'total_explained_variance': float(pca.explained_variance_ratio_[top_k_indices].sum()),
    }

    return opposite_source_pcs, info


def main():
    parser = argparse.ArgumentParser(
        description="Compute opposite-source emotion PCs for diverse isolation data"
    )
    parser.add_argument(
        "--isolation-type",
        type=str,
        required=True,
        choices=["user", "assistant"],
        help="Which isolation type to process",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to diverse isolation data",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default="outputs/data/activations/conversations2_neutral.h5",
        help="Path to neutral baseline data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/orthogonal_pcs/diverse_isolation",
        help="Output directory",
    )
    parser.add_argument(
        "--k-neutral",
        type=int,
        default=20,
        help="Number of neutral PCs",
    )
    parser.add_argument(
        "--k-opposite",
        type=int,
        default=10,
        help="Number of opposite-source emotion PCs",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs='+',
        default=list(range(0, 62, 10)),
        help="Layers to process",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir) / args.isolation_type
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"COMPUTING OPPOSITE-SOURCE EMOTION PCS FOR {args.isolation_type.upper()}")
    print("=" * 80)
    print(f"Data: {args.data_path}")
    print(f"Baseline: {args.baseline_path}")
    print(f"k_neutral: {args.k_neutral}")
    print(f"k_opposite: {args.k_opposite}")
    print(f"Layers: {args.layers}")
    print()

    # Load baseline data (for neutral PCs)
    print("Loading baseline data...")
    with h5py.File(args.baseline_path, 'r') as f:
        metadata = json.loads(f['metadata'][()])
        n_baseline = len(metadata)

    # Process each layer
    for layer in tqdm(args.layers, desc="Processing layers"):
        print(f"\n--- Layer {layer} ---")

        # Load emotional data
        activations, target_emotions, other_emotions = load_diverse_isolation_data(
            args.data_path,
            layer,
        )

        # Load baseline data for this layer
        with h5py.File(args.baseline_path, 'r') as f:
            # Load from global activations
            baseline_acts = []
            for i in range(min(n_baseline, len(activations))):
                act = f[f'activations/{i}/global'][layer]
                baseline_acts.append(act)
            baseline_acts = np.stack(baseline_acts, axis=0)

        # Compute neutral PCs
        print(f"  Computing neutral PCs (k={args.k_neutral})...")
        neutral_pcs, neutral_info = compute_neutral_pcs(
            activations,
            baseline_acts,
            k=args.k_neutral,
        )
        print(f"    Mean ratio: {neutral_info['mean_ratio']:.4f}")

        # Compute opposite-source emotion PCs
        opposite_name = "assistant" if args.isolation_type == "user" else "user"
        print(f"  Computing {opposite_name} emotion PCs (k={args.k_opposite})...")
        opposite_pcs, opposite_info = compute_opposite_source_emotion_pcs(
            activations,
            other_emotions,
            k=args.k_opposite,
        )
        print(f"    Mean F-statistic: {opposite_info['mean_f_statistic']:.4f}")
        print(f"    Total explained variance: {opposite_info['total_explained_variance']:.4f}")

        # Combine orthogonal PCs
        orthogonal_pcs = np.hstack([neutral_pcs, opposite_pcs])  # (hidden_dim, k_neutral + k_opposite)

        # Save
        output_path = output_dir / f"layer_{layer}.npz"
        np.savez(
            output_path,
            orthogonal_pcs=orthogonal_pcs,
            neutral_pcs=neutral_pcs,
            opposite_source_pcs=opposite_pcs,
            neutral_info=neutral_info,
            opposite_info=opposite_info,
        )

        print(f"  ✓ Saved to {output_path}")

    print("\n" + "=" * 80)
    print("✓ Done!")
    print(f"Saved PCs for {len(args.layers)} layers to {output_dir}")


if __name__ == "__main__":
    main()
