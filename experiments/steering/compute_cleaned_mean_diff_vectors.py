#!/usr/bin/env python3
"""Compute mean-diff steering vectors with neutral PC removal.

This creates cleaner steering vectors by:
1. Computing PCs on ALL data (emotional + neutral combined)
2. Computing PCs on NEUTRAL data only
3. Combining both PC sets
4. For each PC: compute ratio = var(neutral_proj) / var(diff_proj)
5. Remove top-k PCs with highest ratio from mean diffs
6. Optionally subtract global emotion direction for contrast

Usage:
    python compute_cleaned_mean_diff_vectors.py --layer 30 --k 20
    python compute_cleaned_mean_diff_vectors.py --layer 30 --k 20 --subtract-global
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from sklearn.decomposition import PCA


EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def load_text_activations(
    h5_path: Path, layer: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load emotional and neutral activations per emotion.

    Returns:
        emotional_acts: Dict[emotion] -> [n_samples, hidden_dim]
        neutral_acts: Dict[emotion] -> [n_samples, hidden_dim]
    """
    print(f"Loading text activations from {h5_path}")
    print(f"Layer: {layer}")

    emotional_acts: Dict[str, List] = {e: [] for e in EMOTIONS}
    neutral_acts: Dict[str, List] = {e: [] for e in EMOTIONS}

    with h5py.File(h5_path, "r") as f:
        activations_group = f["activations"]

        for key in activations_group.keys():
            parts = key.split('_')
            if len(parts) < 4:
                continue

            emotion = parts[-1]
            if emotion not in EMOTIONS:
                continue

            text_group = activations_group[key]
            if "emotional" not in text_group or "neutral" not in text_group:
                continue

            emotional_data = text_group["emotional"][:]
            neutral_data = text_group["neutral"][:]

            if layer >= len(emotional_data):
                continue

            emotional_acts[emotion].append(emotional_data[layer])
            neutral_acts[emotion].append(neutral_data[layer])

    # Convert to arrays
    emotional_arrays = {}
    neutral_arrays = {}
    for emotion in EMOTIONS:
        if len(emotional_acts[emotion]) > 0:
            emotional_arrays[emotion] = np.stack(emotional_acts[emotion]).astype(np.float32)
            neutral_arrays[emotion] = np.stack(neutral_acts[emotion]).astype(np.float32)
            print(f"  {emotion}: {len(emotional_acts[emotion])} samples")

    return emotional_arrays, neutral_arrays


def compute_combined_pcs(
    emotional_acts: Dict[str, np.ndarray],
    neutral_acts: Dict[str, np.ndarray],
    n_pcs_all: int = 50,
    n_pcs_neutral: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCs on combined data and neutral-only data.

    Returns:
        pcs_all: [n_pcs_all, hidden_dim] - PCs from all data
        pcs_neutral: [n_pcs_neutral, hidden_dim] - PCs from neutral only
        combined_pcs: [n_pcs_all + n_pcs_neutral, hidden_dim] - concatenated
    """
    # Stack all emotional and neutral activations
    all_emotional = np.vstack([emotional_acts[e] for e in EMOTIONS])
    all_neutral = np.vstack([neutral_acts[e] for e in EMOTIONS])

    print(f"\nComputing PCs...")
    print(f"  All emotional: {all_emotional.shape}")
    print(f"  All neutral: {all_neutral.shape}")

    # PCA on combined (emotional + neutral)
    combined = np.vstack([all_emotional, all_neutral])
    print(f"  Combined: {combined.shape}")

    pca_all = PCA(n_components=n_pcs_all)
    pca_all.fit(combined)
    pcs_all = pca_all.components_
    print(f"  PCs from all data: {pcs_all.shape}, var explained: {pca_all.explained_variance_ratio_.sum():.3f}")

    # PCA on neutral only
    pca_neutral = PCA(n_components=n_pcs_neutral)
    pca_neutral.fit(all_neutral)
    pcs_neutral = pca_neutral.components_
    print(f"  PCs from neutral: {pcs_neutral.shape}, var explained: {pca_neutral.explained_variance_ratio_.sum():.3f}")

    # Combine
    combined_pcs = np.vstack([pcs_all, pcs_neutral])
    print(f"  Combined PCs: {combined_pcs.shape}")

    return pcs_all, pcs_neutral, combined_pcs


def compute_pc_ratios(
    combined_pcs: np.ndarray,
    all_neutral: np.ndarray,
    all_diffs: np.ndarray,
) -> np.ndarray:
    """Compute neutral/diff variance ratio for each PC.

    Args:
        combined_pcs: [n_pcs, hidden_dim]
        all_neutral: [n_samples, hidden_dim]
        all_diffs: [n_samples, hidden_dim]

    Returns:
        ratios: [n_pcs] - ratio of neutral variance to diff variance per PC
    """
    # Project onto each PC
    neutral_proj = all_neutral @ combined_pcs.T  # [n_samples, n_pcs]
    diff_proj = all_diffs @ combined_pcs.T  # [n_samples, n_pcs]

    # Compute variance
    neutral_var = neutral_proj.var(axis=0)  # [n_pcs]
    diff_var = diff_proj.var(axis=0)  # [n_pcs]

    # Ratio (high = neutral-dominated)
    ratios = neutral_var / (diff_var + 1e-8)

    return ratios


def remove_high_ratio_pcs(
    vector: np.ndarray,
    combined_pcs: np.ndarray,
    ratios: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, Dict]:
    """Remove top-k highest ratio PCs from a vector.

    Args:
        vector: [hidden_dim] - the vector to clean
        combined_pcs: [n_pcs, hidden_dim]
        ratios: [n_pcs] - neutral/diff variance ratios
        k: number of PCs to remove

    Returns:
        cleaned: [hidden_dim] - vector with neutral PCs removed
        info: diagnostic info
    """
    # Find top-k highest ratio PCs
    top_k_indices = np.argsort(ratios)[-k:]
    pcs_to_remove = combined_pcs[top_k_indices]

    # Project vector onto these PCs
    projections = vector @ pcs_to_remove.T  # [k]

    # Subtract projections
    cleaned = vector - projections @ pcs_to_remove

    info = {
        "k": k,
        "removed_pc_indices": top_k_indices.tolist(),
        "removed_pc_ratios": ratios[top_k_indices].tolist(),
        "mean_ratio_removed": float(ratios[top_k_indices].mean()),
        "mean_ratio_kept": float(np.delete(ratios, top_k_indices).mean()),
        "projection_magnitudes": np.abs(projections).tolist(),
    }

    return cleaned, info


def compute_cleaned_mean_diffs(
    emotional_acts: Dict[str, np.ndarray],
    neutral_acts: Dict[str, np.ndarray],
    k: int = 20,
    n_pcs_all: int = 50,
    n_pcs_neutral: int = 50,
    subtract_global: bool = False,
    normalize: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Compute cleaned mean-diff vectors for each emotion.

    Args:
        emotional_acts: Dict[emotion] -> [n_samples, hidden_dim]
        neutral_acts: Dict[emotion] -> [n_samples, hidden_dim]
        k: number of high-ratio PCs to remove
        n_pcs_all: PCs to compute on combined data
        n_pcs_neutral: PCs to compute on neutral data
        subtract_global: whether to subtract global emotion direction
        normalize: whether to normalize final vectors

    Returns:
        vectors: Dict[emotion] -> [hidden_dim] cleaned steering vectors
        diagnostics: diagnostic information
    """
    # Compute combined PCs
    pcs_all, pcs_neutral, combined_pcs = compute_combined_pcs(
        emotional_acts, neutral_acts, n_pcs_all, n_pcs_neutral
    )

    # Stack all data for ratio computation
    all_emotional = np.vstack([emotional_acts[e] for e in EMOTIONS])
    all_neutral = np.vstack([neutral_acts[e] for e in EMOTIONS])
    all_diffs = all_emotional - all_neutral

    # Compute ratios
    print(f"\nComputing PC ratios...")
    ratios = compute_pc_ratios(combined_pcs, all_neutral, all_diffs)
    print(f"  Ratio range: [{ratios.min():.3f}, {ratios.max():.3f}]")
    print(f"  Top-{k} ratios: {sorted(ratios)[-k:][::-1][:5]}...")  # Show top 5

    # Compute raw mean diffs per emotion
    print(f"\nComputing mean diffs...")
    raw_mean_diffs = {}
    for emotion in EMOTIONS:
        diffs = emotional_acts[emotion] - neutral_acts[emotion]
        raw_mean_diffs[emotion] = diffs.mean(axis=0)

    # Clean each mean diff
    print(f"\nRemoving top-{k} neutral-dominated PCs...")
    cleaned_vectors = {}
    removal_info = {}

    for emotion in EMOTIONS:
        cleaned, info = remove_high_ratio_pcs(
            raw_mean_diffs[emotion], combined_pcs, ratios, k
        )
        cleaned_vectors[emotion] = cleaned
        removal_info[emotion] = info
        print(f"  {emotion}: removed {info['mean_ratio_removed']:.3f} avg ratio PCs")

    # Optionally subtract global emotion direction
    if subtract_global:
        print(f"\nSubtracting global emotion direction...")
        global_dir = np.mean([cleaned_vectors[e] for e in EMOTIONS], axis=0)
        global_norm = np.linalg.norm(global_dir)
        print(f"  Global direction norm: {global_norm:.4f}")

        for emotion in EMOTIONS:
            cleaned_vectors[emotion] = cleaned_vectors[emotion] - global_dir

    # Normalize
    if normalize:
        for emotion in EMOTIONS:
            norm = np.linalg.norm(cleaned_vectors[emotion])
            if norm > 1e-8:
                cleaned_vectors[emotion] = cleaned_vectors[emotion] / norm

    # Compute diagnostics
    diagnostics = {
        "k": k,
        "n_pcs_all": n_pcs_all,
        "n_pcs_neutral": n_pcs_neutral,
        "subtract_global": subtract_global,
        "total_combined_pcs": combined_pcs.shape[0],
        "ratio_stats": {
            "min": float(ratios.min()),
            "max": float(ratios.max()),
            "mean": float(ratios.mean()),
            "median": float(np.median(ratios)),
        },
        "removal_info": removal_info,
    }

    return cleaned_vectors, diagnostics


def compute_stds(
    emotional_acts: Dict[str, np.ndarray],
    neutral_acts: Dict[str, np.ndarray],
    vectors: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute std of projections onto each vector."""
    stds = {}
    for emotion, vector in vectors.items():
        all_acts = np.vstack([emotional_acts[emotion], neutral_acts[emotion]])
        projections = all_acts @ vector
        stds[emotion] = float(np.std(projections))
    return stds


def save_vectors(
    vectors: Dict[str, np.ndarray],
    stds: Dict[str, float],
    diagnostics: Dict,
    layer: int,
    output_dir: Path,
    suffix: str,
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
            source=f"cleaned_mean_diff_{suffix}",
            normalized=True,
            std=stds.get(emotion, 0.0),
            k=diagnostics["k"],
        )
        print(f"Saved: {output_path}")

    # Save diagnostics
    import json
    diag_path = output_dir / f"diagnostics_{suffix}_layer{layer}.json"
    with open(diag_path, 'w') as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Saved diagnostics: {diag_path}")


def compare_with_raw(
    cleaned_vectors: Dict[str, np.ndarray],
    raw_dir: Path,
    layer: int,
):
    """Compare cleaned vectors with raw mean-diff vectors."""
    print(f"\nComparing with raw textmeandiff vectors...")

    for emotion, cleaned in cleaned_vectors.items():
        raw_path = raw_dir / f"{emotion}_textmeandiff_layer{layer}.npz"
        if raw_path.exists():
            data = np.load(raw_path)
            raw = data["vector"]
            cosine = np.dot(cleaned, raw) / (np.linalg.norm(cleaned) * np.linalg.norm(raw))
            print(f"  {emotion}: cosine_sim = {cosine:.4f}")


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
        "--k", type=int, default=20, help="Number of high-ratio PCs to remove"
    )
    parser.add_argument(
        "--n-pcs-all", type=int, default=50, help="PCs from combined data"
    )
    parser.add_argument(
        "--n-pcs-neutral", type=int, default=50, help="PCs from neutral data"
    )
    parser.add_argument(
        "--subtract-global",
        action="store_true",
        help="Subtract global emotion direction for contrast",
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
        help="Compare with raw textmeandiff vectors",
    )

    args = parser.parse_args()

    # Resolve data path
    data_path = Path(args.data)
    if not data_path.is_absolute():
        repo_root = Path(__file__).parent.parent.parent
        data_path = repo_root / args.data

    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return

    # Load data
    emotional_acts, neutral_acts = load_text_activations(data_path, args.layer)

    if not emotional_acts:
        print("ERROR: No data loaded!")
        return

    # Compute cleaned vectors
    cleaned_vectors, diagnostics = compute_cleaned_mean_diffs(
        emotional_acts,
        neutral_acts,
        k=args.k,
        n_pcs_all=args.n_pcs_all,
        n_pcs_neutral=args.n_pcs_neutral,
        subtract_global=args.subtract_global,
        normalize=True,
    )

    # Compute stds
    print(f"\nComputing projection standard deviations...")
    stds = compute_stds(emotional_acts, neutral_acts, cleaned_vectors)
    for emotion, std in stds.items():
        print(f"  {emotion}: std={std:.4f}")

    # Determine suffix
    suffix = f"textcleaned_k{args.k}"
    if args.subtract_global:
        suffix += "_subglobal"

    # Save
    print(f"\nSaving vectors...")
    save_vectors(cleaned_vectors, stds, diagnostics, args.layer, args.output_dir, suffix)

    # Compare
    if args.compare:
        compare_with_raw(cleaned_vectors, args.output_dir, args.layer)

    # Print config.py values
    print("\n" + "=" * 60)
    print(f"DIRECTION_STD values for config.py ({suffix}):")
    print("=" * 60)
    for emotion in EMOTIONS:
        std = stds.get(emotion, 0.0)
        print(f'    "{emotion}_{suffix}": {std:.2f},')


if __name__ == "__main__":
    main()
