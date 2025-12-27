#!/usr/bin/env python3
"""Tune k parameter for ratio PCA by maximizing silhouette score.

Similar to alpha tuning in cPCA, but for ratio PCA's k parameter.

Usage:
    # Tune k for layer 30
    python scripts/tune_ratio_pca_k.py \
        --activations data/activations/texts_combined.h5 \
        --layer 30 \
        --k-values 3 5 7 10 15 20 25 30 \
        --n-components 50 \
        --output results/ratio_pca_k_tuning/layer30_results.json

    # Tune across multiple layers
    python scripts/tune_ratio_pca_k.py \
        --activations data/activations/texts_combined.h5 \
        --layers 20 30 40 50 \
        --k-values 5 10 15 20 \
        --output results/ratio_pca_k_tuning/multi_layer_results.json
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import h5py
import numpy as np
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from probes.methods.pca_ratio import ratio_based_neutral_removal


def load_activations_layer(h5_path: Path, layer: int) -> tuple:
    """Load activations for a specific layer.

    Returns:
        emotional: [n_pairs, hidden_dim]
        neutral: [n_pairs, hidden_dim]
        labels: [n_pairs] emotion labels
        pair_ids: [n_pairs] pair IDs
    """
    with h5py.File(h5_path, 'r') as f:
        metadata = json.loads(f['metadata'][()])
        pair_ids = list(f['activations'].keys())

        emotional = []
        neutral = []
        labels = []

        for pair_id in pair_ids:
            emotional.append(f[f'activations/{pair_id}/emotional'][layer])
            neutral.append(f[f'activations/{pair_id}/neutral'][layer])
            # Get emotion label from metadata
            pair_meta = next(m for m in metadata if m['id'] == pair_id)
            labels.append(pair_meta['emotion'])

        emotional = np.array(emotional)
        neutral = np.array(neutral)
        labels = np.array(labels)

    return emotional, neutral, labels, pair_ids


def compute_silhouette_ratio_pca(
    diffs: np.ndarray,
    neutral: np.ndarray,
    emotional: np.ndarray,
    labels: np.ndarray,
    k: int,
    n_components: int = 50,
    n_dims_score: int = 10,
    n_pcs_all: int = 100,
) -> float:
    """Compute silhouette score for ratio PCA with given k.

    Args:
        diffs: Activation differences [n_samples, hidden_dim]
        neutral: Neutral activations [n_samples, hidden_dim]
        emotional: Emotional activations [n_samples, hidden_dim]
        labels: Emotion labels [n_samples]
        k: Number of neutral-dominated PCs to remove
        n_components: Number of final PCs to compute
        n_dims_score: Number of top PCs to use for scoring
        n_pcs_all: Number of PCs for ratio analysis

    Returns:
        Silhouette score (-1 to 1, higher is better)
    """
    # Run ratio-based neutral removal
    cleaned_diffs, removal_info = ratio_based_neutral_removal(
        diffs=diffs,
        neutral=neutral,
        emotional=emotional,
        k=k,
        n_pcs_all=n_pcs_all,
    )

    # Run PCA on cleaned diffs
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    pca.fit(cleaned_diffs)
    components = pca.components_  # [n_components, hidden_dim]

    # Project diffs onto top-N components
    projected = diffs @ components[:n_dims_score].T

    # Compute silhouette score
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError(f"Need at least 2 emotions, got {len(unique_labels)}")

    return silhouette_score(projected, labels)


def tune_k_single_layer(
    activations_path: Path,
    layer: int,
    k_values: List[int],
    n_components: int = 50,
    n_dims_score: int = 10,
    n_pcs_all: int = 100,
    max_samples: int = None,
) -> Dict:
    """Tune k for a single layer.

    Returns:
        results: Dict with scores, best_k, etc.
    """
    print(f"\n{'='*80}")
    print(f"TUNING K FOR LAYER {layer}")
    print(f"{'='*80}")

    # Load activations
    print("Loading activations...")
    emotional, neutral, labels, pair_ids = load_activations_layer(
        activations_path, layer
    )

    if max_samples and len(pair_ids) > max_samples:
        indices = np.random.choice(len(pair_ids), max_samples, replace=False)
        emotional = emotional[indices]
        neutral = neutral[indices]
        labels = labels[indices]
        pair_ids = [pair_ids[i] for i in indices]
        print(f"Limited to {max_samples} samples")

    print(f"Loaded {len(pair_ids)} pairs")
    print(f"Emotions: {np.unique(labels)}")
    print(f"Shape: {emotional.shape}")

    # Compute diffs
    diffs = emotional - neutral

    # Try each k value
    scores = []
    print(f"\nTrying k values: {k_values}")

    for k in tqdm(k_values, desc="Tuning k"):
        try:
            score = compute_silhouette_ratio_pca(
                diffs=diffs,
                neutral=neutral,
                emotional=emotional,
                labels=labels,
                k=k,
                n_components=n_components,
                n_dims_score=n_dims_score,
                n_pcs_all=n_pcs_all,
            )
            scores.append(score)
            print(f"  k={k:3d}: silhouette={score:.4f}")
        except Exception as e:
            print(f"  k={k:3d}: ERROR - {e}")
            scores.append(np.nan)

    scores = np.array(scores)

    # Find best k
    valid_scores = ~np.isnan(scores)
    if not valid_scores.any():
        raise ValueError("All k values failed")

    best_idx = np.nanargmax(scores)
    best_k = k_values[best_idx]
    best_score = scores[best_idx]

    results = {
        "layer": layer,
        "k_values": k_values,
        "silhouette_scores": scores.tolist(),
        "best_k": int(best_k),
        "best_score": float(best_score),
        "n_samples": len(pair_ids),
        "n_components": n_components,
        "n_dims_score": n_dims_score,
    }

    print(f"\n{'='*80}")
    print(f"RESULTS FOR LAYER {layer}")
    print(f"{'='*80}")
    print(f"Best k: {best_k}")
    print(f"Best silhouette score: {best_score:.4f}")
    print(f"Score range: [{np.nanmin(scores):.4f}, {np.nanmax(scores):.4f}]")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Tune k parameter for ratio PCA"
    )
    parser.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="Path to activations HDF5 file",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Single layer to tune (alternative to --layers)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Multiple layers to tune",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[3, 5, 7, 10, 15, 20, 25, 30],
        help="K values to try (default: 3 5 7 10 15 20 25 30)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=50,
        help="Number of final PCs (default: 50)",
    )
    parser.add_argument(
        "--n-dims-score",
        type=int,
        default=10,
        help="Number of top PCs for scoring (default: 10)",
    )
    parser.add_argument(
        "--n-pcs-all",
        type=int,
        default=100,
        help="Number of PCs for ratio analysis (default: 100)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to use (for testing)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file",
    )

    args = parser.parse_args()

    # Determine layers to process
    if args.layer is not None:
        layers = [args.layer]
    elif args.layers is not None:
        layers = args.layers
    else:
        raise ValueError("Must specify either --layer or --layers")

    print("="*80)
    print("K PARAMETER TUNING FOR RATIO PCA")
    print("="*80)
    print(f"Activations: {args.activations}")
    print(f"Layers: {layers}")
    print(f"K values to try: {args.k_values}")
    print(f"Output: {args.output}")

    # Tune each layer
    all_results = []
    for layer in layers:
        results = tune_k_single_layer(
            activations_path=args.activations,
            layer=layer,
            k_values=args.k_values,
            n_components=args.n_components,
            n_dims_score=args.n_dims_score,
            n_pcs_all=args.n_pcs_all,
            max_samples=args.max_samples,
        )
        all_results.append(results)

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "activations": str(args.activations),
            "k_values": args.k_values,
            "n_components": args.n_components,
            "n_dims_score": args.n_dims_score,
            "n_pcs_all": args.n_pcs_all,
            "max_samples": args.max_samples,
        },
        "results": all_results,
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for result in all_results:
        print(f"Layer {result['layer']}: best_k={result['best_k']}, "
              f"best_score={result['best_score']:.4f}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
