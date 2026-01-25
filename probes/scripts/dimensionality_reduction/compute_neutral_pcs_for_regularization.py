#!/usr/bin/env python3
"""Compute neutral PCs for use in orthogonality regularization.

This script uses ratio-based PCA to identify neutral-dominated principal components
that should be orthogonal to emotion probe weights.

The ratio-based approach:
1. Run PCA on combined emotional + neutral activations
2. For each PC, compute ratio = var(neutral_proj) / var(diff_proj)
3. Extract top-k PCs with highest ratio (most neutral-dominated)
4. Save these PCs for use in orthogonality-regularized probe training

Usage:
    # Compute neutral PCs for a single layer
    python compute_neutral_pcs_for_regularization.py \
        --data data/activations/texts.h5 \
        --layer 20 \
        --k 20 \
        --output results/neutral_pcs/

    # Compute for multiple layers
    python compute_neutral_pcs_for_regularization.py \
        --data data/activations/texts.h5 \
        --layers 0 10 20 30 \
        --k 20 \
        --output results/neutral_pcs/

    # Compute for all layers with layer range
    python compute_neutral_pcs_for_regularization.py \
        --data data/activations/texts.h5 \
        --layer-range 0 32 \
        --k 20 \
        --output results/neutral_pcs/
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

import h5py
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


def load_emotion_and_neutral_activations(
    h5_path: Path,
    layer: int,
    tiers: List[str] = None,
    max_samples: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load emotional and neutral activations from texts.h5.

    Args:
        h5_path: Path to texts.h5 file
        layer: Layer index to extract
        tiers: List of tier names to include. If None, includes all tiers
        max_samples: Maximum samples to load per emotion. If None, loads all

    Returns:
        emotional_acts: [n_samples, hidden_dim] emotional activations
        neutral_acts: [n_samples, hidden_dim] corresponding neutral activations
    """
    f = h5py.File(h5_path, "r")
    acts_group = f["activations"]

    emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

    emotional_acts = []
    neutral_acts = []

    for key in acts_group.keys():
        # Parse key: set_X_TIER_EMOTION
        parts = key.split("_")
        if len(parts) < 4:
            continue

        # Extract tier and emotion
        tier = "_".join(parts[2:-1])
        emotion = parts[-1]

        # Filter by tier if specified
        if tiers is not None and tier not in tiers:
            continue

        # Filter by emotion (we want the 6 basic emotions)
        if emotion not in emotions:
            continue

        # Load activations for this layer
        emotional = acts_group[key]["emotional"][layer]
        neutral = acts_group[key]["neutral"][layer]

        emotional_acts.append(emotional)
        neutral_acts.append(neutral)

    f.close()

    emotional_acts = np.array(emotional_acts)
    neutral_acts = np.array(neutral_acts)

    # Subsample if requested
    if max_samples is not None and len(emotional_acts) > max_samples:
        indices = np.random.choice(len(emotional_acts), max_samples, replace=False)
        emotional_acts = emotional_acts[indices]
        neutral_acts = neutral_acts[indices]

    return emotional_acts, neutral_acts


def compute_neutral_pcs_ratio_based(
    emotional_acts: np.ndarray,
    neutral_acts: np.ndarray,
    k: int,
    n_pcs_all: int = 100,
) -> tuple[np.ndarray, Dict]:
    """Compute top-k neutral-dominated PCs using ratio-based method.

    Args:
        emotional_acts: [n_samples, hidden_dim] emotional activations
        neutral_acts: [n_samples, hidden_dim] neutral activations
        k: Number of neutral PCs to extract
        n_pcs_all: Number of PCs to compute for ratio analysis

    Returns:
        neutral_pcs: [hidden_dim, k] matrix of neutral PC directions
        info: Dictionary with diagnostic information
    """
    # Compute differences
    diffs = emotional_acts - neutral_acts  # [n_samples, hidden_dim]

    # Combine neutral and emotional for PCA
    combined = np.vstack([neutral_acts, emotional_acts])  # [2*n_samples, hidden_dim]

    # Run PCA on combined to get "universal" basis
    n_comp = min(n_pcs_all, combined.shape[0] - 1, combined.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(combined)
    all_pcs = pca.components_  # [n_comp, hidden_dim]

    # Project neutral and diffs onto each PC
    neutral_proj = neutral_acts @ all_pcs.T  # [n_samples, n_comp]
    diff_proj = diffs @ all_pcs.T  # [n_samples, n_comp]

    # Compute variance of projections
    neutral_var = neutral_proj.var(axis=0)  # [n_comp]
    diff_var = diff_proj.var(axis=0)  # [n_comp]

    # Compute ratio (neutral variance / diff variance)
    # High ratio = neutral-dominated, low ratio = signal-dominated
    ratio = neutral_var / (diff_var + 1e-8)

    # Find top-k PCs with highest ratio (most neutral-dominated)
    top_k_indices = np.argsort(ratio)[-k:][::-1]  # Sort descending
    top_k_ratios = ratio[top_k_indices]

    # Extract neutral PCs: transpose to [hidden_dim, k]
    neutral_pcs = all_pcs[top_k_indices].T

    # Diagnostic info
    info = {
        "k": k,
        "n_pcs_all": n_comp,
        "neutral_pc_indices": top_k_indices.tolist(),
        "neutral_pc_ratios": top_k_ratios.tolist(),
        "mean_ratio_neutral": float(top_k_ratios.mean()),
        "mean_ratio_all": float(ratio.mean()),
        "neutral_var_explained": float(pca.explained_variance_ratio_[top_k_indices].sum()),
    }

    return neutral_pcs, info


def main():
    parser = argparse.ArgumentParser(
        description="Compute neutral PCs for orthogonality regularization"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/activations/texts.h5",
        help="Path to texts.h5 file",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Single layer to compute neutral PCs for",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Multiple layers to compute neutral PCs for (e.g., 0 10 20 30)",
    )
    parser.add_argument(
        "--layer-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Range of layers to compute (e.g., 0 32 for layers 0-31)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of neutral PCs to extract (default: 20)",
    )
    parser.add_argument(
        "--n-pcs-all",
        type=int,
        default=100,
        help="Number of PCs for ratio analysis (default: 100)",
    )
    parser.add_argument(
        "--tiers",
        type=str,
        nargs="+",
        default=None,
        help="Tiers to include (e.g., direct_address third_person). If not specified, uses all.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to use for PCA computation (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/neutral_pcs",
        help="Output directory (default: results/neutral_pcs)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Determine which layers to process
    if args.layer is not None:
        layers = [args.layer]
    elif args.layers is not None:
        layers = args.layers
    elif args.layer_range is not None:
        layers = list(range(args.layer_range[0], args.layer_range[1]))
    else:
        raise ValueError("Must specify --layer, --layers, or --layer-range")

    print("=" * 80)
    print("COMPUTING NEUTRAL PCs FOR ORTHOGONALITY REGULARIZATION")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Layers: {layers}")
    print(f"Number of neutral PCs (k): {args.k}")
    print(f"PCs for ratio analysis: {args.n_pcs_all}")
    print(f"Tiers: {args.tiers if args.tiers else 'all'}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
    print(f"Output: {args.output}")
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each layer
    all_info = {}

    for layer in tqdm(layers, desc="Layers"):
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        # Load activations
        print("Loading activations...")
        emotional_acts, neutral_acts = load_emotion_and_neutral_activations(
            Path(args.data),
            layer,
            tiers=args.tiers,
            max_samples=args.max_samples,
        )

        print(f"Loaded {len(emotional_acts)} samples")
        print(f"Activation shape: {emotional_acts.shape}")

        # Compute neutral PCs
        print(f"\nComputing neutral PCs (k={args.k})...")
        neutral_pcs, info = compute_neutral_pcs_ratio_based(
            emotional_acts,
            neutral_acts,
            k=args.k,
            n_pcs_all=args.n_pcs_all,
        )

        print(f"Neutral PCs shape: {neutral_pcs.shape}")
        print(f"Mean ratio (neutral PCs): {info['mean_ratio_neutral']:.4f}")
        print(f"Mean ratio (all PCs): {info['mean_ratio_all']:.4f}")
        print(f"Variance explained by neutral PCs: {info['neutral_var_explained']:.4f}")

        # Save neutral PCs
        output_path = output_dir / f"layer_{layer}_neutral_pcs_k{args.k}.npy"
        np.save(output_path, neutral_pcs)
        print(f"Saved to: {output_path}")

        # Store info
        all_info[f"layer_{layer}"] = info

    # Save configuration and info
    config = {
        "data_path": args.data,
        "layers": layers,
        "k": args.k,
        "n_pcs_all": args.n_pcs_all,
        "tiers": args.tiers,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "layer_info": all_info,
    }

    config_path = output_dir / f"config_k{args.k}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*80}")
    print("COMPUTATION COMPLETE")
    print(f"{'='*80}")
    print(f"Saved {len(layers)} layer(s) to: {output_dir}")
    print(f"Configuration saved to: {config_path}")
    print()
    print("Usage in training:")
    print(f"  neutral_pcs = np.load('{output_dir}/layer_<LAYER>_neutral_pcs_k{args.k}.npy')")


if __name__ == "__main__":
    main()
