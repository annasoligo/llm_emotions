#!/usr/bin/env python3
"""Compute neutral and shared emotion PCs for controlled variation data.

This script implements multi-level orthogonal regularization for isolating
user and assistant emotion representations:

Stage 1: Neutral PCs
    - Use ratio-based method to identify neutral-dominated directions
    - ratio = var(baseline_proj) / var(diff_proj)
    - High ratio = neutral noise that should be orthogonal to emotion probes

Stage 2: Shared Emotion PCs
    - After removing neutral PCs, identify shared emotion structure
    - These are directions with high variance in BOTH user and assistant isolation
    - Use correlation-based or minimum-variance-difference methods

Usage:
    # Compute for specific layers
    python compute_controlled_variation_orthogonal_pcs.py \
        --user-data outputs/activations/controlled_variation/user_isolation.h5 \
        --asst-data outputs/activations/controlled_variation/assistant_isolation.h5 \
        --layers 10 20 30 40 50 \
        --k-neutral 20 \
        --k-shared 10 \
        --output outputs/orthogonal_pcs/controlled_variation/

    # Compute for all layers
    python compute_controlled_variation_orthogonal_pcs.py \
        --user-data outputs/activations/controlled_variation/user_isolation.h5 \
        --asst-data outputs/activations/controlled_variation/assistant_isolation.h5 \
        --layer-range 0 62 \
        --k-neutral 20 \
        --k-shared 10 \
        --output outputs/orthogonal_pcs/controlled_variation/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm


def load_activations_for_layer(
    h5_path: Path,
    layer: int,
    max_pairs: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load neutral and emotional activations for a specific layer.

    Args:
        h5_path: Path to isolation h5 file (user_isolation.h5 or assistant_isolation.h5)
        layer: Layer index to extract
        max_pairs: Maximum pairs to load. If None, loads all

    Returns:
        neutral_acts: [n_pairs, hidden_dim] baseline activations
        emotional_acts: [n_pairs, hidden_dim] emotional activations
    """
    f = h5py.File(h5_path, "r")
    acts_group = f["activations"]

    # Get all pair keys
    pair_keys = sorted([k for k in acts_group.keys() if k.startswith("pair_")])

    if max_pairs is not None:
        pair_keys = pair_keys[:max_pairs]

    neutral_acts = []
    emotional_acts = []

    for pair_key in pair_keys:
        pair = acts_group[pair_key]
        neutral_acts.append(pair["neutral"][layer])  # [hidden_dim]
        emotional_acts.append(pair["emotional"][layer])  # [hidden_dim]

    f.close()

    neutral_acts = np.array(neutral_acts)  # [n_pairs, hidden_dim]
    emotional_acts = np.array(emotional_acts)  # [n_pairs, hidden_dim]

    return neutral_acts, emotional_acts


def compute_neutral_pcs_ratio_based(
    neutral_acts: np.ndarray,
    emotional_acts: np.ndarray,
    k: int,
    n_pcs_all: int = 100,
) -> Tuple[np.ndarray, Dict]:
    """Compute top-k neutral-dominated PCs using ratio-based method.

    This is identical to the text-based method but adapted for controlled variation data.

    Args:
        neutral_acts: [n_samples, hidden_dim] baseline activations
        emotional_acts: [n_samples, hidden_dim] emotional activations
        k: Number of neutral PCs to extract
        n_pcs_all: Number of PCs to compute for ratio analysis

    Returns:
        neutral_pcs: [hidden_dim, k] matrix of neutral PC directions
        info: Dictionary with diagnostic information
    """
    # Compute differences (signal)
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


def remove_projection(
    data: np.ndarray,
    pcs: np.ndarray,
) -> np.ndarray:
    """Remove projection onto given PCs from data.

    Args:
        data: [n_samples, hidden_dim] activations
        pcs: [hidden_dim, k] PC directions to remove

    Returns:
        residuals: [n_samples, hidden_dim] data with PCs removed
    """
    # Project onto PCs: [n_samples, k]
    projections = data @ pcs

    # Reconstruct from projections: [n_samples, hidden_dim]
    reconstruction = projections @ pcs.T

    # Remove reconstruction
    residuals = data - reconstruction

    return residuals


def compute_shared_emotion_pcs_correlation_based(
    user_emotional: np.ndarray,
    user_neutral: np.ndarray,
    asst_emotional: np.ndarray,
    asst_neutral: np.ndarray,
    k: int,
    n_pcs_all: int = 100,
) -> Tuple[np.ndarray, Dict]:
    """Compute shared emotion PCs using correlation-based method.

    Strategy: Find PCs with high variance in BOTH user and assistant diffs,
    indicating shared emotion structure regardless of source.

    Args:
        user_emotional: [n_samples, hidden_dim] user emotional activations
        user_neutral: [n_samples, hidden_dim] user neutral activations
        asst_emotional: [n_samples, hidden_dim] assistant emotional activations
        asst_neutral: [n_samples, hidden_dim] assistant neutral activations
        k: Number of shared PCs to extract
        n_pcs_all: Number of PCs to compute

    Returns:
        shared_pcs: [hidden_dim, k] matrix of shared emotion PC directions
        info: Dictionary with diagnostic information
    """
    # Compute diffs for both isolation types
    user_diffs = user_emotional - user_neutral  # [n_samples, hidden_dim]
    asst_diffs = asst_emotional - asst_neutral  # [n_samples, hidden_dim]

    # Combine all diffs for PCA
    combined_diffs = np.vstack([user_diffs, asst_diffs])  # [2*n_samples, hidden_dim]

    # Run PCA on combined diffs
    n_comp = min(n_pcs_all, combined_diffs.shape[0] - 1, combined_diffs.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(combined_diffs)
    all_pcs = pca.components_  # [n_comp, hidden_dim]

    # Project user and assistant diffs onto each PC
    user_proj = user_diffs @ all_pcs.T  # [n_samples, n_comp]
    asst_proj = asst_diffs @ all_pcs.T  # [n_samples, n_comp]

    # Compute variance of projections for each isolation type
    user_var = user_proj.var(axis=0)  # [n_comp]
    asst_var = asst_proj.var(axis=0)  # [n_comp]

    # Compute "sharedness" metric: minimum of the two variances (normalized)
    # PCs with high variance in BOTH user and assistant are most shared
    sharedness = np.minimum(user_var, asst_var) / (np.maximum(user_var, asst_var) + 1e-8)

    # Alternative: Use correlation between user and assistant projections
    correlations = np.array([
        np.corrcoef(user_proj[:, i], asst_proj[:, i])[0, 1]
        for i in range(n_comp)
    ])

    # Combine metrics: high sharedness AND high correlation
    combined_metric = sharedness * np.abs(correlations)

    # Find top-k PCs with highest combined metric (most shared)
    top_k_indices = np.argsort(combined_metric)[-k:][::-1]  # Sort descending
    top_k_sharedness = sharedness[top_k_indices]
    top_k_correlations = correlations[top_k_indices]

    # Extract shared PCs: transpose to [hidden_dim, k]
    shared_pcs = all_pcs[top_k_indices].T

    # Diagnostic info
    info = {
        "k": k,
        "n_pcs_all": n_comp,
        "shared_pc_indices": top_k_indices.tolist(),
        "shared_pc_sharedness": top_k_sharedness.tolist(),
        "shared_pc_correlations": top_k_correlations.tolist(),
        "mean_sharedness": float(top_k_sharedness.mean()),
        "mean_correlation": float(top_k_correlations.mean()),
        "shared_var_explained": float(pca.explained_variance_ratio_[top_k_indices].sum()),
    }

    return shared_pcs, info


def visualize_pc_ratios(
    neutral_info: Dict,
    shared_info: Dict,
    layer: int,
    output_dir: Path,
):
    """Create visualization of PC selection metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Neutral PC ratios
    ax = axes[0]
    ratios = neutral_info["neutral_pc_ratios"]
    ax.bar(range(len(ratios)), ratios, color='steelblue', alpha=0.7)
    ax.set_xlabel("Neutral PC Index")
    ax.set_ylabel("Ratio (neutral_var / diff_var)")
    ax.set_title(f"Layer {layer}: Neutral PC Ratios\nMean: {neutral_info['mean_ratio_neutral']:.2f}")
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Shared PC metrics
    ax = axes[1]
    sharedness = shared_info["shared_pc_sharedness"]
    correlations = shared_info["shared_pc_correlations"]
    x = np.arange(len(sharedness))
    width = 0.35
    ax.bar(x - width/2, sharedness, width, label='Sharedness', color='coral', alpha=0.7)
    ax.bar(x + width/2, np.abs(correlations), width, label='|Correlation|', color='seagreen', alpha=0.7)
    ax.set_xlabel("Shared PC Index")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Layer {layer}: Shared Emotion PC Metrics")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / f"layer_{layer}_pc_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compute neutral and shared emotion PCs for controlled variation"
    )
    parser.add_argument(
        "--user-data",
        type=str,
        required=True,
        help="Path to user_isolation.h5",
    )
    parser.add_argument(
        "--asst-data",
        type=str,
        required=True,
        help="Path to assistant_isolation.h5",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="Single layer to compute PCs for",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Multiple layers to compute PCs for (e.g., 10 20 30 40 50)",
    )
    parser.add_argument(
        "--layer-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Range of layers to compute (e.g., 0 62 for layers 0-61)",
    )
    parser.add_argument(
        "--k-neutral",
        type=int,
        default=20,
        help="Number of neutral PCs to extract (default: 20)",
    )
    parser.add_argument(
        "--k-shared",
        type=int,
        default=10,
        help="Number of shared emotion PCs to extract (default: 10)",
    )
    parser.add_argument(
        "--n-pcs-all",
        type=int,
        default=100,
        help="Number of PCs for analysis (default: 100)",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Max pairs to use for PCA computation (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/orthogonal_pcs/controlled_variation",
        help="Output directory",
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
    print("COMPUTING ORTHOGONAL PCs FOR CONTROLLED VARIATION")
    print("=" * 80)
    print(f"User data: {args.user_data}")
    print(f"Assistant data: {args.asst_data}")
    print(f"Layers: {layers}")
    print(f"Neutral PCs (k): {args.k_neutral}")
    print(f"Shared emotion PCs (k): {args.k_shared}")
    print(f"PCs for analysis: {args.n_pcs_all}")
    print(f"Max pairs: {args.max_pairs if args.max_pairs else 'all'}")
    print(f"Output: {args.output}")
    print()

    # Create output directories
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    neutral_dir = output_dir / "neutral_pcs"
    shared_dir = output_dir / "shared_pcs"
    viz_dir = output_dir / "visualizations"

    neutral_dir.mkdir(exist_ok=True)
    shared_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(exist_ok=True)

    # Process each layer
    all_info = {}

    for layer in tqdm(layers, desc="Layers"):
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        # ========== Stage 1: Compute Neutral PCs ==========
        print("\n[Stage 1] Computing neutral PCs...")

        # Load user isolation data
        print("Loading user isolation activations...")
        user_neutral, user_emotional = load_activations_for_layer(
            Path(args.user_data),
            layer,
            max_pairs=args.max_pairs,
        )
        print(f"  User: {len(user_neutral)} pairs, shape: {user_neutral.shape}")

        # Compute neutral PCs from user data
        neutral_pcs, neutral_info = compute_neutral_pcs_ratio_based(
            user_neutral,
            user_emotional,
            k=args.k_neutral,
            n_pcs_all=args.n_pcs_all,
        )

        print(f"  Neutral PCs shape: {neutral_pcs.shape}")
        print(f"  Mean ratio (neutral PCs): {neutral_info['mean_ratio_neutral']:.4f}")
        print(f"  Mean ratio (all PCs): {neutral_info['mean_ratio_all']:.4f}")
        print(f"  Variance explained: {neutral_info['neutral_var_explained']:.4f}")

        # Save neutral PCs
        neutral_path = neutral_dir / f"layer_{layer}_neutral_pcs_k{args.k_neutral}.npy"
        np.save(neutral_path, neutral_pcs)
        print(f"  Saved to: {neutral_path}")

        # ========== Stage 2: Compute Shared Emotion PCs ==========
        print("\n[Stage 2] Computing shared emotion PCs...")

        # Load assistant isolation data
        print("Loading assistant isolation activations...")
        asst_neutral, asst_emotional = load_activations_for_layer(
            Path(args.asst_data),
            layer,
            max_pairs=args.max_pairs,
        )
        print(f"  Assistant: {len(asst_neutral)} pairs, shape: {asst_neutral.shape}")

        # Remove neutral PCs from both user and assistant data
        print("Removing neutral PCs from activation data...")
        user_emotional_residual = remove_projection(user_emotional, neutral_pcs)
        user_neutral_residual = remove_projection(user_neutral, neutral_pcs)
        asst_emotional_residual = remove_projection(asst_emotional, neutral_pcs)
        asst_neutral_residual = remove_projection(asst_neutral, neutral_pcs)

        # Compute shared emotion PCs
        shared_pcs, shared_info = compute_shared_emotion_pcs_correlation_based(
            user_emotional_residual,
            user_neutral_residual,
            asst_emotional_residual,
            asst_neutral_residual,
            k=args.k_shared,
            n_pcs_all=args.n_pcs_all,
        )

        print(f"  Shared PCs shape: {shared_pcs.shape}")
        print(f"  Mean sharedness: {shared_info['mean_sharedness']:.4f}")
        print(f"  Mean correlation: {shared_info['mean_correlation']:.4f}")
        print(f"  Variance explained: {shared_info['shared_var_explained']:.4f}")

        # Save shared PCs
        shared_path = shared_dir / f"layer_{layer}_shared_pcs_k{args.k_shared}.npy"
        np.save(shared_path, shared_pcs)
        print(f"  Saved to: {shared_path}")

        # Visualize
        viz_path = visualize_pc_ratios(neutral_info, shared_info, layer, viz_dir)
        print(f"  Visualization saved to: {viz_path}")

        # Store info
        all_info[f"layer_{layer}"] = {
            "neutral": neutral_info,
            "shared": shared_info,
        }

    # Save configuration and info
    config = {
        "user_data": args.user_data,
        "asst_data": args.asst_data,
        "layers": layers,
        "k_neutral": args.k_neutral,
        "k_shared": args.k_shared,
        "n_pcs_all": args.n_pcs_all,
        "max_pairs": args.max_pairs,
        "seed": args.seed,
        "layer_info": all_info,
    }

    config_path = output_dir / f"config_k{args.k_neutral}_k{args.k_shared}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*80}")
    print("COMPUTATION COMPLETE")
    print(f"{'='*80}")
    print(f"Processed {len(layers)} layer(s)")
    print(f"Neutral PCs saved to: {neutral_dir}")
    print(f"Shared PCs saved to: {shared_dir}")
    print(f"Visualizations saved to: {viz_dir}")
    print(f"Configuration saved to: {config_path}")
    print()
    print("Usage in training:")
    print(f"  neutral_pcs = np.load('{neutral_dir}/layer_<LAYER>_neutral_pcs_k{args.k_neutral}.npy')")
    print(f"  shared_pcs = np.load('{shared_dir}/layer_<LAYER>_shared_pcs_k{args.k_shared}.npy')")
    print(f"  # Concatenate for full regularization:")
    print(f"  all_pcs = np.concatenate([neutral_pcs, shared_pcs], axis=1)")


if __name__ == "__main__":
    main()
