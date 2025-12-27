#!/usr/bin/env python3
"""Run ratio-based PCA on activation pairs.

This script implements the ratio-based neutral removal approach (PCA v3) from
believe-it-or-not, adapted for direct comparison with cPCA.

Key Differences from cPCA:
- cPCA: Uses alpha-tuned contrastive covariance (C_emo - alpha * C_neu)
- Ratio PCA: Removes high neutral/diff ratio PCs (no alpha tuning needed)

Both methods aim to remove neutral structure while preserving emotional signal.

Usage:
    # Run ratio PCA with k=10 neutral-dominated PCs removed
    python scripts/run_ratio_pca.py \\
        --activations data/activations/texts_combined.h5 \\
        --output results/ratio_pca/ \\
        --k 10 \\
        --n-components 50

    # Compare different k values
    python scripts/run_ratio_pca.py \\
        --activations data/activations/texts_combined.h5 \\
        --output results/ratio_pca_k20/ \\
        --k 20 \\
        --n-components 50

Output Structure (compatible with cPCA):
    results/ratio_pca/
        model_name_ratio_pca.npz  - components, explained variance, removal info
        model_name_summary.json   - metadata and diagnostics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
from tqdm import tqdm

from probes.methods.pca_ratio import run_ratio_pca_per_layer


def load_activations_hdf5(h5_path: Path) -> tuple:
    """Load activations from HDF5 file.

    Args:
        h5_path: Path to HDF5 file

    Returns:
        activations: dict mapping pair_id -> {'neutral': array, 'emotional': array}
        metadata: list of metadata dicts
        attrs: file attributes dict
    """
    activations = {}
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)
        metadata = json.loads(f["metadata"][()])

        for pair_id in f["activations"].keys():
            activations[pair_id] = {
                "neutral": f[f"activations/{pair_id}/neutral"][:],
                "emotional": f[f"activations/{pair_id}/emotional"][:],
            }

    return activations, metadata, attrs


def prepare_arrays(
    activations: Dict,
    pair_ids: List[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare activation arrays for PCA.

    Args:
        activations: dict of activations
        pair_ids: list of pair IDs to include

    Returns:
        emotional_acts: [n_pairs, n_layers, hidden_dim]
        neutral_acts: [n_pairs, n_layers, hidden_dim]
    """
    emotional = []
    neutral = []

    for pair_id in pair_ids:
        emotional.append(activations[pair_id]["emotional"])
        neutral.append(activations[pair_id]["neutral"])

    return np.stack(emotional), np.stack(neutral)


def main():
    parser = argparse.ArgumentParser(
        description="Run ratio-based PCA with neutral removal"
    )
    parser.add_argument(
        "--activations",
        type=Path,
        required=True,
        help="Path to activations HDF5 file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of high-ratio PCs to remove (default: 10)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=50,
        help="Number of final PCs to compute (default: 50)",
    )
    parser.add_argument(
        "--n-pcs-all",
        type=int,
        default=100,
        help="Number of PCs for ratio analysis (default: 100)",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default=None,
        help="Filter to specific tier (optional)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to use (for testing)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("RATIO-BASED PCA (PCA v3)")
    print("=" * 80)
    print(f"Activations: {args.activations}")
    print(f"Output: {args.output}")
    print(f"k (PCs to remove): {args.k}")
    print(f"n_components: {args.n_components}")
    print(f"n_pcs_all: {args.n_pcs_all}")
    print()

    # Load activations
    print("Loading activations...")
    activations, metadata, attrs = load_activations_hdf5(args.activations)
    print(f"Loaded {len(activations)} activation pairs")

    # Filter by tier if specified
    pair_ids = list(activations.keys())
    if args.tier:
        id_to_meta = {m["id"]: m for m in metadata}
        pair_ids = [
            pid for pid in pair_ids if id_to_meta.get(pid, {}).get("tier") == args.tier
        ]
        print(f"Filtered to tier '{args.tier}': {len(pair_ids)} pairs")

    # Limit samples if specified
    if args.max_samples:
        pair_ids = pair_ids[: args.max_samples]
        print(f"Limited to {len(pair_ids)} samples")

    # Prepare arrays
    print("\nPreparing activation arrays...")
    emotional_acts, neutral_acts = prepare_arrays(activations, pair_ids)
    print(f"Shape: {emotional_acts.shape}")

    # Get dimensions
    n_pairs, n_layers, hidden_dim = emotional_acts.shape
    model_name = attrs.get("model_name", "unknown_model")

    print(f"\nModel: {model_name}")
    print(f"Pairs: {n_pairs}")
    print(f"Layers: {n_layers}")
    print(f"Hidden dim: {hidden_dim}")
    print()

    # Run ratio PCA
    print("Running ratio-based PCA...")
    print(f"  Removing top-{args.k} PCs with highest neutral/diff ratio per layer")
    print()

    results = run_ratio_pca_per_layer(
        target_acts=emotional_acts,
        background_acts=neutral_acts,
        k=args.k,
        n_components=args.n_components,
        n_pcs_all=args.n_pcs_all,
    )

    # Print summary statistics
    print("\n" + "=" * 80)
    print("NEUTRAL REMOVAL DIAGNOSTICS")
    print("=" * 80)

    avg_ratio_removed = np.mean(
        [info["mean_ratio_removed"] for info in results["removal_info"].values()]
    )
    avg_ratio_kept = np.mean(
        [info["mean_ratio_kept"] for info in results["removal_info"].values()]
    )

    print(f"Average ratio of removed PCs: {avg_ratio_removed:.4f}")
    print(f"Average ratio of kept PCs: {avg_ratio_kept:.4f}")
    print(f"Ratio improvement: {avg_ratio_removed / (avg_ratio_kept + 1e-8):.2f}x")
    print()

    # Show per-layer stats for a few layers
    print("Per-layer diagnostics (first 5 layers):")
    for layer_idx in range(min(5, n_layers)):
        info = results["removal_info"][layer_idx]
        print(
            f"  Layer {layer_idx:2d}: removed_ratio={info['mean_ratio_removed']:.4f}, "
            f"kept_ratio={info['mean_ratio_kept']:.4f}"
        )

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)

    # Save NPZ file (compatible with cPCA format)
    npz_path = args.output / f"{model_name}_ratio_pca.npz"
    print(f"\nSaving results to {npz_path}")

    # Prepare data for NPZ - match cPCA format
    # Stack all layers into single arrays
    components_all = np.stack(
        [results["components"][i] for i in range(n_layers)], axis=0
    )  # [n_layers, n_components, hidden_dim]

    explained_variance_all = np.stack(
        [results["explained_variance"][i] for i in range(n_layers)], axis=0
    )  # [n_layers, n_components]

    explained_variance_ratio_all = np.stack(
        [results["explained_variance_ratio"][i] for i in range(n_layers)], axis=0
    )  # [n_layers, n_components]

    # Extract removal ratios (analogous to alphas in cPCA)
    removal_ratios_per_layer = np.array([
        results["removal_info"][i]["mean_ratio_removed"] for i in range(n_layers)
    ])

    np.savez_compressed(
        npz_path,
        components=components_all,
        explained_variance=explained_variance_all,
        explained_variance_ratio=explained_variance_ratio_all,
        eigenvalues=explained_variance_all,  # Use explained variance as eigenvalues
        alphas=removal_ratios_per_layer,  # Use removal ratios as "alphas" for compatibility
        config=json.dumps(results["config"]),
        removal_ratios=removal_ratios_per_layer,
    )

    # Save summary JSON
    summary_path = args.output / f"{model_name}_summary.json"
    print(f"Saving summary to {summary_path}")

    summary = {
        "method": "ratio_pca",
        "model_name": model_name,
        "n_pairs": n_pairs,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "config": results["config"],
        "diagnostics": {
            "avg_ratio_removed": float(avg_ratio_removed),
            "avg_ratio_kept": float(avg_ratio_kept),
            "ratio_improvement": float(avg_ratio_removed / (avg_ratio_kept + 1e-8)),
        },
        "removal_info_per_layer": {
            str(k): v for k, v in results["removal_info"].items()
        },
        "pair_ids": pair_ids,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output}")
    print(f"  Components: {npz_path}")
    print(f"  Summary: {summary_path}")
    print()
    print("Next steps:")
    print("  - Compare with cPCA results using compare_pca_methods.py")
    print("  - Run autointerp on ratio PCA components")
    print("  - Train probes on ratio PCA projections")


if __name__ == "__main__":
    main()