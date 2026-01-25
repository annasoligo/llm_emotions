#!/usr/bin/env python3
"""Compare original cPCA PCs with neutral and shared orthogonal PCs.

This script validates that the neutral and shared PCs capture the high-similarity
structure observed in the original user vs assistant cPCA comparison.

Analysis:
1. Load original user and assistant cPCA PCs
2. Load neutral and shared orthogonal PCs
3. Compute cosine similarities:
   - User cPCA PCs vs neutral PCs
   - Assistant cPCA PCs vs neutral PCs
   - User cPCA PCs vs shared PCs
   - Assistant cPCA PCs vs shared PCs
4. Show that neutral/shared PCs align with high-similarity cPCA PCs

Usage:
    python compare_cpca_vs_orthogonal_pcs.py \
        --layers 10 20 30 40 50 \
        --output outputs/orthogonal_pcs/controlled_variation/analysis/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_cpca_components(base_dir: Path, isolation_type: str, layer_idx: int) -> np.ndarray:
    """Load cPCA components for a specific layer."""
    layer_dir = base_dir / f"{isolation_type}_isolation" / "layers_0_61"
    layer_file = layer_dir / f"layer_{layer_idx}.npz"
    data = np.load(layer_file)
    return data['components']  # [n_components, hidden_dim]


def load_orthogonal_pcs(base_dir: Path, pc_type: str, layer_idx: int, k: int) -> np.ndarray:
    """Load neutral or shared PCs."""
    pc_dir = base_dir / f"{pc_type}_pcs"
    pc_file = pc_dir / f"layer_{layer_idx}_{pc_type}_pcs_k{k}.npy"
    return np.load(pc_file)  # [hidden_dim, k]


def compute_cosine_similarity_matrix(
    pcs_a: np.ndarray,
    pcs_b: np.ndarray
) -> np.ndarray:
    """Compute pairwise cosine similarities.

    Args:
        pcs_a: [n_a, hidden_dim] or [hidden_dim, n_a]
        pcs_b: [n_b, hidden_dim] or [hidden_dim, n_b]

    Returns:
        similarity: [n_a, n_b] cosine similarity matrix
    """
    # Ensure shape is [n_components, hidden_dim]
    if pcs_a.shape[0] > pcs_a.shape[1]:
        pcs_a = pcs_a.T
    if pcs_b.shape[0] > pcs_b.shape[1]:
        pcs_b = pcs_b.T

    # Normalize
    pcs_a_norm = pcs_a / np.linalg.norm(pcs_a, axis=1, keepdims=True)
    pcs_b_norm = pcs_b / np.linalg.norm(pcs_b, axis=1, keepdims=True)

    # Compute similarity
    similarity = pcs_a_norm @ pcs_b_norm.T

    return similarity


def analyze_layer(
    layer: int,
    cpca_base_dir: Path,
    orthogonal_base_dir: Path,
    k_neutral: int = 20,
    k_shared: int = 10,
) -> Dict:
    """Analyze alignment between cPCA and orthogonal PCs for a single layer."""

    # Load cPCA components
    user_cpca = load_cpca_components(cpca_base_dir, "user", layer)  # [50, hidden_dim]
    asst_cpca = load_cpca_components(cpca_base_dir, "assistant", layer)  # [50, hidden_dim]

    # Load orthogonal PCs
    neutral_pcs = load_orthogonal_pcs(orthogonal_base_dir, "neutral", layer, k_neutral)  # [hidden_dim, 20]
    shared_pcs = load_orthogonal_pcs(orthogonal_base_dir, "shared", layer, k_shared)  # [hidden_dim, 10]

    # Compute original user vs assistant similarity
    user_asst_sim = compute_cosine_similarity_matrix(user_cpca, asst_cpca)  # [50, 50]
    max_user_asst_sim = np.max(np.abs(user_asst_sim))
    mean_user_asst_sim = np.mean(np.abs(user_asst_sim))

    # Compute similarities with neutral PCs
    user_neutral_sim = compute_cosine_similarity_matrix(user_cpca, neutral_pcs)  # [50, 20]
    asst_neutral_sim = compute_cosine_similarity_matrix(asst_cpca, neutral_pcs)  # [50, 20]

    # Compute similarities with shared PCs
    user_shared_sim = compute_cosine_similarity_matrix(user_cpca, shared_pcs)  # [50, 10]
    asst_shared_sim = compute_cosine_similarity_matrix(asst_cpca, shared_pcs)  # [50, 10]

    # Find which cPCA PCs are most aligned with neutral/shared
    user_max_neutral = np.max(np.abs(user_neutral_sim), axis=1)  # [50] max sim to any neutral PC
    asst_max_neutral = np.max(np.abs(asst_neutral_sim), axis=1)  # [50]
    user_max_shared = np.max(np.abs(user_shared_sim), axis=1)  # [50] max sim to any shared PC
    asst_max_shared = np.max(np.abs(asst_shared_sim), axis=1)  # [50]

    # Analyze first 10 cPCA PCs (which showed high similarity in original analysis)
    first_10_indices = range(10)
    user_first10_neutral = user_max_neutral[first_10_indices]
    asst_first10_neutral = asst_max_neutral[first_10_indices]
    user_first10_shared = user_max_shared[first_10_indices]
    asst_first10_shared = asst_max_shared[first_10_indices]

    results = {
        "layer": layer,
        "user_asst_max_sim": float(max_user_asst_sim),
        "user_asst_mean_sim": float(mean_user_asst_sim),
        "user_neutral": {
            "mean_max_sim_all": float(user_max_neutral.mean()),
            "mean_max_sim_first10": float(user_first10_neutral.mean()),
            "max_sim": float(user_max_neutral.max()),
        },
        "asst_neutral": {
            "mean_max_sim_all": float(asst_max_neutral.mean()),
            "mean_max_sim_first10": float(asst_first10_neutral.mean()),
            "max_sim": float(asst_max_neutral.max()),
        },
        "user_shared": {
            "mean_max_sim_all": float(user_max_shared.mean()),
            "mean_max_sim_first10": float(user_first10_shared.mean()),
            "max_sim": float(user_max_shared.max()),
        },
        "asst_shared": {
            "mean_max_sim_all": float(asst_max_shared.mean()),
            "mean_max_sim_first10": float(asst_first10_shared.mean()),
            "max_sim": float(asst_max_shared.max()),
        },
        "matrices": {
            "user_neutral_sim": user_neutral_sim,
            "asst_neutral_sim": asst_neutral_sim,
            "user_shared_sim": user_shared_sim,
            "asst_shared_sim": asst_shared_sim,
        }
    }

    return results


def visualize_comparison(
    results_dict: Dict[int, Dict],
    output_path: Path,
):
    """Create comprehensive comparison visualization."""
    layers = sorted(results_dict.keys())

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, len(layers), hspace=0.35, wspace=0.3)

    # Row 1: User cPCA vs Neutral PCs
    for idx, layer in enumerate(layers):
        ax = fig.add_subplot(gs[0, idx])
        sim_matrix = results_dict[layer]["matrices"]["user_neutral_sim"]

        im = ax.imshow(np.abs(sim_matrix), cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'Layer {layer}\nUser cPCA vs Neutral', fontsize=10, pad=8)
        ax.set_xlabel('Neutral PC', fontsize=9)
        ax.set_ylabel('User cPCA PC', fontsize=9)
        ax.tick_params(labelsize=8)

        # Mark first 10 cPCA PCs
        ax.axhline(y=9.5, color='cyan', linestyle='--', linewidth=1, alpha=0.7)

        if idx == len(layers) - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('|Cosine Sim|', fontsize=9)

    # Row 2: Assistant cPCA vs Neutral PCs
    for idx, layer in enumerate(layers):
        ax = fig.add_subplot(gs[1, idx])
        sim_matrix = results_dict[layer]["matrices"]["asst_neutral_sim"]

        im = ax.imshow(np.abs(sim_matrix), cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'Assistant cPCA vs Neutral', fontsize=10, pad=8)
        ax.set_xlabel('Neutral PC', fontsize=9)
        ax.set_ylabel('Asst cPCA PC', fontsize=9)
        ax.tick_params(labelsize=8)

        # Mark first 10 cPCA PCs
        ax.axhline(y=9.5, color='cyan', linestyle='--', linewidth=1, alpha=0.7)

        if idx == len(layers) - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('|Cosine Sim|', fontsize=9)

    # Row 3: Comparison with Shared PCs (user and assistant side-by-side)
    for idx, layer in enumerate(layers):
        ax = fig.add_subplot(gs[2, idx])

        user_sim = results_dict[layer]["matrices"]["user_shared_sim"]
        asst_sim = results_dict[layer]["matrices"]["asst_shared_sim"]

        # Stack user and assistant similarities horizontally
        combined = np.hstack([np.abs(user_sim), np.abs(asst_sim)])  # [50, 20]

        im = ax.imshow(combined, cmap='PuBu', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f'cPCA vs Shared PCs', fontsize=10, pad=8)
        ax.set_xlabel('Shared PC (U | A)', fontsize=9)
        ax.set_ylabel('cPCA PC', fontsize=9)
        ax.tick_params(labelsize=8)

        # Draw vertical line separating user and assistant
        ax.axvline(x=9.5, color='white', linestyle='-', linewidth=2, alpha=0.8)
        ax.axhline(y=9.5, color='cyan', linestyle='--', linewidth=1, alpha=0.7)

        if idx == len(layers) - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('|Cosine Sim|', fontsize=9)

    fig.suptitle('Alignment of Original cPCA PCs with Neutral and Shared Orthogonal PCs\n' +
                 'Cyan line marks first 10 cPCA PCs (high similarity in original analysis)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def print_summary(results_dict: Dict[int, Dict]):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY: Alignment Between cPCA and Orthogonal PCs")
    print("="*80)

    layers = sorted(results_dict.keys())

    print("\n{:<8} {:<12} {:<12} {:<12} {:<12}".format(
        "Layer", "User↔Asst", "→Neutral", "→Shared", "Combined"
    ))
    print("-" * 80)

    for layer in layers:
        r = results_dict[layer]

        # Original user-assistant similarity (for first 10 PCs)
        orig_sim = r["user_asst_max_sim"]

        # Alignment with neutral PCs (average of user and assistant first 10)
        neutral_alignment = (r["user_neutral"]["mean_max_sim_first10"] +
                            r["asst_neutral"]["mean_max_sim_first10"]) / 2

        # Alignment with shared PCs (average of user and assistant first 10)
        shared_alignment = (r["user_shared"]["mean_max_sim_first10"] +
                           r["asst_shared"]["mean_max_sim_first10"]) / 2

        # Combined: max of neutral and shared (captures total "explainable" similarity)
        combined = max(neutral_alignment, shared_alignment)

        print("{:<8} {:<12.3f} {:<12.3f} {:<12.3f} {:<12.3f}".format(
            layer, orig_sim, neutral_alignment, shared_alignment, combined
        ))

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("User↔Asst: Maximum similarity between user and assistant cPCA PCs")
    print("→Neutral:  Mean max similarity of first 10 cPCA PCs to neutral PCs")
    print("→Shared:   Mean max similarity of first 10 cPCA PCs to shared PCs")
    print("Combined:  Max of neutral and shared alignment (captures explainable sim)")
    print("\nHigh values in →Neutral and →Shared indicate that orthogonal PCs")
    print("successfully capture the high-similarity structure in original cPCA.")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Compare cPCA PCs with neutral and shared orthogonal PCs"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[10, 20, 30, 40, 50],
        help="Layers to analyze",
    )
    parser.add_argument(
        "--cpca-dir",
        type=str,
        default="outputs/dimensionality_reduction/cpca/controlled_variation",
        help="Directory containing cPCA results",
    )
    parser.add_argument(
        "--orthogonal-dir",
        type=str,
        default="outputs/orthogonal_pcs/controlled_variation",
        help="Directory containing orthogonal PCs",
    )
    parser.add_argument(
        "--k-neutral",
        type=int,
        default=20,
        help="Number of neutral PCs",
    )
    parser.add_argument(
        "--k-shared",
        type=int,
        default=10,
        help="Number of shared PCs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/orthogonal_pcs/controlled_variation/analysis",
        help="Output directory",
    )

    args = parser.parse_args()

    print("="*80)
    print("COMPARING cPCA PCs WITH ORTHOGONAL PCs")
    print("="*80)
    print(f"cPCA directory: {args.cpca_dir}")
    print(f"Orthogonal PCs directory: {args.orthogonal_dir}")
    print(f"Layers: {args.layers}")
    print(f"k_neutral: {args.k_neutral}, k_shared: {args.k_shared}")
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Analyze each layer
    results_dict = {}

    for layer in args.layers:
        print(f"Analyzing layer {layer}...")
        results = analyze_layer(
            layer,
            Path(args.cpca_dir),
            Path(args.orthogonal_dir),
            k_neutral=args.k_neutral,
            k_shared=args.k_shared,
        )
        results_dict[layer] = results

    # Print summary
    print_summary(results_dict)

    # Save detailed results (without matrices)
    results_to_save = {}
    for layer, res in results_dict.items():
        results_to_save[f"layer_{layer}"] = {
            k: v for k, v in res.items() if k != "matrices"
        }

    results_path = output_dir / "cpca_orthogonal_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Create visualization
    viz_path = output_dir / "cpca_orthogonal_alignment.png"
    visualize_comparison(results_dict, viz_path)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
