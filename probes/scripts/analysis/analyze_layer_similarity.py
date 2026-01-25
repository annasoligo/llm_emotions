#!/usr/bin/env python3
"""Analyze detailed pairwise cosine similarities for a specific layer."""

import numpy as np
from pathlib import Path
import argparse

def load_components(base_dir: Path, isolation_type: str, layer_idx: int) -> np.ndarray:
    """Load cPCA components for a specific layer."""
    layer_dir = base_dir / f"{isolation_type}_isolation" / "layers_0_61"
    layer_file = layer_dir / f"layer_{layer_idx}.npz"

    data = np.load(layer_file)
    return data['components']  # [n_components, hidden_dim]


def compute_cosine_similarity_matrix(
    user_comps: np.ndarray,
    asst_comps: np.ndarray
) -> np.ndarray:
    """Compute pairwise cosine similarities."""
    # Normalize
    user_norm = user_comps / np.linalg.norm(user_comps, axis=1, keepdims=True)
    asst_norm = asst_comps / np.linalg.norm(asst_comps, axis=1, keepdims=True)

    # Compute cosine similarity matrix
    similarity = user_norm @ asst_norm.T

    return similarity


def print_similarity_matrix(sim_matrix: np.ndarray, layer_idx: int, top_n: int = 10):
    """Print similarity matrix with statistics."""
    print(f"\n{'='*80}")
    print(f"LAYER {layer_idx}: PAIRWISE COSINE SIMILARITIES")
    print(f"User PCs (rows) × Assistant PCs (columns)")
    print(f"{'='*80}")

    n_user, n_asst = sim_matrix.shape
    print(f"Matrix shape: {n_user} user PCs × {n_asst} assistant PCs")

    # Statistics
    print(f"\nStatistics:")
    print(f"  Max absolute similarity: {np.max(np.abs(sim_matrix)):.4f}")
    print(f"  Mean absolute similarity: {np.mean(np.abs(sim_matrix)):.4f}")
    print(f"  Median absolute similarity: {np.median(np.abs(sim_matrix)):.4f}")
    print(f"  Std absolute similarity: {np.std(np.abs(sim_matrix)):.4f}")

    # Distribution
    abs_sims = np.abs(sim_matrix).ravel()
    print(f"\nDistribution of absolute similarities:")
    print(f"  0.0-0.1: {np.sum(abs_sims < 0.1)} pairs ({100*np.mean(abs_sims < 0.1):.1f}%)")
    print(f"  0.1-0.2: {np.sum((abs_sims >= 0.1) & (abs_sims < 0.2))} pairs ({100*np.mean((abs_sims >= 0.1) & (abs_sims < 0.2)):.1f}%)")
    print(f"  0.2-0.5: {np.sum((abs_sims >= 0.2) & (abs_sims < 0.5))} pairs ({100*np.mean((abs_sims >= 0.2) & (abs_sims < 0.5)):.1f}%)")
    print(f"  0.5-0.7: {np.sum((abs_sims >= 0.5) & (abs_sims < 0.7))} pairs ({100*np.mean((abs_sims >= 0.5) & (abs_sims < 0.7)):.1f}%)")
    print(f"  0.7-0.9: {np.sum((abs_sims >= 0.7) & (abs_sims < 0.9))} pairs ({100*np.mean((abs_sims >= 0.7) & (abs_sims < 0.9)):.1f}%)")
    print(f"  0.9-1.0: {np.sum(abs_sims >= 0.9)} pairs ({100*np.mean(abs_sims >= 0.9):.1f}%)")

    # Top similar pairs
    print(f"\nTop {top_n} most similar pairs:")
    flat_indices = np.argsort(np.abs(sim_matrix).ravel())[::-1][:top_n]
    for rank, idx in enumerate(flat_indices, 1):
        i, j = np.unravel_index(idx, sim_matrix.shape)
        print(f"  {rank:2d}. User PC{i:2d} ↔ Asst PC{j:2d}: {sim_matrix[i, j]:+.4f}")

    # Show full matrix (first 10x10 if large)
    show_n = min(10, n_user, n_asst)
    print(f"\nSimilarity matrix (first {show_n}×{show_n}):")
    print("      ", end="")
    for j in range(show_n):
        print(f"Asst{j:2d} ", end="")
    print()

    for i in range(show_n):
        print(f"User{i:2d}", end=" ")
        for j in range(show_n):
            val = sim_matrix[i, j]
            if abs(val) > 0.7:
                print(f"\033[91m{val:+6.3f}\033[0m", end=" ")  # Red for high
            elif abs(val) > 0.3:
                print(f"\033[93m{val:+6.3f}\033[0m", end=" ")  # Yellow for medium
            else:
                print(f"{val:+6.3f}", end=" ")  # Normal for low
        print()

    # Save full matrix
    output_file = Path(f"/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/controlled_variation/layer_{layer_idx}_similarity_matrix.npy")
    np.save(output_file, sim_matrix)
    print(f"\nFull matrix saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=30, help="Layer index to analyze")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top pairs to show")
    args = parser.parse_args()

    base_dir = Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/controlled_variation")

    print(f"Loading components for layer {args.layer}...")
    user_comps = load_components(base_dir, "user", args.layer)
    asst_comps = load_components(base_dir, "assistant", args.layer)

    print(f"User components shape: {user_comps.shape}")
    print(f"Assistant components shape: {asst_comps.shape}")

    print("\nComputing pairwise cosine similarities...")
    sim_matrix = compute_cosine_similarity_matrix(user_comps, asst_comps)

    print_similarity_matrix(sim_matrix, args.layer, args.top_n)


if __name__ == "__main__":
    main()
