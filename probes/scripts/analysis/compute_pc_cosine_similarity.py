#!/usr/bin/env python3
"""Compute pairwise cosine similarity between user and assistant cPCA components."""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import json

def load_components(base_dir: Path, isolation_type: str) -> Dict[int, np.ndarray]:
    """Load cPCA components for all layers.

    Returns:
        Dict mapping layer_idx -> components array [n_components, hidden_dim]
    """
    layer_dir = base_dir / f"{isolation_type}_isolation" / "layers_0_61"

    components = {}
    for layer_idx in range(62):
        layer_file = layer_dir / f"layer_{layer_idx}.npz"
        if layer_file.exists():
            data = np.load(layer_file)
            components[layer_idx] = data['components']  # [n_components, hidden_dim]

    return components


def compute_cosine_similarity_matrix(
    user_comps: np.ndarray,
    asst_comps: np.ndarray
) -> np.ndarray:
    """Compute pairwise cosine similarities between user and assistant components.

    Args:
        user_comps: [n_components, hidden_dim]
        asst_comps: [n_components, hidden_dim]

    Returns:
        similarity_matrix: [n_user_components, n_asst_components]
    """
    # Normalize
    user_norm = user_comps / np.linalg.norm(user_comps, axis=1, keepdims=True)
    asst_norm = asst_comps / np.linalg.norm(asst_comps, axis=1, keepdims=True)

    # Compute cosine similarity matrix
    similarity = user_norm @ asst_norm.T

    return similarity


def analyze_similarity(
    user_components: Dict[int, np.ndarray],
    asst_components: Dict[int, np.ndarray]
) -> Dict:
    """Analyze cosine similarity across all layers.

    Returns dict with per-layer statistics and summaries.
    """
    results = {
        "per_layer": {},
        "summary": {}
    }

    all_max_sims = []
    all_mean_sims = []

    for layer_idx in sorted(user_components.keys()):
        if layer_idx not in asst_components:
            continue

        user_comps = user_components[layer_idx]
        asst_comps = asst_components[layer_idx]

        # Compute similarity matrix
        sim_matrix = compute_cosine_similarity_matrix(user_comps, asst_comps)

        # Compute statistics
        max_sim = np.max(np.abs(sim_matrix))
        mean_sim = np.mean(np.abs(sim_matrix))

        # Top 5 similar pairs
        flat_indices = np.argsort(np.abs(sim_matrix).ravel())[::-1][:5]
        top_pairs = []
        for idx in flat_indices:
            i, j = np.unravel_index(idx, sim_matrix.shape)
            top_pairs.append({
                "user_pc": int(i),
                "asst_pc": int(j),
                "similarity": float(sim_matrix[i, j])
            })

        results["per_layer"][layer_idx] = {
            "max_abs_similarity": float(max_sim),
            "mean_abs_similarity": float(mean_sim),
            "top_similar_pairs": top_pairs,
            "shape": list(sim_matrix.shape)
        }

        all_max_sims.append(max_sim)
        all_mean_sims.append(mean_sim)

    # Summary statistics
    results["summary"] = {
        "overall_max_similarity": float(np.max(all_max_sims)),
        "overall_mean_max_similarity": float(np.mean(all_max_sims)),
        "overall_mean_similarity": float(np.mean(all_mean_sims)),
        "num_layers": len(all_max_sims)
    }

    return results


def main():
    base_dir = Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/controlled_variation")

    print("Loading user isolation components...")
    user_components = load_components(base_dir, "user")
    print(f"Loaded {len(user_components)} layers for user isolation")

    print("Loading assistant isolation components...")
    asst_components = load_components(base_dir, "assistant")
    print(f"Loaded {len(asst_components)} layers for assistant isolation")

    print("\nComputing pairwise cosine similarities...")
    results = analyze_similarity(user_components, asst_components)

    # Print summary
    print("\n" + "=" * 80)
    print("COSINE SIMILARITY ANALYSIS: USER vs ASSISTANT PCs")
    print("=" * 80)
    print(f"Number of layers analyzed: {results['summary']['num_layers']}")
    print(f"Overall maximum similarity: {results['summary']['overall_max_similarity']:.4f}")
    print(f"Mean of max similarities per layer: {results['summary']['overall_mean_max_similarity']:.4f}")
    print(f"Overall mean similarity: {results['summary']['overall_mean_similarity']:.4f}")

    print("\nPer-layer max similarities:")
    for layer_idx in sorted(results["per_layer"].keys()):
        max_sim = results["per_layer"][layer_idx]["max_abs_similarity"]
        mean_sim = results["per_layer"][layer_idx]["mean_abs_similarity"]
        print(f"  Layer {layer_idx:2d}: max={max_sim:.4f}, mean={mean_sim:.4f}")

    # Show top similar pairs across all layers
    print("\nTop 10 most similar PC pairs (across all layers):")
    all_pairs = []
    for layer_idx, layer_data in results["per_layer"].items():
        for pair in layer_data["top_similar_pairs"]:
            all_pairs.append((layer_idx, pair))
    all_pairs.sort(key=lambda x: abs(x[1]["similarity"]), reverse=True)

    for i, (layer_idx, pair) in enumerate(all_pairs[:10], 1):
        print(f"  {i}. Layer {layer_idx}, User PC{pair['user_pc']} ↔ Asst PC{pair['asst_pc']}: {pair['similarity']:.4f}")

    # Save results
    output_file = base_dir / "pc_cosine_similarity_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
