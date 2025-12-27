#!/usr/bin/env python3
"""Compare ratio-based PCA vs contrastive PCA (cPCA).

This script loads results from both methods and compares:
1. Component similarity (cosine similarity between top PCs)
2. Explained variance
3. Diagnostic metrics (ratio removal stats vs alpha values)
4. Projection distributions

Usage:
    python scripts/compare_pca_methods.py \\
        --ratio-pca results/ratio_pca/model_ratio_pca.npz \\
        --cpca results/cpca/model_cpca.npz \\
        --output results/comparison/

    # With summary JSONs for diagnostics
    python scripts/compare_pca_methods.py \\
        --ratio-pca results/ratio_pca/model_ratio_pca.npz \\
        --cpca results/cpca/model_cpca.npz \\
        --ratio-summary results/ratio_pca/model_summary.json \\
        --cpca-summary results/cpca/model_summary.json \\
        --output results/comparison/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def load_pca_results(npz_path: Path) -> dict:
    """Load PCA results from NPZ file.

    Supports two formats:
    1. Stacked format (cPCA/new ratio PCA): components [n_layers, n_components, hidden_dim]
    2. Per-layer format (old): components as separate keys "0", "1", etc.
    """
    data = np.load(npz_path, allow_pickle=True)

    # Check format
    if "components" in data.keys():
        # Stacked format (cPCA/new ratio PCA)
        components_all = data["components"]  # [n_layers, n_components, hidden_dim]
        n_layers = components_all.shape[0]

        components = {i: components_all[i] for i in range(n_layers)}

        # Extract explained variance if available
        if "explained_variance" in data:
            ev_all = data["explained_variance"]
            explained_variance = {i: ev_all[i] for i in range(n_layers)}
        else:
            explained_variance = {}

        if "explained_variance_ratio" in data:
            evr_all = data["explained_variance_ratio"]
            explained_variance_ratio = {i: evr_all[i] for i in range(n_layers)}
        elif "eigenvalues" in data:
            # Compute explained variance ratio from eigenvalues
            eigenvalues_all = data["eigenvalues"]
            total_var = eigenvalues_all.sum(axis=1, keepdims=True)
            evr_all = eigenvalues_all / total_var
            explained_variance_ratio = {i: evr_all[i] for i in range(n_layers)}
        else:
            explained_variance_ratio = {}
    else:
        # Per-layer format (old)
        components = {}
        explained_variance = {}
        explained_variance_ratio = {}

        for key in data.keys():
            if key.startswith("explained_variance_ratio_"):
                layer_idx = int(key.split("_")[-1])
                explained_variance_ratio[layer_idx] = data[key]
            elif key.startswith("explained_variance_"):
                layer_idx = int(key.split("_")[-1])
                explained_variance[layer_idx] = data[key]
            elif key.isdigit():
                layer_idx = int(key)
                components[layer_idx] = data[key]

    return {
        "components": components,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
    }


def compute_component_similarity(
    ratio_comps: np.ndarray, cpca_comps: np.ndarray, top_k: int = 10
) -> dict:
    """Compute similarity between ratio PCA and cPCA components.

    Args:
        ratio_comps: [n_components, hidden_dim] from ratio PCA
        cpca_comps: [n_components, hidden_dim] from cPCA
        top_k: number of top components to compare

    Returns:
        dict with similarity metrics
    """
    # Limit to top k
    ratio_top = ratio_comps[:top_k]
    cpca_top = cpca_comps[:top_k]

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(ratio_top, cpca_top)

    # For each ratio PC, find best matching cPCA PC
    best_matches = np.max(np.abs(sim_matrix), axis=1)
    avg_similarity = np.mean(best_matches)

    # Alignment: how many ratio PCs have their best match at the same index?
    diagonal_sim = np.abs(np.diag(sim_matrix))
    alignment_score = np.mean(diagonal_sim)

    return {
        "similarity_matrix": sim_matrix,
        "best_match_similarities": best_matches,
        "avg_similarity": avg_similarity,
        "alignment_score": alignment_score,
    }


def plot_component_similarity(
    ratio_results: dict,
    cpca_results: dict,
    output_dir: Path,
    layers: list = None,
):
    """Plot component similarity across layers."""
    if layers is None:
        layers = sorted(ratio_results["components"].keys())

    # Compute similarity per layer
    similarities = []
    alignments = []

    for layer_idx in layers:
        if layer_idx not in ratio_results["components"]:
            continue
        if layer_idx not in cpca_results["components"]:
            continue

        ratio_comps = ratio_results["components"][layer_idx]
        cpca_comps = cpca_results["components"][layer_idx]

        sim_info = compute_component_similarity(ratio_comps, cpca_comps, top_k=10)
        similarities.append(sim_info["avg_similarity"])
        alignments.append(sim_info["alignment_score"])

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Similarity
    ax1.plot(layers, similarities, marker="o", linewidth=2)
    ax1.axhline(y=0.8, color="r", linestyle="--", alpha=0.5, label="0.8 threshold")
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("Avg Best-Match Similarity", fontsize=12)
    ax1.set_title("Component Similarity: Ratio PCA vs cPCA", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Alignment
    ax2.plot(layers, alignments, marker="s", linewidth=2, color="orange")
    ax2.axhline(y=0.7, color="r", linestyle="--", alpha=0.5, label="0.7 threshold")
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("Alignment Score", fontsize=12)
    ax2.set_title("Component Alignment: Same-Index Similarity", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "component_similarity.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved component similarity plot to {output_dir / 'component_similarity.png'}")


def plot_explained_variance(
    ratio_results: dict,
    cpca_results: dict,
    output_dir: Path,
    layers: list = None,
):
    """Plot explained variance comparison."""
    if layers is None:
        layers = sorted(ratio_results["explained_variance"].keys())

    # Check if we have explained variance ratio data
    has_ratio_evr = len(ratio_results["explained_variance_ratio"]) > 0
    has_cpca_evr = len(cpca_results["explained_variance_ratio"]) > 0

    if not has_ratio_evr and not has_cpca_evr:
        print("Skipping explained variance plot - no data available")
        return

    # Compute total variance explained by top-10 PCs
    ratio_var = []
    cpca_var = []

    for layer_idx in layers:
        if has_ratio_evr and layer_idx in ratio_results["explained_variance_ratio"]:
            ratio_var.append(ratio_results["explained_variance_ratio"][layer_idx][:10].sum())
        elif has_ratio_evr:
            ratio_var.append(np.nan)

        if has_cpca_evr and layer_idx in cpca_results["explained_variance_ratio"]:
            cpca_var.append(cpca_results["explained_variance_ratio"][layer_idx][:10].sum())
        elif has_cpca_evr:
            cpca_var.append(np.nan)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    if has_ratio_evr and len(ratio_var) > 0:
        ax.plot(layers, ratio_var, marker="o", linewidth=2, label="Ratio PCA")
    if has_cpca_evr and len(cpca_var) > 0:
        ax.plot(layers, cpca_var, marker="s", linewidth=2, label="cPCA")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Total Variance Explained (Top-10 PCs)", fontsize=12)
    ax.set_title("Explained Variance: Ratio PCA vs cPCA", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "explained_variance.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved explained variance plot to {output_dir / 'explained_variance.png'}")


def plot_similarity_heatmap(
    ratio_results: dict,
    cpca_results: dict,
    output_dir: Path,
    layer_idx: int,
):
    """Plot similarity heatmap for a specific layer."""
    if layer_idx not in ratio_results["components"]:
        print(f"Layer {layer_idx} not found in ratio PCA results")
        return
    if layer_idx not in cpca_results["components"]:
        print(f"Layer {layer_idx} not found in cPCA results")
        return

    ratio_comps = ratio_results["components"][layer_idx][:10]
    cpca_comps = cpca_results["components"][layer_idx][:10]

    sim_matrix = cosine_similarity(ratio_comps, cpca_comps)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Cosine Similarity"},
        xticklabels=[f"cPC{i}" for i in range(10)],
        yticklabels=[f"rPC{i}" for i in range(10)],
        ax=ax,
    )

    ax.set_title(f"Component Similarity Matrix (Layer {layer_idx})", fontsize=14)
    ax.set_xlabel("cPCA Components", fontsize=12)
    ax.set_ylabel("Ratio PCA Components", fontsize=12)

    plt.tight_layout()
    plt.savefig(
        output_dir / f"similarity_heatmap_layer{layer_idx}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Saved similarity heatmap for layer {layer_idx} to {output_dir / f'similarity_heatmap_layer{layer_idx}.png'}"
    )


def main():
    parser = argparse.ArgumentParser(description="Compare ratio PCA vs cPCA")
    parser.add_argument(
        "--ratio-pca", type=Path, required=True, help="Ratio PCA results (.npz)"
    )
    parser.add_argument("--cpca", type=Path, required=True, help="cPCA results (.npz)")
    parser.add_argument(
        "--ratio-summary",
        type=Path,
        default=None,
        help="Ratio PCA summary JSON (optional)",
    )
    parser.add_argument(
        "--cpca-summary", type=Path, default=None, help="cPCA summary JSON (optional)"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory for plots"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layers to analyze (default: all)",
    )
    parser.add_argument(
        "--heatmap-layer",
        type=int,
        default=30,
        help="Layer for detailed heatmap (default: 30)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("COMPARING PCA METHODS")
    print("=" * 80)
    print(f"Ratio PCA: {args.ratio_pca}")
    print(f"cPCA: {args.cpca}")
    print(f"Output: {args.output}")
    print()

    # Load results
    print("Loading results...")
    ratio_results = load_pca_results(args.ratio_pca)
    cpca_results = load_pca_results(args.cpca)

    print(f"Ratio PCA layers: {len(ratio_results['components'])}")
    print(f"cPCA layers: {len(cpca_results['components'])}")
    print()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Determine layers to analyze
    layers = args.layers
    if layers is None:
        common_layers = set(ratio_results["components"].keys()) & set(
            cpca_results["components"].keys()
        )
        layers = sorted(common_layers)
        print(f"Analyzing {len(layers)} common layers")

    # Generate comparison plots
    print("\nGenerating comparison plots...")

    # 1. Component similarity across layers
    plot_component_similarity(ratio_results, cpca_results, args.output, layers)

    # 2. Explained variance comparison
    plot_explained_variance(ratio_results, cpca_results, args.output, layers)

    # 3. Detailed heatmap for one layer
    if args.heatmap_layer in layers:
        plot_similarity_heatmap(
            ratio_results, cpca_results, args.output, args.heatmap_layer
        )

    # 4. Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    avg_similarities = []
    for layer_idx in layers:
        ratio_comps = ratio_results["components"][layer_idx]
        cpca_comps = cpca_results["components"][layer_idx]
        sim_info = compute_component_similarity(ratio_comps, cpca_comps, top_k=10)
        avg_similarities.append(sim_info["avg_similarity"])

    print(f"Average component similarity: {np.mean(avg_similarities):.3f} ± {np.std(avg_similarities):.3f}")
    print(f"Min similarity: {np.min(avg_similarities):.3f} (layer {layers[np.argmin(avg_similarities)]})")
    print(f"Max similarity: {np.max(avg_similarities):.3f} (layer {layers[np.argmax(avg_similarities)]})")
    print()

    # Load summaries if provided
    if args.ratio_summary and args.cpca_summary:
        print("Loading method diagnostics...")
        with open(args.ratio_summary) as f:
            ratio_summary = json.load(f)
        with open(args.cpca_summary) as f:
            cpca_summary = json.load(f)

        print("\nRatio PCA diagnostics:")
        print(f"  k (PCs removed): {ratio_summary['config']['k']}")
        print(
            f"  Avg ratio removed: {ratio_summary['diagnostics']['avg_ratio_removed']:.4f}"
        )
        print(f"  Avg ratio kept: {ratio_summary['diagnostics']['avg_ratio_kept']:.4f}")

        if "alpha_per_layer" in cpca_summary:
            alphas = list(cpca_summary["alpha_per_layer"].values())
            print("\ncPCA diagnostics:")
            print(f"  Avg alpha: {np.mean(alphas):.4f}")
            print(f"  Alpha range: [{np.min(alphas):.4f}, {np.max(alphas):.4f}]")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
