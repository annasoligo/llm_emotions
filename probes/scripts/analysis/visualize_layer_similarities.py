#!/usr/bin/env python3
"""Visualize pairwise cosine similarity matrices across multiple layers."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def load_components(base_dir: Path, isolation_type: str, layer_idx: int) -> np.ndarray:
    """Load cPCA components for a specific layer."""
    layer_dir = base_dir / f"{isolation_type}_isolation" / "layers_0_61"
    layer_file = layer_dir / f"layer_{layer_idx}.npz"
    data = np.load(layer_file)
    return data['components']


def compute_cosine_similarity_matrix(user_comps: np.ndarray, asst_comps: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarities."""
    user_norm = user_comps / np.linalg.norm(user_comps, axis=1, keepdims=True)
    asst_norm = asst_comps / np.linalg.norm(asst_comps, axis=1, keepdims=True)
    return user_norm @ asst_norm.T


def plot_similarity_heatmaps(base_dir: Path, layers: list, output_path: Path):
    """Create subplot heatmaps for multiple layers."""
    n_layers = len(layers)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Global color scale
    vmin, vmax = -1, 1

    for idx, layer_idx in enumerate(layers):
        print(f"Processing layer {layer_idx}...")

        # Load components
        user_comps = load_components(base_dir, "user", layer_idx)
        asst_comps = load_components(base_dir, "assistant", layer_idx)

        # Compute similarity
        sim_matrix = compute_cosine_similarity_matrix(user_comps, asst_comps)

        # Statistics
        max_sim = np.max(np.abs(sim_matrix))
        mean_sim = np.mean(np.abs(sim_matrix))
        high_sim_count = np.sum(np.abs(sim_matrix) > 0.5)

        # Plot heatmap
        ax = axes[idx]
        im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')

        ax.set_title(f'Layer {layer_idx}\nMax: {max_sim:.3f}, Mean: {mean_sim:.3f}\nHigh (>0.5): {high_sim_count} pairs',
                     fontsize=11, pad=10)
        ax.set_xlabel('Assistant PC', fontsize=10)
        ax.set_ylabel('User PC', fontsize=10)

        # Add gridlines
        ax.set_xticks(np.arange(0, 50, 5))
        ax.set_yticks(np.arange(0, 50, 5))
        ax.grid(True, alpha=0.3, linewidth=0.5)

        # Tick labels
        ax.tick_params(labelsize=8)

        # Print summary
        print(f"  Layer {layer_idx}: max={max_sim:.4f}, mean={mean_sim:.4f}, >0.5: {high_sim_count}")

    # Hide the 6th subplot
    axes[5].axis('off')

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal',
                        fraction=0.05, pad=0.08, aspect=40)
    cbar.set_label('Cosine Similarity', fontsize=12)

    plt.suptitle('User vs Assistant PC Cosine Similarities Across Layers',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    # Also create individual detailed plots
    for layer_idx in layers:
        fig_detail, ax = plt.subplots(1, 1, figsize=(10, 9))

        user_comps = load_components(base_dir, "user", layer_idx)
        asst_comps = load_components(base_dir, "assistant", layer_idx)
        sim_matrix = compute_cosine_similarity_matrix(user_comps, asst_comps)

        im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        max_sim = np.max(np.abs(sim_matrix))
        mean_sim = np.mean(np.abs(sim_matrix))
        high_sim_count = np.sum(np.abs(sim_matrix) > 0.5)

        ax.set_title(f'Layer {layer_idx}: User vs Assistant PC Cosine Similarities\n'
                     f'Max: {max_sim:.4f} | Mean: {mean_sim:.4f} | High (>0.5): {high_sim_count} pairs',
                     fontsize=13, pad=15)
        ax.set_xlabel('Assistant PC Index', fontsize=11)
        ax.set_ylabel('User PC Index', fontsize=11)

        # Mark diagonal
        ax.plot([0, 49], [0, 49], 'k--', alpha=0.3, linewidth=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Cosine Similarity', fontsize=11)

        # Grid
        ax.set_xticks(np.arange(0, 50, 5))
        ax.set_yticks(np.arange(0, 50, 5))
        ax.grid(True, alpha=0.3, linewidth=0.5)

        plt.tight_layout()

        detail_path = output_path.parent / f"layer_{layer_idx}_similarity_heatmap.png"
        plt.savefig(detail_path, dpi=150, bbox_inches='tight')
        print(f"Detailed plot saved: {detail_path}")
        plt.close(fig_detail)


def main():
    base_dir = Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/controlled_variation")
    output_dir = Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/controlled_variation/visualizations")
    output_dir.mkdir(exist_ok=True)

    layers = [10, 20, 30, 40, 50]
    output_path = output_dir / "pc_similarity_comparison.png"

    print("Generating similarity heatmaps...")
    plot_similarity_heatmaps(base_dir, layers, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
