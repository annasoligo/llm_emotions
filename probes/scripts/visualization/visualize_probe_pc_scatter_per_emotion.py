#!/usr/bin/env python3
"""
Scatter plots showing each emotion class's probe weight projected onto PC1 vs PC2.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_probe_weights(model_path: str) -> tuple:
    """Load probe weight matrix from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    weights = checkpoint['linear.weight'].numpy()  # (num_classes, hidden_dim)
    emotion_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    return weights, emotion_names


def compute_probe_pcs(weight_matrix: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Compute PCA on probe weight directions.

    Args:
        weight_matrix: (num_classes, hidden_dim) probe weights
        n_components: Number of PCs to compute

    Returns:
        pcs: (hidden_dim, n_components) principal components
    """
    from sklearn.decomposition import PCA

    # Run PCA on probe weights (each emotion is a sample)
    pca = PCA(n_components=n_components)
    pca.fit(weight_matrix)  # (num_classes, hidden_dim)

    # Return components transposed to (hidden_dim, n_components)
    return pca.components_.T


def compute_projections(weight_matrix: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    """Project weight matrix onto PCs."""
    # weight_matrix: (num_classes, hidden_dim)
    # pcs: (hidden_dim, k)
    # Returns: (num_classes, k)
    return weight_matrix @ pcs


def create_scatter_plots(layers: list, lambdas: list, output_dir: str):
    """Create scatter plots with one point per emotion."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Color map for emotions
    emotion_colors = {
        'anger': '#d62728',      # red
        'disgust': '#8c564b',    # brown
        'fear': '#9467bd',       # purple
        'happiness': '#2ca02c',  # green
        'sadness': '#1f77b4',    # blue
        'surprise': '#ff7f0e',   # orange
    }

    # Marker styles
    user_marker = 'o'
    asst_marker = 's'

    for layer in layers:
        fig, axes = plt.subplots(2, len(lambdas), figsize=(7 * len(lambdas), 12))
        if len(lambdas) == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'Emotion-Specific Probe Projections - Layer {layer}', fontsize=16, y=0.995)

        for lambda_idx, lambda_val in enumerate(lambdas):
            # Load probes
            user_probe_path = f"outputs/probes/diverse_isolation/user_layer{layer}_lambda{float(lambda_val)}/model.pt"
            asst_probe_path = f"outputs/probes/diverse_isolation/assistant_layer{layer}_lambda{float(lambda_val)}/model.pt"

            if not Path(user_probe_path).exists() or not Path(asst_probe_path).exists():
                continue

            user_weights, emotion_names = load_probe_weights(user_probe_path)
            asst_weights, _ = load_probe_weights(asst_probe_path)

            # Compute PCs from probe weights
            user_pcs = compute_probe_pcs(user_weights, n_components=2)  # (hidden_dim, 2)
            asst_pcs = compute_probe_pcs(asst_weights, n_components=2)  # (hidden_dim, 2)

            # Project onto PCs
            user_on_user_pcs = compute_projections(user_weights, user_pcs)  # (6, 2)
            asst_on_user_pcs = compute_projections(asst_weights, user_pcs)
            user_on_asst_pcs = compute_projections(user_weights, asst_pcs)
            asst_on_asst_pcs = compute_projections(asst_weights, asst_pcs)

            # Plot 1: User Probe PCs
            ax = axes[0, lambda_idx]
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.2, linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.2, linewidth=0.5)

            for i, emotion in enumerate(emotion_names):
                color = emotion_colors[emotion]
                # User probe
                ax.scatter(user_on_user_pcs[i, 0], user_on_user_pcs[i, 1],
                          s=150, marker=user_marker, color=color,
                          label=f'{emotion}' if lambda_idx == 0 else None,
                          edgecolors='black', linewidth=1.5, alpha=0.7)
                # Assistant probe
                ax.scatter(asst_on_user_pcs[i, 0], asst_on_user_pcs[i, 1],
                          s=150, marker=asst_marker, color=color,
                          edgecolors='black', linewidth=1.5, alpha=0.7)

            ax.set_xlabel('PC1 (User Probe)', fontsize=10)
            ax.set_ylabel('PC2 (User Probe)', fontsize=10)
            ax.set_title(f'λ={lambda_val:.0f}: User Probe PCs\n(○=User probe, □=Asst probe)', fontsize=9)
            if lambda_idx == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

            # Plot 2: Assistant Probe PCs
            ax = axes[1, lambda_idx]
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.2, linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.2, linewidth=0.5)

            for i, emotion in enumerate(emotion_names):
                color = emotion_colors[emotion]
                # User probe
                ax.scatter(user_on_asst_pcs[i, 0], user_on_asst_pcs[i, 1],
                          s=150, marker=user_marker, color=color,
                          edgecolors='black', linewidth=1.5, alpha=0.7)
                # Assistant probe
                ax.scatter(asst_on_asst_pcs[i, 0], asst_on_asst_pcs[i, 1],
                          s=150, marker=asst_marker, color=color,
                          edgecolors='black', linewidth=1.5, alpha=0.7)

            ax.set_xlabel('PC1 (Asst Probe)', fontsize=10)
            ax.set_ylabel('PC2 (Asst Probe)', fontsize=10)
            ax.set_title(f'λ={lambda_val:.0f}: Asst Probe PCs\n(○=User probe, □=Asst probe)', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

        output_path = output_dir / f'probe_pc_scatter_emotions_layer{layer}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Per-emotion scatter plots")
    parser.add_argument("--layers", type=int, nargs='+', default=[20, 30, 50, 60])
    parser.add_argument("--lambdas", type=float, nargs='+', default=[0, 10, 100])
    parser.add_argument("--output", type=str, default="probes/visualizations/pc_scatter_emotions")

    args = parser.parse_args()

    print("=" * 80)
    print("CREATING PER-EMOTION PC SCATTER PLOTS")
    print("=" * 80)
    print(f"Layers: {args.layers}")
    print(f"Lambdas: {args.lambdas}")
    print()

    create_scatter_plots(args.layers, args.lambdas, args.output)

    print("\n✓ Done!")
    print(f"Visualizations saved to: {args.output}")


if __name__ == "__main__":
    main()
