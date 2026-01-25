#!/usr/bin/env python3
"""
Scatter plots of probe weights projected onto PC1 and PC2.
Shows where user and assistant probes land in emotion-specific PC space.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_probe_weights(model_path: str) -> tuple:
    """Load probe weight matrix and emotion labels from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    # Shape: (num_classes, hidden_dim)
    weights = checkpoint['linear.weight'].numpy()

    # Get emotion names - we know there are 6 emotions
    emotion_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

    return weights, emotion_names


def load_opposite_pcs(isolation_type: str, layer: int) -> np.ndarray:
    """Load opposite-source emotion PCs."""
    pc_path = f"outputs/orthogonal_pcs/diverse_isolation/{isolation_type}/layer_{layer}.npz"
    data = np.load(pc_path)
    opposite_pcs = data['orthogonal_pcs']  # Shape: (hidden_dim, k)
    return opposite_pcs


def compute_projections(weight_matrix: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    """Compute projection onto PCs."""
    # weight_matrix: (num_classes, hidden_dim) or (hidden_dim,)
    # pcs: (hidden_dim, k)
    # Returns: (num_classes, k) or (k,) - projection onto each PC
    projections = weight_matrix @ pcs
    return projections


def create_scatter_plots(layers: list, lambdas: list, output_dir: str):
    """Create 2D scatter plots of probe projections."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # For each layer, create a figure with subplots for different lambdas
    for layer in layers:
        fig, axes = plt.subplots(2, len(lambdas), figsize=(6 * len(lambdas), 10))
        if len(lambdas) == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'Probe Projections onto PC1 vs PC2 - Layer {layer}', fontsize=16, y=0.995)

        # Load PCs
        user_pcs = load_opposite_pcs('user', layer)  # Captures assistant emotion
        asst_pcs = load_opposite_pcs('assistant', layer)  # Captures user emotion

        for lambda_idx, lambda_val in enumerate(lambdas):
            # Load probes
            user_probe_path = f"outputs/probes/diverse_isolation/user_layer{layer}_lambda{float(lambda_val)}/model.pt"
            asst_probe_path = f"outputs/probes/diverse_isolation/assistant_layer{layer}_lambda{float(lambda_val)}/model.pt"

            if not Path(user_probe_path).exists() or not Path(asst_probe_path).exists():
                continue

            user_weights = load_probe_weights(user_probe_path)
            asst_weights = load_probe_weights(asst_probe_path)

            # Project onto PCs
            user_on_user_pcs = compute_projections(user_weights, user_pcs)
            asst_on_user_pcs = compute_projections(asst_weights, user_pcs)
            user_on_asst_pcs = compute_projections(user_weights, asst_pcs)
            asst_on_asst_pcs = compute_projections(asst_weights, asst_pcs)

            # Plot 1: User Isolation PCs (assistant emotion variance)
            ax = axes[0, lambda_idx]

            # Plot origin
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)

            # Plot probes
            ax.scatter(user_on_user_pcs[0], user_on_user_pcs[1],
                      s=200, marker='o', color='#1f77b4',
                      label='User Probe', edgecolors='black', linewidth=2, zorder=3)
            ax.scatter(asst_on_user_pcs[0], asst_on_user_pcs[1],
                      s=200, marker='s', color='#ff7f0e',
                      label='Asst Probe', edgecolors='black', linewidth=2, zorder=3)

            # Add arrows from origin
            ax.arrow(0, 0, user_on_user_pcs[0], user_on_user_pcs[1],
                    head_width=0.001, head_length=0.001, fc='#1f77b4', ec='#1f77b4',
                    alpha=0.3, linewidth=1.5)
            ax.arrow(0, 0, asst_on_user_pcs[0], asst_on_user_pcs[1],
                    head_width=0.001, head_length=0.001, fc='#ff7f0e', ec='#ff7f0e',
                    alpha=0.3, linewidth=1.5)

            ax.set_xlabel('PC1 (Asst Emotion)', fontsize=11)
            ax.set_ylabel('PC2 (Asst Emotion)', fontsize=11)
            ax.set_title(f'λ={lambda_val:.0f}: User Isolation PCs\n(captures Assistant emotion)', fontsize=10)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

            # Add distance annotation
            dist = np.linalg.norm(user_on_user_pcs[:2] - asst_on_user_pcs[:2])
            ax.text(0.02, 0.98, f'Distance: {dist:.4f}',
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Plot 2: Assistant Isolation PCs (user emotion variance)
            ax = axes[1, lambda_idx]

            # Plot origin
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.3, linewidth=0.5)

            # Plot probes
            ax.scatter(user_on_asst_pcs[0], user_on_asst_pcs[1],
                      s=200, marker='o', color='#1f77b4',
                      label='User Probe', edgecolors='black', linewidth=2, zorder=3)
            ax.scatter(asst_on_asst_pcs[0], asst_on_asst_pcs[1],
                      s=200, marker='s', color='#ff7f0e',
                      label='Asst Probe', edgecolors='black', linewidth=2, zorder=3)

            # Add arrows from origin
            ax.arrow(0, 0, user_on_asst_pcs[0], user_on_asst_pcs[1],
                    head_width=0.001, head_length=0.001, fc='#1f77b4', ec='#1f77b4',
                    alpha=0.3, linewidth=1.5)
            ax.arrow(0, 0, asst_on_asst_pcs[0], asst_on_asst_pcs[1],
                    head_width=0.001, head_length=0.001, fc='#ff7f0e', ec='#ff7f0e',
                    alpha=0.3, linewidth=1.5)

            ax.set_xlabel('PC1 (User Emotion)', fontsize=11)
            ax.set_ylabel('PC2 (User Emotion)', fontsize=11)
            ax.set_title(f'λ={lambda_val:.0f}: Assistant Isolation PCs\n(captures User emotion)', fontsize=10)
            ax.legend(loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

            # Add distance annotation
            dist = np.linalg.norm(user_on_asst_pcs[:2] - asst_on_asst_pcs[:2])
            ax.text(0.02, 0.98, f'Distance: {dist:.4f}',
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        output_path = output_dir / f'probe_pc_scatter_layer{layer}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Scatter plots of probe projections")
    parser.add_argument("--layers", type=int, nargs='+', default=[20, 30, 50, 60])
    parser.add_argument("--lambdas", type=float, nargs='+', default=[0, 10, 100])
    parser.add_argument("--output", type=str, default="probes/visualizations/pc_scatter")

    args = parser.parse_args()

    print("=" * 80)
    print("CREATING PC SCATTER PLOTS")
    print("=" * 80)
    print(f"Layers: {args.layers}")
    print(f"Lambdas: {args.lambdas}")
    print(f"Output: {args.output}")
    print()

    create_scatter_plots(args.layers, args.lambdas, args.output)

    print("\n✓ Done!")
    print(f"Visualizations saved to: {args.output}")


if __name__ == "__main__":
    main()
