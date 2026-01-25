#!/usr/bin/env python3
"""
Visualize probe weight projections onto opposite-source emotion PCs.

Shows how user and assistant probe weights project onto:
1. User emotion PCs (computed from user emotion variance)
2. Assistant emotion PCs (computed from assistant emotion variance)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_probe_weights(model_path: str) -> np.ndarray:
    """Load probe weight matrix from checkpoint."""
    checkpoint = torch.load(model_path, map_location='cpu')
    # Shape: (num_classes, hidden_dim)
    weights = checkpoint['linear.weight'].numpy()
    # Average across emotion classes to get single direction
    # Shape: (hidden_dim,)
    avg_weight = weights.mean(axis=0)
    return avg_weight


def load_opposite_pcs(isolation_type: str, layer: int) -> tuple:
    """Load opposite-source emotion PCs."""
    pc_path = f"outputs/orthogonal_pcs/diverse_isolation/{isolation_type}/layer_{layer}.npz"
    data = np.load(pc_path)

    # These are the PCs that capture variance in the OPPOSITE person's emotions
    opposite_pcs = data['orthogonal_pcs']  # Shape: (hidden_dim, k)

    return opposite_pcs


def compute_projections(weight_vector: np.ndarray, pcs: np.ndarray) -> np.ndarray:
    """Compute projection strengths onto PCs."""
    # weight_vector: (hidden_dim,)
    # pcs: (hidden_dim, k)
    # Returns: (k,) - projection onto each PC
    projections = weight_vector @ pcs  # (k,)
    return projections


def plot_projections(
    layers: list,
    lambdas: list,
    output_dir: str,
):
    """Create comprehensive projection visualizations."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # For each layer and lambda, create a figure with 2 subplots:
    # 1. Both probes projected onto USER PCs
    # 2. Both probes projected onto ASSISTANT PCs

    for layer in layers:
        fig, axes = plt.subplots(len(lambdas), 2, figsize=(14, 4 * len(lambdas)))
        if len(lambdas) == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle(f'Probe Weight Projections - Layer {layer}', fontsize=16, y=0.995)

        # Load PCs for this layer
        user_pcs = load_opposite_pcs('user', layer)  # These capture ASSISTANT emotion variance
        asst_pcs = load_opposite_pcs('assistant', layer)  # These capture USER emotion variance

        for lambda_idx, lambda_val in enumerate(lambdas):
            # Load probe weights
            user_probe_path = f"outputs/probes/diverse_isolation/user_layer{layer}_lambda{float(lambda_val)}/model.pt"
            asst_probe_path = f"outputs/probes/diverse_isolation/assistant_layer{layer}_lambda{float(lambda_val)}/model.pt"

            if not Path(user_probe_path).exists() or not Path(asst_probe_path).exists():
                continue

            user_weights = load_probe_weights(user_probe_path)
            asst_weights = load_probe_weights(asst_probe_path)

            # Project onto USER isolation PCs (which capture ASSISTANT emotion variance)
            # The user probe should avoid these if regularization works
            user_on_user_pcs = compute_projections(user_weights, user_pcs)
            asst_on_user_pcs = compute_projections(asst_weights, user_pcs)

            # Project onto ASSISTANT isolation PCs (which capture USER emotion variance)
            # The assistant probe should avoid these if regularization works
            user_on_asst_pcs = compute_projections(user_weights, asst_pcs)
            asst_on_asst_pcs = compute_projections(asst_weights, asst_pcs)

            # Plot 1: Projections onto USER isolation PCs (assistant emotion variance)
            ax = axes[lambda_idx, 0]
            x = np.arange(len(user_on_user_pcs))
            width = 0.35

            ax.bar(x - width/2, np.abs(user_on_user_pcs), width,
                   label='User Probe', alpha=0.8, color='#1f77b4')
            ax.bar(x + width/2, np.abs(asst_on_user_pcs), width,
                   label='Asst Probe', alpha=0.8, color='#ff7f0e')

            ax.set_xlabel('PC Index (ordered by F-statistic)')
            ax.set_ylabel('|Projection Strength|')
            ax.set_title(f'λ={lambda_val:.0f}: Projection onto User Isolation PCs\n(captures Assistant emotion variance)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add L2 norms as text
            user_norm = np.linalg.norm(user_on_user_pcs)
            asst_norm = np.linalg.norm(asst_on_user_pcs)
            ax.text(0.98, 0.98, f'User L2: {user_norm:.3f}\nAsst L2: {asst_norm:.3f}',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Plot 2: Projections onto ASSISTANT isolation PCs (user emotion variance)
            ax = axes[lambda_idx, 1]
            x = np.arange(len(user_on_asst_pcs))

            ax.bar(x - width/2, np.abs(user_on_asst_pcs), width,
                   label='User Probe', alpha=0.8, color='#1f77b4')
            ax.bar(x + width/2, np.abs(asst_on_asst_pcs), width,
                   label='Asst Probe', alpha=0.8, color='#ff7f0e')

            ax.set_xlabel('PC Index (ordered by F-statistic)')
            ax.set_ylabel('|Projection Strength|')
            ax.set_title(f'λ={lambda_val:.0f}: Projection onto Assistant Isolation PCs\n(captures User emotion variance)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add L2 norms
            user_norm = np.linalg.norm(user_on_asst_pcs)
            asst_norm = np.linalg.norm(asst_on_asst_pcs)
            ax.text(0.98, 0.98, f'User L2: {user_norm:.3f}\nAsst L2: {asst_norm:.3f}',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        output_path = output_dir / f'probe_projections_layer{layer}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    # Create summary plot: L2 norms across all layers
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('L2 Norm of Probe Projections onto Opposite-Source PCs', fontsize=14)

    for lambda_val in lambdas:
        user_norms_on_opposite = []
        asst_norms_on_opposite = []
        layer_list = []

        for layer in layers:
            user_probe_path = f"outputs/probes/diverse_isolation/user_layer{layer}_lambda{float(lambda_val)}/model.pt"
            asst_probe_path = f"outputs/probes/diverse_isolation/assistant_layer{layer}_lambda{float(lambda_val)}/model.pt"

            if not Path(user_probe_path).exists() or not Path(asst_probe_path).exists():
                continue

            # Load PCs
            user_pcs = load_opposite_pcs('user', layer)  # Assistant emotion PCs
            asst_pcs = load_opposite_pcs('assistant', layer)  # User emotion PCs

            # Load weights
            user_weights = load_probe_weights(user_probe_path)
            asst_weights = load_probe_weights(asst_probe_path)

            # User probe on its OPPOSITE PCs (assistant emotion PCs)
            user_on_opposite = compute_projections(user_weights, user_pcs)
            user_norms_on_opposite.append(np.linalg.norm(user_on_opposite))

            # Assistant probe on its OPPOSITE PCs (user emotion PCs)
            asst_on_opposite = compute_projections(asst_weights, asst_pcs)
            asst_norms_on_opposite.append(np.linalg.norm(asst_on_opposite))

            layer_list.append(layer)

        # Plot user probe projections onto opposite (assistant) PCs
        axes[0].plot(layer_list, user_norms_on_opposite, marker='o', label=f'λ={lambda_val:.0f}')

        # Plot assistant probe projections onto opposite (user) PCs
        axes[1].plot(layer_list, asst_norms_on_opposite, marker='o', label=f'λ={lambda_val:.0f}')

    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('L2 Norm')
    axes[0].set_title('User Probe → Assistant Emotion PCs\n(Should decrease with regularization)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('L2 Norm')
    axes[1].set_title('Assistant Probe → User Emotion PCs\n(Should decrease with regularization)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    summary_path = output_dir / 'probe_projections_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {summary_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize probe projections onto PCs")
    parser.add_argument("--layers", type=int, nargs='+', default=[0, 10, 20, 30, 40, 50, 60])
    parser.add_argument("--lambdas", type=float, nargs='+', default=[0, 100])
    parser.add_argument("--output", type=str, default="probes/visualizations/projections")

    args = parser.parse_args()

    print("=" * 80)
    print("VISUALIZING PROBE WEIGHT PROJECTIONS")
    print("=" * 80)
    print(f"Layers: {args.layers}")
    print(f"Lambdas: {args.lambdas}")
    print(f"Output: {args.output}")
    print()

    plot_projections(args.layers, args.lambdas, args.output)

    print("\n✓ Done!")
    print(f"Visualizations saved to: {args.output}")


if __name__ == "__main__":
    main()
