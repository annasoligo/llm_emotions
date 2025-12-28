#!/usr/bin/env python3
"""Analyze and visualize multi-orthogonal probe search results.

Usage:
    python analyze_multi_orthogonal_results.py \
        --search-results outputs/probes/.../search_layer30_ortho10.0_seed42.pkl \
        --output results/multi_ortho_analysis/
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_k_vs_metrics(results: List[Dict], output_dir: Path):
    """Plot how metrics change with increasing K."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    k_values = [r['n_sets'] for r in results]

    # Extract metrics
    mean_accs = [r['final_metrics']['mean_accuracy'] for r in results]
    std_accs = [r['final_metrics']['std_accuracy'] for r in results]
    ortho_means = [r['final_metrics']['cross_dots_mean'] for r in results]
    cross_agreements = [r['final_metrics']['mean_cross_agreement'] for r in results]

    # 1. Mean accuracy with error bars
    ax = axes[0, 0]
    ax.errorbar(k_values, mean_accs, yerr=std_accs, marker='o', capsize=5, linewidth=2)
    ax.axhline(1/6, color='red', linestyle='--', alpha=0.5, label='Random (16.7%)')
    ax.set_xlabel('Number of Probe Sets (K)', fontsize=12)
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_title('Accuracy vs K', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Orthogonality (lower is better)
    ax = axes[0, 1]
    ax.plot(k_values, ortho_means, marker='s', linewidth=2, color='green')
    ax.set_xlabel('Number of Probe Sets (K)', fontsize=12)
    ax.set_ylabel('Mean |Cross-Dot Product|', fontsize=12)
    ax.set_title('Orthogonality vs K\n(Lower = Better Separation)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Cross-set agreement (should be low if independent)
    ax = axes[1, 0]
    ax.plot(k_values, cross_agreements, marker='^', linewidth=2, color='orange')
    ax.axhline(1/6, color='red', linestyle='--', alpha=0.5, label='Random agreement')
    ax.set_xlabel('Number of Probe Sets (K)', fontsize=12)
    ax.set_ylabel('Cross-Set Agreement', fontsize=12)
    ax.set_title('Prediction Agreement Between Sets\n(Lower = More Independent)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Per-set accuracy ranges
    ax = axes[1, 1]
    min_accs = [r['final_metrics']['min_accuracy'] for r in results]
    max_accs = [r['final_metrics']['max_accuracy'] for r in results]

    ax.fill_between(k_values, min_accs, max_accs, alpha=0.3, color='blue', label='Range')
    ax.plot(k_values, mean_accs, marker='o', linewidth=2, color='blue', label='Mean')
    ax.axhline(1/6, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlabel('Number of Probe Sets (K)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Set Accuracy Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'k_vs_metrics.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'k_vs_metrics.png'}")
    plt.close()


def plot_convergence_curves(results: List[Dict], output_dir: Path):
    """Plot training convergence for each K."""
    n_configs = len(results)
    fig, axes = plt.subplots(n_configs, 3, figsize=(15, 4 * n_configs))

    if n_configs == 1:
        axes = axes.reshape(1, -1)

    for i, res in enumerate(results):
        k = res['n_sets']
        history = res['history']

        epochs = [h['epoch'] for h in history]
        task_losses = [h['task_loss'] for h in history]
        ortho_losses = [h['ortho_loss'] for h in history]
        mean_accs = [h['mean_accuracy'] for h in history]

        # Task loss
        ax = axes[i, 0]
        ax.plot(epochs, task_losses, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Task Loss (Cross-Entropy)')
        ax.set_title(f'K={k}: Task Loss')
        ax.grid(True, alpha=0.3)

        # Orthogonality loss
        ax = axes[i, 1]
        ax.plot(epochs, ortho_losses, linewidth=2, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Orthogonality Loss')
        ax.set_title(f'K={k}: Orthogonality Loss')
        ax.grid(True, alpha=0.3)

        # Mean accuracy
        ax = axes[i, 2]
        ax.plot(epochs, mean_accs, linewidth=2, color='orange')
        ax.axhline(1/6, color='red', linestyle='--', alpha=0.5, label='Random')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Accuracy')
        ax.set_title(f'K={k}: Mean Accuracy')
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'convergence_curves.png'}")
    plt.close()


def plot_probe_similarity_matrices(results: List[Dict], output_dir: Path):
    """Plot similarity matrices between probe sets for each K."""
    n_configs = len(results)
    n_cols = min(3, n_configs)
    n_rows = (n_configs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_configs == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, res in enumerate(results):
        k = res['n_sets']
        probe_sets = res['all_probe_sets']  # [K, n_emotions, hidden_dim]

        # Compute similarity matrix between all pairs of sets
        # Average absolute dot product between probe sets
        sim_matrix = np.zeros((k, k))

        for set_i in range(k):
            for set_j in range(k):
                # Cross-emotion dot products
                cross_dots = probe_sets[set_i] @ probe_sets[set_j].T
                sim_matrix[set_i, set_j] = np.abs(cross_dots).mean()

        # Plot
        ax = axes[i]
        im = ax.imshow(sim_matrix, cmap='RdYlGn_r', vmin=0, vmax=1)
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels([f'Set {j+1}' for j in range(k)])
        ax.set_yticklabels([f'Set {i+1}' for i in range(k)])
        ax.set_title(f'K={k}: Inter-Set Similarity\n(Lower = More Orthogonal)', fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, label='Mean |Dot Product|')

        # Annotate cells
        for set_i in range(k):
            for set_j in range(k):
                text = ax.text(
                    set_j, set_i, f'{sim_matrix[set_i, set_j]:.3f}',
                    ha="center", va="center",
                    color="white" if sim_matrix[set_i, set_j] > 0.5 else "black",
                    fontsize=10
                )

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_matrices.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'similarity_matrices.png'}")
    plt.close()


def print_summary_table(results: List[Dict]):
    """Print a summary table of all results."""
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'K':<4} {'Mean Acc':<12} {'Std Acc':<12} {'Ortho Loss':<12} {'Cross-Agree':<12} {'Converged':<10}")
    print("-"*80)

    for res in results:
        k = res['n_sets']
        m = res['final_metrics']
        converged = "✓" if res['converged'] else "✗"

        print(
            f"{k:<4} "
            f"{m['mean_accuracy']:<12.4f} "
            f"{m['std_accuracy']:<12.4f} "
            f"{m['cross_dots_mean']:<12.4f} "
            f"{m['mean_cross_agreement']:<12.4f} "
            f"{converged:<10}"
        )

    print("="*80)

    # Find best K
    best_k = None
    best_acc = 0.0
    for res in results:
        if res['converged'] and res['final_metrics']['mean_accuracy'] > best_acc:
            best_k = res['n_sets']
            best_acc = res['final_metrics']['mean_accuracy']

    if best_k:
        print(f"\n✓ Best K: {best_k} with mean accuracy {best_acc:.4f}")
    else:
        print("\n⚠ No configuration converged successfully")


def analyze_intrinsic_dimensionality(results: List[Dict]):
    """Analyze what K tells us about intrinsic emotion dimensionality."""
    print("\n" + "="*80)
    print("INTRINSIC DIMENSIONALITY ANALYSIS")
    print("="*80)

    # Find maximum K that still converges with good accuracy
    max_converged_k = 0
    acc_threshold = 0.3  # 30% accuracy (well above random 16.7%)

    for res in results:
        k = res['n_sets']
        converged = res['converged']
        acc = res['final_metrics']['mean_accuracy']

        if converged and acc >= acc_threshold:
            max_converged_k = max(max_converged_k, k)

    print(f"\nMaximum converged K: {max_converged_k}")
    print(f"  (with accuracy >= {acc_threshold:.1%})")

    if max_converged_k > 0:
        print(f"\nInterpretation:")
        print(f"  The representation supports at least {max_converged_k} independent")
        print(f"  emotion subspaces, suggesting the intrinsic dimensionality for")
        print(f"  emotion encoding is >= {max_converged_k * 6} (K × n_emotions).")
        print(f"\n  Note: This is a lower bound - the true dimensionality may be higher.")
    else:
        print("\n  Unable to determine intrinsic dimensionality from these results.")

    # Check if accuracy/orthogonality degrades sharply
    print("\n" + "-"*80)
    print("Degradation Analysis:")
    print("-"*80)

    for i in range(1, len(results)):
        k_prev = results[i-1]['n_sets']
        k_curr = results[i]['n_sets']

        acc_prev = results[i-1]['final_metrics']['mean_accuracy']
        acc_curr = results[i]['final_metrics']['mean_accuracy']
        acc_drop = acc_prev - acc_curr

        ortho_prev = results[i-1]['final_metrics']['cross_dots_mean']
        ortho_curr = results[i]['final_metrics']['cross_dots_mean']
        ortho_increase = ortho_curr - ortho_prev

        print(f"  K {k_prev} → {k_curr}:")
        print(f"    Accuracy change: {acc_drop:+.4f} ({acc_prev:.4f} → {acc_curr:.4f})")
        print(f"    Orthogonality change: {ortho_increase:+.4f} ({ortho_prev:.4f} → {ortho_curr:.4f})")

        if acc_drop > 0.05:
            print(f"    ⚠ Significant accuracy drop!")
        if ortho_increase > 0.1:
            print(f"    ⚠ Significant orthogonality degradation!")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multi-orthogonal probe search results"
    )
    parser.add_argument(
        "--search-results",
        type=str,
        required=True,
        help="Path to search results pickle file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/multi_ortho_analysis",
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Load results
    with open(args.search_results, 'rb') as f:
        data = pickle.load(f)

    results = data['all_results']
    layer = data['layer']
    args_dict = data['args']

    print(f"Loaded results for layer {layer}")
    print(f"Tested K values: {[r['n_sets'] for r in results]}")

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_k_vs_metrics(results, output_dir)
    plot_convergence_curves(results, output_dir)
    plot_probe_similarity_matrices(results, output_dir)

    # Print analyses
    print_summary_table(results)
    analyze_intrinsic_dimensionality(results)

    print(f"\n✓ All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
