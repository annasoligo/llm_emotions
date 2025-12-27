#!/usr/bin/env python3
"""
Generate all visualizations for emotion probe analysis.

This script creates:
1. Dimensionality sweep plot (test accuracy vs # PCs)
2. Weight distribution analysis (for 50 PC probes)
3. Summary tables and statistics

Usage:
    python probes/scripts/generate_all_visualizations.py
"""

import pickle
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Believe-it-or-not color palette
COLORS = {
    "coral": "#D4876A",
    "sky_blue": "#7BA7D7",
    "olive": "#7D9B7D",
    "dusty_rose": "#C17B8D",
    "sage": "#B8CCC8",
    "lavender": "#a59dc9",
}

# Layer colors
LAYER_COLORS = {
    10: COLORS["coral"],
    20: COLORS["sky_blue"],
    30: COLORS["olive"],
    40: COLORS["dusty_rose"],
    50: COLORS["sage"],
}


def plot_dimensionality_sweep():
    """Plot test accuracy vs number of PCs."""
    print("\n" + "="*80)
    print("GENERATING DIMENSIONALITY SWEEP PLOT")
    print("="*80)

    configs = {
        3: 'results/emotion_probes_top3',
        5: 'results/emotion_probes_top5',
        10: 'results/emotion_probes_top10',
        20: 'results/emotion_probes_top20',
        50: 'results/emotion_probes_high_alpha_cpca',
    }

    layers = [10, 20, 30, 40, 50]
    data = {layer: {'n_pcs': [], 'test_acc': [], 'train_acc': []} for layer in layers}

    # Load results
    for n_pcs, results_dir in configs.items():
        results_dir = Path(results_dir)
        if not results_dir.exists():
            continue

        for layer in layers:
            pattern = f"probe_layer{layer}_all_cpca*.pkl"
            files = list(results_dir.glob(pattern))

            if files:
                with open(files[0], 'rb') as f:
                    results = pickle.load(f)
                data[layer]['n_pcs'].append(n_pcs)
                data[layer]['test_acc'].append(results['test_accuracy'])
                data[layer]['train_acc'].append(results['train_accuracy'])

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Test Accuracy
    for layer in layers:
        if data[layer]['n_pcs']:
            ax1.plot(data[layer]['n_pcs'], data[layer]['test_acc'],
                    marker='o', markersize=8, linewidth=2.5,
                    label=f'Layer {layer}', color=LAYER_COLORS[layer])

    ax1.set_xlabel('Number of Principal Components', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
    ax1.set_title('Emotion Probe Test Accuracy vs Dimensionality', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.set_xscale('log')
    ax1.set_xticks([3, 5, 10, 20, 50])
    ax1.set_xticklabels(['3', '5', '10', '20', '50'])
    ax1.set_ylim(0.83, 1.005)

    # Add optimal annotation
    ax1.axvline(x=20, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    ax1.text(20, 0.84, 'Optimal: 20 PCs', ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Train-Test Gap
    for layer in layers:
        if data[layer]['n_pcs'] and data[layer]['train_acc']:
            gaps = [train - test for train, test in zip(data[layer]['train_acc'], data[layer]['test_acc'])]
            ax2.plot(data[layer]['n_pcs'], gaps,
                    marker='o', markersize=8, linewidth=2.5,
                    label=f'Layer {layer}', color=LAYER_COLORS[layer])

    ax2.set_xlabel('Number of Principal Components', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Train-Test Gap (overfitting)', fontsize=13, fontweight='bold')
    ax2.set_title('Overfitting vs Dimensionality', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.set_xscale('log')
    ax2.set_xticks([3, 5, 10, 20, 50])
    ax2.set_xticklabels(['3', '5', '10', '20', '50'])
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

    plt.tight_layout()
    output_path = Path('results/dimensionality_sweep_complete.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: Average Test Accuracy by # PCs")
    print("="*80)
    for n_pcs in [3, 5, 10, 20, 50]:
        accs = []
        for layer in layers:
            if n_pcs in data[layer]['n_pcs']:
                idx = data[layer]['n_pcs'].index(n_pcs)
                accs.append(data[layer]['test_acc'][idx])
        if accs:
            print(f"{n_pcs:2d} PCs: {np.mean(accs):.4f} ± {np.std(accs):.4f}  "
                  f"(min={min(accs):.4f}, max={max(accs):.4f})")


def analyze_weight_distribution():
    """Analyze how probe weights are distributed across PCs."""
    print("\n" + "="*80)
    print("ANALYZING WEIGHT DISTRIBUTION (50 PC PROBES)")
    print("="*80)

    results_dir = Path("results/emotion_probes_high_alpha_cpca")
    layers = [5, 10, 15, 20, 25, 30, 35, 40, 45]

    if not results_dir.exists():
        print(f"⚠ Directory not found: {results_dir}")
        return

    all_layer_data = {}

    for layer in layers:
        probe_path = results_dir / f"probe_layer{layer}_all_cpca.pkl"
        if not probe_path.exists():
            continue

        with open(probe_path, 'rb') as f:
            results = pickle.load(f)

        weights = results['model'].weight.detach().cpu().numpy()
        pc_importance = np.sqrt(np.sum(weights**2, axis=0))
        total_weight = np.sum(pc_importance)
        pc_fraction = pc_importance / total_weight
        cumulative_weight = np.cumsum(pc_fraction)

        n_for_50 = np.argmax(cumulative_weight >= 0.5) + 1
        n_for_80 = np.argmax(cumulative_weight >= 0.8) + 1

        all_layer_data[layer] = {
            'pc_fraction': pc_fraction,
            'cumulative': cumulative_weight,
            'n_for_50': n_for_50,
            'n_for_80': n_for_80,
        }

        print(f"Layer {layer}: {n_for_50} PCs for 50% weight, {n_for_80} PCs for 80% weight")

    if not all_layer_data:
        print("⚠ No results found")
        return

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cumulative weight
    ax = axes[0]
    for layer in sorted(all_layer_data.keys()):
        cumulative = all_layer_data[layer]['cumulative']
        ax.plot(range(1, len(cumulative)+1), cumulative,
                marker='o', markersize=3, label=f'Layer {layer}', alpha=0.7)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Principal Components', fontsize=12)
    ax.set_ylabel('Cumulative Weight Fraction', fontsize=12)
    ax.set_title('Cumulative Probe Weight by PC', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.05)

    # Plot 2: Average PC weight
    ax = axes[1]
    avg_fraction = np.mean([data['pc_fraction'] for data in all_layer_data.values()], axis=0)
    x = np.arange(20)
    ax.bar(x, avg_fraction[:20], color=COLORS['coral'], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Principal Component Index', fontsize=12)
    ax.set_ylabel('Average Weight Fraction', fontsize=12)
    ax.set_title('Average Weight per PC (First 20 PCs)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in x])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = results_dir / 'cpca_probe_weight_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("="*80)
    print("GENERATING ALL VISUALIZATIONS")
    print("="*80)

    try:
        plot_dimensionality_sweep()
        analyze_weight_distribution()

        print("\n" + "="*80)
        print("✓ ALL VISUALIZATIONS COMPLETE!")
        print("="*80)
        print("\nGenerated files:")
        print("  - results/dimensionality_sweep_complete.png")
        print("  - results/emotion_probes_high_alpha_cpca/cpca_probe_weight_distribution.png")
        print("\nSee probes/DIMENSIONALITY_REDUCTION_RESULTS.md for full analysis.")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
