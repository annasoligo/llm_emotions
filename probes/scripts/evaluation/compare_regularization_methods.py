#!/usr/bin/env python3
"""Compare regularization methods and dimensionality reduction results."""

import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Believe-it-or-not color palette
COLORS = {
    "coral": "#D4876A",
    "sky_blue": "#7BA7D7",
    "olive": "#7D9B7D",
    "dusty_rose": "#C17B8D",
    "sage": "#B8CCC8",
    "lavender": "#a59dc9",
}

def load_probe_results(results_dir: Path, pattern: str, layers: list):
    """Load results for a specific configuration."""
    results = {}
    for layer in layers:
        probe_file = list(results_dir.glob(f"probe_layer{layer}_{pattern}"))
        if probe_file:
            with open(probe_file[0], 'rb') as f:
                results[layer] = pickle.load(f)
    return results


def analyze_weight_sparsity(results: dict):
    """Analyze how sparse the probe weights are."""
    sparsity_data = {}

    for layer, res in results.items():
        weights = res['model'].weight.detach().cpu().numpy()  # [n_classes, n_features]

        # Compute L1 norm of each feature (sum across classes)
        feature_importance = np.abs(weights).sum(axis=0)

        # Normalize
        feature_importance = feature_importance / feature_importance.sum()

        # Sort descending
        sorted_importance = np.sort(feature_importance)[::-1]

        # Compute cumulative sum
        cumulative = np.cumsum(sorted_importance)

        # Find how many features for 50%, 80%, 90%, 95% of weight
        n_for_50 = np.argmax(cumulative >= 0.5) + 1
        n_for_80 = np.argmax(cumulative >= 0.8) + 1
        n_for_90 = np.argmax(cumulative >= 0.9) + 1
        n_for_95 = np.argmax(cumulative >= 0.95) + 1

        # Gini coefficient (measure of inequality/sparsity)
        # 0 = perfectly uniform, 1 = perfectly sparse
        n = len(sorted_importance)
        gini = (2 * np.sum((np.arange(n) + 1) * sorted_importance)) / (n * np.sum(sorted_importance)) - (n + 1) / n

        sparsity_data[layer] = {
            'n_features': len(feature_importance),
            'n_for_50': n_for_50,
            'n_for_80': n_for_80,
            'n_for_90': n_for_90,
            'n_for_95': n_for_95,
            'gini': gini,
            'feature_importance': feature_importance,
            'cumulative': cumulative,
        }

    return sparsity_data


def plot_comparison(all_results: dict, output_path: Path):
    """Create comparison plots."""

    configs = list(all_results.keys())
    layers = sorted(list(all_results[configs[0]]['accuracy'].keys()))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Test Accuracy Comparison
    ax = axes[0, 0]
    x = np.arange(len(layers))
    width = 0.2

    for i, config in enumerate(configs):
        test_accs = [all_results[config]['accuracy'][layer]['test'] for layer in layers]
        ax.bar(x + i*width, test_accs, width, label=config, alpha=0.8)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title('Test Accuracy by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(configs)-1) / 2)
    ax.set_xticklabels(layers)
    ax.set_ylim(0.98, 1.005)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Sparsity (Gini coefficient)
    ax = axes[0, 1]
    for i, config in enumerate(configs):
        if 'sparsity' not in all_results[config]:
            continue
        ginis = [all_results[config]['sparsity'][layer]['gini'] for layer in layers]
        ax.plot(layers, ginis, marker='o', label=config, linewidth=2, markersize=6)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Gini Coefficient (Sparsity)', fontsize=12)
    ax.set_title('Weight Sparsity by Configuration', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Features needed for 80% weight
    ax = axes[1, 0]
    x = np.arange(len(layers))
    width = 0.2

    for i, config in enumerate(configs):
        if 'sparsity' not in all_results[config]:
            continue
        n_for_80 = [all_results[config]['sparsity'][layer]['n_for_80'] for layer in layers]
        ax.bar(x + i*width, n_for_80, width, label=config, alpha=0.8)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('# Features for 80% Weight', fontsize=12)
    ax.set_title('Feature Concentration', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(configs)-1) / 2)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Cumulative weight curves (layer 30)
    ax = axes[1, 1]
    layer = 30

    for config in configs:
        if 'sparsity' not in all_results[config]:
            continue
        if layer not in all_results[config]['sparsity']:
            continue
        cumulative = all_results[config]['sparsity'][layer]['cumulative']
        n_features = all_results[config]['sparsity'][layer]['n_features']
        ax.plot(range(1, len(cumulative)+1), cumulative, label=config, linewidth=2)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.8, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Feature Index (sorted by importance)', fontsize=12)
    ax.set_ylabel('Cumulative Weight Fraction', fontsize=12)
    ax.set_title(f'Cumulative Weight Distribution (Layer {layer})', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {output_path}")
    plt.close()


def main():
    layers = [20, 30, 40]

    # Define configurations
    configs = {
        'Baseline (50 PCs, L2)': {
            'dir': Path('results/emotion_probes_high_alpha_cpca'),
            'pattern': 'all_cpca.pkl',
        },
        'L1 (50 PCs)': {
            'dir': Path('results/emotion_probes_l1'),
            'pattern': 'all_cpca_l1.pkl',
        },
        'L2 (20 PCs)': {
            'dir': Path('results/emotion_probes_top20'),
            'pattern': 'all_cpca_top20.pkl',
        },
        'L1 (20 PCs)': {
            'dir': Path('results/emotion_probes_top20_l1'),
            'pattern': 'all_cpca_top20_l1.pkl',
        },
    }

    all_results = {}

    print("=" * 80)
    print("LOADING RESULTS")
    print("=" * 80)

    for config_name, config_info in configs.items():
        if not config_info['dir'].exists():
            print(f"⚠ Skipping {config_name}: directory not found")
            continue

        results = load_probe_results(config_info['dir'], config_info['pattern'], layers)

        if not results:
            print(f"⚠ No results found for {config_name}")
            continue

        print(f"\n{config_name}:")
        print(f"  Found {len(results)}/{len(layers)} layers")

        # Extract accuracy
        accuracy = {}
        for layer, res in results.items():
            accuracy[layer] = {
                'train': res['train_accuracy'],
                'test': res['test_accuracy'],
            }
            print(f"    Layer {layer}: Train={res['train_accuracy']:.4f}, Test={res['test_accuracy']:.4f}")

        # Analyze sparsity
        sparsity = analyze_weight_sparsity(results)

        all_results[config_name] = {
            'accuracy': accuracy,
            'sparsity': sparsity,
            'results': results,
        }

    # Print sparsity comparison
    print("\n" + "=" * 80)
    print("SPARSITY COMPARISON")
    print("=" * 80)

    for config_name in all_results.keys():
        print(f"\n{config_name}:")
        print(f"  {'Layer':<8} {'Gini':<8} {'50%':<6} {'80%':<6} {'90%':<6} {'95%':<6}")
        print(f"  {'-'*40}")

        for layer in layers:
            if layer not in all_results[config_name]['sparsity']:
                continue
            s = all_results[config_name]['sparsity'][layer]
            print(f"  {layer:<8} {s['gini']:.3f}    {s['n_for_50']:<6} {s['n_for_80']:<6} {s['n_for_90']:<6} {s['n_for_95']:<6}")

    # Create plots
    if len(all_results) > 0:
        output_path = Path('results/regularization_comparison.png')
        plot_comparison(all_results, output_path)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Average metrics across layers
    for config_name in all_results.keys():
        test_accs = [all_results[config_name]['accuracy'][layer]['test']
                     for layer in layers if layer in all_results[config_name]['accuracy']]
        ginis = [all_results[config_name]['sparsity'][layer]['gini']
                 for layer in layers if layer in all_results[config_name]['sparsity']]

        if test_accs:
            print(f"\n{config_name}:")
            print(f"  Avg test accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
            if ginis:
                print(f"  Avg Gini (sparsity): {np.mean(ginis):.3f} ± {np.std(ginis):.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
