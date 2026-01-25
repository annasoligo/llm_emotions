#!/usr/bin/env python3
"""
Summarize diverse isolation probe results: same-source and cross-source accuracy.
"""

import json
from pathlib import Path

def load_results():
    """Load all training and cross-source results."""
    results = []

    # Layers and lambdas
    layers = [0, 10, 20, 30, 40, 50, 60]
    lambdas = [0.0, 10.0, 100.0]

    # Load cross-source results
    cross_source_path = Path("outputs/probes/diverse_isolation/cross_source_results.json")
    with open(cross_source_path, 'r') as f:
        cross_source_results = json.load(f)

    # Create lookup for cross-source results
    cross_lookup = {}
    for r in cross_source_results:
        key = (r['layer'], r['lambda'])
        cross_lookup[key] = r

    # Load training results
    for layer in layers:
        for lambda_val in lambdas:
            # User probe results
            user_path = Path(f"outputs/probes/diverse_isolation/user_layer{layer}_lambda{lambda_val}/results.json")
            if user_path.exists():
                with open(user_path, 'r') as f:
                    user_results = json.load(f)
                user_acc = user_results['test_results']['accuracy']
            else:
                user_acc = None

            # Assistant probe results
            asst_path = Path(f"outputs/probes/diverse_isolation/assistant_layer{layer}_lambda{lambda_val}/results.json")
            if asst_path.exists():
                with open(asst_path, 'r') as f:
                    asst_results = json.load(f)
                asst_acc = asst_results['test_results']['accuracy']
            else:
                asst_acc = None

            # Cross-source results
            key = (layer, lambda_val)
            if key in cross_lookup:
                cross = cross_lookup[key]
                user_on_asst = cross['user_on_asst']
                asst_on_user = cross['asst_on_user']
                cos_sim = cross['cos_sim']
            else:
                user_on_asst = None
                asst_on_user = None
                cos_sim = None

            results.append({
                'layer': layer,
                'lambda': lambda_val,
                'user_acc': user_acc,
                'asst_acc': asst_acc,
                'user_on_asst': user_on_asst,
                'asst_on_user': asst_on_user,
                'cos_sim': cos_sim,
            })

    return results

def print_summary(results):
    """Print comprehensive summary table."""
    print("=" * 120)
    print("COMPREHENSIVE DIVERSE ISOLATION RESULTS")
    print("=" * 120)
    print()
    print("Same-Source Accuracy: Training on X, testing on X")
    print("Cross-Source Accuracy: Training on X, testing on Y (should be low for good source specificity)")
    print("Cosine Similarity: Alignment between user and assistant probe weights")
    print()
    print("-" * 120)
    print(f"{'Layer':<6} | {'Lambda':<7} | {'User→User':<10} | {'Asst→Asst':<10} | {'User→Asst':<10} | {'Asst→User':<10} | {'CosSim':<8}")
    print(f"{'':6} | {'':7} | {'(train)':<10} | {'(train)':<10} | {'(cross)':<10} | {'(cross)':<10} | {'':8}")
    print("-" * 120)

    for r in results:
        layer = r['layer']
        lambda_val = r['lambda']
        user_acc = f"{r['user_acc']:.4f}" if r['user_acc'] is not None else "N/A"
        asst_acc = f"{r['asst_acc']:.4f}" if r['asst_acc'] is not None else "N/A"
        user_on_asst = f"{r['user_on_asst']:.4f}" if r['user_on_asst'] is not None else "N/A"
        asst_on_user = f"{r['asst_on_user']:.4f}" if r['asst_on_user'] is not None else "N/A"
        cos_sim = f"{r['cos_sim']:.4f}" if r['cos_sim'] is not None else "N/A"

        print(f"{layer:<6} | {lambda_val:<7.1f} | {user_acc:<10} | {asst_acc:<10} | {user_on_asst:<10} | {asst_on_user:<10} | {cos_sim:<8}")

    print("-" * 120)
    print()

    # Key insights
    print("=" * 120)
    print("KEY INSIGHTS")
    print("=" * 120)
    print()

    # Best same-source accuracy
    best_user = max([r for r in results if r['user_acc'] is not None], key=lambda x: x['user_acc'])
    best_asst = max([r for r in results if r['asst_acc'] is not None], key=lambda x: x['asst_acc'])

    print("BEST SAME-SOURCE ACCURACY (training performance):")
    print(f"  User: Layer {best_user['layer']}, λ={best_user['lambda']:.1f} → {best_user['user_acc']:.2%}")
    print(f"  Asst: Layer {best_asst['layer']}, λ={best_asst['lambda']:.1f} → {best_asst['asst_acc']:.2%}")
    print()

    # Best source specificity (lowest cross-source)
    valid_results = [r for r in results if r['user_on_asst'] is not None and r['asst_on_user'] is not None]
    best_specificity = min(valid_results, key=lambda x: (x['user_on_asst'] + x['asst_on_user']) / 2)

    avg_cross = (best_specificity['user_on_asst'] + best_specificity['asst_on_user']) / 2
    print("BEST SOURCE SPECIFICITY (lowest cross-source accuracy):")
    print(f"  Layer {best_specificity['layer']}, λ={best_specificity['lambda']:.1f}")
    print(f"    User→Asst: {best_specificity['user_on_asst']:.2%}")
    print(f"    Asst→User: {best_specificity['asst_on_user']:.2%}")
    print(f"    Average: {avg_cross:.2%} (random baseline: 16.67%)")
    print()

    # Most orthogonal weights
    most_orthogonal = min(valid_results, key=lambda x: x['cos_sim'])
    print("MOST ORTHOGONAL PROBE WEIGHTS:")
    print(f"  Layer {most_orthogonal['layer']}, λ={most_orthogonal['lambda']:.1f}")
    print(f"    Cosine similarity: {most_orthogonal['cos_sim']:.4f}")
    print(f"    User→User: {most_orthogonal['user_acc']:.2%}")
    print(f"    Asst→Asst: {most_orthogonal['asst_acc']:.2%}")
    print(f"    Cross-source avg: {(most_orthogonal['user_on_asst'] + most_orthogonal['asst_on_user'])/2:.2%}")
    print()

    # Layer analysis
    print("LAYER-BY-LAYER PATTERN:")
    for layer in [0, 10, 20, 30, 40, 50, 60]:
        layer_results = [r for r in valid_results if r['layer'] == layer]
        if layer_results:
            best_lambda = min(layer_results, key=lambda x: (x['user_on_asst'] + x['asst_on_user']) / 2)
            avg_cos = sum(r['cos_sim'] for r in layer_results) / len(layer_results)
            avg_cross = (best_lambda['user_on_asst'] + best_lambda['asst_on_user']) / 2

            print(f"  Layer {layer:2d}: Best λ={best_lambda['lambda']:.0f} → cross={avg_cross:.2%}, "
                  f"user_acc={best_lambda['user_acc']:.2%}, asst_acc={best_lambda['asst_acc']:.2%}, "
                  f"avg_cos_sim={avg_cos:.3f}")

def main():
    results = load_results()
    print_summary(results)

if __name__ == "__main__":
    main()
