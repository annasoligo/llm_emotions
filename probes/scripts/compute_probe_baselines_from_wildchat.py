#!/usr/bin/env python3
"""
Compute probe baseline statistics from saved WildChat activations.

This script loads pre-extracted WildChat activations and applies orthogonal probes
to compute baseline emotion score distributions.

Usage:
    python compute_probe_baselines_from_wildchat.py --layer 31 --ortho-weight 1000.0
"""

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np


EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']


def load_probe(probe_path: Path):
    """Load orthogonal probe from file.

    Returns:
        Tuple of (user_probes, asst_probes) as numpy arrays [n_emotions, hidden_dim]
    """
    print(f"Loading probe from {probe_path.name}")
    with open(probe_path, 'rb') as f:
        probe_data = pickle.load(f)

    user_probes = probe_data['final_user_probes']  # [n_emotions, hidden_dim]
    asst_probes = probe_data['final_asst_probes']  # [n_emotions, hidden_dim]

    print(f"  Layer: {probe_data['layer']}")
    print(f"  Ortho weight: {probe_data['ortho_weight']}")
    print(f"  User probes: {user_probes.shape}")
    print(f"  Assistant probes: {asst_probes.shape}")

    return user_probes, asst_probes, probe_data


def compute_probe_scores(activations: np.ndarray, probes: np.ndarray) -> np.ndarray:
    """Apply probes to activations.

    Args:
        activations: [n_samples, hidden_dim]
        probes: [n_emotions, hidden_dim]

    Returns:
        scores: [n_samples, n_emotions]
    """
    return activations @ probes.T


def compute_baseline_statistics(
    activations_path: Path,
    user_probes: np.ndarray,
    asst_probes: np.ndarray,
    output_path: Path
):
    """Compute baseline statistics for each aggregation type.

    Saves JSON with structure:
    {
        'model_name': ...,
        'layer': ...,
        'num_samples': ...,
        'aggregations': {
            'all_tokens': {
                'user': {emotion: {'mean': ..., 'std': ...}},
                'assistant': {emotion: {'mean': ..., 'std': ...}}
            },
            ...
        }
    }
    """
    print(f"\nLoading activations from {activations_path}")

    with h5py.File(activations_path, 'r') as f:
        # Load metadata
        results = {
            'model_name': f.attrs['model_name'],
            'layer': int(f.attrs['layer']),
            'num_samples': int(f.attrs['num_samples']),
            'source': f.attrs['source'],
            'aggregations': {}
        }

        # Process each aggregation type
        for agg_type in f.keys():
            print(f"\n  Processing '{agg_type}'...")
            acts = f[agg_type][:]  # [n_samples, hidden_dim]

            # Apply probes
            user_scores = compute_probe_scores(acts, user_probes)  # [n_samples, n_emotions]
            asst_scores = compute_probe_scores(acts, asst_probes)  # [n_samples, n_emotions]

            # Compute statistics for each emotion
            user_stats = {}
            asst_stats = {}

            for i, emotion in enumerate(EMOTIONS):
                user_stats[emotion] = {
                    'mean': float(user_scores[:, i].mean()),
                    'std': float(user_scores[:, i].std()),
                    'min': float(user_scores[:, i].min()),
                    'max': float(user_scores[:, i].max()),
                    'median': float(np.median(user_scores[:, i]))
                }
                asst_stats[emotion] = {
                    'mean': float(asst_scores[:, i].mean()),
                    'std': float(asst_scores[:, i].std()),
                    'min': float(asst_scores[:, i].min()),
                    'max': float(asst_scores[:, i].max()),
                    'median': float(np.median(asst_scores[:, i]))
                }

            results['aggregations'][agg_type] = {
                'user': user_stats,
                'assistant': asst_stats
            }

            # Print summary
            print(f"    User probe means:")
            for emotion in EMOTIONS:
                mean = user_stats[emotion]['mean']
                std = user_stats[emotion]['std']
                print(f"      {emotion:12s}: {mean:+7.3f} ± {std:.3f}")

            print(f"    Assistant probe means:")
            for emotion in EMOTIONS:
                mean = asst_stats[emotion]['mean']
                std = asst_stats[emotion]['std']
                print(f"      {emotion:12s}: {mean:+7.3f} ± {std:.3f}")

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Baseline statistics saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute probe baselines from WildChat activations")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it",
                       help="Model name (must match saved activations)")
    parser.add_argument("--layer", type=int, required=True,
                       help="Layer number")
    parser.add_argument("--ortho-weight", type=float, required=True,
                       help="Orthogonality weight used during probe training")
    parser.add_argument("--representation", type=str, default="raw",
                       choices=["raw", "global_cpca", "regional_cpca"],
                       help="Representation type (must match probe)")
    parser.add_argument("--n-components", type=int,
                       help="Number of cPCA components (if applicable)")
    parser.add_argument("--activations-dir", type=str,
                       default="/workspace-vast/annas/git/research-tools/data/baselines/wildchat",
                       help="Directory containing saved activations")
    parser.add_argument("--probes-dir", type=str,
                       default="/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/orthogonal",
                       help="Directory containing trained probes")
    parser.add_argument("--output-dir", type=str,
                       default="/workspace-vast/annas/git/research-tools/data/baselines/wildchat_probe_stats",
                       help="Output directory for probe baseline statistics")

    args = parser.parse_args()

    # Build paths
    model_safe_name = args.model.replace("/", "_").replace("-", "_")

    # Activations path
    activations_path = Path(args.activations_dir) / model_safe_name / f"layer{args.layer}_activations.h5"
    if not activations_path.exists():
        raise FileNotFoundError(
            f"Activations not found: {activations_path}\n"
            f"Run compute_wildchat_baseline_activations.py first:\n"
            f"  python compute_wildchat_baseline_activations.py --layer {args.layer}"
        )

    # Probe path
    if args.representation == "raw":
        probe_filename = f"probe_layer{args.layer}_raw_ortho{args.ortho_weight}.pkl"
    elif args.representation == "global_cpca":
        probe_filename = f"probe_layer{args.layer}_global_nc{args.n_components}_ortho{args.ortho_weight}.pkl"
    else:  # regional_cpca
        probe_filename = f"probe_layer{args.layer}_regional_nc{args.n_components}_ortho{args.ortho_weight}.pkl"

    probe_path = Path(args.probes_dir) / f"ortho_{args.ortho_weight}" / probe_filename

    if not probe_path.exists():
        raise FileNotFoundError(f"Probe not found: {probe_path}")

    # Load probe
    user_probes, asst_probes, probe_data = load_probe(probe_path)

    # Output path
    output_filename = f"layer{args.layer}_{args.representation}_ortho{args.ortho_weight}_baseline_stats.json"
    output_path = Path(args.output_dir) / model_safe_name / output_filename

    # Compute baseline statistics
    compute_baseline_statistics(
        activations_path=activations_path,
        user_probes=user_probes,
        asst_probes=asst_probes,
        output_path=output_path
    )

    print("\n" + "="*80)
    print("PROBE BASELINE STATISTICS COMPLETE")
    print("="*80)
    print(f"Statistics saved to: {output_path}")
    print(f"\nTo use these baselines in your analysis:")
    print(f"  import json")
    print(f"  with open('{output_path}') as f:")
    print(f"      baseline = json.load(f)")
    print(f"  # Access: baseline['aggregations']['all_tokens']['user']['happiness']['mean']")


if __name__ == "__main__":
    main()
