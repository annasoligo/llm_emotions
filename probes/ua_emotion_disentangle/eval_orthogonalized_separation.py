#!/usr/bin/env python3
"""
Evaluate separation statistics for orthogonalized U probes.

Computes Cohen's d effect sizes to see if U emotions still discriminate
after removing M contamination.
"""

import argparse
import h5py
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import FULL_EMOTIONS


def load_activations_and_metadata(h5_path: Path, position: str):
    """Load activations and metadata for a specific position."""
    print(f"Loading data from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        # Load metadata
        metadata_json = f.attrs['metadata']
        all_metadata = json.loads(metadata_json)

        # Load activations for position
        acts_group = f['activations']
        activations = acts_group[position][:]
        print(f"  Loaded {position}: {activations.shape}")

    # Filter metadata for this position
    metadata = [m for m in all_metadata if m['position'] == position]

    return activations, metadata


def load_probes(probes_path: Path) -> Dict[str, np.ndarray]:
    """Load probes from .npz file."""
    print(f"Loading probes from: {probes_path}")
    data = np.load(probes_path)
    probes = {key: data[key] for key in data.files}
    print(f"  Loaded {len(probes)} probes")
    return probes


def compute_cohens_d(
    activations: np.ndarray,
    metadata: List[Dict],
    probe_direction: np.ndarray,
    entity: str,
    emotion1: str,
    emotion2: str
) -> float:
    """Compute Cohen's d effect size using a specific probe direction."""
    indices1 = [i for i, m in enumerate(metadata) if m[entity] == emotion1]
    indices2 = [i for i, m in enumerate(metadata) if m[entity] == emotion2]

    if len(indices1) == 0 or len(indices2) == 0:
        return 0.0

    acts1 = activations[indices1]
    acts2 = activations[indices2]

    # Normalize probe direction
    probe_norm = probe_direction / (np.linalg.norm(probe_direction) + 1e-8)

    # Project onto probe
    proj1 = acts1 @ probe_norm
    proj2 = acts2 @ probe_norm

    # Cohen's d = (mean1 - mean2) / pooled_std
    mean1, mean2 = proj1.mean(), proj2.mean()
    std1, std2 = proj1.std(), proj2.std()
    pooled_std = np.sqrt((std1**2 + std2**2) / 2)

    cohens_d = (mean1 - mean2) / (pooled_std + 1e-8)
    return abs(cohens_d)


def evaluate_separation(
    activations: np.ndarray,
    metadata: List[Dict],
    probes: Dict[str, np.ndarray],
    entity: str
):
    """Evaluate U emotion separation using orthogonalized probes."""
    print(f"\n{'='*80}")
    print(f"Evaluating {entity} emotion separation")
    print(f"{'='*80}")

    entity_probes = {k: v for k, v in probes.items() if k.startswith(f'{entity}_')}
    print(f"Found {len(entity_probes)} {entity} probes")

    effect_sizes = {}
    for emotion, opposite in FULL_EMOTIONS.items():
        probe_name = f"{entity}_{emotion}"
        if probe_name not in entity_probes:
            continue

        probe = entity_probes[probe_name]

        # Compute Cohen's d
        d = compute_cohens_d(
            activations, metadata, probe,
            entity, emotion, opposite
        )

        pair_name = f"{emotion}_vs_{opposite}"
        effect_sizes[pair_name] = d

    # Statistics
    mean_d = np.mean(list(effect_sizes.values()))
    median_d = np.median(list(effect_sizes.values()))
    min_d = np.min(list(effect_sizes.values()))
    max_d = np.max(list(effect_sizes.values()))

    print(f"\nCohen's d effect sizes:")
    # Sort by value
    sorted_pairs = sorted(effect_sizes.items(), key=lambda x: x[1], reverse=True)
    for pair_name, d in sorted_pairs:
        print(f"  {pair_name:35s}: d = {d:.3f}")

    print(f"\nSummary statistics:")
    print(f"  Mean:   {mean_d:.3f}")
    print(f"  Median: {median_d:.3f}")
    print(f"  Min:    {min_d:.3f}")
    print(f"  Max:    {max_d:.3f}")

    return {
        'effect_sizes': effect_sizes,
        'mean': float(mean_d),
        'median': float(median_d),
        'min': float(min_d),
        'max': float(max_d),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate orthogonalized probe separation")
    parser.add_argument("--activations", type=Path,
                       default=Path("probes/ua_emotion_disentangle/data/activations/full_layer30.h5"))
    parser.add_argument("--probes", type=Path,
                       default=Path("probes/ua_emotion_disentangle/full_analysis/probes_first_asst_token_orthogonal.npz"))
    parser.add_argument("--original-probes", type=Path,
                       default=Path("probes/ua_emotion_disentangle/full_analysis/probes_first_asst_token.npz"))
    parser.add_argument("--position", type=str, default="first_asst_token")
    parser.add_argument("--output", type=Path,
                       default=Path("probes/ua_emotion_disentangle/full_analysis/orthogonal_separation_stats.json"))
    args = parser.parse_args()

    print("="*80)
    print("ORTHOGONALIZED PROBE SEPARATION ANALYSIS")
    print("="*80)

    # Load data
    activations, metadata = load_activations_and_metadata(args.activations, args.position)

    # Load orthogonalized probes
    print("\n" + "-"*80)
    print("ORTHOGONALIZED PROBES")
    print("-"*80)
    ortho_probes = load_probes(args.probes)

    # Evaluate M and U separation
    m_results_ortho = evaluate_separation(activations, metadata, ortho_probes, 'M')
    u_results_ortho = evaluate_separation(activations, metadata, ortho_probes, 'U')

    # Load original probes for comparison
    print("\n" + "-"*80)
    print("ORIGINAL PROBES (for comparison)")
    print("-"*80)
    orig_probes = load_probes(args.original_probes)

    m_results_orig = evaluate_separation(activations, metadata, orig_probes, 'M')
    u_results_orig = evaluate_separation(activations, metadata, orig_probes, 'U')

    # Comparison
    print("\n" + "="*80)
    print("COMPARISON: Original vs Orthogonalized")
    print("="*80)
    print(f"\nM emotions:")
    print(f"  Original:       mean d = {m_results_orig['mean']:.3f}")
    print(f"  Orthogonalized: mean d = {m_results_ortho['mean']:.3f}")
    print(f"  Change: {(m_results_ortho['mean'] - m_results_orig['mean']):.3f} ({((m_results_ortho['mean'] / m_results_orig['mean'] - 1) * 100):.1f}%)")

    print(f"\nU emotions:")
    print(f"  Original:       mean d = {u_results_orig['mean']:.3f}")
    print(f"  Orthogonalized: mean d = {u_results_ortho['mean']:.3f}")
    print(f"  Change: {(u_results_ortho['mean'] - u_results_orig['mean']):.3f} ({((u_results_ortho['mean'] / u_results_orig['mean'] - 1) * 100):.1f}%)")

    # Save results
    results = {
        'orthogonalized': {
            'M': m_results_ortho,
            'U': u_results_ortho,
        },
        'original': {
            'M': m_results_orig,
            'U': u_results_orig,
        },
        'position': args.position,
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to: {args.output}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
