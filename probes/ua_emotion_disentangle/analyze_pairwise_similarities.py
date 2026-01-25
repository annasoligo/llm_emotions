#!/usr/bin/env python3
"""
Analyze pairwise cosine similarities between emotion probes.

Computes cosine similarity matrices for M and U emotions separately.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict

# Import config
import sys
sys.path.insert(0, str(Path(__file__).parent))


def load_probes(probes_path: Path) -> Dict[str, np.ndarray]:
    """Load probes from .npz file."""
    data = np.load(probes_path)
    probes = {key: data[key] for key in data.files}
    return probes


def separate_probes(probes: Dict[str, np.ndarray]):
    """Separate M and U probes."""
    m_probes = {k.replace('M_', ''): v for k, v in probes.items() if k.startswith('M_')}
    u_probes = {k.replace('U_', ''): v for k, v in probes.items() if k.startswith('U_')}
    return m_probes, u_probes


def compute_cosine_similarity_matrix(probes: Dict[str, np.ndarray]) -> tuple:
    """Compute pairwise cosine similarity matrix."""
    names = sorted(probes.keys())
    n = len(names)

    sim_matrix = np.zeros((n, n))

    for i, name_i in enumerate(names):
        vec_i = probes[name_i]
        norm_i = np.linalg.norm(vec_i)

        for j, name_j in enumerate(names):
            vec_j = probes[name_j]
            norm_j = np.linalg.norm(vec_j)

            # Cosine similarity
            cos_sim = np.dot(vec_i, vec_j) / (norm_i * norm_j + 1e-8)
            sim_matrix[i, j] = cos_sim

    return sim_matrix, names


def plot_similarity_heatmap(sim_matrix: np.ndarray, names: list, title: str, ax: plt.Axes):
    """Plot cosine similarity heatmap."""
    # Use absolute values for visualization
    abs_sim = np.abs(sim_matrix)

    sns.heatmap(abs_sim, xticklabels=names, yticklabels=names,
                cmap='RdYlBu_r', center=0.5, vmin=0, vmax=1,
                cbar_kws={'label': '|Cosine Similarity|'},
                ax=ax, square=True)

    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=6)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=6)


def analyze_similarity_statistics(sim_matrix: np.ndarray, names: list, entity: str):
    """Compute and print statistics about similarity matrix."""
    n = len(names)

    # Get upper triangle (excluding diagonal)
    upper_tri_indices = np.triu_indices(n, k=1)
    upper_tri_sims = sim_matrix[upper_tri_indices]
    abs_sims = np.abs(upper_tri_sims)

    print(f"\n{entity} Emotion Similarity Statistics:")
    print(f"  Mean |cosine sim|: {abs_sims.mean():.4f}")
    print(f"  Median |cosine sim|: {np.median(abs_sims):.4f}")
    print(f"  Min |cosine sim|: {abs_sims.min():.4f}")
    print(f"  Max |cosine sim|: {abs_sims.max():.4f}")
    print(f"  Std |cosine sim|: {abs_sims.std():.4f}")

    # Find most similar pairs (excluding diagonal and opposites)
    print(f"\n  Top 10 most similar {entity} emotion pairs:")
    sorted_indices = np.argsort(abs_sims)[::-1]

    count = 0
    for idx in sorted_indices:
        i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
        name_i, name_j = names[i], names[j]

        # Skip if they're opposites (will have sim ≈ -1)
        if abs(sim_matrix[i, j] + 1.0) < 0.1:  # close to -1
            continue

        print(f"    {name_i:20s} <-> {name_j:20s}: {sim_matrix[i, j]:6.3f} (|{abs_sims[idx]:.3f}|)")
        count += 1
        if count >= 10:
            break

    # Find most dissimilar pairs (excluding opposites)
    print(f"\n  Top 10 most dissimilar {entity} emotion pairs (excluding opposites):")
    count = 0
    for idx in sorted_indices[::-1]:
        i, j = upper_tri_indices[0][idx], upper_tri_indices[1][idx]
        name_i, name_j = names[i], names[j]

        # Skip if they're opposites
        if abs(sim_matrix[i, j] + 1.0) < 0.1:
            continue

        print(f"    {name_i:20s} <-> {name_j:20s}: {sim_matrix[i, j]:6.3f} (|{abs_sims[idx]:.3f}|)")
        count += 1
        if count >= 10:
            break

    return abs_sims


def main():
    parser = argparse.ArgumentParser(description="Analyze pairwise cosine similarities")
    parser.add_argument("--probes", type=Path,
                       default=Path("probes/ua_emotion_disentangle/full_analysis/probes_first_asst_token_orthogonal.npz"))
    parser.add_argument("--output", type=Path,
                       default=Path("probes/ua_emotion_disentangle/full_analysis/pairwise_similarities.png"))
    args = parser.parse_args()

    print("="*80)
    print("PAIRWISE EMOTION PROBE SIMILARITY ANALYSIS")
    print("="*80)

    # Load probes
    probes = load_probes(args.probes)
    m_probes, u_probes = separate_probes(probes)

    print(f"\nLoaded {len(m_probes)} M probes and {len(u_probes)} U probes")

    # Compute similarity matrices
    print("\nComputing cosine similarity matrices...")
    m_sim, m_names = compute_cosine_similarity_matrix(m_probes)
    u_sim, u_names = compute_cosine_similarity_matrix(u_probes)

    # Statistics
    print("\n" + "="*80)
    m_abs_sims = analyze_similarity_statistics(m_sim, m_names, "M (Assistant)")
    u_abs_sims = analyze_similarity_statistics(u_sim, u_names, "U (User)")
    print("="*80)

    # Compare distributions
    print(f"\nComparison:")
    print(f"  M mean |cos sim|: {m_abs_sims.mean():.4f}")
    print(f"  U mean |cos sim|: {u_abs_sims.mean():.4f}")
    print(f"  Difference: {abs(m_abs_sims.mean() - u_abs_sims.mean()):.4f}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    plot_similarity_heatmap(m_sim, m_names,
                           "M (Assistant) Emotion Cosine Similarities",
                           axes[0])
    plot_similarity_heatmap(u_sim, u_names,
                           "U (User) Emotion Cosine Similarities",
                           axes[1])

    plt.tight_layout()

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to: {args.output}")

    # Distribution plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.hist(m_abs_sims, bins=50, alpha=0.6, label=f'M (Assistant) mean={m_abs_sims.mean():.3f}',
            color='blue', edgecolor='black')
    ax.hist(u_abs_sims, bins=50, alpha=0.6, label=f'U (User) mean={u_abs_sims.mean():.3f}',
            color='red', edgecolor='black')

    ax.set_xlabel('|Cosine Similarity|', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Pairwise Cosine Similarities', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    dist_output = args.output.parent / "similarity_distributions.png"
    plt.savefig(dist_output, dpi=300, bbox_inches='tight')
    print(f"✓ Saved distribution plot to: {dist_output}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
