#!/usr/bin/env python3
"""
Interpret PCA components by analyzing emotion loadings and semantic patterns.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from typing import Dict

# Import helpers
import sys
sys.path.insert(0, str(Path(__file__).parent))


def load_probes(probes_path: Path) -> Dict[str, np.ndarray]:
    """Load probes from .npz file."""
    data = np.load(probes_path)
    return {key: data[key] for key in data.files}


def separate_probes(probes: Dict[str, np.ndarray]):
    """Separate M and U probes."""
    m_probes = {k.replace('M_', ''): v for k, v in probes.items() if k.startswith('M_')}
    u_probes = {k.replace('U_', ''): v for k, v in probes.items() if k.startswith('U_')}
    return m_probes, u_probes


def analyze_pc_loadings(probes: Dict[str, np.ndarray], entity: str, n_components=5):
    """Analyze what each PC represents by looking at loadings."""
    names = sorted(probes.keys())
    matrix = np.stack([probes[k] for k in names])

    # Fit PCA
    pca = PCA(n_components=n_components)
    projections = pca.fit_transform(matrix)

    print(f"\n{'='*80}")
    print(f"{entity} PCA COMPONENT INTERPRETATION")
    print(f"{'='*80}")

    # Analyze each component
    for pc_idx in range(n_components):
        print(f"\n{'─'*80}")
        print(f"PC{pc_idx+1} (explains {pca.explained_variance_ratio_[pc_idx]*100:.1f}% variance)")
        print(f"{'─'*80}")

        # Get loadings (correlation with PC)
        pc_loadings = projections[:, pc_idx]

        # Sort by loading
        sorted_indices = np.argsort(np.abs(pc_loadings))[::-1]

        # Show top positive and negative loadings
        print("\nEmotions with highest positive loading:")
        positive_count = 0
        for idx in sorted_indices:
            if pc_loadings[idx] > 0 and positive_count < 5:
                print(f"  {names[idx]:20s}: {pc_loadings[idx]:7.2f}")
                positive_count += 1

        print("\nEmotions with highest negative loading:")
        negative_count = 0
        for idx in sorted_indices[::-1]:
            if pc_loadings[idx] < 0 and negative_count < 5:
                print(f"  {names[idx]:20s}: {pc_loadings[idx]:7.2f}")
                negative_count += 1

        # Try to identify semantic pattern
        print("\nPotential semantic interpretation:")

        # Get top/bottom emotions
        top_emotions = [names[idx] for idx in sorted_indices[:5] if pc_loadings[idx] > 0]
        bottom_emotions = [names[idx] for idx in sorted_indices[::-1][:5] if pc_loadings[idx] < 0]

        print(f"  Positive end: {', '.join(top_emotions)}")
        print(f"  Negative end: {', '.join(bottom_emotions)}")

        # Check for valence pattern (positive vs negative emotions)
        positive_emotions = {'joy', 'trust', 'anticipation', 'surprise', 'love',
                           'optimism', 'serenity', 'acceptance', 'admiration'}
        negative_emotions = {'sadness', 'disgust', 'anger', 'fear', 'remorse',
                           'disapproval', 'grief', 'loathing', 'rage', 'terror'}

        top_positive = sum(1 for e in top_emotions if e in positive_emotions)
        top_negative = sum(1 for e in top_emotions if e in negative_emotions)
        bottom_positive = sum(1 for e in bottom_emotions if e in positive_emotions)
        bottom_negative = sum(1 for e in bottom_emotions if e in negative_emotions)

        if top_positive > top_negative and bottom_negative > bottom_positive:
            print("  → Likely represents VALENCE (positive vs negative)")
        elif top_negative > top_positive and bottom_positive > bottom_negative:
            print("  → Likely represents VALENCE (negative vs positive)")

        # Check for arousal pattern (calm vs excited)
        high_arousal = {'anger', 'rage', 'terror', 'fear', 'ecstasy', 'amazement',
                       'surprise', 'aggressiveness'}
        low_arousal = {'sadness', 'serenity', 'pensiveness', 'boredom', 'acceptance'}

        top_high = sum(1 for e in top_emotions if e in high_arousal)
        top_low = sum(1 for e in top_emotions if e in low_arousal)

        if top_high > top_low * 1.5:
            print("  → Possibly represents high AROUSAL")
        elif top_low > top_high * 1.5:
            print("  → Possibly represents low AROUSAL")

    return pca, projections, names


def plot_component_loadings(pca, names, projections, entity, output_path):
    """Plot loadings for first 4 components."""
    n_components = min(4, pca.n_components_)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for pc_idx in range(n_components):
        ax = axes[pc_idx]

        loadings = projections[:, pc_idx]
        colors = ['red' if l > 0 else 'blue' for l in loadings]

        # Sort by absolute loading
        sorted_indices = np.argsort(np.abs(loadings))
        sorted_names = [names[i] for i in sorted_indices]
        sorted_loadings = [loadings[i] for i in sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]

        # Plot
        y_pos = np.arange(len(sorted_names))
        ax.barh(y_pos, sorted_loadings, color=sorted_colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=7)
        ax.set_xlabel('Loading', fontsize=10)
        ax.set_title(f'{entity} PC{pc_idx+1} Loadings\n({pca.explained_variance_ratio_[pc_idx]*100:.1f}% variance)',
                    fontsize=11, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved component loadings to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Interpret PCA components")
    parser.add_argument("--probes", type=Path,
                       default=Path("probes/ua_emotion_disentangle/full_analysis/probes_first_asst_token_orthogonal.npz"))
    parser.add_argument("--output-dir", type=Path,
                       default=Path("probes/ua_emotion_disentangle/full_analysis"))
    args = parser.parse_args()

    print("="*80)
    print("PCA COMPONENT INTERPRETATION")
    print("="*80)

    # Load probes
    probes = load_probes(args.probes)
    m_probes, u_probes = separate_probes(probes)

    # Analyze M components
    m_pca, m_proj, m_names = analyze_pc_loadings(m_probes, "M (Assistant)", n_components=5)

    # Analyze U components
    u_pca, u_proj, u_names = analyze_pc_loadings(u_probes, "U (User)", n_components=5)

    # Plot loadings
    plot_component_loadings(m_pca, m_names, m_proj, "M",
                           args.output_dir / "m_component_loadings.png")
    plot_component_loadings(u_pca, u_names, u_proj, "U",
                           args.output_dir / "u_component_loadings.png")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
