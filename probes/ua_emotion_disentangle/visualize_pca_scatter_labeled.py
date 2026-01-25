#!/usr/bin/env python3
"""
Create scatter plots of PCA projections with all emotions labeled.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from typing import Dict
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


def plot_pca_scatter(probes: Dict[str, np.ndarray], entity: str, ax: plt.Axes, pc_x: int = 0, pc_y: int = 1, pca_obj=None):
    """Plot PCA scatter with all points labeled and uniquely colored."""
    names = sorted(probes.keys())
    matrix = np.stack([probes[k] for k in names])

    # Fit PCA if not provided
    if pca_obj is None:
        pca = PCA(n_components=4)
        projections = pca.fit_transform(matrix)
    else:
        pca = pca_obj
        projections = pca.transform(matrix)

    # Assign unique color to each emotion using a colormap
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20', len(names))
    colors = [cmap(i) for i in range(len(names))]

    # Plot points
    ax.scatter(projections[:, pc_x], projections[:, pc_y], c=colors, alpha=0.7, s=120, edgecolors='black', linewidths=0.5)

    # Label all points
    for i, name in enumerate(names):
        ax.annotate(name, (projections[i, pc_x], projections[i, pc_y]),
                   fontsize=7, ha='center', va='bottom',
                   xytext=(0, 3), textcoords='offset points')

    ax.set_xlabel(f'PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}% var)', fontsize=11)
    ax.set_ylabel(f'PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}% var)', fontsize=11)
    ax.set_title(f'{entity} Emotions in PC{pc_x+1}-PC{pc_y+1} Space', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

    return pca


def main():
    parser = argparse.ArgumentParser(description="PCA scatter plots with labels")
    parser.add_argument("--probes", type=Path,
                       default=Path("probes/ua_emotion_disentangle/full_analysis/probes_first_asst_token_orthogonal.npz"))
    parser.add_argument("--output", type=Path,
                       default=Path("probes/ua_emotion_disentangle/full_analysis/pca_scatter_labeled.png"))
    args = parser.parse_args()

    print("="*80)
    print("PCA SCATTER PLOTS WITH LABELS")
    print("="*80)

    # Load probes
    probes = load_probes(args.probes)
    m_probes, u_probes = separate_probes(probes)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))

    # Fit PCA once for each entity
    m_names = sorted(m_probes.keys())
    m_matrix = np.stack([m_probes[k] for k in m_names])
    m_pca = PCA(n_components=4)
    m_pca.fit(m_matrix)

    u_names = sorted(u_probes.keys())
    u_matrix = np.stack([u_probes[k] for k in u_names])
    u_pca = PCA(n_components=4)
    u_pca.fit(u_matrix)

    # Plot M PC1 vs PC2 (top left)
    plot_pca_scatter(m_probes, "M (Assistant)", axes[0, 0], pc_x=0, pc_y=1, pca_obj=m_pca)

    # Plot U PC1 vs PC2 (top right)
    plot_pca_scatter(u_probes, "U (User)", axes[0, 1], pc_x=0, pc_y=1, pca_obj=u_pca)

    # Plot M PC3 vs PC4 (bottom left)
    plot_pca_scatter(m_probes, "M (Assistant)", axes[1, 0], pc_x=2, pc_y=3, pca_obj=m_pca)

    # Plot U PC3 vs PC4 (bottom right)
    plot_pca_scatter(u_probes, "U (User)", axes[1, 1], pc_x=2, pc_y=3, pca_obj=u_pca)

    plt.tight_layout()

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved to: {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
