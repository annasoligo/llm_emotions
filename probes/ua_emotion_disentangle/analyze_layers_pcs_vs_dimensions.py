#!/usr/bin/env python3
"""
Analyze how PCs and psychological dimensions evolve across layers.

Uses PRE-COMPUTED ORTHOGONALIZED PROBES from alternating orthogonalization.

For each layer:
1. Load orthogonalized M and U probes
2. Compute PC1-4 for M and U separately
3. Compute psychological dimension directions (V, A, D, AA)
4. Plot PC similarities to dimensions across layers
5. Plot dimension similarities to each other across layers
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple
import json
from tqdm import tqdm

# Import emotion dimensions
import sys
sys.path.insert(0, str(Path(__file__).parent))
from compare_pcs_to_dimensions import EMOTION_DIMENSIONS, DIMENSION_NAMES, cosine_similarity


def load_orthogonal_probes(layer: int, probes_dir: Path) -> Dict[str, np.ndarray]:
    """Load orthogonalized probes for one layer."""
    probe_file = probes_dir / f"layer_{layer}_orthogonal.npz"
    data = np.load(probe_file)
    return {key: data[key] for key in data.files}


def separate_by_entity(probes: Dict[str, np.ndarray]) -> Tuple[Dict, Dict]:
    """Separate M and U probes."""
    m_probes = {}
    u_probes = {}

    for key, vec in probes.items():
        if key.startswith('M_'):
            emotion = key.replace('M_', '')
            m_probes[emotion] = vec
        elif key.startswith('U_'):
            emotion = key.replace('U_', '')
            u_probes[emotion] = vec

    return m_probes, u_probes


def compute_pcs(emotion_vectors: Dict[str, np.ndarray], n_components: int = 4) -> np.ndarray:
    """
    Compute PCA on emotion vectors.

    Returns: (n_components, hidden_dim) array of PC directions
    """
    names = sorted(emotion_vectors.keys())
    matrix = np.stack([emotion_vectors[name] for name in names])

    pca = PCA(n_components=n_components)
    pca.fit(matrix)

    return pca.components_  # Shape: (n_components, hidden_dim)


def compute_dimension_direction(emotion_vectors: Dict[str, np.ndarray], dimension: str) -> np.ndarray:
    """
    Compute direction for a psychological dimension as: mean(high) - mean(low).
    """
    high_vecs = []
    low_vecs = []

    for emotion, vec in emotion_vectors.items():
        if emotion not in EMOTION_DIMENSIONS:
            continue

        value = EMOTION_DIMENSIONS[emotion].get(dimension)
        if value == "H":
            high_vecs.append(vec)
        elif value == "L":
            low_vecs.append(vec)

    if not high_vecs or not low_vecs:
        return None

    high_mean = np.mean(high_vecs, axis=0)
    low_mean = np.mean(low_vecs, axis=0)

    direction = high_mean - low_mean
    # Normalize
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    return direction


def analyze_layer(layer: int, probes_dir: Path) -> Dict:
    """
    Analyze one layer: compute PCs, dimensions, and their similarities.
    Uses PRE-COMPUTED orthogonalized probes.
    """
    # Load orthogonalized probes
    probes = load_orthogonal_probes(layer, probes_dir)
    m_vectors, u_vectors = separate_by_entity(probes)

    results = {'layer': layer}

    # Compute PCs for M and U
    m_pcs = compute_pcs(m_vectors, n_components=4)
    u_pcs = compute_pcs(u_vectors, n_components=4)

    results['M_pcs'] = m_pcs
    results['U_pcs'] = u_pcs

    # Compute dimension directions for M and U
    m_dims = {}
    u_dims = {}

    for dim_code in ['V', 'A', 'D', 'AA']:
        m_dir = compute_dimension_direction(m_vectors, dim_code)
        u_dir = compute_dimension_direction(u_vectors, dim_code)

        if m_dir is not None:
            m_dims[dim_code] = m_dir
        if u_dir is not None:
            u_dims[dim_code] = u_dir

    results['M_dims'] = m_dims
    results['U_dims'] = u_dims

    # Compute PC-to-dimension similarities
    m_pc_dim_sims = {f"PC{i+1}": {} for i in range(4)}
    u_pc_dim_sims = {f"PC{i+1}": {} for i in range(4)}

    for i, pc in enumerate(m_pcs):
        for dim_code, dim_dir in m_dims.items():
            sim = cosine_similarity(pc, dim_dir)
            m_pc_dim_sims[f"PC{i+1}"][dim_code] = float(sim)

    for i, pc in enumerate(u_pcs):
        for dim_code, dim_dir in u_dims.items():
            sim = cosine_similarity(pc, dim_dir)
            u_pc_dim_sims[f"PC{i+1}"][dim_code] = float(sim)

    results['M_pc_dim_sims'] = m_pc_dim_sims
    results['U_pc_dim_sims'] = u_pc_dim_sims

    # Compute dimension-to-dimension similarities
    m_dim_dim_sims = {}
    u_dim_dim_sims = {}

    for dim1 in ['V', 'A', 'D', 'AA']:
        if dim1 not in m_dims:
            continue
        m_dim_dim_sims[dim1] = {}
        for dim2 in ['V', 'A', 'D', 'AA']:
            if dim2 != dim1 and dim2 in m_dims:
                sim = cosine_similarity(m_dims[dim1], m_dims[dim2])
                m_dim_dim_sims[dim1][dim2] = float(sim)

    for dim1 in ['V', 'A', 'D', 'AA']:
        if dim1 not in u_dims:
            continue
        u_dim_dim_sims[dim1] = {}
        for dim2 in ['V', 'A', 'D', 'AA']:
            if dim2 != dim1 and dim2 in u_dims:
                sim = cosine_similarity(u_dims[dim1], u_dims[dim2])
                u_dim_dim_sims[dim1][dim2] = float(sim)

    results['M_dim_dim_sims'] = m_dim_dim_sims
    results['U_dim_dim_sims'] = u_dim_dim_sims

    return results


def plot_pc_to_dimension_similarities(all_results: List[Dict], output_dir: Path):
    """
    Plot PC similarities to psychological dimensions across layers.

    Creates 2 (M/U) × 4 (PC1-4) = 8 plots, each with 4 lines (one per dimension).
    """
    # Sort results by layer number
    all_results = sorted(all_results, key=lambda x: x['layer'])
    layers = [r['layer'] for r in all_results]

    for entity in ['M', 'U']:
        entity_name = "Assistant (M)" if entity == 'M' else "User (U)"

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{entity_name}: PC Similarities to Psychological Dimensions Across Layers\n(Orthogonalized Probes)',
                     fontsize=16, fontweight='bold')

        for pc_idx in range(4):
            ax = axes[pc_idx // 2, pc_idx % 2]
            pc_name = f"PC{pc_idx + 1}"

            # Collect data for each dimension
            for dim_code in ['V', 'A', 'D', 'AA']:
                sims = []
                for result in all_results:
                    pc_dim_sims = result[f'{entity}_pc_dim_sims']
                    if pc_name in pc_dim_sims and dim_code in pc_dim_sims[pc_name]:
                        sims.append(abs(pc_dim_sims[pc_name][dim_code]))
                    else:
                        sims.append(np.nan)

                ax.plot(layers, sims, marker='o', label=DIMENSION_NAMES[dim_code], linewidth=2, markersize=4)

            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Absolute Cosine Similarity', fontsize=12)
            ax.set_title(f'{pc_name}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

        plt.tight_layout()
        output_file = output_dir / f'{entity.lower()}_pc_to_dimensions.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_file}")


def plot_dimension_to_dimension_similarities(all_results: List[Dict], output_dir: Path):
    """
    Plot dimension-to-dimension similarities across layers.

    Creates 2 (M/U) × 4 (dimensions) = 8 plots, each with 3 lines (similarities to other 3 dimensions).
    """
    # Sort results by layer number
    all_results = sorted(all_results, key=lambda x: x['layer'])
    layers = [r['layer'] for r in all_results]

    for entity in ['M', 'U']:
        entity_name = "Assistant (M)" if entity == 'M' else "User (U)"

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{entity_name}: Pairwise Dimension Similarities Across Layers\n(Orthogonalized Probes)',
                     fontsize=16, fontweight='bold')

        dim_codes = ['V', 'A', 'D', 'AA']

        for idx, dim1 in enumerate(dim_codes):
            ax = axes[idx // 2, idx % 2]

            # Plot similarity to other 3 dimensions
            for dim2 in dim_codes:
                if dim2 == dim1:
                    continue

                sims = []
                for result in all_results:
                    dim_sims = result[f'{entity}_dim_dim_sims']
                    if dim1 in dim_sims and dim2 in dim_sims[dim1]:
                        sims.append(dim_sims[dim1][dim2])
                    else:
                        sims.append(np.nan)

                ax.plot(layers, sims, marker='o', label=DIMENSION_NAMES[dim2], linewidth=2, markersize=4)

            ax.set_xlabel('Layer', fontsize=12)
            ax.set_ylabel('Cosine Similarity', fontsize=12)
            ax.set_title(f'{DIMENSION_NAMES[dim1]}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.set_ylim(-1, 1)

        plt.tight_layout()
        output_file = output_dir / f'{entity.lower()}_dimension_similarities.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {output_file}")


def main():
    probes_dir = Path("probes/ua_emotion_disentangle/orthogonal_probes_all_layers")
    output_dir = Path("probes/ua_emotion_disentangle/layer_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("ANALYZING PCs AND DIMENSIONS ACROSS LAYERS")
    print("(Using Orthogonalized Probes)")
    print("="*80)
    print(f"\nProbes directory: {probes_dir}")
    print(f"Output directory: {output_dir}")

    # Get all probe files
    probe_files = sorted(probes_dir.glob("layer_*_orthogonal.npz"))
    if not probe_files:
        print(f"\n⚠ ERROR: No orthogonalized probe files found in {probes_dir}")
        print("Please run compute_orthogonal_probes_all_layers.py first!")
        return

    layers = [int(f.stem.split('_')[1]) for f in probe_files]
    print(f"\nFound {len(layers)} layers: {min(layers)} to {max(layers)}")

    # Analyze each layer
    print("\nAnalyzing layers...")
    all_results = []

    for layer in tqdm(layers, desc="Processing layers"):
        try:
            results = analyze_layer(layer, probes_dir)
            all_results.append(results)
        except Exception as e:
            print(f"\n⚠ Warning: Failed to analyze layer {layer}: {e}")
            continue

    print(f"\n✓ Successfully analyzed {len(all_results)} layers")

    # Save results
    print("\nSaving results...")
    results_file = output_dir / "layer_analysis_results.json"

    # Convert to JSON-serializable format
    json_results = []
    for r in all_results:
        json_r = {'layer': r['layer']}
        json_r['M_pc_dim_sims'] = r['M_pc_dim_sims']
        json_r['U_pc_dim_sims'] = r['U_pc_dim_sims']
        json_r['M_dim_dim_sims'] = r['M_dim_dim_sims']
        json_r['U_dim_dim_sims'] = r['U_dim_dim_sims']
        json_results.append(json_r)

    with open(results_file, 'w') as f:
        json.dump({
            'results': json_results,
            'dimension_names': DIMENSION_NAMES,
            'note': 'Uses orthogonalized probes from alternating orthogonalization'
        }, f, indent=2)

    print(f"✓ Saved: {results_file}")

    # Create plots
    print("\nCreating plots...")
    print("\n1. PC-to-dimension similarities...")
    plot_pc_to_dimension_similarities(all_results, output_dir)

    print("\n2. Dimension-to-dimension similarities...")
    plot_dimension_to_dimension_similarities(all_results, output_dir)

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - Results JSON: {results_file}")
    print(f"  - PC plots: {output_dir}/m_pc_to_dimensions.png")
    print(f"  - PC plots: {output_dir}/u_pc_to_dimensions.png")
    print(f"  - Dimension plots: {output_dir}/m_dimension_similarities.png")
    print(f"  - Dimension plots: {output_dir}/u_dimension_similarities.png")
    print("="*80)


if __name__ == "__main__":
    main()
