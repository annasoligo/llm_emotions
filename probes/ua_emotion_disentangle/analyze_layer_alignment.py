#!/usr/bin/env python3
"""
Analyze how emotion probe directions evolve across layers.

Measures:
1. Cosine similarity: Compare same emotion across layers
2. Principal angles: Compare emotion subspaces across layers
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import json
from scipy.linalg import subspace_angles

def load_layer_probes(layer: int, probes_dir: Path) -> Dict[str, np.ndarray]:
    """Load orthogonalized probes for one layer."""
    probe_file = probes_dir / f"layer_{layer}_orthogonal.npz"
    data = np.load(probe_file)
    return {key: data[key] for key in data.files}


def compute_emotion_similarity_across_layers(
    emotion_key: str,
    layers: List[int],
    probes_dir: Path
) -> np.ndarray:
    """
    Compute pairwise cosine similarity for one emotion across layers.

    Returns: (n_layers, n_layers) similarity matrix
    """
    # Load emotion vector from each layer
    vectors = []
    for layer in layers:
        probes = load_layer_probes(layer, probes_dir)
        if emotion_key in probes:
            vec = probes[emotion_key]
            # Normalize
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            vectors.append(vec)
        else:
            vectors.append(None)

    # Compute pairwise similarities
    n = len(vectors)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if vectors[i] is not None and vectors[j] is not None:
                similarity_matrix[i, j] = np.dot(vectors[i], vectors[j])
            else:
                similarity_matrix[i, j] = np.nan

    return similarity_matrix


def compute_subspace_alignment(
    layers: List[int],
    probes_dir: Path,
    entity: str = 'U'
) -> np.ndarray:
    """
    Compute principal angles between emotion subspaces across layers.

    Args:
        entity: 'M' or 'U' to analyze assistant or user emotions

    Returns: (n_layers, n_layers) matrix of mean principal angles (in degrees)
    """
    # Load emotion subspaces for each layer
    subspaces = []
    for layer in layers:
        probes = load_layer_probes(layer, probes_dir)

        # Extract vectors for the entity
        vectors = []
        for key, vec in probes.items():
            if key.startswith(f"{entity}_"):
                vectors.append(vec)

        if vectors:
            # Stack into matrix: (n_emotions, hidden_dim)
            subspace = np.stack(vectors)
            subspaces.append(subspace)
        else:
            subspaces.append(None)

    # Compute principal angles between subspaces
    n = len(subspaces)
    angle_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if subspaces[i] is not None and subspaces[j] is not None:
                angles = subspace_angles(subspaces[i].T, subspaces[j].T)
                # Use mean angle as summary metric
                angle_matrix[i, j] = np.mean(angles) * 180 / np.pi  # Convert to degrees
            else:
                angle_matrix[i, j] = np.nan

    return angle_matrix


def plot_emotion_similarity_heatmap(
    similarity_matrix: np.ndarray,
    layers: List[int],
    emotion_key: str,
    output_dir: Path
):
    """Plot heatmap of emotion similarity across layers."""
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(similarity_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Set ticks
    tick_indices = list(range(0, len(layers), 5))
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels([layers[i] for i in tick_indices])
    ax.set_yticklabels([layers[i] for i in tick_indices])

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(f'Cosine Similarity: {emotion_key} Across Layers', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=11)

    plt.tight_layout()
    output_file = output_dir / f'layer_similarity_{emotion_key}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def plot_subspace_alignment_heatmap(
    angle_matrix: np.ndarray,
    layers: List[int],
    entity: str,
    output_dir: Path
):
    """Plot heatmap of subspace principal angles across layers."""
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(angle_matrix, cmap='YlOrRd', vmin=0, vmax=90, aspect='auto')

    # Set ticks
    tick_indices = list(range(0, len(layers), 5))
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels([layers[i] for i in tick_indices])
    ax.set_yticklabels([layers[i] for i in tick_indices])

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)

    entity_name = "User (U)" if entity == 'U' else "Assistant (M)"
    ax.set_title(f'Subspace Alignment: {entity_name} Emotions Across Layers\n(Mean Principal Angle)',
                 fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Principal Angle (degrees)', fontsize=11)

    plt.tight_layout()
    output_file = output_dir / f'subspace_alignment_{entity}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def plot_similarity_to_layer30(
    layers: List[int],
    probes_dir: Path,
    output_dir: Path
):
    """Plot how emotions change relative to layer 30."""
    reference_layer = 30
    reference_probes = load_layer_probes(reference_layer, probes_dir)

    # Select 4 non-opposite emotions (avoid inverse patterns)
    emotions_to_plot = [
        'U_joy', 'U_anger', 'U_trust', 'U_disgust',
        'M_joy', 'M_anger', 'M_trust', 'M_disgust'
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for entity_idx, entity in enumerate(['U', 'M']):
        ax = axes[entity_idx]
        entity_emotions = [e for e in emotions_to_plot if e.startswith(f'{entity}_')]

        print(f"   Plotting {len(entity_emotions)} emotions for {entity}: {entity_emotions}")

        for emotion_key in entity_emotions:
            if emotion_key not in reference_probes:
                print(f"     Skipping {emotion_key} - not in reference probes")
                continue

            print(f"     Processing {emotion_key}...")
            ref_vec = reference_probes[emotion_key]
            ref_vec = ref_vec / (np.linalg.norm(ref_vec) + 1e-8)

            similarities = []
            for layer in layers:
                probes = load_layer_probes(layer, probes_dir)
                if emotion_key in probes:
                    vec = probes[emotion_key]
                    vec = vec / (np.linalg.norm(vec) + 1e-8)
                    sim = np.dot(ref_vec, vec)
                    similarities.append(sim)
                else:
                    similarities.append(np.nan)

            emotion_name = emotion_key.replace(f'{entity}_', '')
            display_name = emotion_name.capitalize()
            print(f"     Plotting {display_name} with {len([s for s in similarities if not np.isnan(s)])} valid points")

            # Simple plotting with default colors
            ax.plot(layers, similarities,
                   label=display_name,
                   linewidth=2,
                   marker='o',
                   markersize=4)

        entity_name = "User (U)" if entity == 'U' else "Assistant (M)"
        ax.set_xlabel('Layer', fontsize=13)
        ax.set_ylabel(f'Cosine Similarity to Layer {reference_layer}', fontsize=13)
        ax.set_title(f'{entity_name} Emotions: Similarity to Layer {reference_layer}',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=12, loc='best', framealpha=0.9, edgecolor='black')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=reference_layer, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.set_ylim(0, 1.05)

    plt.tight_layout()
    output_file = output_dir / f'emotion_evolution_vs_layer{reference_layer}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def main():
    probes_dir = Path("probes/ua_emotion_disentangle/orthogonal_probes_all_layers")
    output_dir = Path("probes/ua_emotion_disentangle/layer_alignment_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LAYER ALIGNMENT ANALYSIS")
    print("=" * 80)
    print(f"\nProbes directory: {probes_dir}")
    print(f"Output directory: {output_dir}")

    # Get all layers (sort numerically!)
    probe_files = list(probes_dir.glob("layer_*_orthogonal.npz"))
    layers = sorted([int(f.stem.split('_')[1]) for f in probe_files])
    print(f"\nFound {len(layers)} layers: {min(layers)} to {max(layers)}")

    # 1. Plot evolution relative to layer 30
    print("\n1. Computing emotion evolution relative to layer 30...")
    plot_similarity_to_layer30(layers, probes_dir, output_dir)

    # 2. Analyze specific emotions
    print("\n2. Computing layer similarity for specific emotions...")
    for emotion_key in ['U_sadness', 'U_joy', 'M_sadness', 'M_joy']:
        print(f"   Processing {emotion_key}...")
        similarity_matrix = compute_emotion_similarity_across_layers(emotion_key, layers, probes_dir)
        plot_emotion_similarity_heatmap(similarity_matrix, layers, emotion_key, output_dir)

    # 3. Compute subspace alignment
    print("\n3. Computing subspace alignment...")
    for entity in ['U', 'M']:
        print(f"   Processing {entity} emotions...")
        angle_matrix = compute_subspace_alignment(layers, probes_dir, entity)
        plot_subspace_alignment_heatmap(angle_matrix, layers, entity, output_dir)

        # Save statistics
        off_diagonal = angle_matrix[~np.eye(len(layers), dtype=bool)]
        stats = {
            'entity': entity,
            'mean_angle_degrees': float(np.nanmean(off_diagonal)),
            'median_angle_degrees': float(np.nanmedian(off_diagonal)),
            'std_angle_degrees': float(np.nanstd(off_diagonal)),
            'min_angle_degrees': float(np.nanmin(off_diagonal)),
            'max_angle_degrees': float(np.nanmax(off_diagonal))
        }

        print(f"\n   {entity} Subspace Alignment Statistics:")
        print(f"     Mean angle: {stats['mean_angle_degrees']:.2f}°")
        print(f"     Median: {stats['median_angle_degrees']:.2f}°")
        print(f"     Range: {stats['min_angle_degrees']:.2f}° - {stats['max_angle_degrees']:.2f}°")

    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files in: {output_dir}")


if __name__ == "__main__":
    main()
