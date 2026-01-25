#!/usr/bin/env python3
"""
Compute orthogonalized emotion probes for all layers using alternating orthogonalization.

Reads raw activations and produces orthogonalized M and U probes for each layer.
Probes are computed as: emotion - mean(all other emotions) [CONTRAST-ALL METHOD]

This differs from the opposite-pair method which computes: emotion - opposite_emotion
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm
import sys


def load_layer_data(layer: int, data_dir: Path) -> Dict:
    """Load activation data for one layer."""
    layer_file = data_dir / f"layer_{layer}.pkl"
    with open(layer_file, 'rb') as f:
        return pickle.load(f)


def compute_emotion_probes_contrast_all(
    activations: np.ndarray,
    metadata: list
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Compute emotion probes as: emotion - mean(all other emotions).

    For each emotion (e.g., joy), computes:
    - M probe: mean(activations where M=joy) - mean(activations where M≠joy)
    - U probe: mean(activations where U=joy) - mean(activations where U≠joy)

    This does NOT enforce opposite-pair structure.

    Returns:
        m_probes: Dict mapping emotion names to probe vectors
        u_probes: Dict mapping emotion names to probe vectors
    """
    # Group activations by M and U emotions
    m_emotion_groups = {}
    u_emotion_groups = {}

    for act, meta in zip(activations, metadata):
        m_emotion = meta['M']
        u_emotion = meta['U']

        if m_emotion not in m_emotion_groups:
            m_emotion_groups[m_emotion] = []
        if u_emotion not in u_emotion_groups:
            u_emotion_groups[u_emotion] = []

        m_emotion_groups[m_emotion].append(act)
        u_emotion_groups[u_emotion].append(act)

    # Compute mean activation for each emotion
    m_means = {emotion: np.mean(acts, axis=0) for emotion, acts in m_emotion_groups.items()}
    u_means = {emotion: np.mean(acts, axis=0) for emotion, acts in u_emotion_groups.items()}

    # Compute probes as: emotion - mean(all other emotions)
    m_probes = {}
    u_probes = {}

    # For M probes
    m_emotions = sorted(m_means.keys())
    for target_emotion in m_emotions:
        other_emotions = [emo for emo in m_emotions if emo != target_emotion]
        if other_emotions:
            target_mean = m_means[target_emotion]
            others_mean = np.mean([m_means[emo] for emo in other_emotions], axis=0)
            m_probes[f"M_{target_emotion}"] = target_mean - others_mean

    # For U probes
    u_emotions = sorted(u_means.keys())
    for target_emotion in u_emotions:
        other_emotions = [emo for emo in u_emotions if emo != target_emotion]
        if other_emotions:
            target_mean = u_means[target_emotion]
            others_mean = np.mean([u_means[emo] for emo in other_emotions], axis=0)
            u_probes[f"U_{target_emotion}"] = target_mean - others_mean

    return m_probes, u_probes


def alternating_orthogonalization(
    m_probes: Dict[str, np.ndarray],
    u_probes: Dict[str, np.ndarray],
    max_iters: int = 10,
    convergence_tol: float = 1e-6,
    damping: float = 0.5
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], list]:
    """
    Alternating orthogonalization with damping.
    """
    def project_away_from_subspace(vectors, basis_vectors):
        """Project vectors away from subspace spanned by basis_vectors."""
        from numpy.linalg import qr
        Q, R = qr(basis_vectors.T)
        orthonormal_basis = Q.T
        projection = vectors @ orthonormal_basis.T @ orthonormal_basis
        return vectors - projection

    def compute_contamination(m, u):
        """Compute mean absolute cosine similarity."""
        M = np.stack([m[k] for k in sorted(m.keys())])
        U = np.stack([u[k] for k in sorted(u.keys())])
        M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
        U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
        return float(np.mean(np.abs(M_norm @ U_norm.T)))

    # Stack into matrices
    m_names = sorted(m_probes.keys())
    u_names = sorted(u_probes.keys())
    M = np.stack([m_probes[k] for k in m_names])
    U = np.stack([u_probes[k] for k in u_names])

    M_current = M.copy()
    U_current = U.copy()

    history = []
    history.append(compute_contamination(
        {k: M_current[i] for i, k in enumerate(m_names)},
        {k: U_current[i] for i, k in enumerate(u_names)}
    ))

    for iteration in range(max_iters):
        M_prev = M_current.copy()
        U_prev = U_current.copy()

        # Project U away from M
        U_new = project_away_from_subspace(U_prev, M_prev)
        U_current = damping * U_prev + (1 - damping) * U_new

        # Project M away from U
        M_new = project_away_from_subspace(M_prev, U_current)
        M_current = damping * M_prev + (1 - damping) * M_new

        # Compute contamination
        contamination = compute_contamination(
            {k: M_current[i] for i, k in enumerate(m_names)},
            {k: U_current[i] for i, k in enumerate(u_names)}
        )
        history.append(contamination)

        # Check convergence
        if abs(history[-1] - history[-2]) < convergence_tol:
            break

    # Convert back to dictionaries
    m_clean = {name: M_current[i] for i, name in enumerate(m_names)}
    u_clean = {name: U_current[i] for i, name in enumerate(u_names)}

    return m_clean, u_clean, history


def compute_orthogonal_probes_for_layer(
    layer: int,
    data_dir: Path,
    position: str = 'first_asst_token',
    max_iters: int = 10,
    damping: float = 0.5
) -> Tuple[Dict[str, np.ndarray], list]:
    """
    Compute orthogonalized probes for one layer using CONTRAST-ALL method.

    Returns:
        all_probes: Dict with orthogonalized M and U probes
        history: Convergence history
    """
    # Load layer data
    layer_data = load_layer_data(layer, data_dir)

    # Extract activations and metadata for the position
    activations = layer_data['activations'][position]
    metadata = layer_data['metadata'][position]

    # Compute raw probes (as emotion - mean(all others))
    m_probes, u_probes = compute_emotion_probes_contrast_all(activations, metadata)

    # Orthogonalize using alternating method
    m_clean, u_clean, history = alternating_orthogonalization(
        m_probes, u_probes,
        max_iters=max_iters,
        convergence_tol=1e-6,
        damping=damping
    )

    # Combine
    all_probes = {**m_clean, **u_clean}

    return all_probes, history


def main():
    data_dir = Path("probes/ua_emotion_disentangle/data/activations/full_all_layers")
    output_dir = Path("probes/ua_emotion_disentangle/orthogonal_probes_all_layers_contrast_all")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("COMPUTING ORTHOGONALIZED PROBES FOR ALL LAYERS")
    print("METHOD: CONTRAST-ALL (emotion - mean(all others))")
    print("="*80)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nMethod: Alternating orthogonalization (damping=0.5, max_iters=10)")
    print(f"Probe computation: emotion - mean(all other emotions)")

    # Get all layer files
    layer_files = sorted(data_dir.glob("layer_*.pkl"))
    layers = [int(f.stem.split('_')[1]) for f in layer_files]
    print(f"\nFound {len(layers)} layers: {min(layers)} to {max(layers)}")

    # Process each layer
    print("\nComputing orthogonalized probes...")

    convergence_stats = []

    for layer in tqdm(layers, desc="Processing layers"):
        try:
            orthogonal_probes, history = compute_orthogonal_probes_for_layer(
                layer, data_dir,
                position='first_asst_token',
                max_iters=10,
                damping=0.5
            )

            # Save probes
            output_file = output_dir / f"layer_{layer}_orthogonal.npz"
            np.savez(output_file, **orthogonal_probes)

            # Track convergence
            convergence_stats.append({
                'layer': layer,
                'initial_contamination': history[0],
                'final_contamination': history[-1],
                'iterations': len(history) - 1,
                'reduction_pct': (1 - history[-1] / history[0]) * 100 if history[0] > 0 else 100
            })

        except Exception as e:
            print(f"\n⚠ Warning: Failed to process layer {layer}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)

    if convergence_stats:
        avg_initial = np.mean([s['initial_contamination'] for s in convergence_stats])
        avg_final = np.mean([s['final_contamination'] for s in convergence_stats])
        avg_reduction = np.mean([s['reduction_pct'] for s in convergence_stats])

        print(f"\nAverage across all layers:")
        print(f"  Initial contamination: {avg_initial:.6f}")
        print(f"  Final contamination:   {avg_final:.10f}")
        print(f"  Reduction:             {avg_reduction:.1f}%")

        # Find layers with best/worst orthogonalization
        best = min(convergence_stats, key=lambda s: s['final_contamination'])
        worst = max(convergence_stats, key=lambda s: s['final_contamination'])

        print(f"\nBest orthogonalization:")
        print(f"  Layer {best['layer']}: {best['final_contamination']:.10f}")

        print(f"\nWorst orthogonalization:")
        print(f"  Layer {worst['layer']}: {worst['final_contamination']:.10f}")

    print(f"\n✓ Saved orthogonalized probes to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
