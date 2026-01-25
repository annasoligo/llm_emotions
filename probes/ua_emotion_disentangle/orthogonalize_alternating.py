#!/usr/bin/env python3
"""
Alternating orthogonalization of M and U emotion probes.

Iteratively projects M away from U and U away from M until convergence.
This is a symmetric approach that treats both entities fairly.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


def load_probes(probes_path: Path) -> Dict[str, np.ndarray]:
    """Load probes from .npz file."""
    print(f"Loading probes from: {probes_path}")
    data = np.load(probes_path)
    probes = {key: data[key] for key in data.files}
    print(f"  Loaded {len(probes)} probes")
    return probes


def compute_cross_contamination(
    m_probes: Dict[str, np.ndarray],
    u_probes: Dict[str, np.ndarray]
) -> float:
    """Compute mean absolute cosine similarity between M and U probes."""
    m_names = sorted(m_probes.keys())
    u_names = sorted(u_probes.keys())
    M = np.stack([m_probes[k] for k in m_names])
    U = np.stack([u_probes[k] for k in u_names])

    # Normalize
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)

    # Pairwise cosine similarities
    cosine_sims = np.abs(M_norm @ U_norm.T)

    return float(np.mean(cosine_sims))


def project_away_from_subspace(
    vectors: np.ndarray,
    basis_vectors: np.ndarray
) -> np.ndarray:
    """
    Project vectors away from subspace spanned by basis_vectors.

    Args:
        vectors: [n, d]
        basis_vectors: [m, d]

    Returns:
        Orthogonalized vectors [n, d]
    """
    # Get orthonormal basis via QR
    Q, R = np.linalg.qr(basis_vectors.T)
    orthonormal_basis = Q.T  # [k, d] where k <= m

    # Project onto subspace
    projection = vectors @ orthonormal_basis.T @ orthonormal_basis

    # Subtract to get orthogonal component
    orthogonal = vectors - projection

    return orthogonal


def alternating_orthogonalization(
    m_probes: Dict[str, np.ndarray],
    u_probes: Dict[str, np.ndarray],
    max_iters: int = 10,
    convergence_tol: float = 1e-6,
    damping: float = 0.5
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], list]:
    """
    Alternating orthogonalization with damping for stability.

    Args:
        m_probes: M emotion probes
        u_probes: U emotion probes
        max_iters: Maximum iterations
        convergence_tol: Convergence tolerance for contamination
        damping: Damping factor (0-1), 0.5 = equal weighting of old and new

    Returns:
        m_clean: Orthogonalized M probes
        u_clean: Orthogonalized U probes
        history: List of contamination at each iteration
    """
    print("\n" + "="*80)
    print("ALTERNATING ORTHOGONALIZATION (with damping)")
    print("="*80)

    # Stack into matrices
    m_names = sorted(m_probes.keys())
    u_names = sorted(u_probes.keys())
    M = np.stack([m_probes[k] for k in m_names])
    U = np.stack([u_probes[k] for k in u_names])

    print(f"\nInput:")
    print(f"  M probes: {M.shape}")
    print(f"  U probes: {U.shape}")
    print(f"  Damping factor: {damping}")

    # Initialize
    M_current = M.copy()
    U_current = U.copy()

    history = []
    initial_contamination = compute_cross_contamination(
        {k: M_current[i] for i, k in enumerate(m_names)},
        {k: U_current[i] for i, k in enumerate(u_names)}
    )
    history.append(initial_contamination)

    print(f"\nInitial contamination: {initial_contamination:.6f}")
    print("\nIterating:")

    for iteration in range(max_iters):
        # Save previous state for damping
        M_prev = M_current.copy()
        U_prev = U_current.copy()

        # Project U away from M
        U_new = project_away_from_subspace(U_prev, M_prev)
        U_current = damping * U_prev + (1 - damping) * U_new

        # Project M away from U
        M_new = project_away_from_subspace(M_prev, U_current)
        M_current = damping * M_prev + (1 - damping) * M_new

        # Compute contamination
        contamination = compute_cross_contamination(
            {k: M_current[i] for i, k in enumerate(m_names)},
            {k: U_current[i] for i, k in enumerate(u_names)}
        )
        history.append(contamination)

        # Check convergence
        change = abs(history[-1] - history[-2])
        print(f"  Iteration {iteration + 1}: contamination = {contamination:.6f}, change = {change:.6f}")

        if change < convergence_tol:
            print(f"  ✓ Converged after {iteration + 1} iterations")
            break
    else:
        print(f"  ! Reached max iterations ({max_iters})")

    # Analyze norm preservation
    print("\nNorm preservation:")
    m_norms_before = np.linalg.norm(M, axis=1)
    m_norms_after = np.linalg.norm(M_current, axis=1)
    u_norms_before = np.linalg.norm(U, axis=1)
    u_norms_after = np.linalg.norm(U_current, axis=1)

    print(f"  M: {np.mean(m_norms_after / m_norms_before) * 100:.1f}% retained (mean)")
    print(f"  U: {np.mean(u_norms_after / u_norms_before) * 100:.1f}% retained (mean)")

    # Convert back to dictionaries
    m_clean = {name: M_current[i] for i, name in enumerate(m_names)}
    u_clean = {name: U_current[i] for i, name in enumerate(u_names)}

    return m_clean, u_clean, history


def main():
    parser = argparse.ArgumentParser(description="Alternating orthogonalization")
    parser.add_argument("--input", type=Path,
                       default=Path("probes/ua_emotion_disentangle/full_analysis/probes_first_asst_token.npz"))
    parser.add_argument("--output-dir", type=Path,
                       default=Path("probes/ua_emotion_disentangle/full_analysis"))
    parser.add_argument("--max-iters", type=int, default=10)
    parser.add_argument("--damping", type=float, default=0.5,
                       help="Damping factor (0-1), 0.5 = equal weighting")
    args = parser.parse_args()

    print("="*80)
    print("ALTERNATING M-U ORTHOGONALIZATION")
    print("="*80)

    # Load probes
    all_probes = load_probes(args.input)
    m_probes = {k: v for k, v in all_probes.items() if k.startswith('M_')}
    u_probes = {k: v for k, v in all_probes.items() if k.startswith('U_')}

    print(f"\nSeparated:")
    print(f"  M probes: {len(m_probes)}")
    print(f"  U probes: {len(u_probes)}")

    # Orthogonalize
    m_clean, u_clean, history = alternating_orthogonalization(
        m_probes, u_probes,
        max_iters=args.max_iters,
        damping=args.damping
    )

    # Analysis
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nContamination reduction: {(1 - history[-1]/history[0]) * 100:.1f}%")
    print(f"  Before: {history[0]:.6f}")
    print(f"  After:  {history[-1]:.6f}")

    # Save
    output_probes = {**m_clean, **u_clean}
    input_stem = args.input.stem
    output_path = args.output_dir / f"{input_stem}_alternating.npz"
    np.savez(output_path, **output_probes)
    print(f"\n✓ Saved to: {output_path}")

    # Save history
    analysis = {
        'history': [float(h) for h in history],
        'reduction_pct': float((1 - history[-1]/history[0]) * 100),
        'damping': args.damping,
    }
    analysis_path = args.output_dir / f"{input_stem}_alternating_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"✓ Saved analysis to: {analysis_path}")

    # Compare with asymmetric
    asym_path = args.output_dir / f"{input_stem}_orthogonal.npz"
    if asym_path.exists():
        asym_probes = load_probes(asym_path)
        asym_m = {k: v for k, v in asym_probes.items() if k.startswith('M_')}
        asym_u = {k: v for k, v in asym_probes.items() if k.startswith('U_')}
        asym_contam = compute_cross_contamination(asym_m, asym_u)

        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        print(f"\nAsymmetric (U away from M): {asym_contam:.6f}")
        print(f"Alternating (symmetric):     {history[-1]:.6f}")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
