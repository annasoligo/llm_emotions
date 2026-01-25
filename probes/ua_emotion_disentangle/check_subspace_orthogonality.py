#!/usr/bin/env python3
"""
Check if alternating orthogonalization achieves full subspace orthogonality
by computing principal angles between the resulting M and U subspaces.
"""

import numpy as np
from pathlib import Path


def load_probes(path: Path):
    data = np.load(path)
    return {key: data[key] for key in data.files}


def compute_principal_angles(M_matrix, U_matrix):
    """
    Compute principal angles between M and U subspaces.

    Returns cosines of principal angles.
    """
    # Get orthonormal bases via SVD
    M_basis, s_m, _ = np.linalg.svd(M_matrix.T, full_matrices=False)
    U_basis, s_u, _ = np.linalg.svd(U_matrix.T, full_matrices=False)

    # Keep numerically significant dimensions
    rank_m = np.sum(s_m > 1e-10 * s_m[0])
    rank_u = np.sum(s_u > 1e-10 * s_u[0])
    M_basis = M_basis[:, :rank_m]
    U_basis = U_basis[:, :rank_u]

    # Compute principal angles via SVD of cross-correlation
    cross = M_basis.T @ U_basis
    _, cosines, _ = np.linalg.svd(cross, full_matrices=False)

    return cosines, rank_m, rank_u


def main():
    base_dir = Path("probes/ua_emotion_disentangle/full_analysis")

    # Load probe sets
    print("="*80)
    print("SUBSPACE ORTHOGONALITY ANALYSIS")
    print("="*80)

    methods = {
        "Raw (no orthogonalization)": "probes_first_asst_token.npz",
        "Asymmetric (U away from M)": "probes_first_asst_token_orthogonal.npz",
        "Alternating (symmetric)": "probes_first_asst_token_alternating.npz",
        "Principal Angles (t=0.3)": "probes_first_asst_token_symmetric.npz",
    }

    for method_name, filename in methods.items():
        path = base_dir / filename
        if not path.exists():
            print(f"\n{method_name}: FILE NOT FOUND")
            continue

        probes = load_probes(path)

        # Separate M and U
        m_names = sorted([k for k in probes.keys() if k.startswith('M_')])
        u_names = sorted([k for k in probes.keys() if k.startswith('U_')])
        M = np.stack([probes[k] for k in m_names])
        U = np.stack([probes[k] for k in u_names])

        # Compute principal angles
        cosines, rank_m, rank_u = compute_principal_angles(M, U)

        # Convert to angles in degrees
        angles_deg = np.arccos(np.clip(cosines, -1, 1)) * 180 / np.pi

        print(f"\n{method_name}:")
        print(f"  M subspace rank: {rank_m}")
        print(f"  U subspace rank: {rank_u}")
        print(f"  Number of principal angles: {len(cosines)}")
        print(f"\n  Principal angle cosines (max = most aligned):")
        print(f"    Max:    {np.max(cosines):.6f} ({np.min(angles_deg):.2f}°)")
        print(f"    Median: {np.median(cosines):.6f} ({np.median(angles_deg):.2f}°)")
        print(f"    Min:    {np.min(cosines):.6f} ({np.max(angles_deg):.2f}°)")

        # Count how many are "nearly orthogonal" (cos < 0.1, angle > 84°)
        n_orthogonal = np.sum(cosines < 0.1)
        print(f"\n  Nearly orthogonal directions (cos < 0.1): {n_orthogonal}/{len(cosines)}")

        # Show top 5 principal angles
        print(f"\n  Top 5 principal angles:")
        for i in range(min(5, len(cosines))):
            print(f"    PA{i+1}: cos = {cosines[i]:.6f}, angle = {angles_deg[i]:.2f}°")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("""
Principal angle cosines close to 0 (angles close to 90°) indicate orthogonal subspaces.
Principal angle cosines close to 1 (angles close to 0°) indicate aligned subspaces.

If alternating achieves FULL subspace orthogonality, all principal angles
should be ≈90° (cosines ≈0).

If alternating only achieves pairwise orthogonality but not full subspace
orthogonality, some principal angles may still be small (cosines > 0).
""")


if __name__ == "__main__":
    main()
