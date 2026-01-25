#!/usr/bin/env python3
"""
Compare probe directions across different orthogonalization methods.

Computes pairwise cosine similarities between corresponding probes
from asymmetric, alternating, and principal angles methods.
"""

import numpy as np
from pathlib import Path
from typing import Dict
import pandas as pd


def load_probes(path: Path) -> Dict[str, np.ndarray]:
    """Load probes from .npz file."""
    data = np.load(path)
    return {key: data[key] for key in data.files}


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)


def compare_probe_sets(
    probes1: Dict[str, np.ndarray],
    probes2: Dict[str, np.ndarray],
    name1: str,
    name2: str
) -> pd.DataFrame:
    """Compare two probe sets by computing cosine similarities."""

    # Get common probe names
    common_names = sorted(set(probes1.keys()) & set(probes2.keys()))

    results = []
    for name in common_names:
        v1 = probes1[name]
        v2 = probes2[name]

        cos_sim = cosine_similarity(v1, v2)

        # Categorize by entity
        entity = 'M' if name.startswith('M_') else 'U'
        emotion = name.replace('M_', '').replace('U_', '')

        results.append({
            'probe': name,
            'entity': entity,
            'emotion': emotion,
            'cosine_sim': cos_sim,
            'method1': name1,
            'method2': name2,
        })

    return pd.DataFrame(results)


def main():
    base_dir = Path("probes/ua_emotion_disentangle/full_analysis")

    # Load all probe sets
    print("Loading probe sets...")
    asymmetric = load_probes(base_dir / "probes_first_asst_token_orthogonal.npz")
    alternating = load_probes(base_dir / "probes_first_asst_token_alternating.npz")

    # Try to load principal angles (may have different thresholds)
    principal_files = list(base_dir.glob("probes_first_asst_token_symmetric*.npz"))
    if principal_files:
        principal = load_probes(principal_files[0])
        print(f"  Loaded principal angles from: {principal_files[0].name}")
    else:
        principal = None
        print("  No principal angles file found")

    print(f"  Asymmetric: {len(asymmetric)} probes")
    print(f"  Alternating: {len(alternating)} probes")
    if principal:
        print(f"  Principal angles: {len(principal)} probes")

    # Compare asymmetric vs alternating
    print("\n" + "="*80)
    print("ASYMMETRIC vs ALTERNATING")
    print("="*80)
    df_asym_alt = compare_probe_sets(asymmetric, alternating, "Asymmetric", "Alternating")

    # Overall statistics
    print(f"\nOverall:")
    print(f"  Mean cosine similarity: {df_asym_alt['cosine_sim'].mean():.4f}")
    print(f"  Median cosine similarity: {df_asym_alt['cosine_sim'].median():.4f}")
    print(f"  Min cosine similarity: {df_asym_alt['cosine_sim'].min():.4f}")
    print(f"  Max cosine similarity: {df_asym_alt['cosine_sim'].max():.4f}")

    # By entity
    print(f"\nBy entity:")
    for entity in ['M', 'U']:
        entity_df = df_asym_alt[df_asym_alt['entity'] == entity]
        print(f"  {entity}: mean={entity_df['cosine_sim'].mean():.4f}, "
              f"median={entity_df['cosine_sim'].median():.4f}")

    # Top 10 most similar
    print(f"\nTop 10 most similar probes:")
    top10 = df_asym_alt.nlargest(10, 'cosine_sim')
    for _, row in top10.iterrows():
        print(f"  {row['probe']:25s}: {row['cosine_sim']:+.4f}")

    # Bottom 10 least similar
    print(f"\nBottom 10 least similar probes:")
    bottom10 = df_asym_alt.nsmallest(10, 'cosine_sim')
    for _, row in bottom10.iterrows():
        print(f"  {row['probe']:25s}: {row['cosine_sim']:+.4f}")

    # Compare with principal angles if available
    if principal:
        print("\n" + "="*80)
        print("ALTERNATING vs PRINCIPAL ANGLES")
        print("="*80)
        df_alt_prin = compare_probe_sets(alternating, principal, "Alternating", "Principal")

        print(f"\nOverall:")
        print(f"  Mean cosine similarity: {df_alt_prin['cosine_sim'].mean():.4f}")
        print(f"  Median cosine similarity: {df_alt_prin['cosine_sim'].median():.4f}")

        print(f"\nBy entity:")
        for entity in ['M', 'U']:
            entity_df = df_alt_prin[df_alt_prin['entity'] == entity]
            print(f"  {entity}: mean={entity_df['cosine_sim'].mean():.4f}")

        print("\n" + "="*80)
        print("ASYMMETRIC vs PRINCIPAL ANGLES")
        print("="*80)
        df_asym_prin = compare_probe_sets(asymmetric, principal, "Asymmetric", "Principal")

        print(f"\nOverall:")
        print(f"  Mean cosine similarity: {df_asym_prin['cosine_sim'].mean():.4f}")
        print(f"  Median cosine similarity: {df_asym_prin['cosine_sim'].median():.4f}")

        print(f"\nBy entity:")
        for entity in ['M', 'U']:
            entity_df = df_asym_prin[df_asym_prin['entity'] == entity]
            print(f"  {entity}: mean={entity_df['cosine_sim'].mean():.4f}")

    # Save detailed results
    output_path = base_dir / "orthogonalization_method_comparison.csv"
    df_asym_alt.to_csv(output_path, index=False)
    print(f"\n✓ Detailed results saved to: {output_path}")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
Interpretation:
- Cosine similarity ≈ 1.0: Probes point in same direction (methods agree)
- Cosine similarity ≈ 0.0: Probes are orthogonal (methods differ in direction)
- Cosine similarity < 0.0: Probes point in opposite directions

High similarity suggests the methods produce similar orthogonalized probes.
Low similarity suggests different trade-offs in removing shared structure.
""")


if __name__ == "__main__":
    main()
