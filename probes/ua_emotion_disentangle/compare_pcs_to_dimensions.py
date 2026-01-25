#!/usr/bin/env python3
"""
Compare PCA components to psychological emotion dimensions.

For each dimension (Valence, Arousal, Dominance, Approach-Avoidance):
- Compute direction as: mean(high emotions) - mean(low emotions)
- Compare to PC1-4 using cosine similarity
"""

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from typing import Dict
import json

# Emotion dimension annotations
EMOTION_DIMENSIONS = {
    # V = Valence, A = Arousal, D = Dominance, AA = Approach-Avoidance
    # H = High, N = Neutral, L = Low

    # Basic emotions
    "joy":          {"V": "H", "A": "H", "D": "H", "AA": "H"},
    "sadness":      {"V": "L", "A": "L", "D": "L", "AA": "N"},
    "trust":        {"V": "H", "A": "N", "D": "H", "AA": "H"},
    "disgust":      {"V": "L", "A": "N", "D": "H", "AA": "L"},
    "fear":         {"V": "L", "A": "H", "D": "L", "AA": "L"},
    "anger":        {"V": "L", "A": "H", "D": "H", "AA": "H"},
    "surprise":     {"V": "N", "A": "H", "D": "L", "AA": "N"},
    "anticipation": {"V": "N", "A": "H", "D": "H", "AA": "H"},

    # Mild emotions
    "serenity":     {"V": "H", "A": "L", "D": "H", "AA": "N"},
    "pensiveness":  {"V": "L", "A": "L", "D": "N", "AA": "N"},
    "acceptance":   {"V": "H", "A": "L", "D": "H", "AA": "H"},
    "boredom":      {"V": "L", "A": "L", "D": "N", "AA": "L"},
    "apprehension": {"V": "L", "A": "N", "D": "L", "AA": "L"},
    "annoyance":    {"V": "L", "A": "N", "D": "H", "AA": "H"},
    "distraction":  {"V": "N", "A": "N", "D": "L", "AA": "N"},
    "interest":     {"V": "H", "A": "N", "D": "H", "AA": "H"},

    # Intense emotions
    "ecstasy":      {"V": "H", "A": "H", "D": "H", "AA": "H"},
    "grief":        {"V": "L", "A": "H", "D": "L", "AA": "N"},
    "admiration":   {"V": "H", "A": "N", "D": "N", "AA": "H"},
    "loathing":     {"V": "L", "A": "H", "D": "H", "AA": "L"},
    "terror":       {"V": "L", "A": "H", "D": "L", "AA": "L"},
    "rage":         {"V": "L", "A": "H", "D": "H", "AA": "H"},
    "amazement":    {"V": "H", "A": "H", "D": "L", "AA": "H"},
    "vigilance":    {"V": "N", "A": "H", "D": "H", "AA": "H"},

    # Dyad emotions
    "love":         {"V": "H", "A": "N", "D": "N", "AA": "H"},
    "remorse":      {"V": "L", "A": "N", "D": "L", "AA": "L"},
    "submission":   {"V": "N", "A": "L", "D": "L", "AA": "N"},
    "contempt":     {"V": "L", "A": "N", "D": "H", "AA": "L"},
    "awe":          {"V": "H", "A": "H", "D": "L", "AA": "H"},
    "aggressiveness": {"V": "L", "A": "H", "D": "H", "AA": "H"},
    "disapproval":  {"V": "L", "A": "N", "D": "H", "AA": "L"},
    "optimism":     {"V": "H", "A": "N", "D": "H", "AA": "H"},
}

DIMENSION_NAMES = {
    "V": "Valence (positive-negative)",
    "A": "Arousal (high-low energy)",
    "D": "Dominance (power-submission)",
    "AA": "Approach-Avoidance"
}


def load_probes(probes_path: Path) -> Dict[str, np.ndarray]:
    """Load probes from .npz file."""
    data = np.load(probes_path)
    return {key: data[key] for key in data.files}


def separate_probes(probes: Dict[str, np.ndarray]):
    """Separate M and U probes."""
    m_probes = {k.replace('M_', ''): v for k, v in probes.items() if k.startswith('M_')}
    u_probes = {k.replace('U_', ''): v for k, v in probes.items() if k.startswith('U_')}
    return m_probes, u_probes


def compute_dimension_direction(probes: Dict[str, np.ndarray], dimension: str) -> np.ndarray:
    """
    Compute direction for a dimension as: mean(high) - mean(low).
    Ignores neutral emotions.
    """
    high_probes = []
    low_probes = []

    for emotion, probe in probes.items():
        if emotion not in EMOTION_DIMENSIONS:
            continue

        value = EMOTION_DIMENSIONS[emotion].get(dimension)
        if value == "H":
            high_probes.append(probe)
        elif value == "L":
            low_probes.append(probe)

    if not high_probes or not low_probes:
        return None

    high_mean = np.mean(high_probes, axis=0)
    low_mean = np.mean(low_probes, axis=0)

    direction = high_mean - low_mean
    # Normalize to unit vector
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    return direction


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def analyze_entity(probes: Dict[str, np.ndarray], entity: str):
    """Analyze PCs vs dimensions for one entity."""
    print(f"\n{'='*80}")
    print(f"{entity.upper()} EMOTIONS")
    print(f"{'='*80}")

    # Fit PCA
    names = sorted(probes.keys())
    matrix = np.stack([probes[k] for k in names])

    pca = PCA(n_components=4)
    pca.fit(matrix)

    print(f"\nPCA Variance Explained:")
    for i in range(4):
        print(f"  PC{i+1}: {pca.explained_variance_ratio_[i]*100:.1f}%")

    # Compute dimension directions
    print(f"\nComputing psychological dimension directions...")
    dimension_dirs = {}
    for dim_code in ["V", "A", "D", "AA"]:
        direction = compute_dimension_direction(probes, dim_code)
        if direction is not None:
            dimension_dirs[dim_code] = direction

            # Count emotions
            high = sum(1 for e in probes.keys()
                      if e in EMOTION_DIMENSIONS and EMOTION_DIMENSIONS[e].get(dim_code) == "H")
            low = sum(1 for e in probes.keys()
                     if e in EMOTION_DIMENSIONS and EMOTION_DIMENSIONS[e].get(dim_code) == "L")
            print(f"  {dim_code} ({DIMENSION_NAMES[dim_code]}): {high} high, {low} low emotions")

    # Compute similarities
    print(f"\n{'='*80}")
    print(f"COSINE SIMILARITIES: {entity} PCs vs Psychological Dimensions")
    print(f"{'='*80}\n")

    results = {}

    # Print header
    print(f"{'Dimension':<35} | {'PC1':>8} | {'PC2':>8} | {'PC3':>8} | {'PC4':>8} | {'Best PC':<10}")
    print(f"{'-'*35} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*10}")

    for dim_code in ["V", "A", "D", "AA"]:
        if dim_code not in dimension_dirs:
            continue

        dim_dir = dimension_dirs[dim_code]
        similarities = []

        for i in range(4):
            pc = pca.components_[i]
            sim = cosine_similarity(dim_dir, pc)
            similarities.append(sim)

        # Find best match
        abs_sims = [abs(s) for s in similarities]
        best_idx = np.argmax(abs_sims)
        best_pc = f"PC{best_idx+1}"
        best_sim = similarities[best_idx]

        results[dim_code] = {
            "similarities": [float(s) for s in similarities],
            "best_pc": best_pc,
            "best_similarity": float(best_sim)
        }

        # Print row
        dim_name = DIMENSION_NAMES[dim_code]
        print(f"{dim_name:<35} | {similarities[0]:>+8.3f} | {similarities[1]:>+8.3f} | "
              f"{similarities[2]:>+8.3f} | {similarities[3]:>+8.3f} | "
              f"{best_pc} ({best_sim:+.3f})")

    print(f"{'='*80}\n")

    # Compute pairwise similarities between dimensions
    print(f"PAIRWISE COSINE SIMILARITIES: {entity} Dimension Directions")
    print(f"{'='*80}\n")

    dim_codes = ["V", "A", "D", "AA"]
    dim_pairs = []

    print(f"{'Dimension Pair':<45} | {'Cosine Sim':>12}")
    print(f"{'-'*45} | {'-'*12}")

    for i, dim1 in enumerate(dim_codes):
        if dim1 not in dimension_dirs:
            continue
        for j, dim2 in enumerate(dim_codes):
            if j <= i or dim2 not in dimension_dirs:
                continue

            sim = cosine_similarity(dimension_dirs[dim1], dimension_dirs[dim2])
            pair_name = f"{DIMENSION_NAMES[dim1]} vs {DIMENSION_NAMES[dim2]}"

            print(f"{pair_name:<45} | {sim:>+12.3f}")
            dim_pairs.append({
                "dim1": dim1,
                "dim2": dim2,
                "similarity": float(sim)
            })

    print(f"{'='*80}\n")

    return results, dim_pairs


def main():
    probe_path = Path("probes/ua_emotion_disentangle/full_analysis/probes_first_asst_token_alternating.npz")

    print("="*80)
    print("COMPARING PCs TO PSYCHOLOGICAL DIMENSIONS")
    print("(Using SYMMETRIC/ALTERNATING orthogonalization)")
    print("="*80)
    print(f"\nLoading probes from: {probe_path}")

    # Load probes
    probes = load_probes(probe_path)
    m_probes, u_probes = separate_probes(probes)

    print(f"✓ Loaded {len(m_probes)} M probes, {len(u_probes)} U probes")

    # Analyze both entities
    m_results, m_dim_pairs = analyze_entity(m_probes, "M (Assistant)")
    u_results, u_dim_pairs = analyze_entity(u_probes, "U (User)")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print("\nM (Assistant) - Best PC matches:")
    for dim in ["V", "A", "D", "AA"]:
        if dim in m_results:
            r = m_results[dim]
            print(f"  {DIMENSION_NAMES[dim]:<35} → {r['best_pc']} (similarity: {r['best_similarity']:+.3f})")

    print("\nU (User) - Best PC matches:")
    for dim in ["V", "A", "D", "AA"]:
        if dim in u_results:
            r = u_results[dim]
            print(f"  {DIMENSION_NAMES[dim]:<35} → {r['best_pc']} (similarity: {r['best_similarity']:+.3f})")

    print("\n" + "="*80)

    # Save results
    output_file = Path("probes/ua_emotion_disentangle/full_analysis/pc_dimension_comparison.json")
    with open(output_file, 'w') as f:
        json.dump({
            "M": {
                "pc_similarities": m_results,
                "dimension_pairs": m_dim_pairs
            },
            "U": {
                "pc_similarities": u_results,
                "dimension_pairs": u_dim_pairs
            },
            "dimension_names": DIMENSION_NAMES
        }, f, indent=2)

    print(f"✓ Results saved to: {output_file}")
    print("="*80)


if __name__ == "__main__":
    main()
