#!/usr/bin/env python3
"""Debug probe directions to understand the negative correlation."""

import pickle
import numpy as np
from pathlib import Path

def load_cpca_components(cpca_path, layer):
    """Load cPCA components for projection."""
    data = np.load(cpca_path, allow_pickle=True)
    components = data["components"][layer]  # [n_components, hidden_dim]
    return components

def get_probe_weights(probe_path):
    """Get raw probe weights."""
    with open(probe_path, 'rb') as f:
        data = pickle.load(f)
    model = data['model']
    weights = model.weight.detach().cpu().numpy()
    return weights

LAYER = 30
EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# Paths
results_dir = Path("/workspace-vast/annas/git/research-tools/results/emotion_probes_multiseed")
cpca_path = Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/tier_based/google/google/gemma-3-27b-it_cpca.npz")

print("="*80)
print("DEBUG: Probe Direction Analysis")
print("="*80)
print()

# Load one probe from each setting
raw_probe_path = results_dir / f"probe_layer{LAYER}_nc0_seed0.pkl"
cpca5_probe_path = results_dir / f"probe_layer{LAYER}_nc5_seed0.pkl"
cpca10_probe_path = results_dir / f"probe_layer{LAYER}_nc10_seed0.pkl"
cpca20_probe_path = results_dir / f"probe_layer{LAYER}_nc20_seed0.pkl"

# Get weights
print("1. Raw probe weights (directly in activation space):")
raw_weights = get_probe_weights(raw_probe_path)
print(f"   Shape: {raw_weights.shape}")
print(f"   Norm per emotion: {np.linalg.norm(raw_weights, axis=1)}")
print()

print("2. cPCA probe weights (in cPCA space, need to project back):")
cpca5_weights = get_probe_weights(cpca5_probe_path)
print(f"   nc5 shape: {cpca5_weights.shape}")
print(f"   nc5 norm per emotion: {np.linalg.norm(cpca5_weights, axis=1)}")

cpca10_weights = get_probe_weights(cpca10_probe_path)
print(f"   nc10 shape: {cpca10_weights.shape}")
print(f"   nc10 norm per emotion: {np.linalg.norm(cpca10_weights, axis=1)}")

cpca20_weights = get_probe_weights(cpca20_probe_path)
print(f"   nc20 shape: {cpca20_weights.shape}")
print(f"   nc20 norm per emotion: {np.linalg.norm(cpca20_weights, axis=1)}")
print()

# Load cPCA components
print("3. Loading cPCA components:")
cpca_full = load_cpca_components(cpca_path, LAYER)
print(f"   Full cPCA shape: {cpca_full.shape}")

cpca5 = cpca_full[:5]
cpca10 = cpca_full[:10]
cpca20 = cpca_full[:20]

print(f"   Component norms (first 5): {np.linalg.norm(cpca5, axis=1)}")
print()

# Project cPCA probes back to full space
print("4. Projecting cPCA probes to full space:")
cpca5_full = cpca5_weights @ cpca5
cpca10_full = cpca10_weights @ cpca10
cpca20_full = cpca20_weights @ cpca20

print(f"   nc5 -> full: {cpca5_full.shape}")
print(f"   nc10 -> full: {cpca10_full.shape}")
print(f"   nc20 -> full: {cpca20_full.shape}")
print()

# Normalize all
print("5. Normalizing all probe directions:")
raw_norm = raw_weights / np.linalg.norm(raw_weights, axis=1, keepdims=True)
cpca5_norm = cpca5_full / np.linalg.norm(cpca5_full, axis=1, keepdims=True)
cpca10_norm = cpca10_full / np.linalg.norm(cpca10_full, axis=1, keepdims=True)
cpca20_norm = cpca20_full / np.linalg.norm(cpca20_full, axis=1, keepdims=True)

print("   All normalized to unit length")
print()

# Check similarities for "anger" emotion
print("6. Detailed comparison for ANGER emotion (index 0):")
print()

anger_raw = raw_norm[0]
anger_cpca5 = cpca5_norm[0]
anger_cpca10 = cpca10_norm[0]
anger_cpca20 = cpca20_norm[0]

print(f"   Raw vs nc5:  {np.dot(anger_raw, anger_cpca5):.4f}")
print(f"   Raw vs nc10: {np.dot(anger_raw, anger_cpca10):.4f}")
print(f"   Raw vs nc20: {np.dot(anger_raw, anger_cpca20):.4f}")
print()

# Check if flipping the sign helps
print("7. What if we flip the cPCA probe signs?:")
print(f"   Raw vs -nc5:  {np.dot(anger_raw, -anger_cpca5):.4f}")
print(f"   Raw vs -nc10: {np.dot(anger_raw, -anger_cpca10):.4f}")
print(f"   Raw vs -nc20: {np.dot(anger_raw, -anger_cpca20):.4f}")
print()

# Check component-wise analysis
print("8. Looking at the first 5 dimensions of each probe:")
print(f"   Raw (first 5 dims):   {raw_norm[0, :5]}")
print(f"   nc5 (first 5 dims):   {cpca5_norm[0, :5]}")
print(f"   nc10 (first 5 dims):  {cpca10_norm[0, :5]}")
print(f"   nc20 (first 5 dims):  {cpca20_norm[0, :5]}")
print()

# Check what the cPCA components look like
print("9. First cPCA component statistics:")
print(f"   Mean: {cpca5[0].mean():.6f}")
print(f"   Std:  {cpca5[0].std():.6f}")
print(f"   Min:  {cpca5[0].min():.6f}")
print(f"   Max:  {cpca5[0].max():.6f}")
print(f"   First 10 values: {cpca5[0, :10]}")
print()

# Project raw probe onto cPCA space
print("10. What if we project RAW probe onto cPCA space?")
raw_on_cpca5 = anger_raw @ cpca5.T  # [5]
raw_on_cpca10 = anger_raw @ cpca10.T  # [10]
raw_on_cpca20 = anger_raw @ cpca20.T  # [20]

print(f"   Raw projected onto nc5:  {raw_on_cpca5}")
print(f"   nc5 probe weights:       {cpca5_weights[0]}")
print(f"   Cosine similarity in cPCA space: {np.dot(raw_on_cpca5, cpca5_weights[0]) / (np.linalg.norm(raw_on_cpca5) * np.linalg.norm(cpca5_weights[0])):.4f}")
print()

print("11. Hypothesis check:")
print("   If cPCA inverts emotion directions, we'd expect:")
print("   - Negative correlation between raw and cPCA in FULL space")
print("   - This could happen if cPCA contrast is computed as (neutral - emotional)")
print("     instead of (emotional - neutral)")
print()

# Check all emotions
print("12. All emotions - Raw vs cPCA (sign-ambiguity aware):")
for i, emotion in enumerate(EMOTIONS):
    raw_vec = raw_norm[i]
    cpca20_vec = cpca20_norm[i]

    sim_pos = np.dot(raw_vec, cpca20_vec)
    sim_neg = np.dot(raw_vec, -cpca20_vec)

    best_sim = max(abs(sim_pos), abs(sim_neg))
    sign = "+" if abs(sim_pos) > abs(sim_neg) else "-"

    print(f"   {emotion:12s}: {sim_pos:7.4f} (as is), {sim_neg:7.4f} (flipped) -> best: {best_sim:.4f} [{sign}]")

print()
print("="*80)
