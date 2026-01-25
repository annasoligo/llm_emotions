#!/usr/bin/env python3
"""
Analyze projections onto PCs and dimensions throughout a conversation.
V6: GLOBAL MEAN + NEUTRAL PROJECTION + BASELINE STANDARDIZATION.

Key changes:
1. Center all emotion means by subtracting global mean first
2. Project out neutral activation component (using Alpaca baseline)
3. Compute baseline mean/std for each emotion on neutral data
4. Standardize projections: (proj - baseline_mean) / baseline_std

Usage:
    python analyze_conversation_projections_v6.py --layer 30 --method opposite --orthogonalize
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnterp import StandardizedTransformer
from typing import List, Dict
import pickle
import sys
import argparse

# Import base functions
sys.path.insert(0, str(Path(__file__).parent))
from analyze_conversation_projections import (
    EMOTION_DIMENSIONS,
    EMOTION_COLORS,
    DISCRETE_EMOTIONS,
    parse_conversation,
    extract_activations_for_conversation,
    smooth_projections,
)
from config import FULL_EMOTIONS as OPPOSITE_PAIRS


def load_layer_data(layer: int, data_dir: Path) -> Dict:
    """Load activation data for one layer."""
    layer_file = data_dir / f"layer_{layer}.pkl"
    with open(layer_file, 'rb') as f:
        return pickle.load(f)


def load_neutral_activations(layer: int, baseline_dir: Path, max_samples: int = 5000, split_for_stats: bool = True):
    """
    Load neutral activations from Alpaca baseline for projecting out neutral component.

    Args:
        layer: Layer number
        baseline_dir: Path to baseline activations directory
        max_samples: Maximum number of neutral samples to use
        split_for_stats: If True, return (projection_data, stats_data) split 50/50
                         If False, return all data as single array

    Returns:
        If split_for_stats=True: Tuple of (projection_acts, stats_acts)
        If split_for_stats=False: Single array of neutral activations
    """
    h5_file = baseline_dir / f"layer{layer}_activations.h5"

    if not h5_file.exists():
        raise FileNotFoundError(f"Neutral baseline not found: {h5_file}")

    print(f"   Loading neutral activations from: {h5_file.name}")

    with h5py.File(h5_file, 'r') as f:
        # Use 'first_assistant_token' aggregation (same as probe training data)
        if 'first_assistant_token' in f:
            neutral_acts = f['first_assistant_token'][:]
        elif 'all_tokens' in f:
            neutral_acts = f['all_tokens'][:]
        else:
            # Fall back to first available key
            key = list(f.keys())[0]
            neutral_acts = f[key][:]
            print(f"   Using aggregation: {key}")

    # Limit to max_samples for efficiency
    if len(neutral_acts) > max_samples:
        indices = np.random.choice(len(neutral_acts), max_samples, replace=False)
        neutral_acts = neutral_acts[indices]

    if split_for_stats:
        # Split 50/50: first half for projection, second half for stats
        split_idx = len(neutral_acts) // 2
        neutral_acts_proj = neutral_acts[:split_idx]
        neutral_acts_stats = neutral_acts[split_idx:]
        print(f"   Loaded {len(neutral_acts)} neutral samples, split into {len(neutral_acts_proj)} (projection) + {len(neutral_acts_stats)} (stats)")
        return neutral_acts_proj, neutral_acts_stats
    else:
        print(f"   Loaded {len(neutral_acts)} neutral samples, shape: {neutral_acts.shape}")
        return neutral_acts


def project_out_neutral(probes: Dict[str, np.ndarray], neutral_acts: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Project out the component of each probe that aligns with neutral activations.

    For each probe:
        coeffs = lstsq(U_neutral, probe).solution
        probe_clean = probe - U_neutral @ coeffs

    This ensures probe ≈ 0 on neutral content.

    Args:
        probes: Dictionary of emotion name -> probe vector
        neutral_acts: Matrix U of neutral activations (n_samples, hidden_dim)

    Returns:
        Cleaned probes with neutral component projected out
    """
    print(f"   Projecting out neutral component from {len(probes)} probes...")

    # Convert to torch for lstsq
    U = torch.tensor(neutral_acts, dtype=torch.float32)

    probes_clean = {}
    norms_before = []
    norms_after = []

    for emotion, probe in probes.items():
        probe_torch = torch.tensor(probe, dtype=torch.float32)

        # Solve: U.T @ coeffs = probe (find how much of probe is explained by neutral)
        # U is (n_samples, hidden_dim), we want U.T @ coeffs = probe where probe is (hidden_dim,)
        coeffs = torch.linalg.lstsq(U.T, probe_torch).solution

        # Remove neutral component
        neutral_component = U.T @ coeffs
        probe_clean = probe_torch - neutral_component

        probes_clean[emotion] = probe_clean.numpy()

        norms_before.append(np.linalg.norm(probe))
        norms_after.append(np.linalg.norm(probe_clean.numpy()))

    avg_norm_before = np.mean(norms_before)
    avg_norm_after = np.mean(norms_after)
    reduction = (1 - avg_norm_after / avg_norm_before) * 100

    print(f"   Probe norm: {avg_norm_before:.1f} → {avg_norm_after:.1f} ({reduction:.1f}% reduction)")

    return probes_clean


def compute_baseline_statistics(m_probes, u_probes, neutral_acts, m_pcs, u_pcs, m_dims, u_dims):
    """
    Compute baseline mean and std for each emotion/PC/dimension on neutral data.

    For each probe, project neutral activations and compute statistics.
    Returns dictionaries with mean and std for standardization.

    Args:
        m_probes, u_probes: Cleaned emotion probes (after neutral projection)
        neutral_acts: Neutral baseline activations
        m_pcs, u_pcs: PC directions
        m_dims, u_dims: Dimension directions

    Returns:
        Tuple of (m_stats, u_stats) where each is a dict with keys like:
            'emotion_mean', 'emotion_std', 'pc1_mean', 'pc1_std', etc.
    """
    print("   Computing baseline statistics on neutral data...")

    # Build subspace bases
    m_probe_matrix = np.stack([m_probes[k] for k in sorted(m_probes.keys())])
    u_probe_matrix = np.stack([u_probes[k] for k in sorted(u_probes.keys())])
    m_basis_Q, _ = np.linalg.qr(m_probe_matrix.T)
    u_basis_Q, _ = np.linalg.qr(u_probe_matrix.T)

    # Normalize probes
    m_probes_norm = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in m_probes.items()}
    u_probes_norm = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in u_probes.items()}
    m_pcs_norm = np.array([pc / (np.linalg.norm(pc) + 1e-8) for pc in m_pcs])
    u_pcs_norm = np.array([pc / (np.linalg.norm(pc) + 1e-8) for pc in u_pcs])
    m_dims_norm = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in m_dims.items()}
    u_dims_norm = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in u_dims.items()}

    m_stats = {}
    u_stats = {}

    # Compute projections for all neutral activations
    from analyze_conversation_projections import DISCRETE_EMOTIONS

    # M emotions
    for emotion in DISCRETE_EMOTIONS:
        if emotion not in m_probes_norm:
            continue
        projections = []
        for act in neutral_acts:
            m_proj = act @ m_basis_Q @ m_basis_Q.T
            m_norm = np.linalg.norm(m_proj)
            if m_norm > 1e-8:
                m_proj_normalized = m_proj / m_norm
            else:
                m_proj_normalized = m_proj
            sim = np.dot(m_proj_normalized, m_probes_norm[emotion])
            projections.append(sim)
        m_stats[f'{emotion}_mean'] = np.mean(projections)
        m_stats[f'{emotion}_std'] = np.std(projections) + 1e-8

    # M PCs
    for i in range(4):
        projections = []
        for act in neutral_acts:
            m_proj = act @ m_basis_Q @ m_basis_Q.T
            m_norm = np.linalg.norm(m_proj)
            if m_norm > 1e-8:
                m_proj_normalized = m_proj / m_norm
            else:
                m_proj_normalized = m_proj
            sim = np.dot(m_proj_normalized, m_pcs_norm[i])
            projections.append(sim)
        m_stats[f'pc{i+1}_mean'] = np.mean(projections)
        m_stats[f'pc{i+1}_std'] = np.std(projections) + 1e-8

    # M Dimensions
    dim_keys = ['V', 'A', 'D', 'AA']
    dim_names = ['valence', 'arousal', 'dominance', 'approach_avoidance']
    for dim_name, dim_key in zip(dim_names, dim_keys):
        if dim_key not in m_dims_norm:
            continue
        projections = []
        for act in neutral_acts:
            m_proj = act @ m_basis_Q @ m_basis_Q.T
            m_norm = np.linalg.norm(m_proj)
            if m_norm > 1e-8:
                m_proj_normalized = m_proj / m_norm
            else:
                m_proj_normalized = m_proj
            sim = np.dot(m_proj_normalized, m_dims_norm[dim_key])
            projections.append(sim)
        m_stats[f'{dim_name}_mean'] = np.mean(projections)
        m_stats[f'{dim_name}_std'] = np.std(projections) + 1e-8

    # U emotions (same process)
    for emotion in DISCRETE_EMOTIONS:
        if emotion not in u_probes_norm:
            continue
        projections = []
        for act in neutral_acts:
            u_proj = act @ u_basis_Q @ u_basis_Q.T
            u_norm = np.linalg.norm(u_proj)
            if u_norm > 1e-8:
                u_proj_normalized = u_proj / u_norm
            else:
                u_proj_normalized = u_proj
            sim = np.dot(u_proj_normalized, u_probes_norm[emotion])
            projections.append(sim)
        u_stats[f'{emotion}_mean'] = np.mean(projections)
        u_stats[f'{emotion}_std'] = np.std(projections) + 1e-8

    # U PCs
    for i in range(4):
        projections = []
        for act in neutral_acts:
            u_proj = act @ u_basis_Q @ u_basis_Q.T
            u_norm = np.linalg.norm(u_proj)
            if u_norm > 1e-8:
                u_proj_normalized = u_proj / u_norm
            else:
                u_proj_normalized = u_proj
            sim = np.dot(u_proj_normalized, u_pcs_norm[i])
            projections.append(sim)
        u_stats[f'pc{i+1}_mean'] = np.mean(projections)
        u_stats[f'pc{i+1}_std'] = np.std(projections) + 1e-8

    # U Dimensions
    for dim_name, dim_key in zip(dim_names, dim_keys):
        if dim_key not in u_dims_norm:
            continue
        projections = []
        for act in neutral_acts:
            u_proj = act @ u_basis_Q @ u_basis_Q.T
            u_norm = np.linalg.norm(u_proj)
            if u_norm > 1e-8:
                u_proj_normalized = u_proj / u_norm
            else:
                u_proj_normalized = u_proj
            sim = np.dot(u_proj_normalized, u_dims_norm[dim_key])
            projections.append(sim)
        u_stats[f'{dim_name}_mean'] = np.mean(projections)
        u_stats[f'{dim_name}_std'] = np.std(projections) + 1e-8

    print(f"   ✓ Computed baseline stats for {len(m_stats)//2} M probes, {len(u_stats)//2} U probes")

    return m_stats, u_stats


def compute_emotion_probes_opposite_centered(activations: np.ndarray, metadata: list) -> tuple:
    """
    V4: Compute emotion probes with GLOBAL MEAN CENTERING.

    Steps:
    1. Compute global mean across ALL activations
    2. Center each emotion mean: emotion_centered = emotion_mean - global_mean
    3. Compute probes from centered means: emotion_centered - opposite_centered
    """
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

    # Compute emotion means
    m_means = {emotion: np.mean(acts, axis=0) for emotion, acts in m_emotion_groups.items()}
    u_means = {emotion: np.mean(acts, axis=0) for emotion, acts in u_emotion_groups.items()}

    # V4 KEY CHANGE: Compute global mean and center
    global_mean = np.mean(activations, axis=0)
    print(f"   Global mean norm: {np.linalg.norm(global_mean):.1f}")

    m_means_centered = {emotion: mean - global_mean for emotion, mean in m_means.items()}
    u_means_centered = {emotion: mean - global_mean for emotion, mean in u_means.items()}

    # Check centering worked
    avg_centered_norm = np.mean([np.linalg.norm(v) for v in m_means_centered.values()])
    print(f"   Average centered emotion norm: {avg_centered_norm:.1f} (was ~50000)")

    # Compute probes from centered means
    m_probes = {}
    u_probes = {}
    for emotion, opposite in OPPOSITE_PAIRS.items():
        if emotion in m_means_centered and opposite in m_means_centered:
            m_probes[emotion] = m_means_centered[emotion] - m_means_centered[opposite]
        if emotion in u_means_centered and opposite in u_means_centered:
            u_probes[emotion] = u_means_centered[emotion] - u_means_centered[opposite]

    return m_probes, u_probes


def compute_emotion_probes_contrast_all_centered(activations: np.ndarray, metadata: list) -> tuple:
    """V4: Compute contrast-all probes with global mean centering."""
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

    m_means = {emotion: np.mean(acts, axis=0) for emotion, acts in m_emotion_groups.items()}
    u_means = {emotion: np.mean(acts, axis=0) for emotion, acts in u_emotion_groups.items()}

    # V4: Center by global mean
    global_mean = np.mean(activations, axis=0)
    print(f"   Global mean norm: {np.linalg.norm(global_mean):.1f}")

    m_means_centered = {emotion: mean - global_mean for emotion, mean in m_means.items()}
    u_means_centered = {emotion: mean - global_mean for emotion, mean in u_means.items()}

    avg_centered_norm = np.mean([np.linalg.norm(v) for v in m_means_centered.values()])
    print(f"   Average centered emotion norm: {avg_centered_norm:.1f}")

    # Compute contrast-all from centered means
    m_probes = {}
    u_probes = {}

    m_emotions = sorted(m_means_centered.keys())
    for target in m_emotions:
        others = [m_means_centered[e] for e in m_emotions if e != target]
        if others:
            m_probes[target] = m_means_centered[target] - np.mean(others, axis=0)

    u_emotions = sorted(u_means_centered.keys())
    for target in u_emotions:
        others = [u_means_centered[e] for e in u_emotions if e != target]
        if others:
            u_probes[target] = u_means_centered[target] - np.mean(others, axis=0)

    return m_probes, u_probes


def alternating_orthogonalization(m_probes, u_probes, max_iters=10, damping=0.5):
    """Apply alternating orthogonalization to M and U probes."""
    def project_away(vectors, basis_vectors):
        Q, R = np.linalg.qr(basis_vectors.T)
        orthonormal_basis = Q.T
        projection = vectors @ orthonormal_basis.T @ orthonormal_basis
        return vectors - projection

    m_names = sorted(m_probes.keys())
    u_names = sorted(u_probes.keys())
    M = np.stack([m_probes[k] for k in m_names])
    U = np.stack([u_probes[k] for k in u_names])

    M_current = M.copy()
    U_current = U.copy()

    for iteration in range(max_iters):
        U_new = project_away(U_current, M_current)
        U_current = damping * U_current + (1 - damping) * U_new

        M_new = project_away(M_current, U_current)
        M_current = damping * M_current + (1 - damping) * M_new

    m_clean = {name: M_current[i] for i, name in enumerate(m_names)}
    u_clean = {name: U_current[i] for i, name in enumerate(u_names)}

    return m_clean, u_clean


def compute_pcs(emotion_vectors: Dict[str, np.ndarray], n_components: int = 4):
    """Compute PCA on emotion vectors."""
    from sklearn.decomposition import PCA
    names = sorted(emotion_vectors.keys())
    matrix = np.stack([emotion_vectors[name] for name in names])
    pca = PCA(n_components=n_components)
    pca.fit(matrix)
    return pca.components_


def compute_dimension_direction(emotion_vectors: Dict[str, np.ndarray], dimension: str):
    """Compute direction for a psychological dimension."""
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
    return direction / (np.linalg.norm(direction) + 1e-8)


def compute_projections_v6(m_probes, u_probes, activations, m_pcs, u_pcs, m_dims, u_dims, roles,
                          m_baseline_stats, u_baseline_stats):
    """
    V6: Project onto subspace, normalize, compute cosine similarities, then standardize
    using baseline mean/std computed on neutral data.

    Standardization: (sim - baseline_mean) / baseline_std
    """
    # Build subspace bases
    m_probe_matrix = np.stack([m_probes[k] for k in sorted(m_probes.keys())])
    u_probe_matrix = np.stack([u_probes[k] for k in sorted(u_probes.keys())])

    m_basis_Q, _ = np.linalg.qr(m_probe_matrix.T)
    u_basis_Q, _ = np.linalg.qr(u_probe_matrix.T)

    # Normalize all probe directions
    m_probes_norm = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in m_probes.items()}
    u_probes_norm = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in u_probes.items()}
    m_pcs_norm = np.array([pc / (np.linalg.norm(pc) + 1e-8) for pc in m_pcs])
    u_pcs_norm = np.array([pc / (np.linalg.norm(pc) + 1e-8) for pc in u_pcs])
    m_dims_norm = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in m_dims.items()}
    u_dims_norm = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in u_dims.items()}

    results = {
        'M_pc1': [], 'M_pc2': [], 'M_pc3': [], 'M_pc4': [],
        'U_pc1': [], 'U_pc2': [], 'U_pc3': [], 'U_pc4': [],
        'M_valence': [], 'M_arousal': [], 'M_dominance': [], 'M_approach_avoidance': [],
        'U_valence': [], 'U_arousal': [], 'U_dominance': [], 'U_approach_avoidance': [],
        'M_subspace_magnitude': [],
        'U_subspace_magnitude': [],
        'roles': roles
    }

    for emotion in DISCRETE_EMOTIONS:
        results[f'M_{emotion}'] = []
        results[f'U_{emotion}'] = []

    for act in activations:
        # === M (Assistant) ===
        m_projection = act @ m_basis_Q @ m_basis_Q.T
        m_proj_norm = np.linalg.norm(m_projection)
        results['M_subspace_magnitude'].append(m_proj_norm)

        if m_proj_norm > 1e-8:
            m_projection_normalized = m_projection / m_proj_norm
        else:
            m_projection_normalized = m_projection

        # Discrete emotions
        m_emotion_sims = []
        for emotion in DISCRETE_EMOTIONS:
            if emotion in m_probes_norm:
                sim = np.dot(m_projection_normalized, m_probes_norm[emotion])
                # V6: Apply baseline standardization
                mean = m_baseline_stats[f'{emotion}_mean']
                std = m_baseline_stats[f'{emotion}_std']
                standardized = (sim - mean) / std
                m_emotion_sims.append(standardized)
                results[f'M_{emotion}'].append(standardized)
            else:
                results[f'M_{emotion}'].append(0.0)

        # PCs
        m_pc_sims = []
        for j in range(4):
            sim = np.dot(m_projection_normalized, m_pcs_norm[j])
            # V6: Apply baseline standardization
            mean = m_baseline_stats[f'pc{j+1}_mean']
            std = m_baseline_stats[f'pc{j+1}_std']
            standardized = (sim - mean) / std
            m_pc_sims.append(standardized)
            results[f'M_pc{j+1}'].append(standardized)

        # Dimensions
        m_dim_sims = []
        dim_keys = ['V', 'A', 'D', 'AA']
        dim_names = ['valence', 'arousal', 'dominance', 'approach_avoidance']
        for dim_name, dim_key in zip(dim_names, dim_keys):
            if dim_key in m_dims_norm:
                sim = np.dot(m_projection_normalized, m_dims_norm[dim_key])
                # V6: Apply baseline standardization
                mean = m_baseline_stats[f'{dim_name}_mean']
                std = m_baseline_stats[f'{dim_name}_std']
                standardized = (sim - mean) / std
                m_dim_sims.append(standardized)
                results[f'M_{dim_name}'].append(standardized)
            else:
                results[f'M_{dim_name}'].append(0.0)

        # === U (User) - same process ===
        u_projection = act @ u_basis_Q @ u_basis_Q.T
        u_proj_norm = np.linalg.norm(u_projection)
        results['U_subspace_magnitude'].append(u_proj_norm)

        if u_proj_norm > 1e-8:
            u_projection_normalized = u_projection / u_proj_norm
        else:
            u_projection_normalized = u_projection

        u_emotion_sims = []
        for emotion in DISCRETE_EMOTIONS:
            if emotion in u_probes_norm:
                sim = np.dot(u_projection_normalized, u_probes_norm[emotion])
                # V6: Apply baseline standardization
                mean = u_baseline_stats[f'{emotion}_mean']
                std = u_baseline_stats[f'{emotion}_std']
                standardized = (sim - mean) / std
                u_emotion_sims.append(standardized)
                results[f'U_{emotion}'].append(standardized)
            else:
                results[f'U_{emotion}'].append(0.0)

        u_pc_sims = []
        for j in range(4):
            sim = np.dot(u_projection_normalized, u_pcs_norm[j])
            # V6: Apply baseline standardization
            mean = u_baseline_stats[f'pc{j+1}_mean']
            std = u_baseline_stats[f'pc{j+1}_std']
            standardized = (sim - mean) / std
            u_pc_sims.append(standardized)
            results[f'U_pc{j+1}'].append(standardized)

        u_dim_sims = []
        for dim_name, dim_key in zip(dim_names, dim_keys):
            if dim_key in u_dims_norm:
                sim = np.dot(u_projection_normalized, u_dims_norm[dim_key])
                # V6: Apply baseline standardization
                mean = u_baseline_stats[f'{dim_name}_mean']
                std = u_baseline_stats[f'{dim_name}_std']
                standardized = (sim - mean) / std
                u_dim_sims.append(standardized)
                results[f'U_{dim_name}'].append(standardized)
            else:
                results[f'U_{dim_name}'].append(0.0)

    return results


def plot_projections(projections, output_dir, title_suffix=""):
    """Create plots for all projections."""
    # Subspace magnitudes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8))
    fig.suptitle(f'Emotion Subspace Alignment Magnitudes {title_suffix}\n(First 5 tokens excluded)',
                 fontsize=16, fontweight='bold')

    positions_m, smoothed_m, neutral_m = smooth_projections(projections['M_subspace_magnitude'], window=100, skip_first_n=5)
    ax1.plot(positions_m, smoothed_m, linewidth=2, color='blue', label='M (Assistant)')
    ax1.axhline(y=neutral_m, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({neutral_m:.1f})')
    ax1.set_xlabel('Token Position', fontsize=12)
    ax1.set_ylabel('L2 Norm of Projection', fontsize=12)
    ax1.set_title('Assistant Emotion Subspace Alignment', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    positions_u, smoothed_u, neutral_u = smooth_projections(projections['U_subspace_magnitude'], window=100, skip_first_n=5)
    ax2.plot(positions_u, smoothed_u, linewidth=2, color='green', label='U (User)')
    ax2.axhline(y=neutral_u, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({neutral_u:.1f})')
    ax2.set_xlabel('Token Position', fontsize=12)
    ax2.set_ylabel('L2 Norm of Projection', fontsize=12)
    ax2.set_title('User Emotion Subspace Alignment', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'subspace_magnitudes.png', dpi=150, bbox_inches='tight')
    plt.close()

    # M PCs
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle(f'Assistant (M) - PC Projections {title_suffix}\n(V6: Baseline Standardized)',
                 fontsize=16, fontweight='bold')

    for i in range(4):
        ax = axes[i]
        positions, smoothed, neutral = smooth_projections(projections[f'M_pc{i+1}'], window=100, skip_first_n=5)
        ax.plot(positions, smoothed, linewidth=2, color=f'C{i}')
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({neutral:.3f})')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Standardized Score (σ)', fontsize=12)
        ax.set_title(f'PC{i+1}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'm_pc_projections.png', dpi=150, bbox_inches='tight')
    plt.close()

    # U PCs
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle(f'User (U) - PC Projections {title_suffix}\n(V6: Baseline Standardized)',
                 fontsize=16, fontweight='bold')

    for i in range(4):
        ax = axes[i]
        positions, smoothed, neutral = smooth_projections(projections[f'U_pc{i+1}'], window=100, skip_first_n=5)
        ax.plot(positions, smoothed, linewidth=2, color=f'C{i}')
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({neutral:.3f})')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Standardized Score (σ)', fontsize=12)
        ax.set_title(f'PC{i+1}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'u_pc_projections.png', dpi=150, bbox_inches='tight')
    plt.close()

    # M Dimensions
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle(f'Assistant (M) - Psychological Dimensions {title_suffix}\n(V6: Baseline Standardized)',
                 fontsize=16, fontweight='bold')

    dim_names = ['valence', 'arousal', 'dominance', 'approach_avoidance']
    dim_labels = ['Valence', 'Arousal', 'Dominance', 'Approach-Avoidance']

    for i, (dim_name, dim_label) in enumerate(zip(dim_names, dim_labels)):
        ax = axes[i]
        positions, smoothed, neutral = smooth_projections(projections[f'M_{dim_name}'], window=100, skip_first_n=5)
        ax.plot(positions, smoothed, linewidth=2, color=f'C{i}')
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({neutral:.3f})')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Standardized Score (σ)', fontsize=12)
        ax.set_title(dim_label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'm_dimension_projections.png', dpi=150, bbox_inches='tight')
    plt.close()

    # U Dimensions
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle(f'User (U) - Psychological Dimensions {title_suffix}\n(V6: Baseline Standardized)',
                 fontsize=16, fontweight='bold')

    for i, (dim_name, dim_label) in enumerate(zip(dim_names, dim_labels)):
        ax = axes[i]
        positions, smoothed, neutral = smooth_projections(projections[f'U_{dim_name}'], window=100, skip_first_n=5)
        ax.plot(positions, smoothed, linewidth=2, color=f'C{i}')
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({neutral:.3f})')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Standardized Score (σ)', fontsize=12)
        ax.set_title(dim_label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'u_dimension_projections.png', dpi=150, bbox_inches='tight')
    plt.close()

    # M Discrete Emotions
    fig, axes = plt.subplots(8, 1, figsize=(20, 32))
    fig.suptitle(f'Assistant (M) - Discrete Emotions {title_suffix}\n(V6: Baseline Standardized)',
                 fontsize=16, fontweight='bold')

    for i, emotion in enumerate(DISCRETE_EMOTIONS):
        ax = axes[i]
        positions, smoothed, neutral = smooth_projections(projections[f'M_{emotion}'], window=100, skip_first_n=5)
        color = EMOTION_COLORS.get(emotion, f'C{i}')
        ax.plot(positions, smoothed, linewidth=2, color=color)
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({neutral:.3f})')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Standardized Score (σ)', fontsize=12)
        ax.set_title(emotion.capitalize(), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'm_emotion_projections.png', dpi=150, bbox_inches='tight')
    plt.close()

    # U Discrete Emotions
    fig, axes = plt.subplots(8, 1, figsize=(20, 32))
    fig.suptitle(f'User (U) - Discrete Emotions {title_suffix}\n(V6: Baseline Standardized)',
                 fontsize=16, fontweight='bold')

    for i, emotion in enumerate(DISCRETE_EMOTIONS):
        ax = axes[i]
        positions, smoothed, neutral = smooth_projections(projections[f'U_{emotion}'], window=100, skip_first_n=5)
        color = EMOTION_COLORS.get(emotion, f'C{i}')
        ax.plot(positions, smoothed, linewidth=2, color=color)
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({neutral:.3f})')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Standardized Score (σ)', fontsize=12)
        ax.set_title(emotion.capitalize(), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'u_emotion_projections.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved all V6 plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, required=True, help='Layer to analyze')
    parser.add_argument('--method', choices=['opposite', 'contrast_all'], default='opposite',
                        help='Probe computation method')
    parser.add_argument('--orthogonalize', action='store_true',
                        help='Apply M-U orthogonalization')
    args = parser.parse_args()

    model_name = "unsloth/gemma-3-27b-it"
    device = "cuda"
    data_dir = Path("probes/ua_emotion_disentangle/data/activations/full_all_layers")
    conversation_file = Path("probes/ua_emotion_disentangle/countdown_conversation.txt")
    baseline_dir = Path("/workspace-vast/annas/git/research-tools/data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it")

    ortho_str = "orthogonalized" if args.orthogonalize else "raw"
    output_dir = Path(f"probes/ua_emotion_disentangle/conversation_analysis_v6_{args.method}_{ortho_str}_layer{args.layer}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"CONVERSATION ANALYSIS V6 (Global Mean + Neutral Projection + Raw Cosine)")
    print("="*80)
    print(f"Layer: {args.layer}")
    print(f"Method: {args.method}")
    print(f"Orthogonalization: {ortho_str}")

    # Load conversation
    print("\n1. Loading conversation...")
    with open(conversation_file, 'r') as f:
        conversation_text = f.read()
    conversation = parse_conversation(conversation_text)
    print(f"   Found {len(conversation)} turns")

    # Load model
    print("\n2. Loading model...")
    model_raw = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device,
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    model = StandardizedTransformer(model_raw, trust_remote_code=True,
                                   check_renaming=False, allow_dispatch=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✓ Model loaded")

    # Compute probes with V4 centering
    print(f"\n3. Computing emotion probes ({args.method}) with GLOBAL MEAN CENTERING...")
    layer_data = load_layer_data(args.layer, data_dir)
    activations_data = layer_data['activations']['first_asst_token']
    metadata = layer_data['metadata']['first_asst_token']

    if args.method == 'opposite':
        m_probes, u_probes = compute_emotion_probes_opposite_centered(activations_data, metadata)
    else:
        m_probes, u_probes = compute_emotion_probes_contrast_all_centered(activations_data, metadata)

    # V6: Load neutral data and split for projection vs stats
    print(f"\n3b. Projecting out neutral baseline (Alpaca)...")
    neutral_acts_proj, neutral_acts_stats = load_neutral_activations(args.layer, baseline_dir, max_samples=5000, split_for_stats=True)
    m_probes = project_out_neutral(m_probes, neutral_acts_proj)
    u_probes = project_out_neutral(u_probes, neutral_acts_proj)

    # Optionally orthogonalize
    if args.orthogonalize:
        print("\n3c. Applying alternating orthogonalization...")
        m_probes, u_probes = alternating_orthogonalization(m_probes, u_probes)

    print(f"   ✓ Final: {len(m_probes)} M probes, {len(u_probes)} U probes")

    # Compute PCs and dimensions
    print("\n4. Computing PCs and dimensions...")
    m_pcs = compute_pcs(m_probes, n_components=4)
    u_pcs = compute_pcs(u_probes, n_components=4)

    m_dims = {}
    u_dims = {}
    for dim_code in ['V', 'A', 'D', 'AA']:
        m_dir = compute_dimension_direction(m_probes, dim_code)
        u_dir = compute_dimension_direction(u_probes, dim_code)
        if m_dir is not None:
            m_dims[dim_code] = m_dir
        if u_dir is not None:
            u_dims[dim_code] = u_dir
    print(f"   ✓ Computed PCs and dimensions")

    # V6: Compute baseline statistics (using held-out neutral data)
    print("\n4b. Computing baseline statistics...")
    m_baseline_stats, u_baseline_stats = compute_baseline_statistics(
        m_probes, u_probes, neutral_acts_stats, m_pcs, u_pcs, m_dims, u_dims
    )

    # Extract activations
    print("\n5. Extracting activations...")
    activations, tokens, roles = extract_activations_for_conversation(
        model, tokenizer, conversation, args.layer
    )
    print(f"   ✓ Extracted {len(activations)} tokens")

    # Compute projections with V6 method (baseline standardized)
    print("\n6. Computing projections (baseline standardization)...")
    projections = compute_projections_v6(m_probes, u_probes, activations,
                                        m_pcs, u_pcs, m_dims, u_dims, roles,
                                        m_baseline_stats, u_baseline_stats)
    print(f"   ✓ Computed projections")

    # Save projection data
    print("\n7. Saving projection data...")
    data_to_save = {
        'projections': projections,
        'config': {
            'layer': args.layer,
            'method': args.method,
            'orthogonalized': args.orthogonalize,
            'version': 'v6_baseline_standardized',
        },
        'baseline_stats': {
            'M': m_baseline_stats,
            'U': u_baseline_stats,
        },
        'tokens': tokens,
    }
    with open(output_dir / 'projection_data.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"   ✓ Saved projection data with baseline stats")

    # Plot
    print("\n8. Creating plots...")
    title_suffix = f"(Layer {args.layer}, {args.method}, {ortho_str})"
    plot_projections(projections, output_dir, title_suffix)

    print("\n" + "="*80)
    print("✓ V6 COMPLETE")
    print("="*80)
    print(f"Output: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
