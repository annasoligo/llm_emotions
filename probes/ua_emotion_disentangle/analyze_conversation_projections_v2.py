#!/usr/bin/env python3
"""
Analyze projections onto PCs and dimensions throughout a conversation.
V2: Uses simple cosine similarity normalization.

Usage:
    python analyze_conversation_projections_v2.py --layer 30 --method opposite --orthogonalize
    python analyze_conversation_projections_v2.py --layer 30 --method opposite  # raw
    python analyze_conversation_projections_v2.py --layer 30 --method contrast_all --orthogonalize
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
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


def compute_emotion_probes_opposite(activations: np.ndarray, metadata: list) -> tuple:
    """Compute emotion probes as differences between opposite pairs."""
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

    m_probes = {}
    u_probes = {}
    for emotion, opposite in OPPOSITE_PAIRS.items():
        if emotion in m_means and opposite in m_means:
            m_probes[emotion] = m_means[emotion] - m_means[opposite]
        if emotion in u_means and opposite in u_means:
            u_probes[emotion] = u_means[emotion] - u_means[opposite]

    return m_probes, u_probes


def compute_emotion_probes_contrast_all(activations: np.ndarray, metadata: list) -> tuple:
    """Compute emotion probes as: emotion - mean(all other emotions)."""
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

    m_probes = {}
    u_probes = {}

    m_emotions = sorted(m_means.keys())
    for target in m_emotions:
        others = [m_means[e] for e in m_emotions if e != target]
        if others:
            m_probes[target] = m_means[target] - np.mean(others, axis=0)

    u_emotions = sorted(u_means.keys())
    for target in u_emotions:
        others = [u_means[e] for e in u_emotions if e != target]
        if others:
            u_probes[target] = u_means[target] - np.mean(others, axis=0)

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


def compute_projections_v2(m_probes, u_probes, activations, m_pcs, u_pcs, m_dims, u_dims, roles):
    """
    Compute projections using SIMPLE COSINE SIMILARITY.

    For each activation and probe:
        score = dot(activation, probe) / (||activation|| * ||probe||)

    This gives similarity in [-1, 1] regardless of magnitude.
    """
    results = {
        'M_pc1': [], 'M_pc2': [], 'M_pc3': [], 'M_pc4': [],
        'U_pc1': [], 'U_pc2': [], 'U_pc3': [], 'U_pc4': [],
        'M_valence': [], 'M_arousal': [], 'M_dominance': [], 'M_approach_avoidance': [],
        'U_valence': [], 'U_arousal': [], 'U_dominance': [], 'U_approach_avoidance': [],
        'roles': roles
    }

    for emotion in DISCRETE_EMOTIONS:
        results[f'M_{emotion}'] = []
        results[f'U_{emotion}'] = []

    # Normalize all probe directions once
    m_probes_norm = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in m_probes.items()}
    u_probes_norm = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in u_probes.items()}
    m_pcs_norm = np.array([pc / (np.linalg.norm(pc) + 1e-8) for pc in m_pcs])
    u_pcs_norm = np.array([pc / (np.linalg.norm(pc) + 1e-8) for pc in u_pcs])
    m_dims_norm = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in m_dims.items()}
    u_dims_norm = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in u_dims.items()}

    for i, act in enumerate(activations):
        # Normalize activation
        act_norm = np.linalg.norm(act)
        if act_norm < 1e-8:
            act_normalized = act
        else:
            act_normalized = act / act_norm

        # Simple cosine similarity with all probes

        # M PCs
        for j in range(4):
            score = np.dot(act_normalized, m_pcs_norm[j])
            results[f'M_pc{j+1}'].append(score)

        # U PCs
        for j in range(4):
            score = np.dot(act_normalized, u_pcs_norm[j])
            results[f'U_pc{j+1}'].append(score)

        # M dimensions
        for dim_name, dim_key in [('valence', 'V'), ('arousal', 'A'),
                                   ('dominance', 'D'), ('approach_avoidance', 'AA')]:
            if dim_key in m_dims_norm:
                score = np.dot(act_normalized, m_dims_norm[dim_key])
                results[f'M_{dim_name}'].append(score)
            else:
                results[f'M_{dim_name}'].append(0.0)

        # U dimensions
        for dim_name, dim_key in [('valence', 'V'), ('arousal', 'A'),
                                   ('dominance', 'D'), ('approach_avoidance', 'AA')]:
            if dim_key in u_dims_norm:
                score = np.dot(act_normalized, u_dims_norm[dim_key])
                results[f'U_{dim_name}'].append(score)
            else:
                results[f'U_{dim_name}'].append(0.0)

        # Discrete emotions
        for emotion in DISCRETE_EMOTIONS:
            if emotion in m_probes_norm:
                score = np.dot(act_normalized, m_probes_norm[emotion])
                results[f'M_{emotion}'].append(score)
            else:
                results[f'M_{emotion}'].append(0.0)

            if emotion in u_probes_norm:
                score = np.dot(act_normalized, u_probes_norm[emotion])
                results[f'U_{emotion}'].append(score)
            else:
                results[f'U_{emotion}'].append(0.0)

    return results


def plot_projections(projections, output_dir, title_suffix=""):
    """Create plots for all projections."""

    # M PCs
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle(f'Assistant (M) - PC Projections {title_suffix}\n(Cosine Similarity, First 5 tokens excluded)',
                 fontsize=16, fontweight='bold')

    for i in range(4):
        ax = axes[i]
        positions, smoothed, neutral = smooth_projections(projections[f'M_pc{i+1}'], window=100, skip_first_n=5)
        ax.plot(positions, smoothed, linewidth=2, color=f'C{i}')
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({neutral:.3f})')
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title(f'PC{i+1}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'm_pc_projections.png', dpi=150, bbox_inches='tight')
    plt.close()

    # U PCs
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle(f'User (U) - PC Projections {title_suffix}\n(Cosine Similarity, First 5 tokens excluded)',
                 fontsize=16, fontweight='bold')

    for i in range(4):
        ax = axes[i]
        positions, smoothed, neutral = smooth_projections(projections[f'U_pc{i+1}'], window=100, skip_first_n=5)
        ax.plot(positions, smoothed, linewidth=2, color=f'C{i}')
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({neutral:.3f})')
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title(f'PC{i+1}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'u_pc_projections.png', dpi=150, bbox_inches='tight')
    plt.close()

    # M Dimensions
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle(f'Assistant (M) - Psychological Dimensions {title_suffix}\n(Cosine Similarity, First 5 tokens excluded)',
                 fontsize=16, fontweight='bold')

    dim_names = ['valence', 'arousal', 'dominance', 'approach_avoidance']
    dim_labels = ['Valence', 'Arousal', 'Dominance', 'Approach-Avoidance']

    for i, (dim_name, dim_label) in enumerate(zip(dim_names, dim_labels)):
        ax = axes[i]
        positions, smoothed, neutral = smooth_projections(projections[f'M_{dim_name}'], window=100, skip_first_n=5)
        ax.plot(positions, smoothed, linewidth=2, color=f'C{i}')
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({neutral:.3f})')
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title(dim_label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'm_dimension_projections.png', dpi=150, bbox_inches='tight')
    plt.close()

    # U Dimensions
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle(f'User (U) - Psychological Dimensions {title_suffix}\n(Cosine Similarity, First 5 tokens excluded)',
                 fontsize=16, fontweight='bold')

    for i, (dim_name, dim_label) in enumerate(zip(dim_names, dim_labels)):
        ax = axes[i]
        positions, smoothed, neutral = smooth_projections(projections[f'U_{dim_name}'], window=100, skip_first_n=5)
        ax.plot(positions, smoothed, linewidth=2, color=f'C{i}')
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({neutral:.3f})')
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title(dim_label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'u_dimension_projections.png', dpi=150, bbox_inches='tight')
    plt.close()

    # M Discrete Emotions
    fig, axes = plt.subplots(8, 1, figsize=(20, 32))
    fig.suptitle(f'Assistant (M) - Discrete Emotions {title_suffix}\n(Cosine Similarity, First 5 tokens excluded)',
                 fontsize=16, fontweight='bold')

    for i, emotion in enumerate(DISCRETE_EMOTIONS):
        ax = axes[i]
        positions, smoothed, neutral = smooth_projections(projections[f'M_{emotion}'], window=100, skip_first_n=5)
        color = EMOTION_COLORS.get(emotion, f'C{i}')
        ax.plot(positions, smoothed, linewidth=2, color=color)
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({neutral:.3f})')
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title(emotion.capitalize(), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'm_emotion_projections.png', dpi=150, bbox_inches='tight')
    plt.close()

    # U Discrete Emotions
    fig, axes = plt.subplots(8, 1, figsize=(20, 32))
    fig.suptitle(f'User (U) - Discrete Emotions {title_suffix}\n(Cosine Similarity, First 5 tokens excluded)',
                 fontsize=16, fontweight='bold')

    for i, emotion in enumerate(DISCRETE_EMOTIONS):
        ax = axes[i]
        positions, smoothed, neutral = smooth_projections(projections[f'U_{emotion}'], window=100, skip_first_n=5)
        color = EMOTION_COLORS.get(emotion, f'C{i}')
        ax.plot(positions, smoothed, linewidth=2, color=color)
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({neutral:.3f})')
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Cosine Similarity', fontsize=12)
        ax.set_title(emotion.capitalize(), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'u_emotion_projections.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved PC, dimension, and discrete emotion plots to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, required=True, help='Layer to analyze (e.g., 20, 30, 40)')
    parser.add_argument('--method', choices=['opposite', 'contrast_all'], default='opposite',
                        help='Probe computation method')
    parser.add_argument('--orthogonalize', action='store_true',
                        help='Apply M-U orthogonalization')
    args = parser.parse_args()

    model_name = "unsloth/gemma-3-27b-it"
    device = "cuda"
    data_dir = Path("probes/ua_emotion_disentangle/data/activations/full_all_layers")
    conversation_file = Path("probes/ua_emotion_disentangle/countdown_conversation.txt")

    ortho_str = "orthogonalized" if args.orthogonalize else "raw"
    output_dir = Path(f"probes/ua_emotion_disentangle/conversation_analysis_v2_{args.method}_{ortho_str}_layer{args.layer}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"CONVERSATION ANALYSIS V2 (Simple Cosine Similarity)")
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

    # Compute probes
    print(f"\n3. Computing emotion probes ({args.method})...")
    layer_data = load_layer_data(args.layer, data_dir)
    activations_data = layer_data['activations']['first_asst_token']
    metadata = layer_data['metadata']['first_asst_token']

    if args.method == 'opposite':
        m_probes, u_probes = compute_emotion_probes_opposite(activations_data, metadata)
    else:
        m_probes, u_probes = compute_emotion_probes_contrast_all(activations_data, metadata)

    # Optionally orthogonalize
    if args.orthogonalize:
        print("   Applying alternating orthogonalization...")
        m_probes, u_probes = alternating_orthogonalization(m_probes, u_probes)

    print(f"   ✓ Computed {len(m_probes)} M probes, {len(u_probes)} U probes")

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

    # Extract activations
    print("\n5. Extracting activations...")
    activations, tokens, roles = extract_activations_for_conversation(
        model, tokenizer, conversation, args.layer
    )
    print(f"   ✓ Extracted {len(activations)} tokens")

    # Compute projections with V2 method
    print("\n6. Computing projections (cosine similarity)...")
    projections = compute_projections_v2(m_probes, u_probes, activations,
                                        m_pcs, u_pcs, m_dims, u_dims, roles)
    print(f"   ✓ Computed projections")

    # Save projection data
    print("\n7. Saving projection data...")
    data_to_save = {
        'projections': projections,
        'config': {
            'layer': args.layer,
            'method': args.method,
            'orthogonalized': args.orthogonalize,
        },
        'tokens': tokens,
    }
    with open(output_dir / 'projection_data.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"   ✓ Saved projection data to {output_dir / 'projection_data.pkl'}")

    # Plot
    print("\n8. Creating plots...")
    title_suffix = f"(Layer {args.layer}, {args.method}, {ortho_str})"
    plot_projections(projections, output_dir, title_suffix)

    print("\n" + "="*80)
    print("✓ COMPLETE")
    print("="*80)
    print(f"Output: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
