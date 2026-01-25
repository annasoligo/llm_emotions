#!/usr/bin/env python3
"""
Analyze projections onto PCs and dimensions throughout a conversation.
RAW CONTRAST-ALL VERSION: No M-U orthogonalization, no dimension orthogonalization.

This shows the natural contamination and correlations before any orthogonalization.
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

# Import functions from the original script
sys.path.insert(0, str(Path(__file__).parent))
from analyze_conversation_projections import (
    compute_pcs,
    EMOTION_DIMENSIONS,
    compute_dimension_direction,
    parse_conversation,
    extract_activations_for_conversation,
    project_onto_subspace,
    compute_projections,
    plot_projections,
    plot_zoomed_projections,
    plot_discrete_emotions,
)


def load_layer_data(layer: int, data_dir: Path) -> Dict:
    """Load activation data for one layer."""
    layer_file = data_dir / f"layer_{layer}.pkl"
    with open(layer_file, 'rb') as f:
        return pickle.load(f)


def compute_raw_emotion_probes_contrast_all(
    activations: np.ndarray,
    metadata: list
) -> tuple:
    """
    Compute RAW emotion probes as: emotion - mean(all other emotions).
    NO ORTHOGONALIZATION applied.
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
            m_probes[target_emotion] = target_mean - others_mean

    # For U probes
    u_emotions = sorted(u_means.keys())
    for target_emotion in u_emotions:
        other_emotions = [emo for emo in u_emotions if emo != target_emotion]
        if other_emotions:
            target_mean = u_means[target_emotion]
            others_mean = np.mean([u_means[emo] for emo in other_emotions], axis=0)
            u_probes[target_emotion] = target_mean - others_mean

    return m_probes, u_probes


def main():
    # Configuration
    layer = 30
    model_name = "unsloth/gemma-3-27b-it"
    device = "cuda"

    # Load raw activation data
    data_dir = Path("probes/ua_emotion_disentangle/data/activations/full_all_layers")
    output_dir = Path("probes/ua_emotion_disentangle/conversation_analysis_raw_contrast_all")
    output_dir.mkdir(parents=True, exist_ok=True)

    conversation_file = Path("probes/ua_emotion_disentangle/countdown_conversation.txt")

    print("="*80)
    print("ANALYZING CONVERSATION PROJECTIONS")
    print("METHOD: CONTRAST-ALL (RAW - NO ORTHOGONALIZATION)")
    print("="*80)
    print(f"\nLayer: {layer}")
    print(f"Model: {model_name}")
    print("\n⚠️  NO M-U orthogonalization")
    print("⚠️  NO dimension orthogonalization")

    # Load conversation
    print("\n1. Loading conversation...")
    with open(conversation_file, 'r') as f:
        conversation_text = f.read()

    conversation = parse_conversation(conversation_text)
    print(f"   Found {len(conversation)} turns")

    # Load model
    print("\n2. Loading model...")
    model_raw = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model = StandardizedTransformer(
        model_raw,
        trust_remote_code=True,
        check_renaming=False,
        allow_dispatch=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("   ✓ Model loaded")

    # Compute RAW probes (no orthogonalization)
    print("\n3. Computing RAW emotion probes (contrast-all, no orthogonalization)...")
    layer_data = load_layer_data(layer, data_dir)
    activations_data = layer_data['activations']['first_asst_token']
    metadata = layer_data['metadata']['first_asst_token']

    m_probes, u_probes = compute_raw_emotion_probes_contrast_all(activations_data, metadata)
    print(f"   ✓ Computed {len(m_probes)} M probes, {len(u_probes)} U probes")

    # Compute contamination
    M = np.stack([m_probes[k] for k in sorted(m_probes.keys())])
    U = np.stack([u_probes[k] for k in sorted(u_probes.keys())])
    M_norm = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-8)
    contamination = float(np.mean(np.abs(M_norm @ U_norm.T)))
    print(f"   M-U contamination: {contamination:.6f}")

    # Compute PCs and dimensions
    print("\n4. Computing PCs and RAW dimensions...")
    m_pcs = compute_pcs(m_probes, n_components=4)
    u_pcs = compute_pcs(u_probes, n_components=4)

    # Compute RAW dimension directions (no orthogonalization)
    m_dims = {}
    u_dims = {}
    for dim_code in ['V', 'A', 'D', 'AA']:
        m_dir = compute_dimension_direction(m_probes, dim_code)
        u_dir = compute_dimension_direction(u_probes, dim_code)
        if m_dir is not None:
            m_dims[dim_code] = m_dir
        if u_dir is not None:
            u_dims[dim_code] = u_dir

    # Check dimension correlations
    print("   Checking M dimension correlations (RAW):")
    for dim1 in ['V', 'A', 'D', 'AA']:
        if dim1 not in m_dims:
            continue
        for dim2 in ['V', 'A', 'D', 'AA']:
            if dim2 <= dim1 or dim2 not in m_dims:
                continue
            dot = np.dot(m_dims[dim1], m_dims[dim2])
            print(f"     {dim1} · {dim2} = {dot:.6f}")

    print(f"   ✓ Computed PCs and RAW (correlated) dimensions")

    # Extract activations
    print("\n5. Extracting activations...")
    activations, tokens, roles = extract_activations_for_conversation(
        model, tokenizer, conversation, layer
    )
    print(f"   ✓ Extracted {len(activations)} token activations")

    # Compute projections
    print("\n6. Computing projections...")
    projections = compute_projections(m_probes, u_probes, activations, m_pcs, u_pcs, m_dims, u_dims, roles)
    print(f"   ✓ Computed projections")

    # Plot
    print("\n7. Creating plots...")
    plot_projections(projections, output_dir)

    print("\n8. Creating zoomed-in plots...")
    plot_zoomed_projections(projections, output_dir, start_token=5000, window_size=200)

    print("\n9. Creating discrete emotion plots...")
    plot_discrete_emotions(projections, output_dir)

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total tokens: {len(activations)}")
    print(f"M-U contamination: {contamination:.6f}")
    print("="*80)


if __name__ == "__main__":
    main()
