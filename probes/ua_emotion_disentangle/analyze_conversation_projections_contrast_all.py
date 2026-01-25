#!/usr/bin/env python3
"""
Analyze projections onto PCs and dimensions throughout a conversation.
CONTRAST-ALL VERSION: Uses probes computed as emotion - mean(all others).

For a given conversation at a specific layer:
1. Extract activations token-by-token
2. Project onto orthogonalized PC1-4 for M and U
3. Project onto psychological dimensions (V, A, D, AA)
4. Plot smoothed values (averaged over 100-token chunks)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnterp import StandardizedTransformer
from typing import List, Dict
import re
from sklearn.decomposition import PCA
import sys

# Import functions from the original script
sys.path.insert(0, str(Path(__file__).parent))
from analyze_conversation_projections import (
    load_orthogonal_probes,
    separate_probes,
    compute_pcs,
    EMOTION_DIMENSIONS,
    compute_dimension_direction,
    orthogonalize_dimensions,
    parse_conversation,
    extract_activations_for_conversation,
    project_onto_subspace,
    compute_projections,
    smooth_projections,
    plot_projections,
    plot_zoomed_projections,
    plot_discrete_emotions,
    EMOTION_COLORS,
    DISCRETE_EMOTIONS
)


def main():
    # Configuration
    layer = 30
    model_name = "unsloth/gemma-3-27b-it"
    device = "cuda"

    # Use contrast-all probes
    probes_file = Path(f"probes/ua_emotion_disentangle/orthogonal_probes_all_layers_contrast_all/layer_{layer}_orthogonal.npz")
    output_dir = Path("probes/ua_emotion_disentangle/conversation_analysis_contrast_all")
    output_dir.mkdir(parents=True, exist_ok=True)

    conversation_file = Path("probes/ua_emotion_disentangle/countdown_conversation.txt")

    print("="*80)
    print("ANALYZING CONVERSATION PROJECTIONS")
    print("METHOD: CONTRAST-ALL (emotion - mean(all others))")
    print("="*80)
    print(f"\nLayer: {layer}")
    print(f"Model: {model_name}")
    print(f"Probes: {probes_file}")

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

    # Load orthogonalized probes (contrast-all)
    print("\n3. Loading orthogonalized probes (contrast-all method)...")
    probes = load_orthogonal_probes(probes_file)
    m_probes, u_probes = separate_probes(probes)
    print(f"   ✓ Loaded {len(m_probes)} M probes, {len(u_probes)} U probes")

    # Compute PCs and dimensions
    print("\n4. Computing PCs and dimensions...")
    m_pcs = compute_pcs(m_probes, n_components=4)
    u_pcs = compute_pcs(u_probes, n_components=4)

    # Compute raw dimension directions
    m_dims_raw = {}
    u_dims_raw = {}
    for dim_code in ['V', 'A', 'D', 'AA']:
        m_dir = compute_dimension_direction(m_probes, dim_code)
        u_dir = compute_dimension_direction(u_probes, dim_code)
        if m_dir is not None:
            m_dims_raw[dim_code] = m_dir
        if u_dir is not None:
            u_dims_raw[dim_code] = u_dir

    # Orthogonalize dimensions using Gram-Schmidt
    print("   Orthogonalizing dimensions (V → A → D → AA)...")
    m_dims = orthogonalize_dimensions(m_dims_raw)
    u_dims = orthogonalize_dimensions(u_dims_raw)

    # Verify orthogonality
    print("   Checking M dimension orthogonality:")
    for dim1 in ['V', 'A', 'D', 'AA']:
        if dim1 not in m_dims:
            continue
        for dim2 in ['V', 'A', 'D', 'AA']:
            if dim2 <= dim1 or dim2 not in m_dims:
                continue
            dot = np.dot(m_dims[dim1], m_dims[dim2])
            print(f"     {dim1} · {dim2} = {dot:.6f}")

    print(f"   ✓ Computed PCs and orthogonalized dimensions")

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
    print("\nFiles generated:")
    print("  Full conversation (smoothed):")
    print("    - m_pc_projections.png")
    print("    - u_pc_projections.png")
    print("    - m_dimension_projections.png")
    print("    - u_dimension_projections.png")
    print("  Zoomed (tokens 5000-5200, raw values):")
    print("    - m_pc_projections_zoomed_5000_5200.png")
    print("    - u_pc_projections_zoomed_5000_5200.png")
    print("    - m_dimension_projections_zoomed_5000_5200.png")
    print("    - u_dimension_projections_zoomed_5000_5200.png")
    print("  Discrete emotions (smoothed):")
    print("    - m_discrete_emotions.png")
    print("    - u_discrete_emotions.png")
    print("="*80)


if __name__ == "__main__":
    main()
