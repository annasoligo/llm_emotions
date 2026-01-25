#!/usr/bin/env python3
"""
Analyze projections onto PCs and dimensions throughout a conversation.

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

# Dashboard emotion colors
EMOTION_COLORS = {
    'anger': '#7BA7D7',      # sky blue
    'disgust': '#7D9B7D',    # olive
    'fear': '#a59dc9',       # lavender
    'joy': '#D4876A',        # coral (happiness equivalent)
    'sadness': '#B8CCC8',    # sage
    'surprise': '#D1728F',   # darker pink
}

# Discrete emotions to plot
DISCRETE_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']


def load_orthogonal_probes(probes_file: Path) -> Dict[str, np.ndarray]:
    """Load orthogonalized probes."""
    data = np.load(probes_file)
    return {key: data[key] for key in data.files}


def separate_probes(probes: Dict[str, np.ndarray]):
    """Separate M and U probes."""
    m_probes = {}
    u_probes = {}

    for key, vec in probes.items():
        if key.startswith('M_'):
            emotion = key.replace('M_', '')
            m_probes[emotion] = vec
        elif key.startswith('U_'):
            emotion = key.replace('U_', '')
            u_probes[emotion] = vec

    return m_probes, u_probes


def compute_pcs(emotion_vectors: Dict[str, np.ndarray], n_components: int = 4) -> np.ndarray:
    """Compute PCA on emotion vectors."""
    names = sorted(emotion_vectors.keys())
    matrix = np.stack([emotion_vectors[name] for name in names])

    pca = PCA(n_components=n_components)
    pca.fit(matrix)

    return pca.components_


# Emotion dimension annotations (from compare_pcs_to_dimensions.py)
EMOTION_DIMENSIONS = {
    "joy": {"V": "H", "A": "H", "D": "H", "AA": "H"},
    "sadness": {"V": "L", "A": "L", "D": "L", "AA": "N"},
    "trust": {"V": "H", "A": "N", "D": "H", "AA": "H"},
    "disgust": {"V": "L", "A": "N", "D": "H", "AA": "L"},
    "fear": {"V": "L", "A": "H", "D": "L", "AA": "L"},
    "anger": {"V": "L", "A": "H", "D": "H", "AA": "H"},
    "surprise": {"V": "N", "A": "H", "D": "L", "AA": "N"},
    "anticipation": {"V": "N", "A": "H", "D": "H", "AA": "H"},
    "serenity": {"V": "H", "A": "L", "D": "H", "AA": "N"},
    "pensiveness": {"V": "L", "A": "L", "D": "N", "AA": "N"},
    "acceptance": {"V": "H", "A": "L", "D": "H", "AA": "H"},
    "boredom": {"V": "L", "A": "L", "D": "N", "AA": "L"},
    "apprehension": {"V": "L", "A": "N", "D": "L", "AA": "L"},
    "annoyance": {"V": "L", "A": "N", "D": "H", "AA": "H"},
    "distraction": {"V": "N", "A": "N", "D": "L", "AA": "N"},
    "interest": {"V": "H", "A": "N", "D": "H", "AA": "H"},
    "ecstasy": {"V": "H", "A": "H", "D": "H", "AA": "H"},
    "grief": {"V": "L", "A": "H", "D": "L", "AA": "N"},
    "admiration": {"V": "H", "A": "N", "D": "N", "AA": "H"},
    "loathing": {"V": "L", "A": "H", "D": "H", "AA": "L"},
    "terror": {"V": "L", "A": "H", "D": "L", "AA": "L"},
    "rage": {"V": "L", "A": "H", "D": "H", "AA": "H"},
    "amazement": {"V": "H", "A": "H", "D": "L", "AA": "H"},
    "vigilance": {"V": "N", "A": "H", "D": "H", "AA": "H"},
    "love": {"V": "H", "A": "N", "D": "N", "AA": "H"},
    "remorse": {"V": "L", "A": "N", "D": "L", "AA": "L"},
    "submission": {"V": "N", "A": "L", "D": "L", "AA": "N"},
    "contempt": {"V": "L", "A": "N", "D": "H", "AA": "L"},
    "awe": {"V": "H", "A": "H", "D": "L", "AA": "H"},
    "aggressiveness": {"V": "L", "A": "H", "D": "H", "AA": "H"},
    "disapproval": {"V": "L", "A": "N", "D": "H", "AA": "L"},
    "optimism": {"V": "H", "A": "N", "D": "H", "AA": "H"},
}


def compute_dimension_direction(emotion_vectors: Dict[str, np.ndarray], dimension: str) -> np.ndarray:
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
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    return direction


def orthogonalize_dimensions(dimensions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Orthogonalize dimension directions using Gram-Schmidt process.
    Order: V, A, D, AA (based on typical psychological priority).
    """
    dim_order = ["V", "A", "D", "AA"]
    orthogonal = {}

    for i, dim in enumerate(dim_order):
        if dim not in dimensions:
            continue

        vec = dimensions[dim].copy()

        # Subtract projections onto all previous orthogonalized vectors
        for prev_dim in dim_order[:i]:
            if prev_dim in orthogonal:
                proj = np.dot(vec, orthogonal[prev_dim]) * orthogonal[prev_dim]
                vec = vec - proj

        # Normalize
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        orthogonal[dim] = vec

    return orthogonal


def parse_conversation(text: str) -> List[Dict]:
    """Parse conversation into turns with role labels."""
    # Remove emojis
    text = re.sub(r'[🤖👤😢]', '', text)

    # Split by turn markers
    turns = []

    # Pattern: "User (Turn N)" or "Assistant (Turn N)"
    pattern = r'(User|Assistant) \(Turn \d+\)'

    splits = re.split(pattern, text)

    # Process splits
    role = None
    for i, part in enumerate(splits):
        if part in ['User', 'Assistant']:
            role = part
        elif role and part.strip():
            # Clean up the text
            content = part.strip()
            if content:
                turns.append({
                    'role': role.lower(),
                    'content': content
                })

    return turns


def extract_activations_for_conversation(
    model,
    tokenizer,
    conversation: List[Dict],
    layer: int
) -> tuple:
    """Extract token-by-token activations for a conversation."""

    # Format conversation
    messages = []
    for turn in conversation:
        role = 'user' if turn['role'] == 'user' else 'assistant'
        messages.append({"role": role, "content": turn['content']})

    # Apply chat template
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    token_ids = inputs['input_ids'][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Extract activations
    with torch.no_grad():
        with model.trace(inputs, scan=False):
            hidden_states = model.layers_output[layer].save()

    activations = hidden_states[0].float().cpu().numpy()  # [seq_len, hidden_dim]

    # Determine which tokens belong to user vs assistant
    # This is approximate - we'll mark tokens based on special tokens
    roles = []
    current_role = 'user'

    for i, token in enumerate(tokens):
        if '<start_of_turn>model' in token or 'Assistant' in token:
            current_role = 'assistant'
        elif '<start_of_turn>user' in token or 'User' in token:
            current_role = 'user'

        roles.append(current_role)

    return activations, tokens, roles


def project_onto_subspace(vector: np.ndarray, basis_vectors: np.ndarray) -> np.ndarray:
    """
    Project vector onto subspace spanned by basis_vectors.

    Args:
        vector: [d] activation vector
        basis_vectors: [n, d] basis vectors spanning the subspace

    Returns:
        Projection of vector onto subspace [d]
    """
    # Get orthonormal basis via QR
    Q, R = np.linalg.qr(basis_vectors.T)
    orthonormal_basis = Q.T  # [k, d] where k <= n

    # Project onto subspace
    projection = vector @ orthonormal_basis.T @ orthonormal_basis

    return projection


def compute_projections(
    m_probes: Dict[str, np.ndarray],
    u_probes: Dict[str, np.ndarray],
    activations: np.ndarray,
    m_pcs: np.ndarray,
    u_pcs: np.ndarray,
    m_dims: Dict[str, np.ndarray],
    u_dims: Dict[str, np.ndarray],
    roles: List[str]
) -> Dict:
    """
    Compute projections onto PCs and dimensions for each token.

    Normalizes by magnitude within each subspace (M or U) rather than total magnitude.
    All probe directions (PCs, dimensions, emotions) are normalized to give cosine similarities.
    """

    results = {
        'M_pc1': [], 'M_pc2': [], 'M_pc3': [], 'M_pc4': [],
        'U_pc1': [], 'U_pc2': [], 'U_pc3': [], 'U_pc4': [],
        'M_valence': [], 'M_arousal': [], 'M_dominance': [], 'M_approach_avoidance': [],
        'U_valence': [], 'U_arousal': [], 'U_dominance': [], 'U_approach_avoidance': [],
        'roles': roles
    }

    # Add discrete emotion keys
    for emotion in DISCRETE_EMOTIONS:
        results[f'M_{emotion}'] = []
        results[f'U_{emotion}'] = []

    # Build M and U subspace bases from all probes
    m_probe_vectors = np.stack([m_probes[k] for k in sorted(m_probes.keys())])
    u_probe_vectors = np.stack([u_probes[k] for k in sorted(u_probes.keys())])

    # Normalize all probe directions to get cosine similarities
    # Normalize M probes
    m_probes_norm = {k: v / np.linalg.norm(v) for k, v in m_probes.items()}
    # Normalize U probes
    u_probes_norm = {k: v / np.linalg.norm(v) for k, v in u_probes.items()}
    # Normalize M PCs (if provided)
    m_pcs_norm = np.array([pc / np.linalg.norm(pc) for pc in m_pcs]) if m_pcs is not None else None
    # Normalize U PCs (if provided)
    u_pcs_norm = np.array([pc / np.linalg.norm(pc) for pc in u_pcs]) if u_pcs is not None else None
    # Normalize M dimensions
    m_dims_norm = {k: v / np.linalg.norm(v) for k, v in m_dims.items()}
    # Normalize U dimensions
    u_dims_norm = {k: v / np.linalg.norm(v) for k, v in u_dims.items()}

    for i, act in enumerate(activations):
        role = roles[i]

        # Project onto M subspace and normalize by magnitude within M subspace
        m_projection = project_onto_subspace(act, m_probe_vectors)
        m_norm = np.linalg.norm(m_projection)
        if m_norm > 1e-8:
            m_normalized = m_projection / m_norm
        else:
            m_normalized = m_projection

        # Project onto U subspace and normalize by magnitude within U subspace
        u_projection = project_onto_subspace(act, u_probe_vectors)
        u_norm = np.linalg.norm(u_projection)
        if u_norm > 1e-8:
            u_normalized = u_projection / u_norm
        else:
            u_normalized = u_projection

        # Project onto M PCs (using M-normalized activation) - skip if None
        if m_pcs_norm is not None:
            for j in range(4):
                proj = np.dot(m_normalized, m_pcs_norm[j])
                results[f'M_pc{j+1}'].append(proj)

        # Project onto U PCs (using U-normalized activation) - skip if None
        if u_pcs_norm is not None:
            for j in range(4):
                proj = np.dot(u_normalized, u_pcs_norm[j])
                results[f'U_pc{j+1}'].append(proj)

        # Project onto M dimensions (using M-normalized activation)
        for dim_name, dim_key in [('valence', 'V'), ('arousal', 'A'),
                                   ('dominance', 'D'), ('approach_avoidance', 'AA')]:
            if dim_key in m_dims_norm:
                proj = np.dot(m_normalized, m_dims_norm[dim_key])
                results[f'M_{dim_name}'].append(proj)
            else:
                results[f'M_{dim_name}'].append(0.0)

        # Project onto U dimensions (using U-normalized activation)
        for dim_name, dim_key in [('valence', 'V'), ('arousal', 'A'),
                                   ('dominance', 'D'), ('approach_avoidance', 'AA')]:
            if dim_key in u_dims_norm:
                proj = np.dot(u_normalized, u_dims_norm[dim_key])
                results[f'U_{dim_name}'].append(proj)
            else:
                results[f'U_{dim_name}'].append(0.0)

        # Project onto discrete emotion directions
        for emotion in DISCRETE_EMOTIONS:
            # M emotions (using M-normalized activation)
            if emotion in m_probes_norm:
                proj = np.dot(m_normalized, m_probes_norm[emotion])
                results[f'M_{emotion}'].append(proj)
            else:
                results[f'M_{emotion}'].append(0.0)

            # U emotions (using U-normalized activation)
            if emotion in u_probes_norm:
                proj = np.dot(u_normalized, u_probes_norm[emotion])
                results[f'U_{emotion}'].append(proj)
            else:
                results[f'U_{emotion}'].append(0.0)

    return results


def smooth_projections(projections: List[float], window: int = 100, skip_first_n: int = 5) -> tuple:
    """
    Apply moving average smoothing (dashboard-style), skipping first N tokens.

    Uses partial windows at the beginning to avoid flat lines.
    For position i, computes average of window [max(0, i-window+1), i+1].

    Returns:
        positions: Token positions (starting from skip_first_n)
        smoothed: Moving-average smoothed projection values
        neutral_baseline: Mean projection (excluding first N tokens)
    """
    # Skip first N tokens (outliers)
    projections_clean = projections[skip_first_n:]

    # Compute neutral baseline (mean of all clean projections)
    neutral_baseline = np.mean(projections_clean)

    # Apply moving average (dashboard style)
    smoothed = np.zeros(len(projections_clean))
    for i in range(len(projections_clean)):
        # Window spans from max(0, i-window+1) to i+1
        start_idx = max(0, i - window + 1)
        smoothed[i] = np.mean(projections_clean[start_idx:i+1])

    # Positions adjusted for skipped tokens
    positions = np.arange(skip_first_n, skip_first_n + len(projections_clean))

    return positions, smoothed, neutral_baseline


def plot_projections(projections: Dict, output_dir: Path):
    """Create plots for all projections."""

    # PC plots for M
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle('Assistant (M) Emotion - PC Projections Throughout Conversation\n(First 5 tokens excluded, neutral baseline shown)',
                 fontsize=16, fontweight='bold')

    for i in range(4):
        ax = axes[i]
        pc_key = f'M_pc{i+1}'

        positions, smoothed, neutral = smooth_projections(projections[pc_key], window=100, skip_first_n=5)

        ax.plot(positions, smoothed, linewidth=2, color=f'C{i}', label='Projection')
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Neutral ({neutral:.2f})')
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Projection Value', fontsize=12)
        ax.set_title(f'PC{i+1}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    output_file = output_dir / 'm_pc_projections.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

    # PC plots for U
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle('User (U) Emotion - PC Projections Throughout Conversation\n(First 5 tokens excluded, neutral baseline shown)',
                 fontsize=16, fontweight='bold')

    for i in range(4):
        ax = axes[i]
        pc_key = f'U_pc{i+1}'

        positions, smoothed, neutral = smooth_projections(projections[pc_key], window=100, skip_first_n=5)

        ax.plot(positions, smoothed, linewidth=2, color=f'C{i}', label='Projection')
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Neutral ({neutral:.2f})')
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Projection Value', fontsize=12)
        ax.set_title(f'PC{i+1}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    output_file = output_dir / 'u_pc_projections.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

    # Dimension plots for M
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle('Assistant (M) Emotion - Orthogonalized Psychological Dimensions Throughout Conversation\n(First 5 tokens excluded, neutral baseline shown)',
                 fontsize=16, fontweight='bold')

    dim_names = ['valence', 'arousal', 'dominance', 'approach_avoidance']
    dim_labels = ['Valence', 'Arousal', 'Dominance', 'Approach-Avoidance']

    for i, (dim_name, dim_label) in enumerate(zip(dim_names, dim_labels)):
        ax = axes[i]
        dim_key = f'M_{dim_name}'

        positions, smoothed, neutral = smooth_projections(projections[dim_key], window=100, skip_first_n=5)

        ax.plot(positions, smoothed, linewidth=2, color=f'C{i}', label='Projection')
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Neutral ({neutral:.2f})')
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Projection Value', fontsize=12)
        ax.set_title(dim_label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    output_file = output_dir / 'm_dimension_projections.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

    # Dimension plots for U
    fig, axes = plt.subplots(4, 1, figsize=(20, 16))
    fig.suptitle('User (U) Emotion - Orthogonalized Psychological Dimensions Throughout Conversation\n(First 5 tokens excluded, neutral baseline shown)',
                 fontsize=16, fontweight='bold')

    for i, (dim_name, dim_label) in enumerate(zip(dim_names, dim_labels)):
        ax = axes[i]
        dim_key = f'U_{dim_name}'

        positions, smoothed, neutral = smooth_projections(projections[dim_key], window=100, skip_first_n=5)

        ax.plot(positions, smoothed, linewidth=2, color=f'C{i}', label='Projection')
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Neutral ({neutral:.2f})')
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Projection Value', fontsize=12)
        ax.set_title(dim_label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    output_file = output_dir / 'u_dimension_projections.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def plot_zoomed_projections(projections: Dict, output_dir: Path, start_token: int = 5000, window_size: int = 200):
    """Create zoomed-in plots showing raw (unsmoothed) token values for a specific window."""

    end_token = start_token + window_size

    # Check if window is valid
    sample_key = list(projections.keys())[0]
    total_tokens = len(projections[sample_key])

    if start_token >= total_tokens:
        print(f"⚠ Warning: Start token {start_token} >= total tokens {total_tokens}. Skipping zoomed plots.")
        return

    # Adjust end_token if needed
    end_token = min(end_token, total_tokens)
    actual_window = end_token - start_token

    print(f"  Creating zoomed plots for tokens {start_token}-{end_token} ({actual_window} tokens)")

    # M PC zoomed plots
    fig, axes = plt.subplots(4, 1, figsize=(22, 20))
    fig.suptitle(f'Assistant (M) Emotion - PC Projections (Tokens {start_token}-{end_token}, Raw Values)',
                 fontsize=16, fontweight='bold')

    for i in range(4):
        ax = axes[i]
        pc_key = f'M_pc{i+1}'

        values = projections[pc_key][start_token:end_token]
        positions = np.arange(start_token, end_token)

        ax.plot(positions, values, linewidth=1, color=f'C{i}', alpha=0.7)
        ax.scatter(positions, values, s=10, color=f'C{i}', alpha=0.5)
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Projection Value', fontsize=12)
        ax.set_title(f'PC{i+1}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f'm_pc_projections_zoomed_{start_token}_{end_token}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file}")

    # U PC zoomed plots
    fig, axes = plt.subplots(4, 1, figsize=(22, 20))
    fig.suptitle(f'User (U) Emotion - PC Projections (Tokens {start_token}-{end_token}, Raw Values)',
                 fontsize=16, fontweight='bold')

    for i in range(4):
        ax = axes[i]
        pc_key = f'U_pc{i+1}'

        values = projections[pc_key][start_token:end_token]
        positions = np.arange(start_token, end_token)

        ax.plot(positions, values, linewidth=1, color=f'C{i}', alpha=0.7)
        ax.scatter(positions, values, s=10, color=f'C{i}', alpha=0.5)
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Projection Value', fontsize=12)
        ax.set_title(f'PC{i+1}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f'u_pc_projections_zoomed_{start_token}_{end_token}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file}")

    # M Dimension zoomed plots
    fig, axes = plt.subplots(4, 1, figsize=(22, 20))
    fig.suptitle(f'Assistant (M) Emotion - Orthogonalized Dimensions (Tokens {start_token}-{end_token}, Raw Values)',
                 fontsize=16, fontweight='bold')

    dim_names = ['valence', 'arousal', 'dominance', 'approach_avoidance']
    dim_labels = ['Valence', 'Arousal', 'Dominance', 'Approach-Avoidance']

    for i, (dim_name, dim_label) in enumerate(zip(dim_names, dim_labels)):
        ax = axes[i]
        dim_key = f'M_{dim_name}'

        values = projections[dim_key][start_token:end_token]
        positions = np.arange(start_token, end_token)

        ax.plot(positions, values, linewidth=1, color=f'C{i}', alpha=0.7)
        ax.scatter(positions, values, s=10, color=f'C{i}', alpha=0.5)
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Projection Value', fontsize=12)
        ax.set_title(dim_label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f'm_dimension_projections_zoomed_{start_token}_{end_token}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file}")

    # U Dimension zoomed plots
    fig, axes = plt.subplots(4, 1, figsize=(22, 20))
    fig.suptitle(f'User (U) Emotion - Orthogonalized Dimensions (Tokens {start_token}-{end_token}, Raw Values)',
                 fontsize=16, fontweight='bold')

    for i, (dim_name, dim_label) in enumerate(zip(dim_names, dim_labels)):
        ax = axes[i]
        dim_key = f'U_{dim_name}'

        values = projections[dim_key][start_token:end_token]
        positions = np.arange(start_token, end_token)

        ax.plot(positions, values, linewidth=1, color=f'C{i}', alpha=0.7)
        ax.scatter(positions, values, s=10, color=f'C{i}', alpha=0.5)
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Projection Value', fontsize=12)
        ax.set_title(dim_label, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f'u_dimension_projections_zoomed_{start_token}_{end_token}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {output_file}")


def plot_discrete_emotions(projections: Dict, output_dir: Path):
    """Plot discrete emotion projections using dashboard colors."""

    # M emotions
    fig, axes = plt.subplots(6, 1, figsize=(20, 24))
    fig.suptitle('Assistant (M) - Discrete Emotion Projections Throughout Conversation\n(First 5 tokens excluded, neutral baseline shown)',
                 fontsize=16, fontweight='bold')

    for i, emotion in enumerate(DISCRETE_EMOTIONS):
        ax = axes[i]
        key = f'M_{emotion}'

        positions, smoothed, neutral = smooth_projections(projections[key], window=100, skip_first_n=5)

        ax.plot(positions, smoothed, linewidth=2, color=EMOTION_COLORS[emotion], label='Projection')
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Neutral ({neutral:.2f})')
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Projection Value', fontsize=12)
        ax.set_title(emotion.capitalize(), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    output_file = output_dir / 'm_discrete_emotions.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")

    # U emotions
    fig, axes = plt.subplots(6, 1, figsize=(20, 24))
    fig.suptitle('User (U) - Discrete Emotion Projections Throughout Conversation\n(First 5 tokens excluded, neutral baseline shown)',
                 fontsize=16, fontweight='bold')

    for i, emotion in enumerate(DISCRETE_EMOTIONS):
        ax = axes[i]
        key = f'U_{emotion}'

        positions, smoothed, neutral = smooth_projections(projections[key], window=100, skip_first_n=5)

        ax.plot(positions, smoothed, linewidth=2, color=EMOTION_COLORS[emotion], label='Projection')
        ax.axhline(y=neutral, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Neutral ({neutral:.2f})')
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Projection Value', fontsize=12)
        ax.set_title(emotion.capitalize(), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    output_file = output_dir / 'u_discrete_emotions.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_file}")


def main():
    # Configuration
    layers = list(range(20, 41))  # Layers 20-40
    model_name = "unsloth/gemma-3-27b-it"
    device = "cuda"

    probes_dir = Path("probes/ua_emotion_disentangle/orthogonal_probes_all_layers")
    output_dir = Path("probes/ua_emotion_disentangle/conversation_analysis_layers20_to_40")
    output_dir.mkdir(parents=True, exist_ok=True)

    conversation_file = Path("probes/ua_emotion_disentangle/countdown_conversation.txt")

    print("="*80)
    print("ANALYZING CONVERSATION PROJECTIONS")
    print("="*80)
    print(f"\nLayers: {layers[0]}-{layers[-1]} (aggregated)")
    print(f"Model: {model_name}")

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

    # Process each layer and aggregate
    print(f"\n3. Processing layers {layers[0]}-{layers[-1]}...")
    all_layer_projections = []

    for layer in layers:
        print(f"   Layer {layer}...")

        # Load probes for this layer
        probes_file = probes_dir / f"layer_{layer}_orthogonal.npz"
        probes = load_orthogonal_probes(probes_file)
        m_probes, u_probes = separate_probes(probes)

        # Compute dimensions (skip PCs to avoid direction flipping issues)
        m_dims_raw = {}
        u_dims_raw = {}
        for dim_code in ['V', 'A', 'D', 'AA']:
            m_dir = compute_dimension_direction(m_probes, dim_code)
            u_dir = compute_dimension_direction(u_probes, dim_code)
            if m_dir is not None:
                m_dims_raw[dim_code] = m_dir
            if u_dir is not None:
                u_dims_raw[dim_code] = u_dir

        m_dims = orthogonalize_dimensions(m_dims_raw)
        u_dims = orthogonalize_dimensions(u_dims_raw)

        # Extract activations
        activations, tokens, roles = extract_activations_for_conversation(
            model, tokenizer, conversation, layer
        )

        # Compute projections (no PCs)
        projections = compute_projections(m_probes, u_probes, activations, None, None, m_dims, u_dims, roles)
        all_layer_projections.append(projections)

    # Average projections across layers
    print(f"\n4. Averaging projections across {len(layers)} layers...")
    averaged_projections = {'roles': all_layer_projections[0]['roles']}
    for key in all_layer_projections[0].keys():
        if key == 'roles' or key.startswith('M_pc') or key.startswith('U_pc'):
            continue
        averaged_projections[key] = np.mean([p[key] for p in all_layer_projections], axis=0).tolist()
    print(f"   ✓ Averaged projections")

    # Plot (only dimensions and emotions, no PCs)
    print("\n5. Creating discrete emotion plots...")
    plot_discrete_emotions(averaged_projections, output_dir)

    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Layers aggregated: {layers[0]}-{layers[-1]}")
    print("\nFiles generated:")
    print("  Discrete emotions (smoothed, averaged across layers):")
    print("    - assistant (m)_discrete_emotions.png")
    print("    - user (u)_discrete_emotions.png")
    print("="*80)


if __name__ == "__main__":
    main()
