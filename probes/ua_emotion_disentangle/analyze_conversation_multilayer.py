#!/usr/bin/env python3
"""
Analyze projections onto dimensions and emotions throughout a conversation.
Supports single layer or aggregated across layers.

Usage:
  # Single layer
  python analyze_conversation_multilayer.py --layer 20
  
  # Aggregated over layers (no PCs)
  python analyze_conversation_multilayer.py --aggregate 20 40
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnterp import StandardizedTransformer
from typing import List, Dict, Optional
import argparse
from sklearn.decomposition import PCA

# Dashboard emotion colors
EMOTION_COLORS = {
    'anger': '#7BA7D7',
    'disgust': '#7D9B7D',
    'fear': '#a59dc9',
    'joy': '#D4876A',
    'sadness': '#B8CCC8',
    'surprise': '#D1728F',
}

DISCRETE_EMOTIONS = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise']


def load_layer_probes(layer: int, probes_dir: Path) -> Dict[str, np.ndarray]:
    """Load probes for a specific layer."""
    probe_file = probes_dir / f"layer_{layer}_orthogonal.npz"
    data = np.load(probe_file)
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


def load_conversation(conv_file: Path) -> List[Dict]:
    """Load conversation from file."""
    with open(conv_file, 'r') as f:
        text = f.read()
    
    turns = []
    current_turn = None
    current_content = []
    
    for line in text.split('\n'):
        if line.startswith('User (Turn'):
            if current_turn is not None:
                turns.append({'role': current_turn, 'content': '\n'.join(current_content)})
            current_turn = 'user'
            current_content = []
        elif line.startswith('Assistant (Turn'):
            if current_turn is not None:
                turns.append({'role': current_turn, 'content': '\n'.join(current_content)})
            current_turn = 'assistant'
            current_content = []
        else:
            current_content.append(line)
    
    if current_turn is not None:
        turns.append({'role': current_turn, 'content': '\n'.join(current_content)})
    
    return turns


def extract_activations(model, tokenizer, conversation: List[Dict], layer: int):
    """Extract activations for conversation at given layer."""
    messages = []
    for turn in conversation:
        role = 'user' if turn['role'] == 'user' else 'assistant'
        messages.append({"role": role, "content": turn['content']})
    
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    token_ids = inputs['input_ids'][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    with torch.no_grad():
        with model.trace(inputs, scan=False):
            hidden_states = model.layers_output[layer].save()
    
    activations = hidden_states[0].float().cpu().numpy()
    
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
    """Project vector onto subspace."""
    Q, R = np.linalg.qr(basis_vectors.T)
    orthonormal_basis = Q.T
    projection = vector @ orthonormal_basis.T @ orthonormal_basis
    return projection


def compute_gram_schmidt_dimensions(probes: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute orthogonalized psychological dimensions."""
    emotion_map = {
        'V': ['joy', 'serenity', 'sadness', 'grief'],
        'A': ['rage', 'anger', 'serenity', 'sadness'],
        'D': ['rage', 'anger', 'fear', 'terror'],
        'AA': ['rage', 'joy', 'fear', 'sadness']
    }
    
    positive_map = {
        'V': ['joy', 'serenity'],
        'A': ['rage', 'anger'],
        'D': ['rage', 'anger'],
        'AA': ['rage', 'joy']
    }
    
    dims = {}
    
    for dim, emotions in emotion_map.items():
        available_emotions = [e for e in emotions if e in probes]
        if not available_emotions:
            continue
        
        pos_emotions = [e for e in positive_map[dim] if e in available_emotions]
        neg_emotions = [e for e in available_emotions if e not in pos_emotions]
        
        if pos_emotions and neg_emotions:
            pos_mean = np.mean([probes[e] for e in pos_emotions], axis=0)
            neg_mean = np.mean([probes[e] for e in neg_emotions], axis=0)
            dims[dim] = pos_mean - neg_mean
    
    if 'V' in dims:
        dims['V'] = dims['V'] / np.linalg.norm(dims['V'])
    
    if 'A' in dims and 'V' in dims:
        dims['A'] = dims['A'] - np.dot(dims['A'], dims['V']) * dims['V']
        dims['A'] = dims['A'] / np.linalg.norm(dims['A'])
    
    if 'D' in dims and 'V' in dims:
        dims['D'] = dims['D'] - np.dot(dims['D'], dims['V']) * dims['V']
        if 'A' in dims:
            dims['D'] = dims['D'] - np.dot(dims['D'], dims['A']) * dims['A']
        dims['D'] = dims['D'] / np.linalg.norm(dims['D'])
    
    if 'AA' in dims and 'V' in dims:
        dims['AA'] = dims['AA'] - np.dot(dims['AA'], dims['V']) * dims['V']
        if 'D' in dims:
            dims['AA'] = dims['AA'] - np.dot(dims['AA'], dims['D']) * dims['D']
        dims['AA'] = dims['AA'] / np.linalg.norm(dims['AA'])
    
    return dims


def compute_projections_single_layer(
    m_probes: Dict[str, np.ndarray],
    u_probes: Dict[str, np.ndarray],
    activations: np.ndarray,
    m_pcs: Optional[np.ndarray],
    u_pcs: Optional[np.ndarray],
    m_dims: Dict[str, np.ndarray],
    u_dims: Dict[str, np.ndarray],
    roles: List[str],
    include_pcs: bool = True
) -> Dict:
    """Compute projections for a single layer."""
    
    results = {
        'M_valence': [], 'M_arousal': [], 'M_dominance': [], 'M_approach_avoidance': [],
        'U_valence': [], 'U_arousal': [], 'U_dominance': [], 'U_approach_avoidance': [],
        'roles': roles
    }
    
    if include_pcs and m_pcs is not None:
        for i in range(4):
            results[f'M_pc{i+1}'] = []
            results[f'U_pc{i+1}'] = []
    
    for emotion in DISCRETE_EMOTIONS:
        results[f'M_{emotion}'] = []
        results[f'U_{emotion}'] = []
    
    # Build subspace bases
    m_probe_vectors = np.stack([m_probes[k] for k in sorted(m_probes.keys())])
    u_probe_vectors = np.stack([u_probes[k] for k in sorted(u_probes.keys())])
    
    # Normalize all directions
    m_probes_norm = {k: v / np.linalg.norm(v) for k, v in m_probes.items()}
    u_probes_norm = {k: v / np.linalg.norm(v) for k, v in u_probes.items()}
    m_dims_norm = {k: v / np.linalg.norm(v) for k, v in m_dims.items()}
    u_dims_norm = {k: v / np.linalg.norm(v) for k, v in u_dims.items()}
    
    if include_pcs and m_pcs is not None:
        m_pcs_norm = np.array([pc / np.linalg.norm(pc) for pc in m_pcs])
        u_pcs_norm = np.array([pc / np.linalg.norm(pc) for pc in u_pcs])
    
    for i, act in enumerate(activations):
        # Project onto subspaces and normalize
        m_projection = project_onto_subspace(act, m_probe_vectors)
        m_norm = np.linalg.norm(m_projection)
        m_normalized = m_projection / m_norm if m_norm > 1e-8 else m_projection
        
        u_projection = project_onto_subspace(act, u_probe_vectors)
        u_norm = np.linalg.norm(u_projection)
        u_normalized = u_projection / u_norm if u_norm > 1e-8 else u_projection
        
        # PCs
        if include_pcs and m_pcs is not None:
            for j in range(4):
                results[f'M_pc{j+1}'].append(np.dot(m_normalized, m_pcs_norm[j]))
                results[f'U_pc{j+1}'].append(np.dot(u_normalized, u_pcs_norm[j]))
        
        # Dimensions
        for dim_name, dim_key in [('valence', 'V'), ('arousal', 'A'),
                                   ('dominance', 'D'), ('approach_avoidance', 'AA')]:
            if dim_key in m_dims_norm:
                results[f'M_{dim_name}'].append(np.dot(m_normalized, m_dims_norm[dim_key]))
            else:
                results[f'M_{dim_name}'].append(0.0)
            
            if dim_key in u_dims_norm:
                results[f'U_{dim_name}'].append(np.dot(u_normalized, u_dims_norm[dim_key]))
            else:
                results[f'U_{dim_name}'].append(0.0)
        
        # Emotions
        for emotion in DISCRETE_EMOTIONS:
            if emotion in m_probes_norm:
                results[f'M_{emotion}'].append(np.dot(m_normalized, m_probes_norm[emotion]))
            else:
                results[f'M_{emotion}'].append(0.0)
            
            if emotion in u_probes_norm:
                results[f'U_{emotion}'].append(np.dot(u_normalized, u_probes_norm[emotion]))
            else:
                results[f'U_{emotion}'].append(0.0)
    
    return results


def smooth_projections(projections: List[float], window: int = 100, skip_first_n: int = 5):
    """Apply moving average smoothing."""
    projections_clean = projections[skip_first_n:]
    neutral_baseline = np.mean(projections_clean)
    
    smoothed = np.zeros(len(projections_clean))
    for i in range(len(projections_clean)):
        start_idx = max(0, i - window + 1)
        smoothed[i] = np.mean(projections_clean[start_idx:i+1])
    
    positions = np.arange(skip_first_n, len(projections))
    return positions, smoothed, neutral_baseline


def create_discrete_emotion_plots(results: Dict, output_dir: Path, prefix: str):
    """Create discrete emotion plots."""
    fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
    fig.suptitle(f'{prefix} - Discrete Emotion Projections Throughout Conversation\n(First 5 tokens excluded, neutral baseline shown)', 
                 fontsize=14, y=0.995)
    
    for idx, emotion in enumerate(DISCRETE_EMOTIONS):
        key = f'{prefix}_{emotion}'
        if key not in results:
            continue
        
        positions, smoothed, neutral = smooth_projections(results[key])
        
        ax = axes[idx]
        color = EMOTION_COLORS[emotion]
        ax.plot(positions, smoothed, color=color, linewidth=1.5, label='Projection')
        ax.axhline(y=neutral, color=color, linestyle='--', linewidth=1, alpha=0.6, label=f'Neutral ({neutral:.2f})')
        ax.set_ylabel('Projection Value', fontsize=10)
        ax.set_title(emotion.capitalize(), fontsize=11, pad=5)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Token Position', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / f'{prefix.lower()}_discrete_emotions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir / f'{prefix.lower()}_discrete_emotions.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, help='Single layer to analyze')
    parser.add_argument('--aggregate', nargs=2, type=int, metavar=('START', 'END'),
                        help='Aggregate over layer range (e.g., 20 40)')
    parser.add_argument('--output-suffix', type=str, default='', help='Suffix for output directory')
    args = parser.parse_args()
    
    if args.layer is None and args.aggregate is None:
        parser.error("Must specify either --layer or --aggregate")
    
    # Paths
    base_dir = Path("probes/ua_emotion_disentangle")
    probes_dir = base_dir / "orthogonal_probes_all_layers"
    conv_file = base_dir / "countdown_conversation.txt"
    
    if args.layer is not None:
        output_dir = base_dir / f"conversation_analysis_layer{args.layer}{args.output_suffix}"
        layer_desc = f"Layer {args.layer}"
    else:
        start, end = args.aggregate
        output_dir = base_dir / f"conversation_analysis_layers{start}_to_{end}{args.output_suffix}"
        layer_desc = f"Layers {start}-{end} (aggregated)"
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*80)
    print("ANALYZING CONVERSATION PROJECTIONS")
    print("="*80)
    print(f"\n{layer_desc}")
    print("Model: unsloth/gemma-3-27b-it\n")
    
    # Load conversation
    print("1. Loading conversation...")
    conversation = load_conversation(conv_file)
    print(f"   Found {len(conversation)} turns\n")
    
    # Load model
    print("2. Loading model...")
    model_name = "unsloth/gemma-3-27b-it"
    model_raw = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model = StandardizedTransformer(model_raw)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("   ✓ Model loaded\n")
    
    if args.layer is not None:
        # Single layer analysis
        layer = args.layer
        
        print(f"3. Loading probes for layer {layer}...")
        probes = load_layer_probes(layer, probes_dir)
        m_probes, u_probes = separate_probes(probes)
        print(f"   ✓ Loaded {len(m_probes)} M probes, {len(u_probes)} U probes\n")
        
        print("4. Computing PCs and dimensions...")
        m_pca = PCA(n_components=4).fit(np.stack(list(m_probes.values())))
        m_pcs = m_pca.components_  # Shape: (4, 5376)
        u_pca = PCA(n_components=4).fit(np.stack(list(u_probes.values())))
        u_pcs = u_pca.components_  # Shape: (4, 5376)
        m_dims = compute_gram_schmidt_dimensions(m_probes)
        u_dims = compute_gram_schmidt_dimensions(u_probes)
        print("   ✓ Computed PCs and dimensions\n")
        
        print("5. Extracting activations...")
        activations, tokens, roles = extract_activations(model, tokenizer, conversation, layer)
        print(f"   ✓ Extracted {len(activations)} token activations\n")
        
        print("6. Computing projections...")
        results = compute_projections_single_layer(m_probes, u_probes, activations, m_pcs, u_pcs, 
                                                    m_dims, u_dims, roles, include_pcs=True)
        print("   ✓ Computed projections\n")
        
    else:
        # Aggregated analysis
        start_layer, end_layer = args.aggregate
        layers = list(range(start_layer, end_layer + 1))
        
        print(f"3. Aggregating over {len(layers)} layers...")
        
        # Initialize accumulators
        all_results = []
        
        for layer in layers:
            if layer % 5 == 0:
                print(f"   Processing layer {layer}...")
            
            probes = load_layer_probes(layer, probes_dir)
            m_probes, u_probes = separate_probes(probes)
            m_dims = compute_gram_schmidt_dimensions(m_probes)
            u_dims = compute_gram_schmidt_dimensions(u_probes)
            
            activations, tokens, roles = extract_activations(model, tokenizer, conversation, layer)
            
            layer_results = compute_projections_single_layer(
                m_probes, u_probes, activations, None, None, 
                m_dims, u_dims, roles, include_pcs=False
            )
            all_results.append(layer_results)
        
        print(f"   ✓ Processed {len(layers)} layers\n")
        
        print("4. Averaging projections across layers...")
        results = {'roles': roles}
        for key in all_results[0].keys():
            if key == 'roles':
                continue
            results[key] = np.mean([r[key] for r in all_results], axis=0).tolist()
        print("   ✓ Averaged projections\n")
    
    # Create plots
    print("5. Creating discrete emotion plots...")
    create_discrete_emotion_plots(results, output_dir, 'Assistant (M)')
    create_discrete_emotion_plots(results, output_dir, 'User (U)')
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
