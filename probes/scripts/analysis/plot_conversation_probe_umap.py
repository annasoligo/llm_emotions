#!/usr/bin/env python3
"""
UMAP analysis for K orthogonal user/assistant conversation probes.
Visualizes probe structure with color per emotion and shape per role (user/assistant).
"""
import sys
sys.path.insert(0, '/home/annas/.local/lib/python3.12/site-packages')

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import umap
from pathlib import Path

def load_and_analyze(probe_path: Path, k_value: int, constraint_type: str):
    """Load probes and compute UMAP."""
    print(f"\n{'='*80}")
    print(f"Analyzing: K={k_value}, {constraint_type}")
    print(f"{'='*80}")
    
    with open(probe_path, 'rb') as f:
        data = pickle.load(f)
    
    # probe_sets: [K, 2, 6, hidden_dim]
    # dim 1: 0=user, 1=assistant
    probe_sets = data['probe_sets']
    emotion_labels = data['label_names']
    
    print(f"Probe shape: {probe_sets.shape}")
    print(f"Emotions: {emotion_labels}")
    
    k, n_roles, n_emotions, hidden_dim = probe_sets.shape
    
    # Flatten: [K*2*6, hidden_dim]
    probes_flat = probe_sets.reshape(k * n_roles * n_emotions, hidden_dim)
    
    # Create labels for each probe
    # For each k, for each role (user/asst), for each emotion
    emotion_indices = []
    role_indices = []  # 0=user, 1=assistant
    
    for k_idx in range(k):
        for role_idx in range(n_roles):
            for emotion_idx in range(n_emotions):
                emotion_indices.append(emotion_idx)
                role_indices.append(role_idx)
    
    emotion_indices = np.array(emotion_indices)
    role_indices = np.array(role_indices)
    
    print(f"Total probes: {len(probes_flat)}")
    
    # Compute orthogonality
    from scipy.spatial.distance import pdist
    cosine_dists = pdist(probes_flat, metric='cosine')
    cosine_sims = 1 - cosine_dists
    print(f"Mean |cosine similarity|: {np.mean(np.abs(cosine_sims)):.6f}")
    
    # Run UMAP
    print("\nRunning UMAP...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42
    )
    
    embedding = reducer.fit_transform(probes_flat)
    print(f"UMAP embedding shape: {embedding.shape}")
    
    # Compute silhouette scores
    # Overall by emotion
    silhouette_emotion = silhouette_score(embedding, emotion_indices, metric='euclidean')
    
    # Separate by role
    user_mask = role_indices == 0
    asst_mask = role_indices == 1
    
    silhouette_user = silhouette_score(embedding[user_mask], emotion_indices[user_mask], metric='euclidean')
    silhouette_asst = silhouette_score(embedding[asst_mask], emotion_indices[asst_mask], metric='euclidean')
    
    print(f"\nSilhouette (all by emotion): {silhouette_emotion:.4f}")
    print(f"Silhouette (user only): {silhouette_user:.4f}")
    print(f"Silhouette (assistant only): {silhouette_asst:.4f}")
    
    # Compute centroids
    centroids = {}
    for emotion_idx, emotion in enumerate(emotion_labels):
        user_points = embedding[(emotion_indices == emotion_idx) & (role_indices == 0)]
        asst_points = embedding[(emotion_indices == emotion_idx) & (role_indices == 1)]
        
        centroids[emotion] = {
            'user': user_points.mean(axis=0) if len(user_points) > 0 else None,
            'asst': asst_points.mean(axis=0) if len(asst_points) > 0 else None,
        }
    
    return {
        'embedding': embedding,
        'emotion_indices': emotion_indices,
        'role_indices': role_indices,
        'emotion_labels': emotion_labels,
        'silhouette_emotion': silhouette_emotion,
        'silhouette_user': silhouette_user,
        'silhouette_asst': silhouette_asst,
        'centroids': centroids,
        'k': k,
    }


def plot_umap_comparison(results_dict, output_path):
    """Create comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Color per emotion
    emotion_colors = {
        'anger': '#e74c3c',
        'disgust': '#9b59b6', 
        'fear': '#3498db',
        'happiness': '#f39c12',
        'sadness': '#2ecc71',
        'surprise': '#1abc9c',
    }
    
    # Plot each result
    plot_configs = [
        ('k10_soft', 'K=10 Soft', axes[0, 0]),
        ('k10_hard', 'K=10 Hard (Gram-Schmidt)', axes[0, 1]),
        ('k20_soft', 'K=20 Soft', axes[1, 0]),
        ('k20_hard', 'K=20 Hard (Gram-Schmidt)', axes[1, 1]),
    ]
    
    for key, title, ax in plot_configs:
        if key not in results_dict:
            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center')
            ax.set_title(title)
            continue
        
        result = results_dict[key]
        embedding = result['embedding']
        emotion_indices = result['emotion_indices']
        role_indices = result['role_indices']
        emotion_labels = result['emotion_labels']
        
        # Plot user probes (circles)
        user_mask = role_indices == 0
        for emotion_idx, emotion in enumerate(emotion_labels):
            mask = (emotion_indices == emotion_idx) & user_mask
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=emotion_colors[emotion],
                marker='o',
                s=60,
                alpha=0.6,
                edgecolors='white',
                linewidth=0.5,
                label=f'{emotion.capitalize()} (user)' if emotion_idx < 3 else None
            )
        
        # Plot assistant probes (triangles)
        asst_mask = role_indices == 1
        for emotion_idx, emotion in enumerate(emotion_labels):
            mask = (emotion_indices == emotion_idx) & asst_mask
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=emotion_colors[emotion],
                marker='^',
                s=80,
                alpha=0.6,
                edgecolors='white',
                linewidth=0.5,
                label=f'{emotion.capitalize()} (asst)' if emotion_idx < 3 else None
            )
        
        ax.set_xlabel('UMAP 1', fontsize=11, fontweight='bold')
        ax.set_ylabel('UMAP 2', fontsize=11, fontweight='bold')
        
        title_text = f"{title}\nSilhouette: {result['silhouette_emotion']:.3f} " \
                     f"(user: {result['silhouette_user']:.3f}, asst: {result['silhouette_asst']:.3f})"
        ax.set_title(title_text, fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
    
    # Create legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = []
    for emotion in ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']:
        legend_elements.append(Patch(facecolor=emotion_colors[emotion], label=emotion.capitalize()))
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                  markersize=8, label='User'))
    legend_elements.append(Line2D([0], [0], marker='^', color='w', markerfacecolor='gray',
                                  markersize=10, label='Assistant'))
    
    fig.legend(handles=legend_elements, loc='upper center', ncol=8, 
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, 0.99))
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")


def main():
    base_dir = Path('/workspace-vast/annas/git/research-tools/probes/emotion_probes/conversation/multi_orthogonal')
    
    results = {}
    
    # K=10 soft
    path = base_dir / 'probe_k10_layer30_ortho100000.0.pkl'
    if path.exists():
        results['k10_soft'] = load_and_analyze(path, 10, 'Soft')
    
    # K=10 hard
    path = base_dir / 'probe_k10_layer30_ortho100000.0_gramschmidt.pkl'
    if path.exists():
        results['k10_hard'] = load_and_analyze(path, 10, 'Hard (Gram-Schmidt)')
    
    # K=20 soft
    path = base_dir / 'probe_k20_layer30_ortho100000.0.pkl'
    if path.exists():
        results['k20_soft'] = load_and_analyze(path, 20, 'Soft')
    
    # K=20 hard
    path = base_dir / 'probe_k20_layer30_ortho100000.0_gramschmidt.pkl'
    if path.exists():
        results['k20_hard'] = load_and_analyze(path, 20, 'Hard (Gram-Schmidt)')
    
    # Create plot
    output_path = '/workspace-vast/annas/git/research-tools/probes/scripts/analysis/conversation_probes_umap_comparison.png'
    plot_umap_comparison(results, output_path)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
