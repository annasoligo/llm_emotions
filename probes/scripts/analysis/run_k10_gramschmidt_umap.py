#!/usr/bin/env python3
"""
UMAP analysis on K=10 Gram-Schmidt orthogonal probes.
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

# Load K=10 Gram-Schmidt probes
probe_path = '/workspace-vast/annas/git/research-tools/probes/emotion_probes/text_based/multi_orthogonal/probe_k10_layer30_ortho100000.0_gramschmidt.pkl'

print("=" * 80)
print("UMAP ANALYSIS: K=10 GRAM-SCHMIDT")
print("=" * 80)

with open(probe_path, 'rb') as f:
    probe_data = pickle.load(f)

all_probe_sets = probe_data['all_probe_sets']  # [10, 6, 5376]
emotion_labels = probe_data['label_names']

print(f"Probe shape: {all_probe_sets.shape}")
print(f"Emotions: {emotion_labels}")
print(f"K value: {all_probe_sets.shape[0]}")

# Flatten to [60, 5376] (10 sets × 6 emotions)
n_sets, n_emotions, hidden_dim = all_probe_sets.shape
probes_flat = all_probe_sets.reshape(n_sets * n_emotions, hidden_dim)

# Create labels for coloring
emotion_indices = np.repeat(np.arange(n_emotions), n_sets)

print(f"\nFlattened shape: {probes_flat.shape}")
print(f"Total vectors: {probes_flat.shape[0]} (should be {n_sets * n_emotions})")

# Compute pairwise cosine similarities
print("\n" + "=" * 80)
print("ORTHOGONALITY CHECK")
print("=" * 80)

from scipy.spatial.distance import pdist
cosine_dists = pdist(probes_flat, metric='cosine')
cosine_sims = 1 - cosine_dists

print(f"Mean |cosine similarity|: {np.mean(np.abs(cosine_sims)):.6f}")
print(f"Max |cosine similarity|: {np.max(np.abs(cosine_sims)):.6f}")
print(f"Fraction with |cos| < 0.01: {np.mean(np.abs(cosine_sims) < 0.01):.4f}")

# Run UMAP
print("\n" + "=" * 80)
print("RUNNING UMAP")
print("=" * 80)

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='cosine',
    random_state=42
)

embedding = reducer.fit_transform(probes_flat)
print(f"✓ UMAP embedding shape: {embedding.shape}")

# Compute silhouette score
silhouette = silhouette_score(embedding, emotion_indices, metric='euclidean')
print(f"\nSilhouette score (emotion clustering): {silhouette:.4f}")

# Compute emotion centroids in UMAP space
print("\n" + "=" * 80)
print("EMOTION CENTROIDS IN UMAP SPACE")
print("=" * 80)

emotion_centroids = {}
for emotion_idx, emotion in enumerate(emotion_labels):
    mask = emotion_indices == emotion_idx
    centroid = embedding[mask].mean(axis=0)
    emotion_centroids[emotion] = centroid
    print(f"{emotion.capitalize():12} - Centroid: ({centroid[0]:>7.3f}, {centroid[1]:>7.3f})")

# Compute within-emotion variance
print("\n" + "=" * 80)
print("WITHIN-EMOTION VARIANCE")
print("=" * 80)

for emotion_idx, emotion in enumerate(emotion_labels):
    mask = emotion_indices == emotion_idx
    points = embedding[mask]
    centroid = emotion_centroids[emotion]

    # Compute variance (mean squared distance to centroid)
    variance = np.mean(np.sum((points - centroid)**2, axis=1))
    print(f"{emotion.capitalize():12} - Variance: {variance:.4f}")

# Compute inter-emotion distances
print("\n" + "=" * 80)
print("INTER-EMOTION DISTANCES")
print("=" * 80)

pairs = [
    ('happiness', 'sadness'),
    ('anger', 'fear'),
    ('surprise', 'disgust'),
    ('happiness', 'anger'),
    ('fear', 'sadness'),
]

for emo1, emo2 in pairs:
    c1 = emotion_centroids[emo1]
    c2 = emotion_centroids[emo2]
    dist = np.linalg.norm(c1 - c2)
    print(f"{emo1.capitalize():12} ↔ {emo2.capitalize():12} = {dist:.4f}")

# Create plot
print("\n" + "=" * 80)
print("CREATING PLOT")
print("=" * 80)

fig, ax = plt.subplots(figsize=(12, 10))

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
emotion_colors = {emotion: colors[i] for i, emotion in enumerate(emotion_labels)}

# Plot individual points
for emotion_idx, emotion in enumerate(emotion_labels):
    mask = emotion_indices == emotion_idx
    ax.scatter(
        embedding[mask, 0],
        embedding[mask, 1],
        c=[emotion_colors[emotion]],
        label=emotion.capitalize(),
        alpha=0.6,
        s=80,
        edgecolors='white',
        linewidth=0.5
    )

# Plot centroids
for emotion in emotion_labels:
    centroid = emotion_centroids[emotion]
    ax.scatter(
        centroid[0],
        centroid[1],
        c=emotion_colors[emotion],
        marker='*',
        s=800,
        edgecolors='black',
        linewidth=2,
        zorder=10
    )

ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
ax.set_title(f'K=10 Gram-Schmidt: UMAP Projection\nSilhouette Score: {silhouette:.4f}',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = '/workspace-vast/annas/git/research-tools/probes/scripts/analysis/k10_gramschmidt_umap.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved to: {output_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
