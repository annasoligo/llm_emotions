#!/usr/bin/env python3
"""Run UMAP analysis on K=50 orthogonal emotion probes."""

import sys
sys.path.insert(0, '/home/annas/.local/lib/python3.12/site-packages')

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.metrics import silhouette_score

# Load K=50 probes
probe_path = '/workspace-vast/annas/git/research-tools/probes/emotion_probes/text_based/multi_orthogonal/probe_k50_layer30_ortho100000.0.pkl'
with open(probe_path, 'rb') as f:
    data = pickle.load(f)

all_probe_sets = data['all_probe_sets']  # [50, 6, 5376]
emotion_labels = data['label_names']

print(f"Loaded probes shape: {all_probe_sets.shape}")
print(f"Emotions: {emotion_labels}")

# Reshape to [300, 5376]
n_sets, n_emotions, hidden_dim = all_probe_sets.shape
all_directions = all_probe_sets.reshape(n_sets * n_emotions, hidden_dim)

# Create labels
emotion_indices = []
probe_set_indices = []

for probe_idx in range(n_sets):
    for emotion_idx, emotion in enumerate(emotion_labels):
        emotion_indices.append(emotion_idx)
        probe_set_indices.append(probe_idx)

emotion_indices = np.array(emotion_indices)
probe_set_indices = np.array(probe_set_indices)

print(f"\nTotal directions: {all_directions.shape[0]}")

# Run UMAP with different parameters
print("\nRunning UMAP (this may take a minute)...")

# UMAP with default parameters
reducer_default = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
embedding_default = reducer_default.fit_transform(all_directions)

# UMAP with more neighbors (broader structure)
reducer_broad = umap.UMAP(n_neighbors=50, min_dist=0.1, metric='cosine', random_state=42)
embedding_broad = reducer_broad.fit_transform(all_directions)

# UMAP with fewer neighbors (local structure)
reducer_local = umap.UMAP(n_neighbors=5, min_dist=0.05, metric='cosine', random_state=42)
embedding_local = reducer_local.fit_transform(all_directions)

print("✓ UMAP complete")

# Create color scheme
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33']

# Create comprehensive plot
fig = plt.figure(figsize=(20, 12))

# 1. UMAP default - colored by emotion
ax1 = plt.subplot(2, 3, 1)
for emotion_idx, emotion in enumerate(emotion_labels):
    mask = emotion_indices == emotion_idx
    ax1.scatter(embedding_default[mask, 0], embedding_default[mask, 1],
               c=[colors[emotion_idx]], label=emotion, alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
ax1.set_xlabel('UMAP 1', fontsize=12)
ax1.set_ylabel('UMAP 2', fontsize=12)
ax1.set_title('UMAP (n_neighbors=15, colored by emotion)', fontsize=14, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.2)

# 2. UMAP broad structure
ax2 = plt.subplot(2, 3, 2)
for emotion_idx, emotion in enumerate(emotion_labels):
    mask = emotion_indices == emotion_idx
    ax2.scatter(embedding_broad[mask, 0], embedding_broad[mask, 1],
               c=[colors[emotion_idx]], label=emotion, alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
ax2.set_xlabel('UMAP 1', fontsize=12)
ax2.set_ylabel('UMAP 2', fontsize=12)
ax2.set_title('UMAP (n_neighbors=50, broad structure)', fontsize=14, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.2)

# 3. UMAP local structure
ax3 = plt.subplot(2, 3, 3)
for emotion_idx, emotion in enumerate(emotion_labels):
    mask = emotion_indices == emotion_idx
    ax3.scatter(embedding_local[mask, 0], embedding_local[mask, 1],
               c=[colors[emotion_idx]], label=emotion, alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
ax3.set_xlabel('UMAP 1', fontsize=12)
ax3.set_ylabel('UMAP 2', fontsize=12)
ax3.set_title('UMAP (n_neighbors=5, local structure)', fontsize=14, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.2)

# 4. UMAP default - colored by probe set (show first 10 probe sets)
ax4 = plt.subplot(2, 3, 4)
cmap = plt.cm.get_cmap('tab10')
for probe_idx in range(min(10, n_sets)):
    mask = probe_set_indices == probe_idx
    ax4.scatter(embedding_default[mask, 0], embedding_default[mask, 1],
               c=[cmap(probe_idx)], label=f'Set {probe_idx}', alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
ax4.set_xlabel('UMAP 1', fontsize=12)
ax4.set_ylabel('UMAP 2', fontsize=12)
ax4.set_title('UMAP colored by probe set (first 10 sets)', fontsize=14, fontweight='bold')
ax4.legend(loc='best', fontsize=8, ncol=2)
ax4.grid(True, alpha=0.2)

# 5. Density plot
ax5 = plt.subplot(2, 3, 5)
ax5.hexbin(embedding_default[:, 0], embedding_default[:, 1], gridsize=30, cmap='YlOrRd', mincnt=1)
ax5.set_xlabel('UMAP 1', fontsize=12)
ax5.set_ylabel('UMAP 2', fontsize=12)
ax5.set_title('UMAP Density', fontsize=14, fontweight='bold')

# 6. Emotion centroids
ax6 = plt.subplot(2, 3, 6)
centroids = []
for emotion_idx, emotion in enumerate(emotion_labels):
    mask = emotion_indices == emotion_idx
    centroid = np.mean(embedding_default[mask], axis=0)
    centroids.append(centroid)
    ax6.scatter(embedding_default[mask, 0], embedding_default[mask, 1],
               c=[colors[emotion_idx]], alpha=0.3, s=30)
    ax6.scatter(centroid[0], centroid[1], c=[colors[emotion_idx]], s=500,
               marker='*', edgecolors='k', linewidth=2, label=emotion)
    ax6.annotate(emotion, centroid, fontsize=12, fontweight='bold',
                ha='center', va='center')

ax6.set_xlabel('UMAP 1', fontsize=12)
ax6.set_ylabel('UMAP 2', fontsize=12)
ax6.set_title('UMAP with Emotion Centroids', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.2)

plt.tight_layout()
output_path = '/workspace-vast/annas/git/research-tools/probes/emotion_probes/text_based/multi_orthogonal/k50_umap_analysis.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Compute statistics
print("\n" + "=" * 80)
print("UMAP Analysis Statistics:")
print("=" * 80)

# Compute within-emotion and between-emotion distances
centroids = np.array(centroids)

print("\nEmotion centroid distances (in UMAP space):")
for i, emotion1 in enumerate(emotion_labels):
    for j, emotion2 in enumerate(emotion_labels):
        if i < j:
            dist = np.linalg.norm(centroids[i] - centroids[j])
            print(f"  {emotion1:12} <-> {emotion2:12}: {dist:.3f}")

print("\nWithin-emotion spread (std deviation in UMAP space):")
for emotion_idx, emotion in enumerate(emotion_labels):
    mask = emotion_indices == emotion_idx
    std = np.std(embedding_default[mask], axis=0)
    mean_std = np.mean(std)
    print(f"  {emotion:12}: {mean_std:.3f}")

# Check if emotions cluster or are distributed
print("\nClustering analysis:")
sil_score = silhouette_score(embedding_default, emotion_indices)
print(f"  Silhouette score (emotion labels): {sil_score:.4f}")
print(f"    (Range: -1 to 1, higher = better separated clusters)")

print("\n✓ Analysis complete!")
