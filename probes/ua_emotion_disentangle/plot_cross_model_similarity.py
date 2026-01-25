#!/usr/bin/env python3
"""
Plot cosine similarity of emotion directions across models.

Creates:
1. Main plot: Mean cross-model similarity of M directions across layers
2. Subplots: Per-emotion cross-model similarity (Qwen vs OLMo)
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Plutchik emotion pairs
EMOTION_PAIRS = [
    ('joy', 'sadness'),
    ('trust', 'disgust'),
    ('fear', 'anger'),
    ('surprise', 'anticipation'),
    ('ecstasy', 'grief'),
    ('admiration', 'loathing'),
    ('terror', 'rage'),
    ('amazement', 'vigilance'),
    ('serenity', 'pensiveness'),
    ('acceptance', 'boredom'),
    ('apprehension', 'annoyance'),
    ('distraction', 'interest'),
    ('awe', 'contempt'),
    ('submission', 'aggressiveness'),
    ('love', 'remorse'),
    ('optimism', 'disappointment'),
]

OPPOSITES = {}
for e1, e2 in EMOTION_PAIRS:
    OPPOSITES[e1] = e2
    OPPOSITES[e2] = e1


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_m_directions_for_layer(
    h5_path: Path,
    layer: int,
    position: str = 'first_asst_token'
) -> Dict[str, np.ndarray]:
    """Compute M direction vectors for a single layer."""
    with h5py.File(h5_path, 'r') as f:
        layer_group = f[f'layer_{layer}']
        activations = layer_group[f'{position}_activations'][:]
        M_values = [m.decode() for m in layer_group[f'{position}_M'][:]]

    # Group by emotion
    M_groups = defaultdict(list)
    for i, m in enumerate(M_values):
        M_groups[m].append(activations[i])

    # Compute means
    M_means = {e: np.mean(acts, axis=0) for e, acts in M_groups.items()}

    # Compute directions (emotion - opposite)
    M_directions = {}
    for emotion in M_means:
        if emotion in OPPOSITES and OPPOSITES[emotion] in M_means:
            M_directions[emotion] = M_means[emotion] - M_means[OPPOSITES[emotion]]

    return M_directions


def compute_cross_model_similarity(
    model1_path: Path,
    model2_path: Path,
    layers: List[int],
) -> Dict[str, Dict[int, float]]:
    """
    Compute cosine similarity between model1 and model2 M directions.

    Returns:
        {emotion: {layer: cosine_sim}}
    """
    results = defaultdict(dict)

    for layer in layers:
        m1_dirs = compute_m_directions_for_layer(model1_path, layer)
        m2_dirs = compute_m_directions_for_layer(model2_path, layer)

        for emotion in m1_dirs:
            if emotion in m2_dirs:
                sim = cosine_similarity(m1_dirs[emotion], m2_dirs[emotion])
                results[emotion][layer] = sim

    return dict(results)


def compute_within_model_emotion_similarity(
    h5_path: Path,
    layers: List[int],
) -> Dict[int, float]:
    """
    Compute mean pairwise cosine similarity between emotion directions within a model.

    Returns:
        {layer: mean_pairwise_sim}
    """
    results = {}

    for layer in layers:
        m_dirs = compute_m_directions_for_layer(h5_path, layer)
        emotions = list(m_dirs.keys())

        # Compute pairwise similarities
        sims = []
        for i, e1 in enumerate(emotions):
            for e2 in emotions[i+1:]:
                sim = cosine_similarity(m_dirs[e1], m_dirs[e2])
                sims.append(sim)

        results[layer] = np.mean(sims) if sims else 0.0

    return results


def plot_cross_model_similarity(
    qwen_path: Path,
    olmo_path: Path,
    layers: List[int],
    output_path: Path,
):
    """Create the cross-model similarity visualization."""
    print("Computing cross-model M direction similarity...")
    cross_sim = compute_cross_model_similarity(qwen_path, olmo_path, layers)

    print("Computing within-model emotion similarity...")
    qwen_within = compute_within_model_emotion_similarity(qwen_path, layers)
    olmo_within = compute_within_model_emotion_similarity(olmo_path, layers)

    # Set up figure
    fig = plt.figure(figsize=(14, 10))

    # Main plot: Cross-model similarity
    ax_main = fig.add_subplot(2, 2, (1, 2))

    # Compute mean cross-model similarity per layer
    mean_cross_sims = []
    std_cross_sims = []
    for layer in layers:
        sims = [cross_sim[e].get(layer, np.nan) for e in cross_sim]
        sims = [s for s in sims if not np.isnan(s)]
        mean_cross_sims.append(np.mean(sims) if sims else np.nan)
        std_cross_sims.append(np.std(sims) if sims else np.nan)

    ax_main.plot(layers, mean_cross_sims, 'o-', color='#E24A33', linewidth=2,
                 markersize=5, label='Qwen-OLMo (cross-model)')
    ax_main.fill_between(layers,
                         np.array(mean_cross_sims) - np.array(std_cross_sims),
                         np.array(mean_cross_sims) + np.array(std_cross_sims),
                         alpha=0.2, color='#E24A33')

    # Add within-model baselines
    ax_main.plot(layers, [qwen_within[l] for l in layers], '--', color='#348ABD',
                 linewidth=1.5, alpha=0.7, label='Qwen (within-model emotion pairs)')
    ax_main.plot(layers, [olmo_within[l] for l in layers], '--', color='#988ED5',
                 linewidth=1.5, alpha=0.7, label='OLMo (within-model emotion pairs)')

    ax_main.set_xlabel('Layer', fontsize=12)
    ax_main.set_ylabel('Cosine Similarity', fontsize=12)
    ax_main.set_title('Cross-Model M Direction Similarity (Qwen vs OLMo)', fontsize=14)
    ax_main.legend(loc='best')
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Subplot: Per-emotion cross-model similarity
    ax_emotions = fig.add_subplot(2, 2, 3)

    emotions = sorted(cross_sim.keys())
    cmap = plt.cm.tab20

    for j, emotion in enumerate(emotions):
        layer_sims = cross_sim[emotion]
        x = sorted(layer_sims.keys())
        y = [layer_sims[l] for l in x]
        ax_emotions.plot(x, y, alpha=0.6, linewidth=1, color=cmap(j % 20), label=emotion)

    # Mean line
    ax_emotions.plot(layers, mean_cross_sims, 'k-', linewidth=2.5, label='Mean')

    ax_emotions.set_xlabel('Layer', fontsize=10)
    ax_emotions.set_ylabel('Cosine Similarity', fontsize=10)
    ax_emotions.set_title('Per-Emotion Cross-Model Similarity', fontsize=12)
    ax_emotions.grid(True, alpha=0.3)
    ax_emotions.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Subplot: Heatmap of final layer similarities
    ax_heatmap = fig.add_subplot(2, 2, 4)

    # Get similarities at layer 30 for heatmap
    target_layer = 30
    emotion_sims = [(e, cross_sim[e].get(target_layer, 0)) for e in emotions]
    emotion_sims.sort(key=lambda x: x[1], reverse=True)

    y_pos = np.arange(len(emotion_sims))
    sims = [s for _, s in emotion_sims]
    labels = [e for e, _ in emotion_sims]

    colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in sims]
    ax_heatmap.barh(y_pos, sims, color=colors, alpha=0.7)
    ax_heatmap.set_yticks(y_pos)
    ax_heatmap.set_yticklabels(labels, fontsize=8)
    ax_heatmap.set_xlabel('Cosine Similarity', fontsize=10)
    ax_heatmap.set_title(f'Cross-Model Similarity at Layer {target_layer}', fontsize=12)
    ax_heatmap.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax_heatmap.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Cross-model (Qwen-OLMo) M direction similarity:")
    print(f"  Mean: {np.mean(mean_cross_sims):.3f}")
    print(f"  Std: {np.std(mean_cross_sims):.3f}")
    print(f"  Range: {np.min(mean_cross_sims):.3f} to {np.max(mean_cross_sims):.3f}")
    print()
    print("Top 5 most similar emotions (layer 30):")
    for e, s in emotion_sims[:5]:
        print(f"  {e}: {s:.3f}")
    print()
    print("Bottom 5 least similar emotions (layer 30):")
    for e, s in emotion_sims[-5:]:
        print(f"  {e}: {s:.3f}")


def main():
    data_dir = Path('probes/ua_emotion_disentangle/data')
    output_dir = Path('probes/ua_emotion_disentangle/plots')
    output_dir.mkdir(exist_ok=True)

    qwen_path = data_dir / 'qwen3_32b_ua_emotions_v2.h5'
    olmo_path = data_dir / 'olmo_32b_ua_emotions_v2.h5'

    if not qwen_path.exists():
        print(f"Qwen data not found: {qwen_path}")
        return
    if not olmo_path.exists():
        print(f"OLMo data not found: {olmo_path}")
        return

    # Get layers
    with h5py.File(qwen_path, 'r') as f:
        layers = [int(l) for l in f.attrs['layers']]

    print(f"Comparing Qwen and OLMo across {len(layers)} layers ({min(layers)}-{max(layers)})")

    output_path = output_dir / 'cross_model_m_direction_similarity.png'
    plot_cross_model_similarity(qwen_path, olmo_path, layers, output_path)


if __name__ == '__main__':
    main()
