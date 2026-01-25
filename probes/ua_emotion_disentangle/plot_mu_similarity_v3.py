#!/usr/bin/env python3
"""
Plot M vs U direction cosine similarity across layers for all three models.

Uses v3 data with full 32x32 M-U factorial design.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List

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


def load_layer_data(h5_path: Path, layer: int, position: str = 'first_asst_token'):
    """Load activations and metadata for a single layer."""
    with h5py.File(h5_path, 'r') as f:
        layer_group = f[f'layer_{layer}']
        activations = layer_group[f'{position}_activations'][:]
        M_values = [m.decode() for m in layer_group[f'{position}_M'][:]]
        U_values = [u.decode() for u in layer_group[f'{position}_U'][:]]
    return activations, M_values, U_values


def compute_directions(activations: np.ndarray, M_values: list, U_values: list):
    """Compute M and U direction vectors for each emotion."""
    # Group by M and U values
    M_groups = defaultdict(list)
    U_groups = defaultdict(list)

    for i, (m, u) in enumerate(zip(M_values, U_values)):
        M_groups[m].append(activations[i])
        U_groups[u].append(activations[i])

    # Compute means
    M_means = {e: np.mean(acts, axis=0) for e, acts in M_groups.items()}
    U_means = {e: np.mean(acts, axis=0) for e, acts in U_groups.items()}

    # Compute directions (emotion - opposite)
    M_directions = {}
    U_directions = {}

    for emotion in M_means:
        if emotion not in OPPOSITES:
            continue
        opposite = OPPOSITES[emotion]
        if opposite not in M_means:
            continue

        # M direction (normalized)
        m_dir = M_means[emotion] - M_means[opposite]
        m_norm = np.linalg.norm(m_dir)
        if m_norm > 1e-8:
            M_directions[emotion] = m_dir / m_norm

        # U direction (normalized)
        if emotion in U_means and opposite in U_means:
            u_dir = U_means[emotion] - U_means[opposite]
            u_norm = np.linalg.norm(u_dir)
            if u_norm > 1e-8:
                U_directions[emotion] = u_dir / u_norm

    return M_directions, U_directions


def compute_mu_similarity(M_directions: Dict, U_directions: Dict) -> Dict[str, float]:
    """Compute cosine similarity between M and U directions for each emotion."""
    similarities = {}
    for emotion in M_directions:
        if emotion in U_directions:
            # Directions are already normalized
            sim = np.dot(M_directions[emotion], U_directions[emotion])
            similarities[emotion] = float(sim)
    return similarities


def analyze_model(h5_path: Path, layers: List[int]) -> Tuple[Dict, Dict]:
    """Analyze a single model across all layers."""
    layer_mean_sims = {}
    layer_emotion_sims = {}

    for layer in layers:
        activations, M_values, U_values = load_layer_data(h5_path, layer)
        M_dirs, U_dirs = compute_directions(activations, M_values, U_values)
        sims = compute_mu_similarity(M_dirs, U_dirs)

        layer_mean_sims[layer] = np.mean(list(sims.values()))
        layer_emotion_sims[layer] = sims

    return layer_mean_sims, layer_emotion_sims


def main():
    data_dir = Path(__file__).parent / 'data'
    output_dir = Path(__file__).parent / 'plots'
    output_dir.mkdir(exist_ok=True)

    # Model configs - try v4 (all layers) first, fall back to v3
    models = {
        'Qwen-3-32B': {
            'path': data_dir / 'qwen3_32b_ua_emotions_v4_alllayers.h5',
            'fallback': data_dir / 'qwen3_32b_ua_emotions_v3.h5',
            'color': '#1f77b4',
        },
        'OLMo-3-32B': {
            'path': data_dir / 'olmo_32b_ua_emotions_v4_alllayers.h5',
            'fallback': data_dir / 'olmo_32b_ua_emotions_v3.h5',
            'color': '#ff7f0e',
        },
        'Gemma-3-27B': {
            'path': data_dir / 'gemma3_27b_ua_emotions_v4_alllayers.h5',
            'fallback': data_dir / 'gemma3_27b_ua_emotions_v3.h5',
            'color': '#2ca02c',
        },
    }

    # Resolve paths and get available layers
    for name, config in models.items():
        if config['path'].exists():
            pass  # Use v4
        elif config['fallback'].exists():
            config['path'] = config['fallback']

        # Get layers from file
        if config['path'].exists():
            with h5py.File(config['path'], 'r') as f:
                config['layers'] = sorted([int(k.split('_')[1]) for k in f.keys() if k.startswith('layer_')])

    # Analyze each model
    results = {}
    for name, config in models.items():
        print(f"Analyzing {name}...")
        if not config['path'].exists():
            print(f"  WARNING: File not found: {config['path']}")
            continue
        mean_sims, emotion_sims = analyze_model(config['path'], config['layers'])
        results[name] = {
            'mean_sims': mean_sims,
            'emotion_sims': emotion_sims,
            'layers': config['layers'],
            'color': config['color'],
        }
        print(f"  Mean M-U similarity: {np.mean(list(mean_sims.values())):.3f}")

    # Define consistent colors for each emotion pair
    # Using a colormap to get distinct colors for 16 pairs
    cmap = plt.cm.tab20
    emotion_colors = {}
    for i, (e1, e2) in enumerate(EMOTION_PAIRS):
        color = cmap(i / len(EMOTION_PAIRS))
        emotion_colors[e1] = color
        emotion_colors[e2] = color  # Same color for opposite pair

    # Create figure with space for legend
    fig = plt.figure(figsize=(18, 10))

    # Use GridSpec for better layout control
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1, 1, 0.4])

    # Main plot: mean similarity across layers
    ax_main = fig.add_subplot(gs[0, 0])
    for name, data in results.items():
        layers = sorted(data['mean_sims'].keys())
        sims = [data['mean_sims'][l] for l in layers]
        ax_main.plot(layers, sims, 'o-', label=name, color=data['color'], linewidth=2, markersize=4)

    ax_main.set_xlabel('Layer', fontsize=12)
    ax_main.set_ylabel('Mean M-U Cosine Similarity', fontsize=12)
    ax_main.set_title('Model vs User Emotion Direction Similarity\n(Mean across all emotion pairs)', fontsize=14)
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_main.set_ylim(-0.2, 1.0)

    # Per-model subplots showing per-emotion similarities
    subplot_positions = [(0, 1), (1, 0), (1, 1)]  # positions in grid
    for idx, (name, data) in enumerate(results.items()):
        row, col = subplot_positions[idx]
        ax = fig.add_subplot(gs[row, col])

        # Get all emotions
        sample_layer = list(data['emotion_sims'].keys())[0]
        emotions = sorted(data['emotion_sims'][sample_layer].keys())

        # Plot each emotion with consistent colors
        layers = sorted(data['emotion_sims'].keys())
        for emotion in emotions:
            sims = [data['emotion_sims'][l].get(emotion, np.nan) for l in layers]
            ax.plot(layers, sims, '-', alpha=0.6, linewidth=1.2,
                   color=emotion_colors.get(emotion, 'gray'), label=emotion)

        # Add mean line
        mean_sims = [data['mean_sims'][l] for l in layers]
        ax.plot(layers, mean_sims, 'k-', linewidth=3, label='Mean', zorder=10)

        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('M-U Cosine Similarity', fontsize=10)
        ax.set_title(f'{name}: Per-Emotion M-U Similarity', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(-0.2, 1.0)

    # Legend subplot on the right
    ax_legend = fig.add_subplot(gs[:, 2])
    ax_legend.axis('off')

    # Create legend entries for emotion pairs
    from matplotlib.lines import Line2D
    legend_elements = []
    for i, (e1, e2) in enumerate(EMOTION_PAIRS):
        color = cmap(i / len(EMOTION_PAIRS))
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=2, label=f'{e1} / {e2}')
        )
    # Add mean line to legend
    legend_elements.append(
        Line2D([0], [0], color='black', linewidth=3, label='Mean')
    )

    ax_legend.legend(handles=legend_elements, loc='center', fontsize=10,
                    title='Emotion Pairs', title_fontsize=12, frameon=True)

    plt.tight_layout()

    # Save
    output_path = output_dir / 'mu_cosine_similarity_v3.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot to {output_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for name, data in results.items():
        mean_sims = list(data['mean_sims'].values())
        print(f"\n{name}:")
        print(f"  Mean M-U similarity: {np.mean(mean_sims):.3f} +/- {np.std(mean_sims):.3f}")
        print(f"  Range: [{np.min(mean_sims):.3f}, {np.max(mean_sims):.3f}]")

        # Find layer with highest/lowest similarity
        layers = sorted(data['mean_sims'].keys())
        sims = [data['mean_sims'][l] for l in layers]
        max_idx = np.argmax(sims)
        min_idx = np.argmin(sims)
        print(f"  Max at layer {layers[max_idx]}: {sims[max_idx]:.3f}")
        print(f"  Min at layer {layers[min_idx]}: {sims[min_idx]:.3f}")


if __name__ == '__main__':
    main()
