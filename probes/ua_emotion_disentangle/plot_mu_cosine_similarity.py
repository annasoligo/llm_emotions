#!/usr/bin/env python3
"""
Plot cosine similarity between M (model) and U (user) emotion directions across layers.

Creates:
1. Main plot: Mean cosine sim across emotions, one line per model
2. 3 subplots: Per-emotion cosine sim for each model
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


def compute_mu_directions_for_layer(
    h5_path: Path,
    layer: int,
    position: str = 'first_asst_token'
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute M and U direction vectors for a single layer."""
    with h5py.File(h5_path, 'r') as f:
        layer_group = f[f'layer_{layer}']
        activations = layer_group[f'{position}_activations'][:]
        M_values = [m.decode() for m in layer_group[f'{position}_M'][:]]
        U_values = [u.decode() for u in layer_group[f'{position}_U'][:]]

    # Group by emotion
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
        if emotion in OPPOSITES and OPPOSITES[emotion] in M_means:
            M_directions[emotion] = M_means[emotion] - M_means[OPPOSITES[emotion]]
        if emotion in OPPOSITES and OPPOSITES[emotion] in U_means:
            U_directions[emotion] = U_means[emotion] - U_means[OPPOSITES[emotion]]

    return M_directions, U_directions


def compute_mu_similarity_across_layers(
    h5_path: Path,
    layers: List[int],
) -> Dict[str, Dict[int, float]]:
    """
    Compute cosine similarity between M and U directions for each emotion across layers.

    Returns:
        {emotion: {layer: cosine_sim}}
    """
    results = defaultdict(dict)

    for layer in layers:
        M_dirs, U_dirs = compute_mu_directions_for_layer(h5_path, layer)

        for emotion in M_dirs:
            if emotion in U_dirs:
                sim = cosine_similarity(M_dirs[emotion], U_dirs[emotion])
                results[emotion][layer] = sim

    return dict(results)


def plot_mu_similarity(
    model_data: Dict[str, Dict[str, Dict[int, float]]],
    output_path: Path,
):
    """
    Create the visualization.

    Args:
        model_data: {model_name: {emotion: {layer: cosine_sim}}}
    """
    models = list(model_data.keys())
    n_models = len(models)

    # Get all layers (assume same for all models)
    first_model = models[0]
    first_emotion = list(model_data[first_model].keys())[0]
    layers = sorted(model_data[first_model][first_emotion].keys())

    # Set up figure: main plot + one subplot per model
    fig = plt.figure(figsize=(14, 10))

    # Main plot on top
    ax_main = fig.add_subplot(2, 1, 1)

    # Model colors
    colors = {'Qwen': '#E24A33', 'OLMo': '#348ABD', 'Gemma': '#988ED5'}

    # Plot mean similarity for each model
    for model in models:
        emotion_data = model_data[model]

        # Compute mean across emotions for each layer
        mean_sims = []
        std_sims = []
        for layer in layers:
            sims = [emotion_data[e].get(layer, np.nan) for e in emotion_data]
            sims = [s for s in sims if not np.isnan(s)]
            mean_sims.append(np.mean(sims) if sims else np.nan)
            std_sims.append(np.std(sims) if sims else np.nan)

        color = colors.get(model, '#333333')
        ax_main.plot(layers, mean_sims, 'o-', label=model, color=color, linewidth=2, markersize=4)
        ax_main.fill_between(layers,
                             np.array(mean_sims) - np.array(std_sims),
                             np.array(mean_sims) + np.array(std_sims),
                             alpha=0.2, color=color)

    ax_main.set_xlabel('Layer', fontsize=12)
    ax_main.set_ylabel('Cosine Similarity (M vs U)', fontsize=12)
    ax_main.set_title('Mean M-U Direction Similarity Across Layers', fontsize=14)
    ax_main.legend(loc='best')
    ax_main.grid(True, alpha=0.3)
    ax_main.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax_main.set_ylim(-0.2, 1.0)

    # Per-model subplots
    axes = []
    for i, model in enumerate(models):
        ax = fig.add_subplot(2, n_models, n_models + i + 1)
        axes.append(ax)

        emotion_data = model_data[model]

        # Get a colormap for emotions
        emotions = sorted(emotion_data.keys())
        cmap = plt.cm.tab20

        for j, emotion in enumerate(emotions):
            layer_sims = emotion_data[emotion]
            x = sorted(layer_sims.keys())
            y = [layer_sims[l] for l in x]
            ax.plot(x, y, alpha=0.6, linewidth=1, color=cmap(j % 20))

        # Compute and plot mean
        mean_sims = []
        for layer in layers:
            sims = [emotion_data[e].get(layer, np.nan) for e in emotion_data]
            sims = [s for s in sims if not np.isnan(s)]
            mean_sims.append(np.mean(sims) if sims else np.nan)
        ax.plot(layers, mean_sims, 'k-', linewidth=2.5, label='Mean')

        ax.set_xlabel('Layer', fontsize=10)
        ax.set_ylabel('Cosine Sim' if i == 0 else '', fontsize=10)
        ax.set_title(f'{model}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylim(-0.2, 1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    data_dir = Path('probes/ua_emotion_disentangle/data')
    output_dir = Path('probes/ua_emotion_disentangle/plots')
    output_dir.mkdir(exist_ok=True)

    # Define models and their data files
    models = {
        'Qwen': data_dir / 'qwen3_32b_ua_emotions_v2.h5',
        'OLMo': data_dir / 'olmo_32b_ua_emotions_v2.h5',
    }

    # Check for Gemma data
    gemma_path = data_dir / 'gemma3_27b_ua_emotions_v2.h5'
    if gemma_path.exists():
        models['Gemma'] = gemma_path

    # Get layers from first model
    with h5py.File(list(models.values())[0], 'r') as f:
        layers = [int(l) for l in f.attrs['layers']]

    print(f"Computing M-U similarity for {len(models)} models across {len(layers)} layers")
    print(f"Layers: {min(layers)} to {max(layers)}")

    # Compute similarities for each model
    model_data = {}
    for model_name, h5_path in models.items():
        print(f"\nProcessing {model_name}...")
        if not h5_path.exists():
            print(f"  Skipping - file not found: {h5_path}")
            continue

        model_data[model_name] = compute_mu_similarity_across_layers(h5_path, layers)
        n_emotions = len(model_data[model_name])
        print(f"  Computed for {n_emotions} emotions")

    # Plot
    output_path = output_dir / 'mu_cosine_similarity_across_layers.png'
    plot_mu_similarity(model_data, output_path)

    # Print summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_name, emotion_data in model_data.items():
        all_sims = []
        for emotion, layer_sims in emotion_data.items():
            all_sims.extend(layer_sims.values())
        print(f"{model_name}:")
        print(f"  Mean M-U similarity: {np.mean(all_sims):.3f}")
        print(f"  Std: {np.std(all_sims):.3f}")
        print(f"  Min: {np.min(all_sims):.3f}, Max: {np.max(all_sims):.3f}")


if __name__ == '__main__':
    main()
