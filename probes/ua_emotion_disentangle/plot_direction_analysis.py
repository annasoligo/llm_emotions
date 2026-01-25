#!/usr/bin/env python3
"""
Analyze M emotion directions across layers.

Creates:
1. Main plot: Direction norm (signal strength) across layers per model
2. Subplots: Layer-to-layer consistency (how stable are directions across layers)
3. Within-model emotion similarity (are different emotions pointing same direction?)
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

EMOTION_PAIRS = [
    ('joy', 'sadness'), ('trust', 'disgust'), ('fear', 'anger'),
    ('surprise', 'anticipation'), ('ecstasy', 'grief'), ('admiration', 'loathing'),
    ('terror', 'rage'), ('amazement', 'vigilance'), ('serenity', 'pensiveness'),
    ('acceptance', 'boredom'), ('apprehension', 'annoyance'), ('distraction', 'interest'),
    ('awe', 'contempt'), ('submission', 'aggressiveness'), ('love', 'remorse'),
    ('optimism', 'disappointment'),
]

OPPOSITES = {e1: e2 for e1, e2 in EMOTION_PAIRS}
OPPOSITES.update({e2: e1 for e1, e2 in EMOTION_PAIRS})


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_m_directions(h5_path: Path, layer: int) -> Dict[str, np.ndarray]:
    """Compute M direction vectors for a layer."""
    with h5py.File(h5_path, 'r') as f:
        lg = f[f'layer_{layer}']
        acts = lg['first_asst_token_activations'][:]
        M_vals = [m.decode() for m in lg['first_asst_token_M'][:]]

    groups = defaultdict(list)
    for i, m in enumerate(M_vals):
        groups[m].append(acts[i])

    means = {e: np.mean(a, axis=0) for e, a in groups.items()}
    dirs = {}
    for e in means:
        if e in OPPOSITES and OPPOSITES[e] in means:
            dirs[e] = means[e] - means[OPPOSITES[e]]
    return dirs


def analyze_model(h5_path: Path, layers: List[int], model_name: str):
    """Analyze directions for a model."""
    print(f"\nAnalyzing {model_name}...")

    # Store directions for all layers
    all_dirs = {l: compute_m_directions(h5_path, l) for l in layers}
    emotions = list(all_dirs[layers[0]].keys())

    # 1. Direction norms per layer
    norms = {e: [] for e in emotions}
    for l in layers:
        for e in emotions:
            norms[e].append(np.linalg.norm(all_dirs[l][e]))

    # 2. Layer-to-layer consistency
    consistency = {e: [] for e in emotions}
    for i, l in enumerate(layers[:-1]):
        next_l = layers[i + 1]
        for e in emotions:
            sim = cosine_similarity(all_dirs[l][e], all_dirs[next_l][e])
            consistency[e].append(sim)

    # 3. Within-model emotion pairwise similarity
    within_sim = []
    for l in layers:
        sims = []
        for i, e1 in enumerate(emotions):
            for e2 in emotions[i+1:]:
                sims.append(abs(cosine_similarity(all_dirs[l][e1], all_dirs[l][e2])))
        within_sim.append(np.mean(sims))

    return {
        'norms': norms,
        'consistency': consistency,
        'within_sim': within_sim,
        'emotions': emotions,
    }


def plot_analysis(results: Dict, layers: List[int], output_path: Path):
    """Create visualization."""
    models = list(results.keys())
    colors = {'Qwen': '#E24A33', 'OLMo': '#348ABD'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Top-left: Mean direction norm across layers
    ax = axes[0, 0]
    for model in models:
        data = results[model]
        mean_norms = [np.mean([data['norms'][e][i] for e in data['emotions']]) for i in range(len(layers))]
        std_norms = [np.std([data['norms'][e][i] for e in data['emotions']]) for i in range(len(layers))]
        ax.plot(layers, mean_norms, 'o-', label=model, color=colors.get(model, '#333'), linewidth=2)
        ax.fill_between(layers, np.array(mean_norms) - np.array(std_norms),
                        np.array(mean_norms) + np.array(std_norms), alpha=0.2, color=colors.get(model, '#333'))
    ax.set_xlabel('Layer')
    ax.set_ylabel('Direction Norm')
    ax.set_title('Mean M Direction Norm (Signal Strength)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Top-right: Layer-to-layer consistency
    ax = axes[0, 1]
    for model in models:
        data = results[model]
        mean_cons = [np.mean([data['consistency'][e][i] for e in data['emotions']]) for i in range(len(layers)-1)]
        ax.plot(layers[:-1], mean_cons, 'o-', label=model, color=colors.get(model, '#333'), linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Sim (layer N → N+1)')
    ax.set_title('Layer-to-Layer Direction Consistency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # 3. Bottom-left: Within-model emotion similarity
    ax = axes[1, 0]
    for model in models:
        data = results[model]
        ax.plot(layers, data['within_sim'], 'o-', label=model, color=colors.get(model, '#333'), linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean |Cosine Sim|')
    ax.set_title('Within-Model Emotion Direction Overlap')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Bottom-right: Per-emotion norms at layer 30
    ax = axes[1, 1]
    l30_idx = layers.index(30)
    width = 0.35
    emotions = results[models[0]]['emotions']
    x = np.arange(len(emotions))

    for i, model in enumerate(models):
        norms_l30 = [results[model]['norms'][e][l30_idx] for e in emotions]
        ax.bar(x + i*width, norms_l30, width, label=model, color=colors.get(model, '#333'), alpha=0.7)

    ax.set_xlabel('Emotion')
    ax.set_ylabel('Direction Norm')
    ax.set_title('Direction Norm per Emotion (Layer 30)')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(emotions, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    data_dir = Path('probes/ua_emotion_disentangle/data')
    output_dir = Path('probes/ua_emotion_disentangle/plots')
    output_dir.mkdir(exist_ok=True)

    models = {
        'Qwen': data_dir / 'qwen3_32b_ua_emotions_v2.h5',
        'OLMo': data_dir / 'olmo_32b_ua_emotions_v2.h5',
    }

    # Get layers
    with h5py.File(list(models.values())[0], 'r') as f:
        layers = [int(l) for l in f.attrs['layers']]

    results = {}
    for name, path in models.items():
        if path.exists():
            results[name] = analyze_model(path, layers, name)

    output_path = output_dir / 'direction_analysis.png'
    plot_analysis(results, layers, output_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY (Layer 30)")
    print("=" * 60)
    l30_idx = layers.index(30)
    for model, data in results.items():
        mean_norm = np.mean([data['norms'][e][l30_idx] for e in data['emotions']])
        mean_consistency = np.mean([data['consistency'][e][l30_idx-1] for e in data['emotions']]) if l30_idx > 0 else 0
        print(f"{model}:")
        print(f"  Mean direction norm: {mean_norm:.2f}")
        print(f"  Layer consistency (29→30): {mean_consistency:.3f}")
        print(f"  Within-model overlap: {data['within_sim'][l30_idx]:.3f}")


if __name__ == '__main__':
    main()
