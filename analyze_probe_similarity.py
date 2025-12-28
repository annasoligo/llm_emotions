#!/usr/bin/env python3
"""Analyze cosine similarity between emotion probes at layer 30.

Compares:
1. Within-setting: Same setting (e.g., nc10) across different seeds
2. Between-setting: Different settings (raw, nc5, nc10, nc20) for the same emotion
"""

import pickle
import numpy as np
from pathlib import Path
from collections import defaultdict

EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
LAYER = 30

def load_cpca_components(cpca_path, layer):
    """Load cPCA components for projection."""
    data = np.load(cpca_path, allow_pickle=True)
    components = data["components"][layer]  # [n_components, hidden_dim]
    return components

def get_probe_direction_in_full_space(probe_data, cpca_components=None):
    """Get probe direction in full activation space.

    Args:
        probe_data: Loaded probe pickle with 'model' containing probe weights
        cpca_components: [n_components, hidden_dim] cPCA components (if used)

    Returns:
        directions: [n_emotions, hidden_dim] probe directions in full space
    """
    model = probe_data['model']

    # Extract probe weights from the PyTorch Linear model
    # The model is a torch.nn.Linear, so get weights via state_dict or direct access
    if hasattr(model, 'weight'):
        probe_weights = model.weight.detach().cpu().numpy()  # [n_emotions, feature_dim]
    elif hasattr(model, 'state_dict'):
        state_dict = model.state_dict()
        probe_weights = state_dict['weight'].cpu().numpy()
    else:
        raise ValueError(f"Could not extract weights from model of type {type(model)}")

    if cpca_components is not None:
        # Project back to full space: probe_full = probe_cpca @ components
        # probe_weights: [n_emotions, n_components]
        # cpca_components: [n_components, hidden_dim]
        # result: [n_emotions, hidden_dim]
        probe_directions = probe_weights @ cpca_components
    else:
        # Already in full space
        probe_directions = probe_weights

    # Normalize
    probe_directions = probe_directions / np.linalg.norm(probe_directions, axis=1, keepdims=True)

    return probe_directions

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    results_dir = Path("/workspace-vast/annas/git/research-tools/results/emotion_probes_multiseed")
    cpca_path = Path("/workspace-vast/annas/git/research-tools/outputs/dimensionality_reduction/cpca/tier_based/google/google/gemma-3-27b-it_cpca.npz")

    print("="*80)
    print(f"PROBE SIMILARITY ANALYSIS - Layer {LAYER}")
    print("="*80)
    print()

    # Load cPCA components for all settings
    print("Loading cPCA components...")
    cpca_components_full = load_cpca_components(cpca_path, LAYER)
    cpca_5 = cpca_components_full[:5]
    cpca_10 = cpca_components_full[:10]
    cpca_20 = cpca_components_full[:20]
    print(f"Full cPCA shape: {cpca_components_full.shape}")
    print()

    # Load all probes for layer 30
    print("Loading probes...")
    probes_by_setting = {
        'nc0': {},  # seed -> probe_directions
        'nc5': {},
        'nc10': {},
        'nc20': {},
    }

    for nc_setting in ['nc0', 'nc5', 'nc10', 'nc20']:
        cpca_comps = None
        if nc_setting == 'nc5':
            cpca_comps = cpca_5
        elif nc_setting == 'nc10':
            cpca_comps = cpca_10
        elif nc_setting == 'nc20':
            cpca_comps = cpca_20

        for seed in range(10):
            probe_path = results_dir / f"probe_layer{LAYER}_{nc_setting}_seed{seed}.pkl"
            if not probe_path.exists():
                print(f"Warning: {probe_path} not found")
                continue

            with open(probe_path, 'rb') as f:
                probe_data = pickle.load(f)

            directions = get_probe_direction_in_full_space(probe_data, cpca_comps)
            probes_by_setting[nc_setting][seed] = directions

        print(f"Loaded {len(probes_by_setting[nc_setting])} probes for {nc_setting}")

    print()
    print("="*80)
    print("ANALYSIS 1: Within-Setting Similarity (Across Seeds)")
    print("="*80)
    print()

    # For each setting, compute pairwise cosine similarity across seeds for each emotion
    for nc_setting in ['nc0', 'nc5', 'nc10', 'nc20']:
        print(f"--- {nc_setting} ---")
        seeds = sorted(probes_by_setting[nc_setting].keys())

        if len(seeds) < 2:
            print("  Not enough seeds for comparison")
            print()
            continue

        # For each emotion, compute all pairwise similarities
        emotion_sims = {emotion: [] for emotion in EMOTIONS}

        for i, seed_i in enumerate(seeds):
            for seed_j in seeds[i+1:]:
                dirs_i = probes_by_setting[nc_setting][seed_i]
                dirs_j = probes_by_setting[nc_setting][seed_j]

                for emo_idx, emotion in enumerate(EMOTIONS):
                    sim = cosine_similarity(dirs_i[emo_idx], dirs_j[emo_idx])
                    emotion_sims[emotion].append(sim)

        # Print statistics
        print(f"  Emotion-wise similarity (mean ± std):")
        for emotion in EMOTIONS:
            sims = emotion_sims[emotion]
            print(f"    {emotion:12s}: {np.mean(sims):.4f} ± {np.std(sims):.4f} (n={len(sims)})")

        # Overall stats
        all_sims = [sim for sims in emotion_sims.values() for sim in sims]
        print(f"  Overall:        {np.mean(all_sims):.4f} ± {np.std(all_sims):.4f}")
        print(f"  Min: {np.min(all_sims):.4f}, Max: {np.max(all_sims):.4f}")
        print()

    print()
    print("="*80)
    print("ANALYSIS 2: Between-Setting Similarity (Same Emotion, Different Settings)")
    print("="*80)
    print()

    # Average probe direction per setting (across seeds)
    avg_probes = {}
    for nc_setting in ['nc0', 'nc5', 'nc10', 'nc20']:
        seeds = sorted(probes_by_setting[nc_setting].keys())
        if not seeds:
            continue

        # Average across seeds
        all_dirs = np.stack([probes_by_setting[nc_setting][s] for s in seeds])  # [n_seeds, n_emotions, hidden_dim]
        avg_dirs = all_dirs.mean(axis=0)  # [n_emotions, hidden_dim]
        # Re-normalize
        avg_dirs = avg_dirs / np.linalg.norm(avg_dirs, axis=1, keepdims=True)
        avg_probes[nc_setting] = avg_dirs

    # Compare all pairs of settings
    settings = ['nc0', 'nc5', 'nc10', 'nc20']
    setting_names = {'nc0': 'Raw', 'nc5': 'Top-5 cPCA', 'nc10': 'Top-10 cPCA', 'nc20': 'Top-20 cPCA'}

    for i, setting_i in enumerate(settings):
        for setting_j in settings[i+1:]:
            if setting_i not in avg_probes or setting_j not in avg_probes:
                continue

            print(f"--- {setting_names[setting_i]} vs {setting_names[setting_j]} ---")

            dirs_i = avg_probes[setting_i]
            dirs_j = avg_probes[setting_j]

            for emo_idx, emotion in enumerate(EMOTIONS):
                sim = cosine_similarity(dirs_i[emo_idx], dirs_j[emo_idx])
                print(f"  {emotion:12s}: {sim:.4f}")

            # Overall similarity
            all_sims = [cosine_similarity(dirs_i[e], dirs_j[e]) for e in range(len(EMOTIONS))]
            print(f"  Mean:           {np.mean(all_sims):.4f} ± {np.std(all_sims):.4f}")
            print()

    print()
    print("="*80)
    print("ANALYSIS 3: Cross-Emotion Similarity Within Each Setting")
    print("="*80)
    print()

    # Check how orthogonal different emotions are within same setting
    for nc_setting in ['nc0', 'nc5', 'nc10', 'nc20']:
        if nc_setting not in avg_probes:
            continue

        print(f"--- {setting_names[nc_setting]} ---")
        dirs = avg_probes[nc_setting]

        # Compute cross-emotion similarity matrix
        cross_sims = []
        for i, emo_i in enumerate(EMOTIONS):
            for j, emo_j in enumerate(EMOTIONS):
                if i < j:  # Only upper triangle
                    sim = cosine_similarity(dirs[i], dirs[j])
                    cross_sims.append(abs(sim))

        print(f"  Mean |cross-emotion similarity|: {np.mean(cross_sims):.4f} ± {np.std(cross_sims):.4f}")
        print(f"  Min: {np.min(cross_sims):.4f}, Max: {np.max(cross_sims):.4f}")
        print()

    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
