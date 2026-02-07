#!/usr/bin/env python3
"""
Generate combined subplot figures for PCA analysis across all vector types per model.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Psychological axes
AXES = ['Valence', 'Arousal', 'Dominance', 'Approach_Withdraw', 'Trust', 'Certainty', 'Agency', 'Social_Engagement']
AXES_SHORT = ['Val', 'Aro', 'Dom', 'App', 'Tru', 'Cer', 'Age', 'Soc']

ENCODING = {
    'Low': -1.0, 'Mid': 0.0, 'Mid-High': 0.5, 'Mid-Low': -0.5,
    'High': 1.0, 'Neutral': 0.0, 'Approach': 1.0, 'Withdraw': -1.0,
    'Self': 1.0, 'Other': -1.0, 'Self/Other': 0.0,
}

EMOTION_AXES = {
    'fear':        {'Valence': 'Low', 'Arousal': 'High', 'Dominance': 'Low', 'Approach_Withdraw': 'Withdraw', 'Trust': 'Low', 'Certainty': 'Low', 'Agency': 'Other', 'Social_Engagement': 'Low'},
    'anxiety':     {'Valence': 'Low', 'Arousal': 'High', 'Dominance': 'Low', 'Approach_Withdraw': 'Withdraw', 'Trust': 'Low', 'Certainty': 'Low', 'Agency': 'Self/Other', 'Social_Engagement': 'Low'},
    'anger':       {'Valence': 'Low', 'Arousal': 'High', 'Dominance': 'High', 'Approach_Withdraw': 'Approach', 'Trust': 'Low', 'Certainty': 'High', 'Agency': 'Other', 'Social_Engagement': 'Neutral'},
    'frustration': {'Valence': 'Low', 'Arousal': 'Mid-High', 'Dominance': 'Low', 'Approach_Withdraw': 'Approach', 'Trust': 'Low', 'Certainty': 'Mid', 'Agency': 'Self/Other', 'Social_Engagement': 'Low'},
    'sadness':     {'Valence': 'Low', 'Arousal': 'Low', 'Dominance': 'Low', 'Approach_Withdraw': 'Withdraw', 'Trust': 'Neutral', 'Certainty': 'High', 'Agency': 'Other', 'Social_Engagement': 'Low'},
    'guilt':       {'Valence': 'Low', 'Arousal': 'Mid', 'Dominance': 'Low', 'Approach_Withdraw': 'Withdraw', 'Trust': 'Neutral', 'Certainty': 'High', 'Agency': 'Self', 'Social_Engagement': 'Low'},
    'shame':       {'Valence': 'Low', 'Arousal': 'Mid', 'Dominance': 'Low', 'Approach_Withdraw': 'Withdraw', 'Trust': 'Low', 'Certainty': 'High', 'Agency': 'Self', 'Social_Engagement': 'Low'},
    'disgust':     {'Valence': 'Low', 'Arousal': 'Mid', 'Dominance': 'Mid', 'Approach_Withdraw': 'Withdraw', 'Trust': 'Low', 'Certainty': 'High', 'Agency': 'Other', 'Social_Engagement': 'Low'},
    'contempt':    {'Valence': 'Low', 'Arousal': 'Low', 'Dominance': 'Mid', 'Approach_Withdraw': 'Withdraw', 'Trust': 'Low', 'Certainty': 'High', 'Agency': 'Other', 'Social_Engagement': 'Low'},
    'boredom':     {'Valence': 'Low', 'Arousal': 'Low', 'Dominance': 'Neutral', 'Approach_Withdraw': 'Withdraw', 'Trust': 'Neutral', 'Certainty': 'Low', 'Agency': 'Self', 'Social_Engagement': 'Low'},
    'despair':     {'Valence': 'Low', 'Arousal': 'Low', 'Dominance': 'Low', 'Approach_Withdraw': 'Withdraw', 'Trust': 'Low', 'Certainty': 'Low', 'Agency': 'Other', 'Social_Engagement': 'Low'},
    'confusion':   {'Valence': 'Neutral', 'Arousal': 'Mid', 'Dominance': 'Low', 'Approach_Withdraw': 'Neutral', 'Trust': 'Low', 'Certainty': 'Low', 'Agency': 'Self/Other', 'Social_Engagement': 'Neutral'},
    'surprise':    {'Valence': 'Neutral', 'Arousal': 'High', 'Dominance': 'Low', 'Approach_Withdraw': 'Neutral', 'Trust': 'Neutral', 'Certainty': 'Low', 'Agency': 'Other', 'Social_Engagement': 'Neutral'},
    'curiosity':   {'Valence': 'Mid-High', 'Arousal': 'Mid', 'Dominance': 'Neutral', 'Approach_Withdraw': 'Approach', 'Trust': 'Neutral', 'Certainty': 'Low', 'Agency': 'Self', 'Social_Engagement': 'Neutral'},
    'interest':    {'Valence': 'Mid-High', 'Arousal': 'Mid', 'Dominance': 'Neutral', 'Approach_Withdraw': 'Approach', 'Trust': 'Neutral', 'Certainty': 'Low', 'Agency': 'Self/Other', 'Social_Engagement': 'Neutral'},
    'hope':        {'Valence': 'High', 'Arousal': 'Mid', 'Dominance': 'Neutral', 'Approach_Withdraw': 'Approach', 'Trust': 'High', 'Certainty': 'Low', 'Agency': 'Self/Other', 'Social_Engagement': 'Neutral'},
    'relief':      {'Valence': 'High', 'Arousal': 'Low', 'Dominance': 'Mid', 'Approach_Withdraw': 'Neutral', 'Trust': 'High', 'Certainty': 'High', 'Agency': 'Other', 'Social_Engagement': 'Neutral'},
    'calm':        {'Valence': 'High', 'Arousal': 'Low', 'Dominance': 'High', 'Approach_Withdraw': 'Neutral', 'Trust': 'High', 'Certainty': 'High', 'Agency': 'Self', 'Social_Engagement': 'Neutral'},
    'contentment': {'Valence': 'High', 'Arousal': 'Low', 'Dominance': 'High', 'Approach_Withdraw': 'Neutral', 'Trust': 'High', 'Certainty': 'High', 'Agency': 'Self', 'Social_Engagement': 'High'},
    'joy':         {'Valence': 'High', 'Arousal': 'High', 'Dominance': 'High', 'Approach_Withdraw': 'Approach', 'Trust': 'High', 'Certainty': 'High', 'Agency': 'Self/Other', 'Social_Engagement': 'High'},
    'excitement':  {'Valence': 'High', 'Arousal': 'High', 'Dominance': 'Mid', 'Approach_Withdraw': 'Approach', 'Trust': 'High', 'Certainty': 'Low', 'Agency': 'Self/Other', 'Social_Engagement': 'High'},
    'pride':       {'Valence': 'High', 'Arousal': 'Mid', 'Dominance': 'High', 'Approach_Withdraw': 'Approach', 'Trust': 'High', 'Certainty': 'High', 'Agency': 'Self', 'Social_Engagement': 'Mid'},
    'gratitude':   {'Valence': 'High', 'Arousal': 'Mid', 'Dominance': 'Neutral', 'Approach_Withdraw': 'Approach', 'Trust': 'High', 'Certainty': 'High', 'Agency': 'Other', 'Social_Engagement': 'High'},
    'admiration':  {'Valence': 'High', 'Arousal': 'Mid', 'Dominance': 'Low', 'Approach_Withdraw': 'Approach', 'Trust': 'High', 'Certainty': 'High', 'Agency': 'Other', 'Social_Engagement': 'High'},
}

MODELS = {
    'gemma12b': {'middle_layer': 24, 'n_layers': 48},
    'gemma3_27b': {'middle_layer': 31, 'n_layers': 62},
    'qwen14b': {'middle_layer': 20, 'n_layers': 40},
    'qwen32b': {'middle_layer': 32, 'n_layers': 64},
    'qwen235b': {'middle_layer': 47, 'n_layers': 94},
}

# Vector set display names
VECTOR_SET_NAMES = {
    'base_emotion_vs_others': 'Base\n(vs others)',
    'high_emotion_vs_others': 'High\n(vs others)',
    'text_pairs_emotion_vs_neutral': 'TextPairs\n(vs neutral)',
    'text_pairs_emotion_vs_others': 'TextPairs\n(vs others)',
    'text_pairs_emotion_vs_opposite': 'TextPairs\n(vs opposite)',
}


def encode_psychological_axes() -> Tuple[np.ndarray, List[str]]:
    emotions = sorted(EMOTION_AXES.keys())
    matrix = np.zeros((len(emotions), len(AXES)))
    for i, emotion in enumerate(emotions):
        for j, axis in enumerate(AXES):
            value = EMOTION_AXES[emotion][axis]
            matrix[i, j] = ENCODING[value]
    return matrix, emotions


def load_vectors_for_layer(vectors_dir: Path, layer_idx: int, representation: str = None) -> Optional[Dict[str, np.ndarray]]:
    if representation:
        layer_path = vectors_dir / representation / f"layer_{layer_idx:02d}.pkl"
    else:
        layer_path = vectors_dir / "layers" / f"layer_{layer_idx:02d}.pkl"
    if not layer_path.exists():
        return None
    with open(layer_path, 'rb') as f:
        return pickle.load(f)


def analyze_vectors(vectors: Dict[str, np.ndarray], emotion_order: List[str], psych_matrix: np.ndarray, n_pcs: int = 10):
    """Run PCA and compute correlations."""
    available = [e for e in emotion_order if e in vectors]
    matrix = np.stack([vectors[e] for e in available])
    emotion_indices = [emotion_order.index(e) for e in available]
    filtered_psych = psych_matrix[emotion_indices]

    n_components = min(n_pcs, len(available), matrix.shape[1])
    pca = PCA(n_components=n_components)
    emotion_coords = pca.fit_transform(matrix)

    # Compute correlations
    correlations = np.zeros((n_components, len(AXES)))
    for pc_idx in range(n_components):
        for axis_idx in range(len(AXES)):
            pc_scores = emotion_coords[:, pc_idx]
            axis_values = filtered_psych[:, axis_idx]
            if np.std(pc_scores) > 0 and np.std(axis_values) > 0:
                correlations[pc_idx, axis_idx] = np.corrcoef(pc_scores, axis_values)[0, 1]

    return {
        'emotion_coords': emotion_coords,
        'emotions': available,
        'correlations': correlations,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'filtered_psych': filtered_psych,
    }


def plot_combined_heatmaps(model_name: str, all_results: Dict, output_path: Path):
    """Plot correlation heatmaps for all vector sets as subplots."""
    n_sets = len(all_results)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, (vec_set_name, results) in enumerate(all_results.items()):
        if idx >= 8:
            break
        ax = axes[idx]

        correlations = results['correlations'][:5]  # Top 5 PCs

        im = ax.imshow(correlations, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        ax.set_xticks(range(len(AXES_SHORT)))
        ax.set_xticklabels(AXES_SHORT, fontsize=8)
        ax.set_yticks(range(5))
        ax.set_yticklabels([f'PC{i+1}' for i in range(5)], fontsize=9)

        # Add text annotations
        for i in range(5):
            for j in range(len(AXES)):
                color = 'white' if abs(correlations[i, j]) > 0.5 else 'black'
                ax.text(j, i, f'{correlations[i, j]:.2f}', ha='center', va='center',
                       color=color, fontsize=7)

        display_name = VECTOR_SET_NAMES.get(vec_set_name, vec_set_name)
        ax.set_title(display_name, fontsize=10)

    # Hide unused subplots
    for idx in range(len(all_results), 8):
        axes[idx].axis('off')

    fig.suptitle(f'{model_name} - PC vs Psychological Axis Correlations', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=axes, shrink=0.6, label='Correlation')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_combined_scatters(model_name: str, all_results: Dict, output_path: Path):
    """Plot PC1 vs PC2 scatter for all vector sets as subplots."""
    n_sets = len(all_results)
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    axes = axes.flatten()

    for idx, (vec_set_name, results) in enumerate(all_results.items()):
        if idx >= 8:
            break
        ax = axes[idx]

        emotion_coords = results['emotion_coords']
        emotions = results['emotions']
        valence = results['filtered_psych'][:, 0]

        scatter = ax.scatter(
            emotion_coords[:, 0],
            emotion_coords[:, 1],
            c=valence,
            cmap='RdYlGn',
            s=60,
            alpha=0.8,
            vmin=-1, vmax=1,
        )

        for i, emotion in enumerate(emotions):
            ax.annotate(emotion, (emotion_coords[i, 0], emotion_coords[i, 1]),
                       fontsize=6, ha='center', va='bottom')

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('PC1', fontsize=9)
        ax.set_ylabel('PC2', fontsize=9)

        # Add variance explained
        var1 = results['explained_variance_ratio'][0]
        var2 = results['explained_variance_ratio'][1]

        display_name = VECTOR_SET_NAMES.get(vec_set_name, vec_set_name)
        ax.set_title(f"{display_name}\nPC1:{var1:.0%} PC2:{var2:.0%}", fontsize=9)

    # Hide unused subplots
    for idx in range(len(all_results), 8):
        axes[idx].axis('off')

    fig.suptitle(f'{model_name} - Emotions in PC Space (colored by Valence)', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(scatter, cax=cbar_ax, label='Valence')

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_combined_variance(model_name: str, all_results: Dict, output_path: Path):
    """Plot variance explained for all vector sets as subplots."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()

    for idx, (vec_set_name, results) in enumerate(all_results.items()):
        if idx >= 8:
            break
        ax = axes[idx]

        var_ratio = results['explained_variance_ratio'][:10]
        cumulative = np.cumsum(var_ratio)
        n_pcs = len(var_ratio)

        ax.bar(range(1, n_pcs + 1), var_ratio, alpha=0.7, color='steelblue')
        ax.plot(range(1, n_pcs + 1), cumulative, 'ro-', markersize=4)

        ax.set_xlabel('PC', fontsize=9)
        ax.set_ylabel('Variance', fontsize=9)
        ax.set_xticks(range(1, n_pcs + 1))
        ax.set_ylim(0, 1)

        display_name = VECTOR_SET_NAMES.get(vec_set_name, vec_set_name)
        ax.set_title(display_name, fontsize=9)

    # Hide unused subplots
    for idx in range(len(all_results), 8):
        axes[idx].axis('off')

    fig.suptitle(f'{model_name} - Variance Explained by PCs', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pc1_summary(all_model_results: Dict, output_path: Path):
    """Plot PC1 correlation with all axes across models and vector sets."""
    models = list(all_model_results.keys())

    # Collect all unique vector sets
    all_vec_sets = set()
    for model_results in all_model_results.values():
        all_vec_sets.update(model_results.keys())
    vec_sets = sorted(all_vec_sets)

    fig, axes = plt.subplots(len(models), 1, figsize=(14, 4 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for model_idx, model_name in enumerate(models):
        ax = axes[model_idx]
        model_results = all_model_results[model_name]

        x = np.arange(len(AXES))
        width = 0.12

        for i, vec_set in enumerate(vec_sets):
            if vec_set in model_results:
                pc1_corrs = model_results[vec_set]['correlations'][0]
                offset = (i - len(vec_sets)/2) * width
                short_name = VECTOR_SET_NAMES.get(vec_set, vec_set).replace('\n', ' ')
                ax.bar(x + offset, pc1_corrs, width, label=short_name, alpha=0.8)

        ax.set_ylabel('PC1 Correlation')
        ax.set_xticks(x)
        ax.set_xticklabels(AXES, rotation=45, ha='right')
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7, ncol=2)

    fig.suptitle('PC1 Correlation with Psychological Axes', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    vectors_root = Path('steering_tests/vectors')
    output_root = Path('steering_tests/analysis/pca')
    output_root.mkdir(parents=True, exist_ok=True)

    psych_matrix, emotion_order = encode_psychological_axes()

    all_model_results = {}

    for model_name, config in MODELS.items():
        model_dir = vectors_root / model_name
        if not model_dir.exists():
            print(f"Skipping {model_name}: directory not found")
            continue

        print(f"\n=== {model_name} ===")
        model_results = {}

        for vec_set_dir in sorted(model_dir.iterdir()):
            if not vec_set_dir.is_dir():
                continue

            vec_set_name = vec_set_dir.name

            # Determine representation(s)
            if (vec_set_dir / 'last_token').exists():
                # Use special_mean for chat mode (more consistent with valence)
                rep = 'special_mean'
                vectors = load_vectors_for_layer(vec_set_dir, config['middle_layer'], rep)
                if vectors:
                    result = analyze_vectors(vectors, emotion_order, psych_matrix)
                    model_results[f"{vec_set_name}\n({rep})"] = result
                    print(f"  {vec_set_name}/{rep}")

                # Also do last_token
                rep = 'last_token'
                vectors = load_vectors_for_layer(vec_set_dir, config['middle_layer'], rep)
                if vectors:
                    result = analyze_vectors(vectors, emotion_order, psych_matrix)
                    model_results[f"{vec_set_name}\n({rep})"] = result
                    print(f"  {vec_set_name}/{rep}")
            else:
                vectors = load_vectors_for_layer(vec_set_dir, config['middle_layer'], None)
                if vectors:
                    result = analyze_vectors(vectors, emotion_order, psych_matrix)
                    model_results[vec_set_name] = result
                    print(f"  {vec_set_name}")

        if model_results:
            all_model_results[model_name] = model_results

            # Generate combined plots
            model_output = output_root / model_name
            model_output.mkdir(exist_ok=True)

            print(f"  Generating combined plots...")
            plot_combined_heatmaps(model_name, model_results, model_output / 'combined_heatmaps.png')
            plot_combined_scatters(model_name, model_results, model_output / 'combined_scatters.png')
            plot_combined_variance(model_name, model_results, model_output / 'combined_variance.png')

    # Cross-model PC1 summary
    print("\nGenerating cross-model summary...")
    plot_pc1_summary(all_model_results, output_root / 'cross_model_pc1_summary.png')

    print(f"\nDone! Combined plots saved to {output_root}")


if __name__ == '__main__':
    main()
