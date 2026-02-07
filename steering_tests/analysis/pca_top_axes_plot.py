#!/usr/bin/env python3
"""
Visualize top 3 aligned psychological axes for each PC across models.
For high_emotion_vs_others and text_pairs_emotion_vs_neutral.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Axes and encoding (same as before)
AXES = ['Valence', 'Arousal', 'Dominance', 'Approach_Withdraw', 'Trust', 'Certainty', 'Agency', 'Social_Engagement']
AXES_SHORT = ['Valence', 'Arousal', 'Domin.', 'Appr/Withd', 'Trust', 'Certain.', 'Agency', 'Social']

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

# Colors for axes - muted palette
AXIS_COLORS = {
    'Valence': '#D4876A',           # Coral/Terra Cotta
    'Arousal': '#7BA7D7',           # Sky Blue
    'Dominance': '#7D9B7D',         # Olive Green
    'Approach_Withdraw': '#C17B8D', # Dusty Rose/Pink
    'Trust': '#B8CCC8',             # Sage Green
    'Certainty': '#D4D0E5',         # Soft Lavender
    'Agency': '#E8C07D',            # Muted Gold
    'Social_Engagement': '#9B8AA6', # Dusty Purple
}


def encode_psychological_axes():
    emotions = sorted(EMOTION_AXES.keys())
    matrix = np.zeros((len(emotions), len(AXES)))
    for i, emotion in enumerate(emotions):
        for j, axis in enumerate(AXES):
            value = EMOTION_AXES[emotion][axis]
            matrix[i, j] = ENCODING[value]
    return matrix, emotions


def load_vectors_for_layer(vectors_dir: Path, layer_idx: int, representation: str = None):
    if representation:
        layer_path = vectors_dir / representation / f"layer_{layer_idx:02d}.pkl"
    else:
        layer_path = vectors_dir / "layers" / f"layer_{layer_idx:02d}.pkl"
    if not layer_path.exists():
        return None
    with open(layer_path, 'rb') as f:
        return pickle.load(f)


def analyze_vectors(vectors, emotion_order, psych_matrix, n_pcs=10):
    available = [e for e in emotion_order if e in vectors]
    matrix = np.stack([vectors[e] for e in available])
    emotion_indices = [emotion_order.index(e) for e in available]
    filtered_psych = psych_matrix[emotion_indices]

    n_components = min(n_pcs, len(available), matrix.shape[1])
    pca = PCA(n_components=n_components)
    emotion_coords = pca.fit_transform(matrix)

    correlations = np.zeros((n_components, len(AXES)))
    for pc_idx in range(n_components):
        for axis_idx in range(len(AXES)):
            pc_scores = emotion_coords[:, pc_idx]
            axis_values = filtered_psych[:, axis_idx]
            if np.std(pc_scores) > 0 and np.std(axis_values) > 0:
                correlations[pc_idx, axis_idx] = np.corrcoef(pc_scores, axis_values)[0, 1]

    return correlations, pca.explained_variance_ratio_


def get_top_axes(correlations, n_top=3):
    """Get top n axes by absolute correlation for each PC."""
    results = []
    for pc_idx in range(correlations.shape[0]):
        abs_corrs = np.abs(correlations[pc_idx])
        top_indices = np.argsort(abs_corrs)[::-1][:n_top]
        top_axes = [(AXES[i], correlations[pc_idx, i]) for i in top_indices]
        results.append(top_axes)
    return results


def plot_top_axes_comparison(all_results: Dict, title: str, output_path: Path, n_pcs: int = 6):
    """
    Create a visualization showing top 3 axes for each PC across models.

    Layout: rows = PCs, columns = models
    Each cell shows top 3 axes with colored bars
    """
    models = list(all_results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(n_pcs, n_models, figsize=(4.5 * n_models, 2.8 * n_pcs))

    for col, model in enumerate(models):
        correlations = all_results[model]['correlations']
        var_explained = all_results[model]['variance']
        top_axes = get_top_axes(correlations, n_top=3)

        for row in range(min(n_pcs, len(top_axes))):
            ax = axes[row, col] if n_models > 1 else axes[row]

            # Get top 3 axes for this PC
            pc_top = top_axes[row]

            # Create horizontal bar chart
            y_pos = np.arange(3)
            axis_names = [a[0] for a in pc_top]
            corr_values = [a[1] for a in pc_top]
            colors = [AXIS_COLORS[a] for a in axis_names]

            bars = ax.barh(y_pos, corr_values, color=colors, alpha=0.9, height=0.6)

            # Add correlation values as text - always black
            for i, (bar, val) in enumerate(zip(bars, corr_values)):
                # Position text inside or outside bar depending on value
                if abs(val) > 0.3:
                    x_pos = val / 2
                else:
                    x_pos = val + 0.08 * np.sign(val) if val != 0 else 0.08
                ax.text(x_pos, i, f'{val:+.2f}', va='center', ha='center',
                       fontsize=12, fontweight='bold', color='black')

            # Styling
            ax.set_yticks(y_pos)
            ax.set_yticklabels([AXES_SHORT[AXES.index(a)] for a in axis_names], fontsize=12)
            ax.set_xlim(-1.1, 1.1)
            ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

            # Add variance explained
            var = var_explained[row] if row < len(var_explained) else 0

            if row == 0:
                ax.set_title(f'{model}\n', fontsize=14, fontweight='bold')

            # Add PC label on the left
            if col == 0:
                ax.set_ylabel(f'PC{row+1}\n({var:.0%})', fontsize=13, fontweight='bold')
            else:
                # Just show variance for other columns
                ax.text(-1.35, 1, f'({var:.0%})', fontsize=11, va='center',
                       transform=ax.transData, color='gray')

            ax.tick_params(axis='x', labelsize=11)

            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)

    # Add legend for axis colors - MUCH bigger
    legend_elements = [plt.Rectangle((0,0), 1, 1, facecolor=AXIS_COLORS[axis], alpha=0.9, label=AXES_SHORT[i])
                      for i, axis in enumerate(AXES)]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.01),
              ncol=4, fontsize=16, frameon=True, fancybox=True, shadow=True,
              handlelength=2.5, handleheight=1.8, handletextpad=0.8,
              columnspacing=2.0, borderpad=1.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_summary_table(high_results: Dict, neutral_results: Dict, output_path: Path):
    """Create a summary table visualization comparing high vs neutral across models."""
    models = list(high_results.keys())
    n_pcs = 5

    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    for plot_idx, (results, title) in enumerate([
        (high_results, 'High Intensity (emotion vs others)'),
        (neutral_results, 'Text Pairs (emotion vs neutral)')
    ]):
        ax = axes[plot_idx]

        # Create table data
        cell_text = []
        row_labels = []

        for pc_idx in range(n_pcs):
            row = []
            for model in models:
                if model not in results:
                    row.append('N/A')
                    continue
                corrs = results[model]['correlations']
                var = results[model]['variance'][pc_idx] if pc_idx < len(results[model]['variance']) else 0

                # Get top axis
                abs_corrs = np.abs(corrs[pc_idx])
                top_idx = np.argmax(abs_corrs)
                top_axis = AXES_SHORT[top_idx]
                top_r = corrs[pc_idx, top_idx]

                row.append(f'{top_axis}\nr={top_r:+.2f}\n({var:.0%})')

            cell_text.append(row)
            row_labels.append(f'PC{pc_idx+1}')

        # Create table
        ax.axis('off')
        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=models,
            loc='center',
            cellLoc='center',
        )

        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.3, 2.8)

        # Color cells based on top axis
        import matplotlib.colors as mcolors
        for i in range(n_pcs):
            for j, model in enumerate(models):
                if model not in results:
                    table[(i+1, j)].set_facecolor('#e0e0e0')  # Gray for N/A
                    continue
                corrs = results[model]['correlations']
                abs_corrs = np.abs(corrs[i])
                top_idx = np.argmax(abs_corrs)
                color = AXIS_COLORS[AXES[top_idx]]

                # Lighten the color slightly
                rgb = mcolors.to_rgb(color)
                light_rgb = tuple(0.2 + 0.8 * c for c in rgb)

                table[(i+1, j)].set_facecolor(light_rgb)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    plt.suptitle('Top Aligned Psychological Axis per PC\n(Axis name, correlation r, variance explained)',
                fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    vectors_root = Path('steering_tests/vectors')
    output_root = Path('steering_tests/analysis/pca')
    output_root.mkdir(parents=True, exist_ok=True)

    psych_matrix, emotion_order = encode_psychological_axes()

    high_results = {}
    neutral_results = {}

    print("Loading and analyzing vectors...")

    for model_name, config in MODELS.items():
        model_dir = vectors_root / model_name
        if not model_dir.exists():
            continue

        # High emotion vs others (special_mean)
        high_dir = model_dir / 'high_emotion_vs_others'
        if high_dir.exists():
            vectors = load_vectors_for_layer(high_dir, config['middle_layer'], 'special_mean')
            if vectors:
                corrs, var = analyze_vectors(vectors, emotion_order, psych_matrix)
                high_results[model_name] = {'correlations': corrs, 'variance': var}
                print(f"  {model_name} high: loaded")

        # Text pairs emotion vs neutral
        neutral_dir = model_dir / 'text_pairs_emotion_vs_neutral'
        if neutral_dir.exists():
            vectors = load_vectors_for_layer(neutral_dir, config['middle_layer'], None)
            if vectors:
                corrs, var = analyze_vectors(vectors, emotion_order, psych_matrix)
                neutral_results[model_name] = {'correlations': corrs, 'variance': var}
                print(f"  {model_name} neutral: loaded")

    # Generate plots
    print("\nGenerating plots...")

    if high_results:
        plot_top_axes_comparison(
            high_results,
            'High Intensity Prompts (emotion vs others) - Top 3 Axes per PC',
            output_root / 'top_axes_high.png',
            n_pcs=6
        )
        print("  Saved top_axes_high.png")

    if neutral_results:
        plot_top_axes_comparison(
            neutral_results,
            'Text Pairs (emotion vs neutral) - Top 3 Axes per PC',
            output_root / 'top_axes_neutral.png',
            n_pcs=6
        )
        print("  Saved top_axes_neutral.png")

    if high_results and neutral_results:
        plot_summary_table(high_results, neutral_results, output_root / 'top_axes_summary_table.png')
        print("  Saved top_axes_summary_table.png")

    print(f"\nDone! Plots saved to {output_root}")


if __name__ == '__main__':
    main()
