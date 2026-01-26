#!/usr/bin/env python3
"""
PCA analysis of emotion steering vectors with psychological axis alignment.

Analyzes how principal components of learned steering vectors align with
known psychological dimensions (Valence, Arousal, Dominance, etc.)
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Psychological axes encoding
AXES = ['Valence', 'Arousal', 'Dominance', 'Approach_Withdraw', 'Trust', 'Certainty', 'Agency', 'Social_Engagement']

ENCODING = {
    'Low': -1.0,
    'Mid': 0.0,
    'Mid-High': 0.5,
    'Mid-Low': -0.5,
    'High': 1.0,
    'Neutral': 0.0,
    'Approach': 1.0,
    'Withdraw': -1.0,
    'Self': 1.0,
    'Other': -1.0,
    'Self/Other': 0.0,
}

# Emotion -> Axis values (from user's table)
EMOTION_AXES = {
    # Negative emotions
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
    # Positive/Neutral emotions
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

# Model configurations
MODELS = {
    'gemma12b': {'middle_layer': 24, 'n_layers': 48},
    'gemma3_27b': {'middle_layer': 31, 'n_layers': 62},
    'qwen14b': {'middle_layer': 20, 'n_layers': 40},
    'qwen32b': {'middle_layer': 32, 'n_layers': 64},
}


def encode_psychological_axes() -> Tuple[np.ndarray, List[str]]:
    """
    Encode emotion axes into numerical matrix.

    Returns:
        matrix: (24 emotions, 8 axes) array
        emotions: List of emotion names in order
    """
    emotions = sorted(EMOTION_AXES.keys())
    matrix = np.zeros((len(emotions), len(AXES)))

    for i, emotion in enumerate(emotions):
        for j, axis in enumerate(AXES):
            value = EMOTION_AXES[emotion][axis]
            matrix[i, j] = ENCODING[value]

    return matrix, emotions


def load_vectors_for_layer(
    vectors_dir: Path,
    layer_idx: int,
    representation: str = None,
) -> Dict[str, np.ndarray]:
    """Load vectors for a specific layer."""

    if representation:
        layer_path = vectors_dir / representation / f"layer_{layer_idx:02d}.pkl"
    else:
        layer_path = vectors_dir / "layers" / f"layer_{layer_idx:02d}.pkl"

    if not layer_path.exists():
        return None

    with open(layer_path, 'rb') as f:
        return pickle.load(f)


def vectors_to_matrix(vectors: Dict[str, np.ndarray], emotions: List[str]) -> np.ndarray:
    """Convert vectors dict to matrix with consistent emotion ordering."""
    available = [e for e in emotions if e in vectors]
    if len(available) != len(emotions):
        missing = set(emotions) - set(available)
        print(f"  Warning: Missing emotions: {missing}")

    matrix = np.stack([vectors[e] for e in available])
    return matrix, available


def compute_correlations(
    emotion_coords: np.ndarray,
    psych_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute correlation between PC scores and psychological axes.

    Args:
        emotion_coords: (n_emotions, n_pcs) - emotion coordinates in PC space
        psych_matrix: (n_emotions, n_axes) - psychological axis values

    Returns:
        correlations: (n_pcs, n_axes) correlation matrix
    """
    n_pcs = emotion_coords.shape[1]
    n_axes = psych_matrix.shape[1]
    correlations = np.zeros((n_pcs, n_axes))

    for pc_idx in range(n_pcs):
        for axis_idx in range(n_axes):
            pc_scores = emotion_coords[:, pc_idx]
            axis_values = psych_matrix[:, axis_idx]
            # Handle constant arrays
            if np.std(pc_scores) > 0 and np.std(axis_values) > 0:
                correlations[pc_idx, axis_idx] = np.corrcoef(pc_scores, axis_values)[0, 1]

    return correlations


def plot_correlation_heatmap(
    correlations: np.ndarray,
    output_path: Path,
    title: str,
):
    """Plot heatmap of PC vs psychological axis correlations."""
    fig, ax = plt.subplots(figsize=(12, 8))

    n_pcs, n_axes = correlations.shape
    im = ax.imshow(correlations, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation')

    # Set ticks
    ax.set_xticks(range(n_axes))
    ax.set_xticklabels(AXES, rotation=45, ha='right')
    ax.set_yticks(range(n_pcs))
    ax.set_yticklabels([f'PC{i+1}' for i in range(n_pcs)])

    # Add text annotations
    for i in range(n_pcs):
        for j in range(n_axes):
            text_color = 'white' if abs(correlations[i, j]) > 0.5 else 'black'
            ax.text(j, i, f'{correlations[i, j]:.2f}', ha='center', va='center',
                   color=text_color, fontsize=8)

    ax.set_title(title)
    ax.set_xlabel('Psychological Axis')
    ax.set_ylabel('Principal Component')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_emotion_scatter(
    emotion_coords: np.ndarray,
    emotions: List[str],
    psych_matrix: np.ndarray,
    output_path: Path,
    title: str,
):
    """Plot emotions in PC1-PC2 space, colored by valence."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Color by valence (first axis)
    valence = psych_matrix[:, 0]
    colors = plt.cm.RdYlGn((valence + 1) / 2)  # Map [-1, 1] to [0, 1]

    scatter = ax.scatter(
        emotion_coords[:, 0],
        emotion_coords[:, 1],
        c=valence,
        cmap='RdYlGn',
        s=100,
        alpha=0.7,
    )

    # Add labels
    for i, emotion in enumerate(emotions):
        ax.annotate(
            emotion,
            (emotion_coords[i, 0], emotion_coords[i, 1]),
            fontsize=9,
            ha='center',
            va='bottom',
        )

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    plt.colorbar(scatter, label='Valence')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_variance_explained(
    explained_variance_ratio: np.ndarray,
    output_path: Path,
    title: str,
):
    """Plot variance explained by each PC."""
    fig, ax = plt.subplots(figsize=(10, 6))

    n_pcs = len(explained_variance_ratio)
    cumulative = np.cumsum(explained_variance_ratio)

    ax.bar(range(1, n_pcs + 1), explained_variance_ratio, alpha=0.7, label='Individual')
    ax.plot(range(1, n_pcs + 1), cumulative, 'ro-', label='Cumulative')

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title(title)
    ax.set_xticks(range(1, n_pcs + 1))
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def analyze_vector_set(
    vectors_dir: Path,
    output_dir: Path,
    model_name: str,
    vector_set_name: str,
    middle_layer: int,
    psych_matrix: np.ndarray,
    emotion_order: List[str],
    n_pcs: int = 10,
    representation: str = None,
):
    """Run PCA analysis on a single vector set."""

    # Load vectors
    vectors = load_vectors_for_layer(vectors_dir, middle_layer, representation)
    if vectors is None:
        print(f"  Skipping {vector_set_name}/{representation or 'layers'}: layer file not found")
        return None

    # Convert to matrix
    matrix, available_emotions = vectors_to_matrix(vectors, emotion_order)

    # Filter psych_matrix to match available emotions
    emotion_indices = [emotion_order.index(e) for e in available_emotions]
    filtered_psych = psych_matrix[emotion_indices]

    # PCA
    n_components = min(n_pcs, len(available_emotions), matrix.shape[1])
    pca = PCA(n_components=n_components)
    emotion_coords = pca.fit_transform(matrix)

    # Compute correlations
    correlations = compute_correlations(emotion_coords, filtered_psych)

    # Create output directory
    if representation:
        out_subdir = output_dir / model_name / vector_set_name / representation
    else:
        out_subdir = output_dir / model_name / vector_set_name
    out_subdir.mkdir(parents=True, exist_ok=True)

    # Plot heatmap
    plot_correlation_heatmap(
        correlations,
        out_subdir / 'correlation_heatmap.png',
        f'{model_name} - {vector_set_name}' + (f' ({representation})' if representation else ''),
    )

    # Plot emotion scatter
    plot_emotion_scatter(
        emotion_coords,
        available_emotions,
        filtered_psych,
        out_subdir / 'emotion_pca_scatter.png',
        f'{model_name} - {vector_set_name} (PC1 vs PC2)',
    )

    # Plot variance explained
    plot_variance_explained(
        pca.explained_variance_ratio_,
        out_subdir / 'variance_explained.png',
        f'{model_name} - {vector_set_name} Variance Explained',
    )

    # Find best-matching axis for each PC
    best_matches = []
    for pc_idx in range(n_components):
        abs_corrs = np.abs(correlations[pc_idx])
        best_axis_idx = np.argmax(abs_corrs)
        best_matches.append({
            'pc': pc_idx + 1,
            'best_axis': AXES[best_axis_idx],
            'correlation': float(correlations[pc_idx, best_axis_idx]),
            'abs_correlation': float(abs_corrs[best_axis_idx]),
        })

    # Save results
    results = {
        'model': model_name,
        'vector_set': vector_set_name,
        'representation': representation,
        'layer': middle_layer,
        'n_emotions': len(available_emotions),
        'n_pcs': n_components,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'correlations': {
            f'PC{i+1}': {axis: float(correlations[i, j]) for j, axis in enumerate(AXES)}
            for i in range(n_components)
        },
        'best_matches': best_matches,
        'emotions': available_emotions,
    }

    with open(out_subdir / 'pca_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    vectors_root = Path('steering_tests/vectors')
    output_root = Path('steering_tests/analysis/pca')
    output_root.mkdir(parents=True, exist_ok=True)

    # Encode psychological axes
    psych_matrix, emotion_order = encode_psychological_axes()

    # Save axis encoding
    with open(output_root / 'psychological_axes.json', 'w') as f:
        json.dump({
            'axes': AXES,
            'encoding': ENCODING,
            'emotion_axes': EMOTION_AXES,
        }, f, indent=2)

    all_results = []

    # Process each model
    for model_name, config in MODELS.items():
        model_dir = vectors_root / model_name
        if not model_dir.exists():
            print(f"Skipping {model_name}: directory not found")
            continue

        print(f"\n=== {model_name} (middle layer: {config['middle_layer']}) ===")

        # Process each vector set
        for vec_set_dir in sorted(model_dir.iterdir()):
            if not vec_set_dir.is_dir():
                continue

            vec_set_name = vec_set_dir.name
            print(f"\n  {vec_set_name}")

            # Check if it's a chat mode (has last_token/special_mean) or text mode (has layers)
            if (vec_set_dir / 'last_token').exists():
                # Chat mode - analyze both representations
                for rep in ['last_token', 'special_mean']:
                    print(f"    {rep}...")
                    result = analyze_vector_set(
                        vec_set_dir,
                        output_root,
                        model_name,
                        vec_set_name,
                        config['middle_layer'],
                        psych_matrix,
                        emotion_order,
                        n_pcs=10,
                        representation=rep,
                    )
                    if result:
                        all_results.append(result)
            else:
                # Text mode - analyze layers
                print(f"    layers...")
                result = analyze_vector_set(
                    vec_set_dir,
                    output_root,
                    model_name,
                    vec_set_name,
                    config['middle_layer'],
                    psych_matrix,
                    emotion_order,
                    n_pcs=10,
                    representation=None,
                )
                if result:
                    all_results.append(result)

    # Save all results summary
    with open(output_root / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Best-matching psychological axis for PC1")
    print("="*60)
    for result in all_results:
        best = result['best_matches'][0]
        rep_str = f"/{result['representation']}" if result['representation'] else ""
        print(f"{result['model']}/{result['vector_set']}{rep_str}:")
        print(f"  PC1 -> {best['best_axis']} (r={best['correlation']:.3f})")
        print(f"  Variance explained: {result['explained_variance_ratio'][0]:.1%}")

    print(f"\nResults saved to: {output_root}")


if __name__ == '__main__':
    main()
