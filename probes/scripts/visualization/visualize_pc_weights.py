#!/usr/bin/env python3
"""
Visualize PC weights for emotion probes with subplots.

Creates bar charts showing how each PC contributes to each emotion,
using the emotion color mapping from believe-it-or-not.
"""

import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Emotion color mapping from believe-it-or-not
COLORS = {
    "coral": "#D4876A",
    "sky_blue": "#7BA7D7",
    "olive": "#7D9B7D",
    "dusty_rose": "#C17B8D",
    "sage": "#B8CCC8",
    "lavender": "#a59dc9",
}

EMOTION_COLORS = {
    'anger': COLORS['sky_blue'],
    'disgust': COLORS['olive'],
    'fear': COLORS['lavender'],
    'happiness': COLORS['coral'],
    'sadness': COLORS['sage'],
    'surprise': '#D1728F',  # Darker pink for surprise
    'neutral': '#808080'  # Gray for neutral
}


def load_probe(probe_path: Path) -> dict:
    """Load trained probe."""
    with open(probe_path, 'rb') as f:
        return pickle.load(f)


def load_autointerp(autointerp_path: Path) -> dict:
    """Load autointerp results."""
    with open(autointerp_path, 'r') as f:
        return json.load(f)


def visualize_pc_weights_subplots(layer: int, n_components_list: list, output_dir: Path,
                                   autointerp_path: Path):
    """
    Create subplot visualizations for each n_components configuration.

    Each configuration gets its own figure with a 3-column grid of subplots,
    showing weights for all emotions with PC names from autointerp.
    """

    # Load autointerp
    print(f"Loading autointerp from {autointerp_path}")
    autointerp = load_autointerp(autointerp_path)

    # Extract interpretations
    if 'interpretations' in autointerp:
        interpretations = autointerp['interpretations']
    else:
        interpretations = autointerp

    # Filter to target layer and create PC name lookup
    layer_interpretations = [item for item in interpretations if item['layer'] == layer]
    pc_names = {}
    for pc_data in layer_interpretations:
        pc_idx = pc_data['pc_index']
        pc_names[pc_idx] = pc_data['dimension_name']

    for n_pcs in n_components_list:
        print(f"\n=== Creating visualization for {n_pcs} PCs ===")

        # Load probe
        if n_pcs == 50:
            probe_dir = Path('/workspace-vast/annas/git/research-tools/results/emotion_probes_high_alpha_cpca')
            probe_path = probe_dir / f'probe_layer{layer}_all_cpca.pkl'
        else:
            probe_dir = Path(f'/workspace-vast/annas/git/research-tools/results/emotion_probes_top{n_pcs}')
            probe_path = probe_dir / f'probe_layer{layer}_all_cpca_top{n_pcs}.pkl'

        if not probe_path.exists():
            print(f"  Warning: {probe_path} not found, skipping")
            continue

        probe_results = load_probe(probe_path)
        weights = probe_results['model'].weight.detach().cpu().numpy()  # [n_emotions, n_components]
        emotion_names = probe_results['label_names']
        test_acc = probe_results['test_accuracy']

        # Create grid layout: 3 columns
        n_cols = 3
        n_rows = int(np.ceil(n_pcs / n_cols))

        # Make figure with square-ish subplots
        fig_width = 18  # 6 inches per column
        fig_height = n_rows * 5  # 5 inches per row
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

        # Flatten axes array for easier indexing
        if n_pcs == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        fig.suptitle(f'Layer {layer} - Top {n_pcs} PCs - PC Weights by Emotion\nTest Accuracy: {test_acc:.2%}',
                     fontsize=16, fontweight='bold', y=0.995)

        # For each PC, create a bar chart
        for pc_idx in range(n_pcs):
            ax = axes[pc_idx]

            # Get weights for this PC across all emotions
            pc_weights = weights[:, pc_idx]  # [n_emotions]

            # Create bar positions
            x = np.arange(len(emotion_names))

            # Color bars by emotion
            colors = [EMOTION_COLORS.get(emotion, '#808080') for emotion in emotion_names]

            # Create bars
            bars = ax.bar(x, pc_weights, color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)

            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2, alpha=0.4)

            # Style
            ax.set_ylabel('Weight', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(emotion_names, rotation=45, ha='right', fontsize=10)

            # Title with PC name
            pc_name = pc_names.get(pc_idx, 'Unknown')
            ax.set_title(f'PC {pc_idx}: {pc_name}', fontsize=11, fontweight='bold', pad=10)

            ax.grid(True, alpha=0.25, axis='y', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Set y-axis limits with some padding
            max_abs = np.max(np.abs(pc_weights))
            if max_abs > 0:
                ax.set_ylim(-max_abs * 1.2, max_abs * 1.2)

            # Add value labels on bars
            for i, (bar, weight) in enumerate(zip(bars, pc_weights)):
                if abs(weight) > max_abs * 0.08:  # Only label significant weights
                    label_y = weight + (max_abs * 0.06 if weight > 0 else -max_abs * 0.06)
                    ax.text(bar.get_x() + bar.get_width()/2, label_y, f'{weight:.3f}',
                           ha='center', va='bottom' if weight > 0 else 'top',
                           fontsize=9, fontweight='bold')

        # Hide unused subplots
        for idx in range(n_pcs, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()

        # Save
        output_path = output_dir / f'pc_weights_layer{layer}_top{n_pcs}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()


def main():
    """Generate PC weight visualizations."""

    # Configuration
    layer = 30
    n_components_list = [3, 5, 10]
    output_dir = Path('/workspace-vast/annas/git/research-tools/results/pc_weight_visualizations')
    autointerp_path = Path('/workspace-vast/annas/git/research-tools/probes/results/autointerp/gemma_layer30_top20.json')

    print("="*80)
    print("VISUALIZING PC WEIGHTS")
    print("="*80)
    print(f"\nLayer: {layer}")
    print(f"Configurations: {n_components_list} PCs")
    print(f"Output directory: {output_dir}")
    print(f"Autointerp: {autointerp_path}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    visualize_pc_weights_subplots(layer, n_components_list, output_dir, autointerp_path)

    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
