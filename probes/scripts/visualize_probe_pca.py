#!/usr/bin/env python3
"""
Visualize emotion probe directions in 2D using PCA.
For each probe type at layer 30:
- Collect all emotion direction vectors
- Run PCA to get first 2 components
- Project and plot emotion directions
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import torch

# Emotion color mapping from probe_configs.py
EMOTION_COLORS = {
    'anger': '#7BA7D7',      # sky_blue
    'disgust': '#7D9B7D',    # olive
    'fear': '#a59dc9',       # lavender
    'happiness': '#D4876A',  # coral
    'sadness': '#B8CCC8',    # sage
    'surprise': '#D1728F',   # darker pink
    'neutral': '#95a5a6',    # gray for neutral
}

def load_probe_weights(probe_path):
    """Load probe and extract weights and labels."""
    with open(probe_path, 'rb') as f:
        data = pickle.load(f)

    # Convert torch tensors to numpy if needed
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    result = {}

    # Extract emotion labels
    if 'label_names' in data:
        result['labels'] = data['label_names']
    elif 'emotion_labels' in data:
        result['labels'] = data['emotion_labels']
    else:
        print(f"Warning: No labels found in {probe_path}")
        result['labels'] = None

    # Extract weights based on probe structure
    if 'user_probes' in data and 'assistant_probes' in data:
        # Orthogonal probe with user/assistant separation (old format)
        result['user_weights'] = to_numpy(data['user_probes'])
        result['assistant_weights'] = to_numpy(data['assistant_probes'])
        result['type'] = 'orthogonal_split'

    elif 'final_user_probes' in data and 'final_asst_probes' in data:
        # Conversation-based orthogonal probe (newer format)
        result['user_weights'] = to_numpy(data['final_user_probes'])
        result['assistant_weights'] = to_numpy(data['final_asst_probes'])
        result['type'] = 'orthogonal_split'

    elif 'probe_sets' in data and isinstance(data['probe_sets'], np.ndarray):
        # Multi-orthogonal conversation probe: [n_sets, 2, n_emotions, hidden_dim]
        probe_sets = data['probe_sets']
        if len(probe_sets.shape) == 4 and probe_sets.shape[1] == 2:
            # Has user/assistant split (dim 1 has size 2)
            # Average over probe sets (dim 0) to get single user/assistant probes
            result['user_weights'] = probe_sets[:, 0, :, :].mean(axis=0)  # [n_emotions, hidden_dim]
            result['assistant_weights'] = probe_sets[:, 1, :, :].mean(axis=0)
            result['type'] = 'orthogonal_split'
        else:
            print(f"Warning: Unexpected probe_sets shape {probe_sets.shape}")
            return None

    elif 'all_probe_sets' in data and isinstance(data['all_probe_sets'], np.ndarray):
        # Multi-orthogonal text probe: [n_sets, n_emotions, hidden_dim]
        probe_sets = data['all_probe_sets']
        if len(probe_sets.shape) == 3:
            # Average over probe sets to get single set of weights
            result['weights'] = probe_sets.mean(axis=0)  # [n_emotions, hidden_dim]
            result['type'] = 'linear'
        else:
            print(f"Warning: Unexpected all_probe_sets shape {probe_sets.shape}")
            return None

    elif 'model' in data:
        # OrthogonalRegularizedProbe or standard linear probe
        model = data['model']
        if hasattr(model, 'state_dict'):
            state = model.state_dict()
            if 'linear.weight' in state:
                # OrthogonalRegularizedProbe
                result['weights'] = to_numpy(state['linear.weight'])
                result['type'] = 'linear'
            elif 'weight' in state:
                result['weights'] = to_numpy(state['weight'])
                result['type'] = 'linear'
            else:
                print(f"Warning: No weight in model state_dict")
                return None
        elif hasattr(model, 'weight'):
            weights = to_numpy(model.weight)
            result['weights'] = weights
            result['type'] = 'linear'
        else:
            print(f"Warning: Unknown model structure in {probe_path}")
            return None
    else:
        print(f"Warning: Unknown probe structure in {probe_path}")
        print(f"  Available keys: {list(data.keys())}")
        return None

    return result

def adjust_text_positions(positions, labels, min_distance=0.15, point_distance=0.08):
    """Simple algorithm to adjust overlapping text labels.

    Args:
        positions: Original positions of points
        labels: Label text (for reference)
        min_distance: Minimum distance between labels
        point_distance: Minimum distance from labels to points

    Returns adjusted positions in data coordinates.
    """
    positions = np.array(positions).astype(float)
    n = len(positions)

    if n == 0:
        return positions

    # Start with small offsets from original positions
    adjusted = positions + np.random.randn(n, 2) * 0.03

    # Iteratively push apart overlapping labels and away from points
    for iteration in range(150):
        moved = False

        # Push labels away from each other
        for i in range(n):
            for j in range(i + 1, n):
                diff = adjusted[i] - adjusted[j]
                dist = np.linalg.norm(diff)

                if dist < min_distance and dist > 1e-6:
                    push = (min_distance - dist) / 2 * 1.2
                    direction = diff / dist
                    adjusted[i] += direction * push
                    adjusted[j] -= direction * push
                    moved = True

        # Push labels away from ALL points (not just their own)
        for i in range(n):
            for j in range(n):
                diff = adjusted[i] - positions[j]
                dist = np.linalg.norm(diff)

                if dist < point_distance and dist > 1e-6:
                    push = (point_distance - dist) * 0.8
                    direction = diff / dist
                    adjusted[i] += direction * push
                    moved = True

        if not moved:
            break

    return adjusted

def plot_pca_projection(weights, labels, title, ax, markers=None, pca_basis_weights=None):
    """Run PCA and plot 2D projection.

    Args:
        weights: [n_emotions, hidden_dim] probe weights to project
        labels: List of emotion labels
        title: Plot title
        ax: Matplotlib axis
        markers: Optional list of marker styles for each point (for distinguishing user/asst)
        pca_basis_weights: Optional weights to compute PCA basis from (if different from weights)
    """
    # weights shape: [n_emotions, hidden_dim]
    if weights.shape[0] < 2:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        ax.set_title(title)
        return

    # Run PCA on specified basis (or on weights itself)
    if pca_basis_weights is not None:
        pca = PCA(n_components=min(2, pca_basis_weights.shape[0]))
        pca.fit(pca_basis_weights)
        projected = pca.transform(weights)
    else:
        pca = PCA(n_components=min(2, weights.shape[0]))
        projected = pca.fit_transform(weights)

    # Default to circles if no markers specified
    if markers is None:
        markers = ['o'] * len(labels)

    # Plot each emotion with its color and marker
    for i, label in enumerate(labels):
        # Extract base emotion name (remove user/asst suffix if present)
        base_emotion = label.split(' (')[0]
        color = EMOTION_COLORS.get(base_emotion, '#333333')
        marker = markers[i]

        ax.scatter(projected[i, 0], projected[i, 1],
                  s=150, alpha=0.7, color=color, marker=marker,
                  edgecolors='black', linewidth=1.5)

    # Adjust label positions to avoid overlap
    adjusted_positions = adjust_text_positions(projected, labels)

    # Add labels at adjusted positions
    for i, label in enumerate(labels):
        # Only draw arrow if label moved significantly
        moved_distance = np.linalg.norm(adjusted_positions[i] - projected[i])

        if moved_distance > 0.05:
            # Draw thin line from point to label
            ax.plot([projected[i, 0], adjusted_positions[i, 0]],
                   [projected[i, 1], adjusted_positions[i, 1]],
                   'k-', linewidth=0.5, alpha=0.3, zorder=1)

        # Add text label with semi-transparent background
        ax.text(adjusted_positions[i, 0], adjusted_positions[i, 1], label,
               fontsize=7, ha='center', va='center', fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                       edgecolor='gray', linewidth=0.5, alpha=0.85),
               zorder=3)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=10)
    if len(pca.explained_variance_ratio_) > 1:
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)

def main():
    # Define probe paths to visualize (NO multi-orthogonal probes)
    base_path = Path('/workspace-vast/annas/git/research-tools')

    probes_to_plot = {
        # Orthogonal regularized probes (different lambda values)
        'Ortho-Reg Baseline':
            'probes/results/orthogonal_regularized_probes/probe_layer30_all_baseline.pkl',
        'Ortho-Reg λ=0.1':
            'probes/results/orthogonal_regularized_probes/probe_layer30_all_ortho_k20_lambda0.1.pkl',
        'Ortho-Reg λ=1.0':
            'probes/results/orthogonal_regularized_probes/probe_layer30_all_ortho_k20_lambda1.0.pkl',
        'Ortho-Reg λ=10.0':
            'probes/results/orthogonal_regularized_probes/probe_layer30_all_ortho_k20_lambda10.0.pkl',
        'Ortho-Reg λ=100.0':
            'probes/results/orthogonal_regularized_probes/probe_layer30_all_ortho_k20_lambda100.0.pkl',

        # Text-based linear probes (raw)
        'Text Raw (seed 0)':
            'outputs/probes/emotion_probes/text_based/multiseed/probe_layer30_nc0_seed0.pkl',
        'Text Raw (seed 5)':
            'outputs/probes/emotion_probes/text_based/multiseed/probe_layer30_nc0_seed5.pkl',

        # Text-based with cPCA
        'Text cPCA-10 (seed 0)':
            'outputs/probes/emotion_probes/text_based/multiseed/probe_layer30_nc10_seed0.pkl',
        'Text cPCA Top-10':
            'outputs/probes/emotion_probes/text_based/cpca/top10/probe_layer30_all_cpca_top10.pkl',
        'Text cPCA Top-20':
            'outputs/probes/emotion_probes/text_based/cpca/top20/probe_layer30_all_cpca_top20.pkl',

        # Conversation-based orthogonal probes (with user/assistant split)
        'Conv Orthogonal Raw':
            'outputs/probes/emotion_probes/conversation_based/orthogonal/ortho_1000.0/probe_layer30_raw_ortho1000.0.pkl',
        'Conv Orthogonal cPCA-10':
            'outputs/probes/emotion_probes/conversation_based/orthogonal/ortho_1000.0/probe_layer30_global_cpca_top10_ortho1000.0.pkl',
        'Conv Orthogonal cPCA-20':
            'outputs/probes/emotion_probes/conversation_based/orthogonal/ortho_1000.0/probe_layer30_global_cpca_top20_ortho1000.0.pkl',
    }

    # Load all probes
    probe_data = {}
    for name, rel_path in probes_to_plot.items():
        full_path = base_path / rel_path
        if not full_path.exists():
            print(f"Warning: {full_path} not found")
            continue

        print(f"Loading {name}...")
        data = load_probe_weights(full_path)
        if data is not None:
            probe_data[name] = data

    # Count total subplots needed (skip None entries)
    n_plots = 0
    for name, data in probe_data.items():
        if data is None:
            continue
        if data['type'] == 'orthogonal_split':
            n_plots += 3  # both on user PCs, both on asst PCs, both on combined PCs
        else:
            n_plots += 1

    # Create figure with subplots
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    plot_idx = 0
    for name, data in probe_data.items():
        if data is None:
            continue

        labels = data['labels']

        if data['type'] == 'orthogonal_split':
            # Prepare combined data
            combined = np.vstack([data['user_weights'], data['assistant_weights']])
            combined_labels = [f"{l} (user)" for l in labels] + \
                            [f"{l} (asst)" for l in labels]
            combined_markers = ['o'] * len(labels) + ['s'] * len(labels)

            # Plot both on user PCs
            plot_pca_projection(
                combined, combined_labels,
                f"{name}\n(Both on User PCs)", axes[plot_idx],
                markers=combined_markers,
                pca_basis_weights=data['user_weights']
            )
            plot_idx += 1

            # Plot both on assistant PCs
            plot_pca_projection(
                combined, combined_labels,
                f"{name}\n(Both on Asst PCs)", axes[plot_idx],
                markers=combined_markers,
                pca_basis_weights=data['assistant_weights']
            )
            plot_idx += 1

            # Plot both on combined PCs (PCA on both user and assistant together)
            plot_pca_projection(
                combined, combined_labels,
                f"{name}\n(Both on Combined PCs)", axes[plot_idx],
                markers=combined_markers,
                pca_basis_weights=combined
            )
            plot_idx += 1

        else:
            # Single plot
            plot_pca_projection(
                data['weights'], labels,
                name, axes[plot_idx]
            )
            plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save figure
    output_path = base_path / 'probes' / 'visualizations' / 'probe_pca_comparison_layer30.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")

    plt.show()

if __name__ == '__main__':
    main()
