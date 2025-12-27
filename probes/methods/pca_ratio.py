#!/usr/bin/env python3
"""
Ratio-based neutral removal PCA (PCA v3 approach).

This implements the ratio-based neutral removal method from believe-it-or-not/pc_probes/pca_v3.py,
adapted for the research-tools codebase to enable direct comparison with cPCA.

Key Insight:
Instead of using alpha-tuned contrastive covariance, we:
1. Run PCA on ALL activations (neutral + emotional combined)
2. Compute variance of neutral and diffs projected onto each PC
3. Remove top-k PCs with highest neutral/diff variance ratio
4. Run PCA on cleaned diffs

This identifies and removes "neutral-dominated" directions while preserving emotional signal.

Comparison with cPCA:
- cPCA: C_emo - alpha * C_neu (requires tuning alpha per layer)
- Ratio PCA: Remove high neutral/diff ratio PCs (no tuning required)

Both aim to remove neutral structure, but ratio method is more direct and interpretable.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional


def ratio_based_neutral_removal(
    diffs: np.ndarray,
    neutral: np.ndarray,
    emotional: np.ndarray,
    k: int,
    n_pcs_all: int = 100,
) -> Tuple[np.ndarray, Dict]:
    """Remove top-k PCs with highest neutral/diff variance ratio.

    The idea: run PCA on ALL activations (neutral + emotional), then identify
    which PCs have high neutral variance relative to diff variance. These are
    the "neutral-dominated" directions we want to remove.

    Args:
        diffs: [n_pairs, hidden_dim] - activation differences at one layer
        neutral: [n_pairs, hidden_dim] - neutral activations at one layer
        emotional: [n_pairs, hidden_dim] - emotional activations at one layer
        k: number of high-ratio PCs to remove
        n_pcs_all: number of PCs to compute on combined activations (default: 100)

    Returns:
        cleaned_diffs: [n_pairs, hidden_dim] - diffs with neutral-dominated PCs removed
        info: dict with diagnostic information
            - k: number of PCs removed
            - removed_pc_indices: indices of removed PCs
            - removed_pc_ratios: ratio values of removed PCs
            - mean_ratio_removed: average ratio of removed PCs
            - mean_ratio_kept: average ratio of kept PCs
    """
    if k <= 0:
        return diffs, {
            "k": 0,
            "removed_pc_indices": [],
            "removed_pc_ratios": [],
            "mean_ratio_removed": 0.0,
            "mean_ratio_kept": 0.0,
        }

    # Combine neutral and emotional for PCA
    combined = np.vstack([neutral, emotional])  # [2*n_pairs, hidden_dim]

    # Run PCA on combined to get "universal" basis
    n_comp = min(n_pcs_all, combined.shape[0] - 1, combined.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(combined)
    all_pcs = pca.components_  # [n_comp, hidden_dim]

    # Project neutral and diffs onto each PC
    neutral_proj = neutral @ all_pcs.T  # [n_pairs, n_comp]
    diff_proj = diffs @ all_pcs.T  # [n_pairs, n_comp]

    # Compute variance of projections
    neutral_var = neutral_proj.var(axis=0)  # [n_comp]
    diff_var = diff_proj.var(axis=0)  # [n_comp]

    # Compute ratio (neutral variance / diff variance)
    # High ratio = neutral-dominated, low ratio = signal-dominated
    ratio = neutral_var / (diff_var + 1e-8)

    # Find top-k PCs with highest ratio (most neutral-dominated)
    top_k_indices = np.argsort(ratio)[-k:]

    # Get the PCs to remove
    pcs_to_remove = all_pcs[top_k_indices]  # [k, hidden_dim]

    # Project diffs onto these PCs
    projections = diffs @ pcs_to_remove.T  # [n_pairs, k]

    # Subtract projections (remove neutral-dominated directions)
    cleaned = diffs - projections @ pcs_to_remove

    # Compute diagnostic info
    info = {
        "k": k,
        "removed_pc_indices": top_k_indices.tolist(),
        "removed_pc_ratios": ratio[top_k_indices].tolist(),
        "mean_ratio_removed": float(ratio[top_k_indices].mean()),
        "mean_ratio_kept": float(np.delete(ratio, top_k_indices).mean()) if k < n_comp else 0.0,
    }

    return cleaned, info


def run_pca_with_ratio_removal(
    target_acts: np.ndarray,
    background_acts: np.ndarray,
    k: int = 10,
    n_components: int = 50,
    n_pcs_all: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Run PCA with ratio-based neutral removal.

    This is the main entry point compatible with the research-tools cPCA interface.

    Args:
        target_acts: [n_samples, hidden_dim] - emotional activations
        background_acts: [n_samples, hidden_dim] - neutral activations
        k: number of high-ratio PCs to remove (default: 10)
        n_components: number of final PCs to return (default: 50)
        n_pcs_all: number of PCs for ratio analysis (default: 100)

    Returns:
        components: [n_components, hidden_dim] - principal components
        explained_variance: [n_components] - variance explained by each PC
        explained_variance_ratio: [n_components] - ratio of variance explained
        removal_info: dict with diagnostic information about neutral removal
    """
    # Compute diffs
    diffs = target_acts - background_acts

    # Remove neutral-dominated directions
    cleaned_diffs, removal_info = ratio_based_neutral_removal(
        diffs=diffs,
        neutral=background_acts,
        emotional=target_acts,
        k=k,
        n_pcs_all=n_pcs_all,
    )

    # Run PCA on cleaned diffs
    n_comp = min(n_components, cleaned_diffs.shape[0] - 1, cleaned_diffs.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(cleaned_diffs)

    components = pca.components_
    explained_variance = pca.explained_variance_
    explained_variance_ratio = pca.explained_variance_ratio_

    return components, explained_variance, explained_variance_ratio, removal_info


def run_ratio_pca_per_layer(
    target_acts: np.ndarray,
    background_acts: np.ndarray,
    k: int = 10,
    n_components: int = 50,
    n_pcs_all: int = 100,
) -> Dict:
    """Run ratio-based PCA on each layer independently.

    Compatible interface with cpca.run_cpca for easy comparison.

    Args:
        target_acts: [n_samples, num_layers, hidden_dim] - emotional activations
        background_acts: [n_samples, num_layers, hidden_dim] - neutral activations
        k: number of high-ratio PCs to remove per layer
        n_components: number of final PCs per layer
        n_pcs_all: number of PCs for ratio analysis

    Returns:
        results dict with:
            - components: dict mapping layer_idx -> [n_components, hidden_dim]
            - explained_variance: dict mapping layer_idx -> [n_components]
            - explained_variance_ratio: dict mapping layer_idx -> [n_components]
            - removal_info: dict mapping layer_idx -> removal diagnostics
            - config: dict with k, n_components, n_pcs_all
    """
    n_layers = target_acts.shape[1]

    results = {
        "components": {},
        "explained_variance": {},
        "explained_variance_ratio": {},
        "removal_info": {},
        "config": {
            "method": "ratio_pca",
            "k": k,
            "n_components": n_components,
            "n_pcs_all": n_pcs_all,
        },
    }

    for layer_idx in range(n_layers):
        # Extract this layer
        target_layer = target_acts[:, layer_idx, :]
        background_layer = background_acts[:, layer_idx, :]

        # Run ratio PCA
        comps, var, var_ratio, removal_info = run_pca_with_ratio_removal(
            target_acts=target_layer,
            background_acts=background_layer,
            k=k,
            n_components=n_components,
            n_pcs_all=n_pcs_all,
        )

        results["components"][layer_idx] = comps
        results["explained_variance"][layer_idx] = var
        results["explained_variance_ratio"][layer_idx] = var_ratio
        results["removal_info"][layer_idx] = removal_info

    return results