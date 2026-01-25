#!/usr/bin/env python3
"""Test script for orthogonality-regularized probes with synthetic data.

This script creates synthetic data to test the full orthogonal regularization pipeline:
1. Generate synthetic emotional and neutral activations
2. Compute neutral PCs using ratio-based method
3. Train baseline probe (no regularization)
4. Train orthogonality-regularized probe
5. Compare results

Usage:
    python test_orthogonal_regularized_probe.py
"""

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from probes.methods.orthogonal_regularized_probe import train_orthogonal_regularized_probe


def generate_synthetic_data(
    n_samples: int = 1000,
    hidden_dim: int = 512,
    n_classes: int = 7,
    n_neutral_dims: int = 50,
    seed: int = 42,
):
    """Generate synthetic emotion and neutral activations.

    Args:
        n_samples: Number of samples per class
        hidden_dim: Activation dimensionality
        n_classes: Number of emotion classes
        n_neutral_dims: Number of dimensions dominated by neutral content
        seed: Random seed

    Returns:
        activations: [n_samples * n_classes, hidden_dim]
        labels: [n_samples * n_classes]
        neutral_pcs: [hidden_dim, n_neutral_dims] ground truth neutral directions
    """
    np.random.seed(seed)

    # Create ground truth neutral directions (random orthonormal basis)
    neutral_pcs = np.random.randn(hidden_dim, n_neutral_dims)
    neutral_pcs, _ = np.linalg.qr(neutral_pcs)  # Orthonormalize

    # Create emotion-specific directions (orthogonal to neutral)
    emotion_directions = np.random.randn(hidden_dim, n_classes)
    # Project out neutral components
    for i in range(n_classes):
        emotion_directions[:, i] -= neutral_pcs @ (neutral_pcs.T @ emotion_directions[:, i])
    # Normalize
    emotion_directions = emotion_directions / np.linalg.norm(emotion_directions, axis=0, keepdims=True)

    # Generate activations
    all_activations = []
    all_labels = []

    for class_idx in range(n_classes):
        for _ in range(n_samples):
            # Base neutral component (varies randomly)
            neutral_component = neutral_pcs @ np.random.randn(n_neutral_dims) * 5.0

            # Emotion-specific component
            emotion_component = emotion_directions[:, class_idx] * 10.0

            # Add noise
            noise = np.random.randn(hidden_dim) * 0.5

            # Combine
            activation = neutral_component + emotion_component + noise

            all_activations.append(activation)
            all_labels.append(class_idx)

    activations = np.array(all_activations)
    labels = np.array(all_labels)

    return activations, labels, neutral_pcs


def compute_neutral_pcs_synthetic(
    emotional_acts: np.ndarray,
    neutral_acts: np.ndarray,
    k: int,
):
    """Compute neutral PCs using ratio-based method (simplified for synthetic data)."""
    from sklearn.decomposition import PCA

    # Compute differences
    diffs = emotional_acts - neutral_acts

    # Combine for PCA
    combined = np.vstack([neutral_acts, emotional_acts])

    # Run PCA
    n_comp = min(100, combined.shape[0] - 1, combined.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(combined)
    all_pcs = pca.components_

    # Project
    neutral_proj = neutral_acts @ all_pcs.T
    diff_proj = diffs @ all_pcs.T

    # Compute ratio
    neutral_var = neutral_proj.var(axis=0)
    diff_var = diff_proj.var(axis=0)
    ratio = neutral_var / (diff_var + 1e-8)

    # Get top-k
    top_k_indices = np.argsort(ratio)[-k:]
    neutral_pcs = all_pcs[top_k_indices].T  # [hidden_dim, k]

    return neutral_pcs


def main():
    print("=" * 80)
    print("TESTING ORTHOGONAL REGULARIZED PROBES WITH SYNTHETIC DATA")
    print("=" * 80)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    activations, labels, true_neutral_pcs = generate_synthetic_data(
        n_samples=200,
        hidden_dim=256,
        n_classes=7,
        n_neutral_dims=20,
        seed=42,
    )
    print(f"   Generated {len(activations)} samples")
    print(f"   Activation shape: {activations.shape}")
    print(f"   True neutral PCs: {true_neutral_pcs.shape}")

    # Split data
    print("\n2. Splitting train/test...")
    train_acts, test_acts, train_labels, test_labels = train_test_split(
        activations, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"   Train: {len(train_acts)}, Test: {len(test_acts)}")

    # Compute neutral PCs using ratio-based method
    print("\n3. Computing neutral PCs using ratio-based method...")
    # For this, we need to simulate having emotional/neutral pairs
    # We'll use a simple approximation: treat half the data as "neutral baseline"
    half = len(train_acts) // 2
    estimated_neutral_pcs = compute_neutral_pcs_synthetic(
        train_acts[:half],
        train_acts[half : 2 * half],
        k=15,
    )
    print(f"   Estimated neutral PCs: {estimated_neutral_pcs.shape}")

    # Measure alignment with true neutral PCs
    alignment = np.linalg.norm(
        true_neutral_pcs.T @ estimated_neutral_pcs, "fro"
    ) / np.sqrt(true_neutral_pcs.shape[1] * estimated_neutral_pcs.shape[1])
    print(f"   Alignment with true neutral PCs: {alignment:.4f} (1.0 = perfect)")

    # Train baseline probe (no regularization)
    print("\n4. Training baseline probe (λ=0)...")
    baseline_results = train_orthogonal_regularized_probe(
        train_activations=train_acts,
        train_labels=train_labels,
        test_activations=test_acts,
        test_labels=test_labels,
        neutral_pcs=None,
        lambda_ortho=0.0,
        max_epochs=100,
        patience=10,
        device="cpu",
        verbose=False,
    )
    print(f"   Train accuracy: {baseline_results['train_accuracy']:.4f}")
    print(f"   Test accuracy: {baseline_results['test_accuracy']:.4f}")

    # Check overlap with neutral PCs
    baseline_weights = baseline_results['model'].linear.weight.detach().cpu().numpy()
    baseline_overlap = np.linalg.norm(
        baseline_weights @ estimated_neutral_pcs
    ) / np.linalg.norm(baseline_weights)
    print(f"   Overlap with neutral PCs: {baseline_overlap:.4f}")

    # Train orthogonality-regularized probes with different λ values
    print("\n5. Training orthogonality-regularized probes...")
    lambda_values = [0.1, 1.0, 10.0]

    for lambda_ortho in lambda_values:
        print(f"\n   λ = {lambda_ortho}")
        results = train_orthogonal_regularized_probe(
            train_activations=train_acts,
            train_labels=train_labels,
            test_activations=test_acts,
            test_labels=test_labels,
            neutral_pcs=estimated_neutral_pcs,
            lambda_ortho=lambda_ortho,
            max_epochs=100,
            patience=10,
            device="cpu",
            verbose=False,
        )

        print(f"      Train accuracy: {results['train_accuracy']:.4f}")
        print(f"      Test accuracy: {results['test_accuracy']:.4f}")
        print(f"      Ortho normalized: {results['ortho_metrics']['ortho_normalized']:.4f}")
        print(f"      Neutral fraction: {results['ortho_metrics']['neutral_fraction']:.4f}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nExpected behavior:")
    print("  - Baseline has high overlap with neutral PCs")
    print("  - Higher λ → lower neutral fraction (more orthogonal)")
    print("  - Test accuracy should remain reasonably high")
    print("  - Some accuracy drop is expected as λ increases")


if __name__ == "__main__":
    main()
