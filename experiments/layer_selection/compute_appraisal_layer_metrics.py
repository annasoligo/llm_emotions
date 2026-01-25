#!/usr/bin/env python3
"""Compute layer metrics for appraisal steering vectors (valence, uncertainty, agency).

This script computes layer selection metrics for appraisal-based steering:
- Variance Ratio: var(steering projection) / mean(var(random projections))
- PCA Alignment: Fraction of steering vector in top-k PC subspace

Appraisal vectors are computed as mean-difference between high and low conditions
for each axis (valence, uncertainty, agency).

Usage:
    python -m experiments.layer_selection.compute_appraisal_layer_metrics \
        --h5-path /workspace-vast/annas/appraisal_data/full_run/activations.h5 \
        --output-dir experiments/layer_selection/results/appraisal/ \
        --model-name gemma27b
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Appraisal axes
AXES = ["valence", "uncertainty", "agency"]

# Token position to use
TOKEN_POSITION = "assistant_start_last_token"


def load_appraisal_steering_vectors(
    h5_path: Path,
    n_layers: int,
    axes: List[str] = None,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, np.ndarray]]:
    """Load appraisal activations and compute steering vectors.

    Steering vector = mean(high_condition) - mean(low_condition)

    Returns:
        (steering_vectors, all_activations) where:
        - steering_vectors: {layer: {axis: unit_vector}}
        - all_activations: {layer: [n_samples, hidden_dim]}
    """
    if axes is None:
        axes = AXES

    logger.info(f"Loading appraisal activations from {h5_path}")
    logger.info(f"Axes: {axes}")

    steering_vectors: Dict[int, Dict[str, np.ndarray]] = {layer: {} for layer in range(n_layers)}
    all_activations: Dict[int, List[np.ndarray]] = {layer: [] for layer in range(n_layers)}

    with h5py.File(h5_path, "r") as f:
        acts_group = f["activations"]
        sample_names = list(acts_group.keys())

        logger.info(f"Total samples: {len(sample_names)}")

        # Group samples by axis and condition (high=_a_, low=_b_)
        axis_samples = {axis: {"high": [], "low": []} for axis in axes}

        for name in sample_names:
            for axis in axes:
                if name.startswith(f"{axis}_"):
                    # _a_ is high, _b_ is low based on minimal pair convention
                    if "_a_" in name:
                        axis_samples[axis]["high"].append(name)
                    elif "_b_" in name:
                        axis_samples[axis]["low"].append(name)

        for axis in axes:
            logger.info(f"  {axis}: {len(axis_samples[axis]['high'])} high, {len(axis_samples[axis]['low'])} low")

        # Load activations and compute steering vectors
        for layer in tqdm(range(n_layers), desc="Processing layers"):
            layer_activations = []

            for axis in axes:
                high_acts = []
                low_acts = []

                for name in axis_samples[axis]["high"]:
                    sample = acts_group[name]
                    if TOKEN_POSITION in sample:
                        act = sample[TOKEN_POSITION][layer, :]
                        high_acts.append(act)
                        layer_activations.append(act)

                for name in axis_samples[axis]["low"]:
                    sample = acts_group[name]
                    if TOKEN_POSITION in sample:
                        act = sample[TOKEN_POSITION][layer, :]
                        low_acts.append(act)
                        layer_activations.append(act)

                if high_acts and low_acts:
                    high_mean = np.mean(high_acts, axis=0)
                    low_mean = np.mean(low_acts, axis=0)
                    diff = high_mean - low_mean
                    norm = np.linalg.norm(diff)
                    if norm > 1e-8:
                        diff = diff / norm
                    steering_vectors[layer][axis] = diff.astype(np.float32)

            if layer_activations:
                all_activations[layer] = np.stack(layer_activations).astype(np.float32)

    n_vectors = sum(len(v) for v in steering_vectors.values())
    logger.info(f"Loaded {n_vectors} steering vectors")

    return steering_vectors, all_activations


def compute_variance_ratio(
    activations: np.ndarray,
    steering_vector: np.ndarray,
    n_random: int = 100,
) -> float:
    """Compute variance ratio: var(steering projection) / mean(var(random projections))."""
    proj_steering = activations @ steering_vector
    var_steering = np.var(proj_steering)

    hidden_dim = activations.shape[1]
    random_vars = []

    for _ in range(n_random):
        random_vec = np.random.randn(hidden_dim).astype(np.float32)
        random_vec = random_vec / np.linalg.norm(random_vec)
        proj_random = activations @ random_vec
        random_vars.append(np.var(proj_random))

    mean_random_var = np.mean(random_vars)

    if mean_random_var < 1e-10:
        return 1.0

    return var_steering / mean_random_var


def compute_pca_alignment(
    activations: np.ndarray,
    steering_vector: np.ndarray,
    k: int = 50,
) -> float:
    """Compute PCA alignment: fraction of steering vector in top-k PC subspace."""
    n_samples = activations.shape[0]
    k = min(k, n_samples - 1, activations.shape[1])

    if k < 1:
        return 0.0

    pca = PCA(n_components=k)
    pca.fit(activations)

    pcs = pca.components_
    coeffs = pcs @ steering_vector
    projection = coeffs @ pcs

    proj_norm = np.linalg.norm(projection)
    vec_norm = np.linalg.norm(steering_vector)

    if vec_norm < 1e-10:
        return 0.0

    return proj_norm / vec_norm


def main():
    parser = argparse.ArgumentParser(
        description="Compute layer metrics for appraisal steering vectors"
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        required=True,
        help="Path to appraisal activations H5 file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/layer_selection/results/appraisal"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name for output files (e.g., gemma27b, qwen32b)",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=100,
        help="Number of random vectors for variance ratio",
    )
    parser.add_argument(
        "--pca-k",
        type=int,
        default=50,
        help="Number of top PCs for alignment metric",
    )

    args = parser.parse_args()

    # Resolve paths
    repo_root = Path(__file__).parent.parent.parent
    h5_path = args.h5_path
    if not h5_path.is_absolute():
        h5_path = repo_root / h5_path

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect number of layers from h5 file
    with h5py.File(h5_path, "r") as f:
        acts_group = f["activations"]
        sample_name = list(acts_group.keys())[0]
        sample = acts_group[sample_name]
        if TOKEN_POSITION in sample:
            n_layers = sample[TOKEN_POSITION].shape[0]
            hidden_dim = sample[TOKEN_POSITION].shape[1]
        else:
            raise ValueError(f"Token position {TOKEN_POSITION} not found")

    logger.info(f"Model: {args.model_name}")
    logger.info(f"Layers: {n_layers}, Hidden dim: {hidden_dim}")

    # Load steering vectors and activations
    steering_vectors, all_activations = load_appraisal_steering_vectors(
        h5_path, n_layers, AXES
    )

    # Initialize results
    results = {
        'variance_ratio': {axis: np.zeros(n_layers) for axis in AXES},
        'pca_alignment': {axis: np.zeros(n_layers) for axis in AXES},
    }

    # Compute metrics
    logger.info("Computing metrics...")
    for axis in tqdm(AXES, desc="Axes"):
        for layer in tqdm(range(n_layers), desc=f"  {axis} layers", leave=False):
            vec = steering_vectors[layer].get(axis)
            acts = all_activations.get(layer)

            if vec is None or acts is None or len(acts) == 0:
                continue

            results['variance_ratio'][axis][layer] = compute_variance_ratio(
                acts, vec, n_random=args.n_random
            )
            results['pca_alignment'][axis][layer] = compute_pca_alignment(
                acts, vec, k=args.pca_k
            )

    # Save results
    output_path = output_dir / f"appraisal_layer_metrics_{args.model_name}.npz"
    save_data = {}

    for metric in ['variance_ratio', 'pca_alignment']:
        for axis in AXES:
            save_data[f"{metric}_{axis}"] = results[metric][axis]

        # Compute average
        avg = np.mean([results[metric][axis] for axis in AXES], axis=0)
        save_data[f"{metric}_average"] = avg

    np.savez(output_path, **save_data, axes=AXES, n_layers=n_layers,
             model_name=args.model_name)
    logger.info(f"Saved results to {output_path}")

    # Generate summary
    summary_path = output_dir / f"appraisal_summary_{args.model_name}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Appraisal Layer Selection Metrics Summary\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Layers: {n_layers}\n")
        f.write("=" * 60 + "\n\n")

        for metric in ['variance_ratio', 'pca_alignment']:
            f.write(f"\n{metric.upper()}\n")
            f.write("-" * 40 + "\n")

            for axis in AXES:
                values = results[metric][axis]
                top_5 = np.argsort(values)[-5:][::-1]
                f.write(f"\n{axis}:\n")
                for i, layer in enumerate(top_5):
                    f.write(f"  {i+1}. Layer {layer}: {values[layer]:.4f}\n")

            avg = np.mean([results[metric][axis] for axis in AXES], axis=0)
            top_5_avg = np.argsort(avg)[-5:][::-1]
            f.write(f"\nAVERAGE:\n")
            for i, layer in enumerate(top_5_avg):
                f.write(f"  {i+1}. Layer {layer}: {avg[layer]:.4f}\n")

    logger.info(f"Saved summary to {summary_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"RESULTS SUMMARY - {args.model_name}")
    print("=" * 60)

    for metric in ['variance_ratio', 'pca_alignment']:
        print(f"\n{metric.upper()}:")
        for axis in AXES:
            values = results[metric][axis]
            top_5 = np.argsort(values)[-5:][::-1]
            print(f"  {axis}: Top 5 = {list(top_5)}")

        avg = np.mean([results[metric][axis] for axis in AXES], axis=0)
        top_5_avg = np.argsort(avg)[-5:][::-1]
        print(f"  AVERAGE: Top 5 = {list(top_5_avg)}")


if __name__ == "__main__":
    main()
