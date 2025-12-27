#!/usr/bin/env python3
"""Run cPCA on a specific range of layers for parallel processing.

Usage:
    python scripts/run_cpca_layer_range.py config.yaml --start-layer 0 --end-layer 4
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from probes.experiments.cpca_experiment import CPCAConfig
from probes.core import load_activations_hdf5, save_json
from probes.methods.cpca import run_cpca, tune_alpha_silhouette


def run_cpca_layer_range(
    config: CPCAConfig,
    start_layer: int,
    end_layer: int,
) -> None:
    """Run cPCA on a specific range of layers.

    Args:
        config: cPCA configuration
        start_layer: First layer index (inclusive)
        end_layer: Last layer index (exclusive)
    """
    # Create output directory for this range
    range_name = f"layers_{start_layer}_{end_layer-1}"
    output_dir = config.output_dir / range_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing layers {start_layer} to {end_layer-1}")
    print(f"Output directory: {output_dir}")

    # Load data
    print(f"Loading activations from: {config.data_path}")
    activations, metadata, attrs = load_activations_hdf5(config.data_path)
    print(f"Loaded {len(activations)} activation pairs")

    # Get dimensions
    pair_ids = list(activations.keys())
    first_pair = activations[pair_ids[0]]
    num_layers = first_pair["neutral"].shape[0]
    hidden_dim = first_pair["neutral"].shape[1]

    # Validate layer range
    if start_layer < 0 or start_layer >= num_layers:
        raise ValueError(f"start_layer {start_layer} out of range [0, {num_layers})")
    if end_layer <= start_layer or end_layer > num_layers:
        raise ValueError(f"end_layer {end_layer} out of range ({start_layer}, {num_layers}]")

    # Prepare metadata for alpha tuning
    id_to_meta = {m["id"]: m for m in metadata}
    emotion_labels = np.array([id_to_meta[pid]["emotion"] for pid in pair_ids])

    # Alpha values for tuning
    tune = config.alpha is None
    if tune:
        alphas = np.geomspace(config.alpha_range[0], config.alpha_range[1], config.n_alphas)
        print(f"Auto-tuning alpha per layer (range: {config.alpha_range}, n_alphas: {config.n_alphas})")
    else:
        print(f"Using fixed alpha: {config.alpha}")

    # Process each layer in range
    results = {
        "components": {},
        "eigenvalues": {},
        "alpha_per_layer": {},
        "pair_ids": pair_ids,
        "metadata": metadata,
        "start_layer": start_layer,
        "end_layer": end_layer,
    }

    for layer_idx in tqdm(range(start_layer, end_layer), desc=f"Layers {start_layer}-{end_layer-1}"):
        # Extract activations for this layer
        emotional_acts = np.stack(
            [activations[pid]["emotional"][layer_idx] for pid in pair_ids]
        ).astype(np.float32)

        neutral_acts = np.stack(
            [activations[pid]["neutral"][layer_idx] for pid in pair_ids]
        ).astype(np.float32)

        # Choose target data
        if config.use_diffs:
            target_acts = emotional_acts - neutral_acts
        else:
            target_acts = emotional_acts

        # Compute diffs for scoring
        diffs = emotional_acts - neutral_acts

        # Tune alpha if not provided
        if tune:
            layer_alpha, tuning_info = tune_alpha_silhouette(
                target_acts, neutral_acts, diffs, emotion_labels, alphas, config.n_components
            )
        else:
            layer_alpha = config.alpha

        results["alpha_per_layer"][layer_idx] = layer_alpha

        # Run cPCA with selected alpha
        components, eigenvalues = run_cpca(
            target_acts, neutral_acts, layer_alpha, config.n_components
        )

        results["components"][layer_idx] = components
        results["eigenvalues"][layer_idx] = eigenvalues

        # Save checkpoint
        checkpoint_file = output_dir / f"layer_{layer_idx}.npz"
        np.savez_compressed(
            checkpoint_file,
            components=components,
            eigenvalues=eigenvalues,
            alpha=layer_alpha,
        )

    # Save summary
    summary = {
        "model_name": config.model_name,
        "n_components": config.n_components,
        "start_layer": start_layer,
        "end_layer": end_layer,
        "num_layers": end_layer - start_layer,
        "num_pairs": len(pair_ids),
        "hidden_dim": hidden_dim,
        "use_diffs": config.use_diffs,
        "alpha": config.alpha,
        "alpha_range": list(config.alpha_range) if tune else None,
        "alphas_used": {k: float(v) for k, v in results["alpha_per_layer"].items()},
    }

    summary_file = output_dir / "summary.json"
    save_json(summary, summary_file)

    print(f"\nCompleted layers {start_layer} to {end_layer-1}")
    print(f"Checkpoints saved to: {output_dir}")
    print(f"Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Run cPCA on layer range")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        required=True,
        help="First layer index (inclusive)",
    )
    parser.add_argument(
        "--end-layer",
        type=int,
        required=True,
        help="Last layer index (exclusive)",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = CPCAConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Run cPCA on layer range
    try:
        run_cpca_layer_range(config, args.start_layer, args.end_layer)
    except Exception as e:
        print(f"Error running cPCA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
