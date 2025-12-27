#!/usr/bin/env python3
"""Combine cPCA results from multiple layer ranges into a single file."""

import argparse
import json
from pathlib import Path

import numpy as np

from probes.core import save_cpca_results


def combine_layer_ranges(
    base_dir: Path,
    output_file: Path,
    num_layers: int = 62,
    layers_per_task: int = 4,
):
    """Combine layer range results into single file.

    Args:
        base_dir: Directory containing layers_X_Y subdirectories
        output_file: Output path for combined results
        num_layers: Total number of layers
        layers_per_task: Layers per task (for calculating ranges)
    """
    print(f"Combining cPCA results from {base_dir}")
    print(f"Total layers: {num_layers}")

    # Load all layer results
    all_components = {}
    all_eigenvalues = {}
    all_alphas = {}
    metadata = None
    pair_ids = None
    config = None

    for layer_idx in range(num_layers):
        # Find which task this layer belongs to
        task_id = layer_idx // layers_per_task
        start_layer = task_id * layers_per_task
        if task_id == 15:  # Last task handles remaining layers
            end_layer = num_layers
        else:
            end_layer = start_layer + layers_per_task

        # Load layer checkpoint
        layer_dir = base_dir / f"layers_{start_layer}_{end_layer-1}"
        layer_file = layer_dir / f"layer_{layer_idx}.npz"

        if not layer_file.exists():
            raise FileNotFoundError(f"Missing layer file: {layer_file}")

        data = np.load(layer_file)
        all_components[layer_idx] = data['components']
        all_eigenvalues[layer_idx] = data['eigenvalues']
        all_alphas[layer_idx] = float(data['alpha'])

        print(f"  Loaded layer {layer_idx} from {layer_dir.name}")

        # Load metadata and config from first task's summary
        if metadata is None:
            summary_file = layer_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file) as f:
                    summary = json.load(f)
                    config = {
                        'model_name': summary['model_name'],
                        'n_components': summary['n_components'],
                        'num_layers': num_layers,
                        'hidden_dim': summary['hidden_dim'],
                        'use_diffs': summary['use_diffs'],
                    }

    # Load metadata and pair_ids from one of the layer directories
    # (they should be the same across all)
    first_dir = base_dir / f"layers_0_{layers_per_task-1}"
    summary_file = first_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
            # Metadata and pair_ids are not in summary, need to reconstruct
            # For now, just use what we have
            pass

    # Stack into arrays
    components_array = np.stack([all_components[i] for i in range(num_layers)])
    eigenvalues_array = np.stack([all_eigenvalues[i] for i in range(num_layers)])
    alphas_array = np.array([all_alphas[i] for i in range(num_layers)])

    print(f"\nCombined shapes:")
    print(f"  Components: {components_array.shape}")
    print(f"  Eigenvalues: {eigenvalues_array.shape}")
    print(f"  Alphas: {alphas_array.shape}")

    # Create combined results
    results = {
        'components': components_array,
        'eigenvalues': eigenvalues_array,
        'alphas': alphas_array,
        'config': config,
    }

    # Save combined results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **results)

    print(f"\nSaved combined results to: {output_file}")
    print(f"File size: {output_file.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Combine cPCA layer ranges")
    parser.add_argument(
        "base_dir",
        type=Path,
        help="Directory containing layers_X_Y subdirectories",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Output file path",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=62,
        help="Total number of layers (default: 62)",
    )

    args = parser.parse_args()

    combine_layer_ranges(
        args.base_dir,
        args.output_file,
        args.num_layers,
    )


if __name__ == "__main__":
    main()
