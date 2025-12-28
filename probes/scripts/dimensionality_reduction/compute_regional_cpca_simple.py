#!/usr/bin/env python3
"""Compute cPCA for regional activation files.

Regional files have structure:
  activations/{id}/emotional: [62, 5376]
  activations/{id}/neutral: [62, 5376]

This script runs cPCA on each regional file separately using the standard
load_activations_hdf5() function and run_cpca_all_layers().
"""

import argparse
import numpy as np
from pathlib import Path

from probes.core.data_loading import load_activations_hdf5
from probes.core.saving import save_cpca_results
from probes.methods.cpca import run_cpca_all_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--n_components', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=5.0)
    args = parser.parse_args()

    print(f"Loading activations from {args.input}...")
    activations, metadata, attrs = load_activations_hdf5(args.input)

    print(f"Loaded {len(activations)} activation pairs")
    print(f"Running cPCA with alpha={args.alpha}, n_components={args.n_components}...")

    # Run cPCA on all layers
    results = run_cpca_all_layers(
        activations=activations,
        metadata=metadata,
        alpha=args.alpha,
        n_components=args.n_components,
        use_diffs=True
    )

    print(f"cPCA complete: {results['config']['num_layers']} layers processed")

    # Save results
    args.output.mkdir(parents=True, exist_ok=True)
    output_file = args.output / f"{args.model.replace('/', '_')}_cpca.npz"

    # Stack components and eigenvalues
    num_layers = results['config']['num_layers']
    components_array = []
    eigenvalues_array = []
    alphas_array = []

    for layer_idx in range(num_layers):
        components_array.append(results['components'][layer_idx])
        eigenvalues_array.append(results['eigenvalues'][layer_idx])
        alphas_array.append(results['alpha_per_layer'][layer_idx])

    save_results = {
        'components': np.stack(components_array),
        'eigenvalues': np.stack(eigenvalues_array),
        'alphas': np.array(alphas_array),
        'config': results['config'],
        'metadata': results['metadata'],
        'pair_ids': results['pair_ids']
    }

    save_cpca_results(save_results, output_file)
    print(f"Saved to {output_file}")


if __name__ == '__main__':
    main()
