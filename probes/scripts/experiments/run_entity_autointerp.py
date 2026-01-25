#!/usr/bin/env python3
"""Run auto-interpretation on entity-specific cPCA results (controlled variation).

Combines per-layer cPCA files into a single structure and runs auto-interpretation
to discover entity-specific affective dimensions.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


def combine_layer_cpca_files(layer_dir: Path, output_path: Path, layers: list[int]):
    """Combine individual layer cPCA files into single NPZ.

    Creates full-size array (62 layers) with only requested layers filled in.
    This allows autointerp script to access layers by their original indices.
    """
    print(f"Combining {len(layers)} layer files from {layer_dir}")

    # Load first layer to get dimensions
    first_layer_file = layer_dir / f"layer_{layers[0]}.npz"
    if not first_layer_file.exists():
        raise FileNotFoundError(f"First layer file not found: {first_layer_file}")

    first_data = np.load(first_layer_file)
    n_components, hidden_dim = first_data['components'].shape

    # Create full arrays (assume 62 layers total for Gemma-27B)
    max_layer = max(layers) + 1
    components = np.zeros((max_layer, n_components, hidden_dim), dtype=np.float32)
    eigenvalues = np.zeros((max_layer, n_components), dtype=np.float32)
    alphas = np.zeros(max_layer, dtype=np.float32)

    # Fill in the requested layers
    for layer_idx in sorted(layers):
        layer_file = layer_dir / f"layer_{layer_idx}.npz"

        if not layer_file.exists():
            print(f"Warning: {layer_file} not found, skipping")
            continue

        data = np.load(layer_file)
        components[layer_idx] = data['components']
        eigenvalues[layer_idx] = data['eigenvalues']
        alphas[layer_idx] = data['alpha']

    print(f"Combined shape: {components.shape}")
    print(f"Filled layers: {layers}")
    print(f"Saving to: {output_path}")

    # Create config dict matching expected format
    config = {
        'n_components': n_components,
        'num_layers': max_layer,
        'hidden_dim': hidden_dim,
        'layers': layers,
    }

    np.savez(
        output_path,
        components=components,
        eigenvalues=eigenvalues,
        alphas=alphas,
        config=config,
        pair_ids=np.array([]),
        metadata=[],
    )

    return output_path


def run_autointerp(
    entity: str,  # 'user' or 'assistant'
    layer_dir: Path,
    activations_path: Path,
    texts_path: Path,
    output_path: Path,
    layers: list[int],
    pcs: list[int],
    combined_cpca_dir: Path,
    dual_mode: bool = True,
):
    """Run auto-interpretation for entity-specific cPCA."""
    print("=" * 80)
    print(f"AUTO-INTERPRETATION: {entity.upper()} ISOLATION")
    print("=" * 80)

    # Use entity-specific text file with proper ID mapping
    entity_texts_path = texts_path.parent / f"{entity}_isolation_for_autointerp.jsonl"
    if not entity_texts_path.exists():
        print(f"Creating text mapping for {entity}...")
        subprocess.run([
            'python',
            'probes/scripts/experiments/create_autointerp_text_mapping.py',
            '--h5', str(activations_path),
            '--jsonl', str(texts_path),
            '--output', str(entity_texts_path),
        ], check=True)

    # Combine layer files
    combined_cpca_dir.mkdir(parents=True, exist_ok=True)
    combined_cpca_path = combined_cpca_dir / f"{entity}_isolation_combined.npz"

    combine_layer_cpca_files(layer_dir, combined_cpca_path, layers)

    # Run auto-interpretation
    cmd = [
        'python',
        'probes/scripts/evaluation/autointerp_pcs.py',
        '--source', f'region:{entity}_isolation',
        '--cpca', str(combined_cpca_path),
        '--activations', str(activations_path),
        '--texts', str(entity_texts_path),  # Use entity-specific text file
        '--output', str(output_path),
        '--layers', *[str(l) for l in layers],
        '--pcs', *[str(p) for p in pcs],
        '--n-samples', '5',
        '--use-diffs',
        '--save-every', '5',
    ]

    if dual_mode:
        cmd.append('--dual-mode')

    print(f"\nRunning command:")
    print(' '.join(cmd))
    print()

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\nError: Auto-interpretation failed with code {result.returncode}")
        return False

    print(f"\nSuccess! Results saved to: {output_path}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run auto-interpretation on entity-specific cPCA results"
    )
    parser.add_argument(
        '--entity',
        type=str,
        choices=['user', 'assistant', 'both'],
        default='both',
        help='Which entity to run (user, assistant, or both)'
    )
    parser.add_argument(
        '--cpca-base-dir',
        type=Path,
        default=Path('outputs/dimensionality_reduction/cpca/controlled_variation'),
        help='Base directory for cPCA layer files'
    )
    parser.add_argument(
        '--activations-dir',
        type=Path,
        default=Path('outputs/activations/controlled_variation'),
        help='Directory containing activation H5 files'
    )
    parser.add_argument(
        '--texts',
        type=Path,
        default=Path('outputs/data/controlled_variation/controlled_variation_all.jsonl'),
        help='Text/metadata JSONL file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/interpretations/autointerp/controlled_variation'),
        help='Output directory for interpretation results'
    )
    parser.add_argument(
        '--combined-cpca-dir',
        type=Path,
        default=Path('outputs/dimensionality_reduction/cpca/controlled_variation/combined'),
        help='Directory to save combined cPCA files'
    )
    parser.add_argument(
        '--layers',
        type=int,
        nargs='+',
        default=[20, 30, 40],
        help='Layers to interpret'
    )
    parser.add_argument(
        '--pcs',
        type=int,
        nargs='+',
        default=list(range(10)),
        help='PC indices to interpret (default: 0-9)'
    )
    parser.add_argument(
        '--no-dual-mode',
        action='store_true',
        help='Disable dual interpretation mode (standard + comparison)'
    )

    args = parser.parse_args()

    entities = ['user', 'assistant'] if args.entity == 'both' else [args.entity]

    for entity in entities:
        layer_dir = args.cpca_base_dir / f"{entity}_isolation" / "layers_0_61"
        activations_path = args.activations_dir / f"{entity}_isolation.h5"
        output_path = args.output_dir / f"{entity}_isolation_layers_{min(args.layers)}-{max(args.layers)}.json"

        if not layer_dir.exists():
            print(f"ERROR: Layer directory not found: {layer_dir}")
            sys.exit(1)

        if not activations_path.exists():
            print(f"ERROR: Activations file not found: {activations_path}")
            sys.exit(1)

        if not args.texts.exists():
            print(f"ERROR: Text file not found: {args.texts}")
            sys.exit(1)

        # Run interpretation
        success = run_autointerp(
            entity=entity,
            layer_dir=layer_dir,
            activations_path=activations_path,
            texts_path=args.texts,
            output_path=output_path,
            layers=args.layers,
            pcs=args.pcs,
            combined_cpca_dir=args.combined_cpca_dir,
            dual_mode=not args.no_dual_mode,
        )

        if not success:
            sys.exit(1)

    print("\n" + "=" * 80)
    print("ALL ENTITY INTERPRETATIONS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
