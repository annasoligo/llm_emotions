#!/usr/bin/env python3
"""
Compute M and U direction vectors from Qwen 235B UA emotion disentanglement data.

Handles the multi-file format where layers are split across files:
- qwen3_235b_a22b_ua_layers_0_19_*.h5
- qwen3_235b_a22b_ua_layers_20_39_*.h5
- etc.

For each emotion pair:
- M direction: mean(activations where M=emotion) - mean(activations where M=opposite)
- U direction: mean(activations where U=emotion) - mean(activations where U=opposite)

Usage:
    python compute_qwen235b_directions.py --layers 50 60 70  # Specific layers
    python compute_qwen235b_directions.py --all-layers       # All 94 layers
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import glob


# Plutchik emotion pairs (opposites)
EMOTION_PAIRS = [
    ('joy', 'sadness'),
    ('trust', 'disgust'),
    ('fear', 'anger'),
    ('surprise', 'anticipation'),
    ('ecstasy', 'grief'),
    ('admiration', 'loathing'),
    ('terror', 'rage'),
    ('amazement', 'vigilance'),
    ('serenity', 'pensiveness'),
    ('acceptance', 'boredom'),
    ('apprehension', 'annoyance'),
    ('distraction', 'interest'),
    ('awe', 'contempt'),
    ('submission', 'aggressiveness'),
    ('love', 'remorse'),
    ('optimism', 'disappointment'),
]

# Create lookup for opposites
OPPOSITES = {}
for e1, e2 in EMOTION_PAIRS:
    OPPOSITES[e1] = e2
    OPPOSITES[e2] = e1


def find_layer_file(data_dir: Path, layer: int) -> Optional[Path]:
    """Find the HDF5 file containing a specific layer."""
    pattern = str(data_dir / "qwen3_235b_a22b_ua_layers_*.h5")
    files = glob.glob(pattern)

    for filepath in files:
        # Parse layer range from filename (e.g., layers_0_19)
        parts = Path(filepath).stem.split('_')
        for i, part in enumerate(parts):
            if part == 'layers' and i + 2 < len(parts):
                try:
                    start_layer = int(parts[i + 1])
                    end_layer = int(parts[i + 2].split('_')[0])  # Handle timestamp suffix
                    if start_layer <= layer <= end_layer:
                        return Path(filepath)
                except ValueError:
                    continue
    return None


def get_all_layer_files(data_dir: Path) -> Dict[Tuple[int, int], Path]:
    """Get all layer files with their layer ranges."""
    pattern = str(data_dir / "qwen3_235b_a22b_ua_layers_*.h5")
    files = glob.glob(pattern)

    layer_files = {}
    for filepath in files:
        parts = Path(filepath).stem.split('_')
        for i, part in enumerate(parts):
            if part == 'layers' and i + 2 < len(parts):
                try:
                    start_layer = int(parts[i + 1])
                    # Handle the end layer which might have timestamp attached
                    end_part = parts[i + 2]
                    if len(end_part) > 4:  # Has timestamp
                        end_layer = int(end_part[:2])
                    else:
                        end_layer = int(end_part)
                    layer_files[(start_layer, end_layer)] = Path(filepath)
                except ValueError:
                    continue

    return layer_files


def load_layer_data(
    data_dir: Path,
    layer: int,
    position: str = 'first_asst_token'
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load activations and metadata for a specific layer."""
    filepath = find_layer_file(data_dir, layer)
    if filepath is None:
        raise FileNotFoundError(f"No file found containing layer {layer}")

    with h5py.File(filepath, 'r') as f:
        layer_group = f[f'layer_{layer}']

        activations = layer_group[f'{position}_activations'][:]
        M_values = [m.decode() if isinstance(m, bytes) else m
                    for m in layer_group[f'{position}_M'][:]]
        U_values = [u.decode() if isinstance(u, bytes) else u
                    for u in layer_group[f'{position}_U'][:]]

    return activations, M_values, U_values


def compute_mean_diff_vectors(
    activations: np.ndarray,
    M_values: List[str],
    U_values: List[str],
    normalize: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
    """
    Compute mean-diff vectors for M and U emotions.

    Returns:
        M_vectors: {emotion: direction_vector}
        U_vectors: {emotion: direction_vector}
        M_norms: {emotion: pre-normalization L2 norm}
        U_norms: {emotion: pre-normalization L2 norm}
    """
    # Group activations by M and U values
    M_groups = defaultdict(list)
    U_groups = defaultdict(list)

    for i, (m, u) in enumerate(zip(M_values, U_values)):
        M_groups[m].append(activations[i])
        U_groups[u].append(activations[i])

    # Compute means
    M_means = {emotion: np.mean(acts, axis=0) for emotion, acts in M_groups.items()}
    U_means = {emotion: np.mean(acts, axis=0) for emotion, acts in U_groups.items()}

    # Compute direction vectors
    M_vectors = {}
    U_vectors = {}
    M_norms = {}
    U_norms = {}

    for emotion in M_means.keys():
        if emotion not in OPPOSITES:
            continue
        opposite = OPPOSITES[emotion]
        if opposite not in M_means:
            continue

        # M direction: emotion - opposite
        m_dir = M_means[emotion] - M_means[opposite]
        m_norm = float(np.linalg.norm(m_dir))
        M_norms[emotion] = m_norm
        if normalize and m_norm > 1e-8:
            m_dir = m_dir / m_norm
        M_vectors[emotion] = m_dir

        # U direction
        if emotion in U_means and opposite in U_means:
            u_dir = U_means[emotion] - U_means[opposite]
            u_norm = float(np.linalg.norm(u_dir))
            U_norms[emotion] = u_norm
            if normalize and u_norm > 1e-8:
                u_dir = u_dir / u_norm
            U_vectors[emotion] = u_dir

    return M_vectors, U_vectors, M_norms, U_norms


def compute_projection_std(
    activations: np.ndarray,
    direction: np.ndarray,
) -> float:
    """Compute standard deviation of projections onto direction."""
    projections = activations @ direction
    return float(np.std(projections))


def compute_layer_norm(activations: np.ndarray) -> float:
    """Compute mean L2 norm of activations at this layer."""
    norms = np.linalg.norm(activations, axis=1)
    return float(np.mean(norms))


def main():
    parser = argparse.ArgumentParser(description='Compute Qwen 235B UA direction vectors')
    parser.add_argument('--data-dir', type=Path,
                        default=Path('probes/ua_emotion_disentangle/data'),
                        help='Directory containing HDF5 files')
    parser.add_argument('--layers', type=int, nargs='+',
                        help='Specific layers to process')
    parser.add_argument('--all-layers', action='store_true',
                        help='Process all 94 layers')
    parser.add_argument('--position', type=str, default='first_asst_token',
                        choices=['last_user_token', 'first_asst_token', 'final_token'],
                        help='Token position to use')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('probes/ua_emotion_disentangle/vectors'),
                        help='Output directory for vectors')
    parser.add_argument('--no-normalize', action='store_true',
                        help='Skip L2 normalization')
    args = parser.parse_args()

    # Determine layers to process
    if args.all_layers:
        layers = list(range(94))
    elif args.layers:
        layers = args.layers
    else:
        # Default: middle and late layers commonly used for steering
        layers = [30, 40, 50, 60, 70, 80]

    print("=" * 70)
    print("QWEN 235B UA DIRECTION VECTOR COMPUTATION")
    print("=" * 70)
    print(f"Data dir: {args.data_dir}")
    print(f"Layers: {layers}")
    print(f"Position: {args.position}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 70)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Track layer norms for config
    layer_norms = {}

    # Process each layer
    all_results = {}
    for layer in layers:
        print(f"\n--- Layer {layer} ---")

        try:
            # Load data
            activations, M_values, U_values = load_layer_data(
                args.data_dir, layer, args.position
            )
            print(f"  Loaded {len(activations)} samples, shape {activations.shape}")

            # Compute layer norm
            layer_norm = compute_layer_norm(activations)
            layer_norms[layer] = layer_norm
            print(f"  Layer norm (mean activation L2): {layer_norm:.2f}")

            # Compute vectors
            M_vectors, U_vectors, M_norms, U_norms = compute_mean_diff_vectors(
                activations, M_values, U_values,
                normalize=not args.no_normalize
            )

            # Compute projection STDs
            M_stds = {}
            U_stds = {}
            for emotion, vec in M_vectors.items():
                M_stds[emotion] = compute_projection_std(activations, vec)
            for emotion, vec in U_vectors.items():
                U_stds[emotion] = compute_projection_std(activations, vec)

            # Store results
            all_results[layer] = {
                'M_vectors': M_vectors,
                'U_vectors': U_vectors,
                'M_norms': M_norms,
                'U_norms': U_norms,
                'M_stds': M_stds,
                'U_stds': U_stds,
            }

            # Print summary
            print(f"  M emotions: {len(M_vectors)}")
            print(f"  U emotions: {len(U_vectors)}")

            # Show a few example norms
            sample_emotions = ['joy', 'anger', 'fear', 'surprise']
            for emo in sample_emotions:
                if emo in M_norms:
                    print(f"    {emo}: M_norm={M_norms[emo]:.1f}, M_std={M_stds[emo]:.1f}, "
                          f"U_norm={U_norms.get(emo, 0):.1f}, U_std={U_stds.get(emo, 0):.1f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save vectors
    print("\n" + "=" * 70)
    print("SAVING VECTORS")
    print("=" * 70)

    for layer, results in all_results.items():
        # Save combined file for this layer
        output_path = args.output_dir / f"qwen235b_ua_directions_layer{layer}.npz"

        save_dict = {
            'layer': layer,
            'model': 'Qwen/Qwen3-235B-A22B',
            'position': args.position,
            'normalized': not args.no_normalize,
            'layer_norm': layer_norms[layer],
        }

        # Add M vectors
        for emotion, vec in results['M_vectors'].items():
            save_dict[f'M_{emotion}'] = vec.astype(np.float32)
            save_dict[f'M_{emotion}_norm'] = results['M_norms'][emotion]
            save_dict[f'M_{emotion}_std'] = results['M_stds'][emotion]

        # Add U vectors
        for emotion, vec in results['U_vectors'].items():
            save_dict[f'U_{emotion}'] = vec.astype(np.float32)
            save_dict[f'U_{emotion}_norm'] = results['U_norms'][emotion]
            save_dict[f'U_{emotion}_std'] = results['U_stds'][emotion]

        np.savez(output_path, **save_dict)
        print(f"  Saved: {output_path.name}")

    # Print layer norms for config.py
    print("\n" + "=" * 70)
    print("LAYER NORMS (for config.py)")
    print("=" * 70)
    print("\"Qwen/Qwen3-235B-A22B\": {")
    for layer in sorted(layer_norms.keys()):
        print(f"    {layer}: {layer_norms[layer]:.2f},")
    print("},")

    # Print direction STDs for config.py
    print("\n" + "=" * 70)
    print("DIRECTION STDS (for config.py, layer 50)")
    print("=" * 70)
    if 50 in all_results:
        results = all_results[50]
        for emotion in sorted(results['M_stds'].keys()):
            print(f"    \"{emotion}_ua_model_qwen235b\": {results['M_stds'][emotion]:.2f},")
        print()
        for emotion in sorted(results['U_stds'].keys()):
            print(f"    \"{emotion}_ua_user_qwen235b\": {results['U_stds'][emotion]:.2f},")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
