"""
Compute mean-diff steering vectors along each axis (valence, arousal, dominance, trust).

For each axis, computes: mean(high) - mean(low) across all combinations.
"""
import argparse
import h5py
import numpy as np
from pathlib import Path
from collections import defaultdict

# Axis order in the keys: valence, arousal, dominance, trust
AXES = ["valence", "arousal", "dominance", "trust"]
LEVELS = ["low", "neutral", "high"]


def parse_key(key: str) -> dict:
    """Parse activation key into components.

    Key format: neutral_{n}_combo_{c}_{valence}_{arousal}_{dominance}_{trust}
    """
    parts = key.split('_')
    # Last 4 parts are axis values
    return {
        "neutral_idx": int(parts[1]),
        "combo_idx": int(parts[3]),
        "valence": parts[4],
        "arousal": parts[5],
        "dominance": parts[6],
        "trust": parts[7],
    }


def compute_axis_vectors(
    h5_path: Path,
    output_dir: Path,
    layer_indices: list = None,
    layer_offset: int = 20,
):
    """Compute mean-diff vectors for each axis.

    Args:
        h5_path: Path to HDF5 file with activations
        output_dir: Directory to save vectors
        layer_indices: Specific h5 indices to save (default: all)
        layer_offset: First model layer in h5 file (default: 20)
                      h5 index 0 = model layer (layer_offset)
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        acts = f['activations']
        keys = list(acts.keys())

        print(f"Total activations: {len(keys)}")

        # Get shape info
        sample_act = acts[keys[0]][:]
        n_layers, hidden_dim = sample_act.shape
        print(f"Activation shape: {n_layers} h5 indices × {hidden_dim} dims")
        print(f"Layer offset: {layer_offset} (h5 idx 0 = model layer {layer_offset})")
        print(f"Model layers: {layer_offset} to {layer_offset + n_layers - 1}")

        if layer_indices is None:
            # Default: use all layers
            layer_indices = list(range(n_layers))

        # Group activations by axis level
        # For each axis, we want high vs low
        axis_activations = {
            axis: {level: [] for level in LEVELS}
            for axis in AXES
        }

        # Also track per-layer stats
        print("\nLoading activations...")
        for i, key in enumerate(keys):
            if i % 1000 == 0:
                print(f"  {i}/{len(keys)}")

            parsed = parse_key(key)
            act = acts[key][:]  # Shape: (n_layers, hidden_dim)

            for axis in AXES:
                level = parsed[axis]
                axis_activations[axis][level].append(act)

        # Compute mean-diff for each axis
        print("\nComputing mean-diff vectors...")
        for axis in AXES:
            high_acts = np.array(axis_activations[axis]["high"])  # (N, layers, dim)
            low_acts = np.array(axis_activations[axis]["low"])
            neutral_acts = np.array(axis_activations[axis]["neutral"])

            print(f"\n{axis}:")
            print(f"  High samples: {len(high_acts)}")
            print(f"  Low samples: {len(low_acts)}")
            print(f"  Neutral samples: {len(neutral_acts)}")

            # Mean across samples
            high_mean = high_acts.mean(axis=0)  # (layers, dim)
            low_mean = low_acts.mean(axis=0)
            neutral_mean = neutral_acts.mean(axis=0)

            # Mean-diff vector: high - low
            meandiff = high_mean - low_mean  # (layers, dim)

            # Also compute high - neutral and neutral - low for reference
            high_neutral_diff = high_mean - neutral_mean
            neutral_low_diff = neutral_mean - low_mean

            # Compute norms (print model layer numbers)
            for h5_idx in [0, 5, 10, 15, 20, 25, 30]:
                if h5_idx < n_layers:
                    model_layer = h5_idx + layer_offset
                    norm = np.linalg.norm(meandiff[h5_idx])
                    print(f"  Model layer {model_layer} (h5 idx {h5_idx}): norm {norm:.1f}")

            # Save per-layer vectors (use model layer numbers in filenames)
            for h5_idx in layer_indices:
                model_layer = h5_idx + layer_offset
                vec = meandiff[h5_idx]

                # Normalize to unit vector
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec_normalized = vec / norm
                else:
                    vec_normalized = vec

                # Save with model layer number in filename
                output_file = output_dir / f"{axis}_meandiff_layer{model_layer}.npz"
                np.savez(
                    output_file,
                    vector=vec_normalized.astype(np.float32),
                    vector_unnormalized=vec.astype(np.float32),
                    norm=norm,
                    model_layer=model_layer,
                    h5_index=h5_idx,
                    axis=axis,
                    n_high=len(high_acts),
                    n_low=len(low_acts),
                )

            # Also save combined file for all layers
            output_file = output_dir / f"{axis}_meandiff_all_layers.npz"
            np.savez(
                output_file,
                vectors=meandiff.astype(np.float32),  # (n_h5_indices, dim)
                high_mean=high_mean.astype(np.float32),
                low_mean=low_mean.astype(np.float32),
                neutral_mean=neutral_mean.astype(np.float32),
                axis=axis,
                n_high=len(high_acts),
                n_low=len(low_acts),
                n_neutral=len(neutral_acts),
                layer_offset=layer_offset,
                model_layers=np.arange(layer_offset, layer_offset + n_layers),
            )
            print(f"  Saved to {output_file}")

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Compute axis mean-diff vectors")
    parser.add_argument("--h5-path", type=Path,
                        default=Path("/workspace-vast/annas/git/research-tools/probes/data/axis_paraphrases/activations_trust_v1.h5"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/workspace-vast/annas/git/research-tools/experiments/steering/vectors/axis_meandiff"))
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific h5 indices to save (default: all)")
    parser.add_argument("--layer-offset", type=int, default=20,
                        help="First model layer in h5 file (default: 20)")
    args = parser.parse_args()

    compute_axis_vectors(
        h5_path=args.h5_path,
        output_dir=args.output_dir,
        layer_indices=args.layers,
        layer_offset=args.layer_offset,
    )


if __name__ == "__main__":
    main()
