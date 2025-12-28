#!/usr/bin/env python3
"""Run regional cPCA on pre-extracted regional activations.

This script loads activations that have already been pooled by region
(user, asst, special1, special2) and runs cPCA INDEPENDENTLY on each region
across all layers.

Expected HDF5 structure:
    /activations/{pair_id}/neutral/regional/{region_name} -> [num_layers, hidden_dim]
    /activations/{pair_id}/emotional/regional/{region_name} -> [num_layers, hidden_dim]

Usage:
    python scripts/run_regional_cpca.py \\
        --input activations/regional.h5 \\
        --output results/cpca_regional \\
        --model gemma-2-9b-it \\
        --regions user asst special1 special2 \\
        --n_components 50 \\
        --alpha 5.0

    # Auto-tune alpha per region:
    python scripts/run_regional_cpca.py \\
        --input activations/regional.h5 \\
        --output results/cpca_regional \\
        --model gemma-2-9b-it \\
        --regions user asst \\
        --n_components 50 \\
        --tune_alpha \\
        --alpha_range 0.1 1000 \\
        --n_alphas 40
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm


from probes.methods.cpca import run_cpca, tune_alpha_silhouette


def load_regional_activations_hdf5(
    h5_path: Path,
    regions: List[str],
) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], List[Dict], Dict]:
    """Load regional activations from HDF5 file.

    Args:
        h5_path: Path to HDF5 file
        regions: List of region names to load

    Returns:
        regional_acts: Dict mapping pair_id -> {
            "neutral": {region: [num_layers, hidden_dim]},
            "emotional": {region: [num_layers, hidden_dim]}
        }
        metadata: List of metadata dicts
        attrs: File attributes dict

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If required regions missing
        ValueError: If data format invalid
    """
    if not h5_path.exists():
        raise FileNotFoundError(f"Activation file not found: {h5_path}")

    if not regions:
        raise ValueError("regions list is empty")

    regional_acts = {}

    with h5py.File(h5_path, "r") as f:
        # Load attributes
        attrs = dict(f.attrs)

        # Load metadata
        if "metadata" not in f:
            raise KeyError(f"Missing 'metadata' key in {h5_path}")

        try:
            metadata = json.loads(f["metadata"][()])
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid metadata JSON in {h5_path}: {e}")

        # Load activations
        if "activations" not in f:
            raise KeyError(f"Missing 'activations' key in {h5_path}")

        for pair_id in f["activations"].keys():
            pair_group = f[f"activations/{pair_id}"]

            if "neutral" not in pair_group:
                raise KeyError(f"Missing 'neutral' for pair {pair_id}")
            if "emotional" not in pair_group:
                raise KeyError(f"Missing 'emotional' for pair {pair_id}")

            # Load regional activations
            regional_acts[pair_id] = {
                "neutral": {},
                "emotional": {},
            }

            for condition in ["neutral", "emotional"]:
                condition_group = pair_group[condition]

                # Check for regional structure
                if "regional" not in condition_group:
                    raise KeyError(
                        f"Missing 'regional' group for pair {pair_id}/{condition}. "
                        f"This script expects pre-extracted regional activations."
                    )

                regional_group = condition_group["regional"]

                # Load each region
                for region in regions:
                    if region not in regional_group:
                        raise KeyError(
                            f"Missing region '{region}' for pair {pair_id}/{condition}. "
                            f"Available regions: {list(regional_group.keys())}"
                        )

                    acts = regional_group[region][:]

                    # Validate shape: [num_layers, hidden_dim]
                    if acts.ndim != 2:
                        raise ValueError(
                            f"Region activations must be 2D [num_layers, hidden_dim], "
                            f"got shape {acts.shape} for {pair_id}/{condition}/{region}"
                        )

                    # Check for NaN/Inf
                    if not np.all(np.isfinite(acts)):
                        raise ValueError(
                            f"Pair {pair_id}/{condition}/{region} has NaN/Inf"
                        )

                    regional_acts[pair_id][condition][region] = acts

    if not regional_acts:
        raise ValueError(f"No activations loaded from {h5_path}")

    return regional_acts, metadata, attrs


def filter_by_metadata(
    regional_acts: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    metadata: List[Dict],
    filter_criteria: Optional[Dict[str, any]] = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], List[Dict], List[str]]:
    """Filter pairs by metadata criteria.

    Args:
        regional_acts: Regional activations dict
        metadata: List of metadata dicts
        filter_criteria: Dict with keys like 'emotion', 'tier', etc.

    Returns:
        Filtered regional_acts, metadata, and list of filtered pair_ids

    Raises:
        ValueError: If filter results in empty dataset
    """
    if filter_criteria is None or not filter_criteria:
        pair_ids = list(regional_acts.keys())
        return regional_acts, metadata, pair_ids

    # Build pair_id to metadata mapping
    id_to_meta = {m["id"]: m for m in metadata}

    # Filter pair_ids
    filtered_ids = []
    for pair_id in regional_acts.keys():
        if pair_id not in id_to_meta:
            continue

        meta = id_to_meta[pair_id]
        match = True

        for key, value in filter_criteria.items():
            if key not in meta:
                match = False
                break

            if isinstance(value, (list, tuple)):
                if meta[key] not in value:
                    match = False
                    break
            else:
                if meta[key] != value:
                    match = False
                    break

        if match:
            filtered_ids.append(pair_id)

    if not filtered_ids:
        raise ValueError(
            f"No pairs match filter criteria {filter_criteria}. "
            f"Available metadata keys: {list(id_to_meta[list(id_to_meta.keys())[0]].keys())}"
        )

    # Filter data
    filtered_acts = {pid: regional_acts[pid] for pid in filtered_ids}
    filtered_meta = [m for m in metadata if m["id"] in filtered_ids]

    return filtered_acts, filtered_meta, filtered_ids


def run_regional_cpca_all_layers(
    regional_acts: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    regions: List[str],
    metadata: List[Dict],
    alpha_per_region: Optional[Dict[str, float]] = None,
    n_components: int = 50,
    alpha_range: Tuple[float, float] = (0.1, 1000),
    n_alphas: int = 40,
    use_diffs: bool = True,
    tune_alpha: bool = False,
) -> Dict:
    """Run cPCA independently on each region across all layers.

    Args:
        regional_acts: Dict mapping pair_id -> condition -> region -> activations
        regions: List of region names
        metadata: List of metadata dicts
        alpha_per_region: Fixed alpha per region (None = use 5.0 for all)
        n_components: Number of cPCs per region
        alpha_range: Range for alpha search (if tuning)
        n_alphas: Number of alpha values to try (if tuning)
        use_diffs: If True, use diffs as target
        tune_alpha: If True, auto-tune alpha per region per layer

    Returns:
        Results dict with per-region, per-layer components

    Raises:
        ValueError: If data format invalid
    """
    if not regional_acts:
        raise ValueError("regional_acts dict is empty")

    if not regions:
        raise ValueError("regions list is empty")

    if not metadata:
        raise ValueError("metadata list is empty")

    # Get pair IDs
    pair_ids = list(regional_acts.keys())

    # Validate all pairs have all regions
    first_pair = next(iter(regional_acts.values()))
    for condition in ["neutral", "emotional"]:
        for region in regions:
            if region not in first_pair[condition]:
                raise KeyError(f"Region '{region}' not found in first pair/{condition}")

    # Get shape info from first region
    first_region_acts = first_pair["neutral"][regions[0]]
    num_layers = first_region_acts.shape[0]
    hidden_dim = first_region_acts.shape[1]

    # Default alpha per region
    if alpha_per_region is None:
        alpha_per_region = {r: 5.0 for r in regions}

    # Validate alpha_per_region
    for region in regions:
        if region not in alpha_per_region:
            raise KeyError(f"alpha_per_region missing region '{region}'")
        if alpha_per_region[region] < 0:
            raise ValueError(
                f"alpha for region '{region}' must be non-negative, "
                f"got {alpha_per_region[region]}"
            )

    # Get emotion labels for tuning
    id_to_meta = {m["id"]: m for m in metadata}
    emotion_labels = []
    for pid in pair_ids:
        if pid not in id_to_meta:
            raise KeyError(f"Pair {pid} not found in metadata")
        meta = id_to_meta[pid]
        # Support both text format (emotion) and conversation format (user_emotion)
        if "emotion" in meta:
            emotion_labels.append(meta["emotion"])
        elif "user_emotion" in meta:
            emotion_labels.append(meta["user_emotion"])
        else:
            raise KeyError(f"Metadata for {pid} missing 'emotion' or 'user_emotion' field")
    emotion_labels = np.array(emotion_labels)

    # Alpha values for tuning
    alphas = np.geomspace(alpha_range[0], alpha_range[1], n_alphas) if tune_alpha else None

    results = {
        "components_per_region": {},  # region -> {layer -> components}
        "eigenvalues_per_region": {},  # region -> {layer -> eigenvalues}
        "alpha_per_region_per_layer": {},  # region -> {layer -> alpha}
        "tuning_per_region": {},  # region -> {layer -> tuning_info}
        "pair_ids": pair_ids,
        "metadata": metadata,
        "config": {
            "regions": regions,
            "n_components": n_components,
            "alpha_per_region": alpha_per_region if not tune_alpha else None,
            "alpha_range": alpha_range if tune_alpha else None,
            "n_alphas": n_alphas if tune_alpha else None,
            "num_pairs": len(pair_ids),
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "use_diffs": use_diffs,
            "tune_alpha": tune_alpha,
        },
    }

    print(f"Running regional cPCA on {num_layers} layers, {len(regions)} regions...")
    print(f"  Pairs: {len(pair_ids)}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Components per region: {n_components}")
    print(f"  Auto-tune alpha: {tune_alpha}")
    print()

    # Process each region separately (INDEPENDENT cPCA)
    for region in regions:
        print(f"Region: {region}")
        if not tune_alpha:
            print(f"  Fixed alpha: {alpha_per_region[region]}")
        else:
            print(f"  Tuning alpha range: {alpha_range}")

        results["components_per_region"][region] = {}
        results["eigenvalues_per_region"][region] = {}
        results["alpha_per_region_per_layer"][region] = {}
        if tune_alpha:
            results["tuning_per_region"][region] = {}

        # Process each layer
        for layer_idx in tqdm(range(num_layers), desc=f"  {region}"):
            # Extract activations for this layer and region
            emotional_acts = []
            neutral_acts = []

            for pid in pair_ids:
                emotional_acts.append(
                    regional_acts[pid]["emotional"][region][layer_idx]
                )
                neutral_acts.append(
                    regional_acts[pid]["neutral"][region][layer_idx]
                )

            emotional_acts = np.stack(emotional_acts).astype(np.float32)
            neutral_acts = np.stack(neutral_acts).astype(np.float32)

            # Compute diffs for scoring
            diffs = emotional_acts - neutral_acts

            # Choose target data
            if use_diffs:
                target_acts = diffs
            else:
                target_acts = emotional_acts

            # Tune alpha if requested
            if tune_alpha:
                layer_alpha, tuning_info = tune_alpha_silhouette(
                    target_acts,
                    neutral_acts,
                    diffs,
                    emotion_labels,
                    alphas,
                    n_components,
                )
                results["tuning_per_region"][region][layer_idx] = tuning_info
            else:
                layer_alpha = alpha_per_region[region]

            results["alpha_per_region_per_layer"][region][layer_idx] = float(layer_alpha)

            # Run cPCA for this region and layer
            components, eigenvalues = run_cpca(
                target_acts,
                neutral_acts,
                layer_alpha,
                n_components,
            )

            results["components_per_region"][region][layer_idx] = components
            results["eigenvalues_per_region"][region][layer_idx] = eigenvalues

        print()

    return results


def save_regional_cpca_results(
    results: Dict,
    output_dir: Path,
    model_name: str,
) -> List[Path]:
    """Save regional cPCA results (one file per region).

    Args:
        results: Results dict from run_regional_cpca_all_layers
        output_dir: Output directory
        model_name: Model identifier (for filename)

    Returns:
        List of saved file paths

    Raises:
        OSError: If cannot write files
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    regions = results["config"]["regions"]

    for region in regions:
        # Prepare data for this region
        region_data = {
            "components": results["components_per_region"][region],
            "eigenvalues": results["eigenvalues_per_region"][region],
            "alpha_per_layer": results["alpha_per_region_per_layer"][region],
            "pair_ids": results["pair_ids"],
            "metadata": results["metadata"],
            "config": {
                **results["config"],
                "region": region,
            },
        }

        # Add tuning info if available
        if "tuning_per_region" in results and region in results["tuning_per_region"]:
            region_data["tuning"] = results["tuning_per_region"][region]

        # Convert to numpy arrays for NPZ
        components_dict = region_data["components"]
        eigenvalues_dict = region_data["eigenvalues"]

        # Stack into arrays: [num_layers, n_components, hidden_dim] for components
        num_layers = len(components_dict)
        n_components = components_dict[0].shape[0]
        hidden_dim = components_dict[0].shape[1]

        components_array = np.zeros((num_layers, n_components, hidden_dim), dtype=np.float32)
        eigenvalues_array = np.zeros((num_layers, n_components), dtype=np.float32)
        alphas_array = np.zeros(num_layers, dtype=np.float32)

        for layer_idx in range(num_layers):
            components_array[layer_idx] = components_dict[layer_idx]
            eigenvalues_array[layer_idx] = eigenvalues_dict[layer_idx]
            alphas_array[layer_idx] = region_data["alpha_per_layer"][layer_idx]

        # Build save dict
        save_dict = {
            "components": components_array,
            "eigenvalues": eigenvalues_array,
            "alphas": alphas_array,
            "config": json.dumps(region_data["config"]),
            "metadata": json.dumps(region_data["metadata"]),
            "pair_ids": np.array(region_data["pair_ids"]),
        }

        # Add tuning info if available
        if "tuning" in region_data:
            # Convert tuning dict to JSON
            save_dict["tuning"] = json.dumps(region_data["tuning"])

        # Save
        output_path = output_dir / f"{model_name}_{region}_cpca.npz"
        try:
            np.savez(output_path, **save_dict)
            saved_paths.append(output_path)
        except Exception as e:
            raise OSError(f"Failed to save {output_path}: {e}")

    return saved_paths


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict]:
    """Load checkpoint if exists.

    Args:
        checkpoint_path: Path to checkpoint NPZ file

    Returns:
        Checkpoint dict or None if doesn't exist
    """
    if not checkpoint_path.exists():
        return None

    try:
        data = np.load(checkpoint_path, allow_pickle=True)
        config = json.loads(str(data["config"]))
        metadata = json.loads(str(data["metadata"]))

        return {
            "components_per_region": data["components_per_region"].item(),
            "eigenvalues_per_region": data["eigenvalues_per_region"].item(),
            "alpha_per_region_per_layer": data["alpha_per_region_per_layer"].item(),
            "pair_ids": data["pair_ids"].tolist(),
            "metadata": metadata,
            "config": config,
            "last_completed_layer": int(data["last_completed_layer"]),
        }
    except Exception as e:
        print(f"Warning: Failed to load checkpoint: {e}")
        return None


def save_checkpoint(
    results: Dict,
    checkpoint_path: Path,
    last_completed_layer: int,
) -> None:
    """Save checkpoint.

    Args:
        results: Partial results dict
        checkpoint_path: Path to save checkpoint
        last_completed_layer: Last completed layer index
    """
    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "components_per_region": results["components_per_region"],
            "eigenvalues_per_region": results["eigenvalues_per_region"],
            "alpha_per_region_per_layer": results["alpha_per_region_per_layer"],
            "pair_ids": np.array(results["pair_ids"]),
            "config": json.dumps(results["config"]),
            "metadata": json.dumps(results["metadata"]),
            "last_completed_layer": last_completed_layer,
        }

        np.savez(checkpoint_path, **save_dict)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run regional cPCA on pre-extracted regional activations"
    )

    # Required arguments
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input HDF5 file with regional activations",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (used in output filenames)",
    )

    # Regional configuration
    parser.add_argument(
        "--regions",
        nargs="+",
        default=["user", "asst", "special1", "special2"],
        help="Regions to process (default: user asst special1 special2)",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=50,
        help="Number of components per region (default: 50)",
    )

    # Alpha configuration
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Fixed alpha for all regions (if not tuning)",
    )
    parser.add_argument(
        "--alpha_user",
        type=float,
        default=None,
        help="Fixed alpha for user region (overrides --alpha)",
    )
    parser.add_argument(
        "--alpha_asst",
        type=float,
        default=None,
        help="Fixed alpha for asst region (overrides --alpha)",
    )
    parser.add_argument(
        "--alpha_special",
        type=float,
        default=None,
        help="Fixed alpha for special regions (overrides --alpha)",
    )
    parser.add_argument(
        "--tune_alpha",
        action="store_true",
        help="Auto-tune alpha per region per layer using silhouette score",
    )
    parser.add_argument(
        "--alpha_range",
        nargs=2,
        type=float,
        default=[0.1, 1000],
        help="Alpha range for tuning (default: 0.1 1000)",
    )
    parser.add_argument(
        "--n_alphas",
        type=int,
        default=40,
        help="Number of alpha values to try when tuning (default: 40)",
    )

    # Other options
    parser.add_argument(
        "--use_diffs",
        action="store_true",
        default=True,
        help="Use diffs as target (default: True)",
    )
    parser.add_argument(
        "--no_use_diffs",
        dest="use_diffs",
        action="store_false",
        help="Use raw emotional as target",
    )

    # Filtering
    parser.add_argument(
        "--filter_emotion",
        type=str,
        default=None,
        help="Filter to specific emotion",
    )
    parser.add_argument(
        "--filter_tier",
        type=str,
        default=None,
        help="Filter to specific tier",
    )

    # Checkpoint/resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if exists",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save checkpoint every N layers (0 = no checkpoints)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("REGIONAL CONTRASTIVE PCA")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"Regions: {args.regions}")
    print(f"Components per region: {args.n_components}")
    print()

    # Validate arguments
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if args.tune_alpha and args.alpha is not None:
        print("Warning: Both --tune_alpha and --alpha specified. Using auto-tuning.")

    # Load activations
    print("Loading regional activations...")
    try:
        regional_acts, metadata, attrs = load_regional_activations_hdf5(
            args.input, args.regions
        )
    except Exception as e:
        print(f"Error loading activations: {e}")
        sys.exit(1)

    print(f"  Loaded {len(regional_acts)} pairs")
    print(f"  Model: {attrs.get('model_name', 'unknown')}")
    print()

    # Filter by metadata if requested
    filter_criteria = {}
    if args.filter_emotion:
        filter_criteria["emotion"] = args.filter_emotion
    if args.filter_tier:
        filter_criteria["tier"] = args.filter_tier

    if filter_criteria:
        print(f"Filtering by: {filter_criteria}")
        try:
            regional_acts, metadata, pair_ids = filter_by_metadata(
                regional_acts, metadata, filter_criteria
            )
            print(f"  Retained {len(pair_ids)} pairs after filtering")
            print()
        except Exception as e:
            print(f"Error filtering: {e}")
            sys.exit(1)

    # Build alpha_per_region
    alpha_per_region = None
    if not args.tune_alpha:
        default_alpha = args.alpha if args.alpha is not None else 5.0
        alpha_per_region = {r: default_alpha for r in args.regions}

        # Override with region-specific alphas
        if args.alpha_user is not None and "user" in args.regions:
            alpha_per_region["user"] = args.alpha_user
        if args.alpha_asst is not None and "asst" in args.regions:
            alpha_per_region["asst"] = args.alpha_asst
        if args.alpha_special is not None:
            for region in args.regions:
                if region.startswith("special"):
                    alpha_per_region[region] = args.alpha_special

        print("Alpha configuration:")
        for region in args.regions:
            print(f"  {region}: {alpha_per_region[region]}")
        print()

    # Run regional cPCA
    print("Running regional cPCA...")
    try:
        results = run_regional_cpca_all_layers(
            regional_acts=regional_acts,
            regions=args.regions,
            metadata=metadata,
            alpha_per_region=alpha_per_region,
            n_components=args.n_components,
            alpha_range=tuple(args.alpha_range),
            n_alphas=args.n_alphas,
            use_diffs=args.use_diffs,
            tune_alpha=args.tune_alpha,
        )
    except Exception as e:
        print(f"Error running cPCA: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save results
    print("\nSaving results...")
    try:
        saved_paths = save_regional_cpca_results(
            results, args.output, args.model
        )
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)

    print(f"\nSaved {len(saved_paths)} result files:")
    for path in saved_paths:
        print(f"  {path}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Processed {len(results['pair_ids'])} pairs")
    print(f"Regions: {len(args.regions)}")
    print(f"Layers: {results['config']['num_layers']}")
    print(f"Components per region: {args.n_components}")
    print()

    # Print per-region stats
    for region in args.regions:
        print(f"{region}:")
        if args.tune_alpha:
            alphas = [
                results["alpha_per_region_per_layer"][region][i]
                for i in range(results["config"]["num_layers"])
            ]
            print(f"  Alpha range: [{min(alphas):.2f}, {max(alphas):.2f}]")
            print(f"  Alpha mean: {np.mean(alphas):.2f}")
        else:
            print(f"  Alpha: {alpha_per_region[region]}")

        # Show top eigenvalues for first/middle/last layers
        num_layers = results["config"]["num_layers"]
        for layer_idx in [0, num_layers // 2, num_layers - 1]:
            evals = results["eigenvalues_per_region"][region][layer_idx][:5]
            print(f"  Layer {layer_idx} top-5 eigenvalues: {evals}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
