"""Saving utilities - no silent failures."""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import h5py


def save_cpca_results(
    results: dict,
    output_path: Path,
    *,
    create_parent: bool = True,
) -> None:
    """Save cPCA results to NPZ file.

    Args:
        results: Dict with keys: components, eigenvalues, alphas, config, etc.
        output_path: Path to save NPZ file
        create_parent: If True, create parent directories

    Raises:
        KeyError: If required keys missing
        ValueError: If data format invalid
        OSError: If cannot write file
    """
    required_keys = ["components", "eigenvalues", "alphas", "config"]
    missing = [k for k in required_keys if k not in results]
    if missing:
        raise KeyError(f"Missing required keys in results: {missing}")

    # Validate shapes
    components = results["components"]
    eigenvalues = results["eigenvalues"]
    alphas = results["alphas"]

    if not isinstance(components, np.ndarray):
        raise ValueError(f"components must be ndarray, got {type(components)}")
    if not isinstance(eigenvalues, np.ndarray):
        raise ValueError(f"eigenvalues must be ndarray, got {type(eigenvalues)}")
    if not isinstance(alphas, np.ndarray):
        raise ValueError(f"alphas must be ndarray, got {type(alphas)}")

    if components.ndim != 3:
        raise ValueError(
            f"components must be 3D [layers, n_components, hidden_dim], "
            f"got shape {components.shape}"
        )

    # Create parent dir if needed
    if create_parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for saving
    save_data = {
        "components": components,
        "eigenvalues": eigenvalues,
        "alphas": alphas,
        "config": json.dumps(results["config"]),
    }

    # Optional keys
    if "metadata" in results and results["metadata"] is not None:
        save_data["metadata"] = json.dumps(results["metadata"])
    if "pair_ids" in results:
        save_data["pair_ids"] = np.array(results["pair_ids"])

    # Save
    try:
        np.savez(output_path, **save_data)
    except Exception as e:
        raise OSError(f"Failed to save to {output_path}: {e}")


def save_probe_results(
    results: dict,
    output_path: Path,
    *,
    create_parent: bool = True,
) -> None:
    """Save probe training results to pickle file.

    Args:
        results: Dict with probe training results
        output_path: Path to save pickle file
        create_parent: If True, create parent directories

    Raises:
        ValueError: If results format invalid
        OSError: If cannot write file
    """
    import pickle

    if not isinstance(results, dict):
        raise ValueError(f"results must be dict, got {type(results)}")

    # Create parent dir if needed
    if create_parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    try:
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
    except Exception as e:
        raise OSError(f"Failed to save to {output_path}: {e}")


def save_json(
    data: dict,
    output_path: Path,
    *,
    create_parent: bool = True,
    indent: int = 2,
) -> None:
    """Save dict to JSON file.

    Args:
        data: Dict to save
        output_path: Path to save JSON file
        create_parent: If True, create parent directories
        indent: JSON indentation

    Raises:
        ValueError: If data cannot be serialized
        OSError: If cannot write file
    """
    if not isinstance(data, dict):
        raise ValueError(f"data must be dict, got {type(data)}")

    # Create parent dir if needed
    if create_parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save
    try:
        with open(output_path, "w") as f:
            json.dump(data, f, indent=indent)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Failed to serialize data: {e}")
    except OSError as e:
        raise OSError(f"Failed to write to {output_path}: {e}")


def save_activations_hdf5(
    activations: Dict[str, Dict[str, np.ndarray]],
    metadata: List[Dict],
    output_path: Path,
    model_name: str,
    position: str = "last_token",
    *,
    create_parent: bool = True,
) -> None:
    """Save activations to HDF5 file.

    Args:
        activations: Dict mapping pair_id to {"neutral": arr, "emotional": arr}
                    where arr is [num_layers, hidden_size]
        metadata: List of metadata dicts (one per pair)
        output_path: Path to save HDF5 file
        model_name: Model identifier
        position: Position type ("last_token", "regional", etc.)
        create_parent: If True, create parent directories

    Raises:
        ValueError: If data format invalid
        OSError: If cannot write file
    """
    if not activations:
        raise ValueError("activations dict is empty")

    if not metadata:
        raise ValueError("metadata list is empty")

    if len(activations) != len(metadata):
        raise ValueError(
            f"activations length {len(activations)} != metadata length {len(metadata)}"
        )

    # Validate all activations have same shape
    first_pair = next(iter(activations.values()))
    if "neutral" not in first_pair or "emotional" not in first_pair:
        raise ValueError("Each pair must have 'neutral' and 'emotional' keys")

    expected_shape = first_pair["neutral"].shape
    for pair_id, pair_data in activations.items():
        if pair_data["neutral"].shape != expected_shape:
            raise ValueError(
                f"Pair {pair_id} neutral shape {pair_data['neutral'].shape} "
                f"!= expected {expected_shape}"
            )
        if pair_data["emotional"].shape != expected_shape:
            raise ValueError(
                f"Pair {pair_id} emotional shape {pair_data['emotional'].shape} "
                f"!= expected {expected_shape}"
            )

        # Check for NaN/Inf
        if np.isnan(pair_data["neutral"]).any() or np.isinf(pair_data["neutral"]).any():
            raise ValueError(f"Pair {pair_id} neutral contains NaN or Inf")
        if np.isnan(pair_data["emotional"]).any() or np.isinf(pair_data["emotional"]).any():
            raise ValueError(f"Pair {pair_id} emotional contains NaN or Inf")

    # Create parent dir if needed
    if create_parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to HDF5
    try:
        with h5py.File(output_path, "w") as f:
            # Save attributes
            f.attrs["model_name"] = model_name
            f.attrs["position"] = position
            f.attrs["num_pairs"] = len(activations)

            # Save activations
            activations_group = f.create_group("activations")
            for pair_id, pair_data in activations.items():
                pair_group = activations_group.create_group(pair_id)
                pair_group.create_dataset("neutral", data=pair_data["neutral"])
                pair_group.create_dataset("emotional", data=pair_data["emotional"])

            # Save metadata as JSON string
            f.create_dataset("metadata", data=json.dumps(metadata))

    except Exception as e:
        raise OSError(f"Failed to save activations to {output_path}: {e}")


def save_regional_activations_hdf5(
    regional_activations: Dict[str, Dict],
    metadata: List[Dict],
    output_path: Path,
    model_name: str,
    *,
    create_parent: bool = True,
) -> None:
    """Save regional activations to HDF5 file (for multi-turn conversations).

    Args:
        regional_activations: Dict mapping pair_id -> {
            "emotional": {"regional": {region_name: array, ...}, "global": array, "special_tokens": {...}},
            "neutral": {"regional": {region_name: array, ...}, "global": array, "special_tokens": {...}}
        }
        metadata: List of metadata dicts (one per pair)
        output_path: Path to save HDF5 file
        model_name: Model identifier
        create_parent: If True, create parent directories

    Raises:
        ValueError: If data format invalid
        OSError: If cannot write file
    """
    if not regional_activations:
        raise ValueError("regional_activations dict is empty")

    if not metadata:
        raise ValueError("metadata list is empty")

    if len(regional_activations) != len(metadata):
        raise ValueError(
            f"regional_activations length {len(regional_activations)} != metadata length {len(metadata)}"
        )

    # Validate structure
    for pair_id, pair_data in regional_activations.items():
        if "neutral" not in pair_data or "emotional" not in pair_data:
            raise ValueError(f"Pair {pair_id} missing 'neutral' or 'emotional'")

        for variant in ["neutral", "emotional"]:
            variant_data = pair_data[variant]

            if "regional" not in variant_data:
                raise ValueError(f"Pair {pair_id}, variant {variant} missing 'regional'")
            if "global" not in variant_data:
                raise ValueError(f"Pair {pair_id}, variant {variant} missing 'global'")

            # Check regional activations
            regional = variant_data["regional"]
            if not isinstance(regional, dict):
                raise ValueError(f"Pair {pair_id}, variant {variant}: regional must be dict")

            for region_name, acts in regional.items():
                if not isinstance(acts, np.ndarray):
                    raise ValueError(f"Pair {pair_id}, variant {variant}, region {region_name}: must be ndarray")
                if not np.all(np.isfinite(acts)):
                    raise ValueError(f"Pair {pair_id}, variant {variant}, region {region_name}: has NaN/Inf")

            # Check global activations
            global_acts = variant_data["global"]
            if not isinstance(global_acts, np.ndarray):
                raise ValueError(f"Pair {pair_id}, variant {variant}: global must be ndarray")
            if not np.all(np.isfinite(global_acts)):
                raise ValueError(f"Pair {pair_id}, variant {variant}: global has NaN/Inf")

            # Check special tokens (optional)
            if "special_tokens" in variant_data:
                special_tokens = variant_data["special_tokens"]
                if not isinstance(special_tokens, dict):
                    raise ValueError(f"Pair {pair_id}, variant {variant}: special_tokens must be dict")

                for token_name, token_acts in special_tokens.items():
                    if not isinstance(token_acts, np.ndarray):
                        raise ValueError(
                            f"Pair {pair_id}, variant {variant}, token {token_name}: must be ndarray"
                        )
                    if not np.all(np.isfinite(token_acts)):
                        raise ValueError(
                            f"Pair {pair_id}, variant {variant}, token {token_name}: has NaN/Inf"
                        )

    # Create parent dir if needed
    if create_parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to HDF5
    try:
        with h5py.File(output_path, "w") as f:
            # Save attributes
            f.attrs["model_name"] = model_name
            f.attrs["position"] = "regional"
            f.attrs["num_pairs"] = len(regional_activations)

            # Save activations
            activations_group = f.create_group("activations")

            for pair_id, pair_data in regional_activations.items():
                pair_group = activations_group.create_group(pair_id)

                for variant in ["neutral", "emotional"]:
                    variant_group = pair_group.create_group(variant)
                    variant_data = pair_data[variant]

                    # Save regional activations
                    regional_group = variant_group.create_group("regional")
                    for region_name, acts in variant_data["regional"].items():
                        regional_group.create_dataset(region_name, data=acts)

                    # Save global activations
                    variant_group.create_dataset("global", data=variant_data["global"])

                    # Save special tokens (if present)
                    if "special_tokens" in variant_data:
                        special_group = variant_group.create_group("special_tokens")
                        for token_name, token_acts in variant_data["special_tokens"].items():
                            special_group.create_dataset(token_name, data=token_acts)

            # Save metadata as JSON string
            f.create_dataset("metadata", data=json.dumps(metadata))

    except Exception as e:
        raise OSError(f"Failed to save regional activations to {output_path}: {e}")


def save_pca_results(
    results: dict,
    output_path: Path,
    *,
    create_parent: bool = True,
) -> None:
    """Save standard PCA results to NPZ file.

    Args:
        results: Dict with keys: components, eigenvalues, explained_variance_ratio, config, etc.
        output_path: Path to save NPZ file
        create_parent: If True, create parent directories

    Raises:
        KeyError: If required keys missing
        ValueError: If data format invalid
        OSError: If cannot write file
    """
    required_keys = ["components", "eigenvalues", "explained_variance_ratio", "config"]
    missing = [k for k in required_keys if k not in results]
    if missing:
        raise KeyError(f"Missing required keys in results: {missing}")

    # Validate shapes
    components = results["components"]
    eigenvalues = results["eigenvalues"]
    explained_variance_ratio = results["explained_variance_ratio"]

    if not isinstance(components, np.ndarray):
        raise ValueError(f"components must be ndarray, got {type(components)}")
    if not isinstance(eigenvalues, np.ndarray):
        raise ValueError(f"eigenvalues must be ndarray, got {type(eigenvalues)}")
    if not isinstance(explained_variance_ratio, np.ndarray):
        raise ValueError(f"explained_variance_ratio must be ndarray, got {type(explained_variance_ratio)}")

    if components.ndim != 3:
        raise ValueError(
            f"components must be 3D [layers, n_components, hidden_dim], "
            f"got shape {components.shape}"
        )

    # Create parent dir if needed
    if create_parent:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data for saving
    save_data = {
        "components": components,
        "eigenvalues": eigenvalues,
        "explained_variance_ratio": explained_variance_ratio,
        "config": json.dumps(results["config"]),
    }

    # Optional keys
    if "metadata" in results and results["metadata"] is not None:
        save_data["metadata"] = json.dumps(results["metadata"])
    if "pair_ids" in results:
        save_data["pair_ids"] = np.array(results["pair_ids"])

    # Save
    try:
        np.savez(output_path, **save_data)
    except Exception as e:
        raise OSError(f"Failed to save to {output_path}: {e}")
