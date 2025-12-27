"""Data loading utilities - no fallbacks, fail fast."""

import json
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def load_activations_hdf5(h5_path: Path) -> tuple[dict, list[dict], dict]:
    """Load activations from HDF5 file.

    Args:
        h5_path: Path to HDF5 file

    Returns:
        activations: Dict mapping pair_id -> {neutral: array, emotional: array}
        metadata: List of metadata dicts
        attrs: File attributes dict

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If required keys missing
        ValueError: If data format invalid
    """
    if not h5_path.exists():
        raise FileNotFoundError(f"Activation file not found: {h5_path}")

    activations = {}

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

            neutral = pair_group["neutral"][:]
            emotional = pair_group["emotional"][:]

            # Check for NaN/Inf
            if not np.all(np.isfinite(neutral)):
                raise ValueError(f"Pair {pair_id} has NaN/Inf in neutral activations")
            if not np.all(np.isfinite(emotional)):
                raise ValueError(f"Pair {pair_id} has NaN/Inf in emotional activations")

            # Check shape match
            if neutral.shape != emotional.shape:
                raise ValueError(
                    f"Shape mismatch for pair {pair_id}: "
                    f"neutral {neutral.shape} vs emotional {emotional.shape}"
                )

            activations[pair_id] = {
                "neutral": neutral,
                "emotional": emotional,
            }

    if not activations:
        raise ValueError(f"No activations loaded from {h5_path}")

    return activations, metadata, attrs


def load_cpca_results(npz_path: Path) -> dict:
    """Load cPCA results from NPZ file.

    Args:
        npz_path: Path to NPZ file

    Returns:
        Dict with keys: components, eigenvalues, alphas, pair_ids, config, metadata

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If required keys missing
        ValueError: If data format invalid
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"cPCA results not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    required_keys = ["components", "eigenvalues", "alphas", "config"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in {npz_path}: {missing}")

    # Parse config
    try:
        config_data = data["config"]
        # Handle both dict objects and JSON strings
        if isinstance(config_data, dict):
            config = config_data
        elif isinstance(config_data, np.ndarray):
            # numpy 0-d array containing a dict object
            if config_data.ndim == 0:
                config = config_data.item()  # Extract the dict from 0-d array
            else:
                config = dict(config_data)
        else:
            # Try parsing as JSON string
            config = json.loads(str(config_data))
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        raise ValueError(f"Invalid config in {npz_path}: {e}")

    # Parse metadata if present
    metadata = None
    if "metadata" in data:
        try:
            metadata = json.loads(str(data["metadata"]))
        except (json.JSONDecodeError, TypeError):
            pass  # Metadata is optional

    result = {
        "components": data["components"],
        "eigenvalues": data["eigenvalues"],
        "alphas": data["alphas"],
        "config": config,
        "metadata": metadata,
    }

    # Optional: pair_ids
    if "pair_ids" in data:
        result["pair_ids"] = data["pair_ids"]

    return result


def load_regional_activations_hdf5(h5_path: Path) -> tuple[dict, list[dict], dict]:
    """Load regional activations from HDF5 file (for multi-turn conversations).

    Regional activations are split by conversation regions (user turns, assistant turns, special tokens).

    Args:
        h5_path: Path to HDF5 file with regional activations

    Returns:
        regional_activations: Dict mapping pair_id -> {
            "emotional": {"regional": {region_name: array, ...}, "global": array, ...},
            "neutral": {"regional": {region_name: array, ...}, "global": array, ...}
        }
        metadata: List of metadata dicts
        attrs: File attributes dict

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If required keys missing
        ValueError: If data format invalid
    """
    if not h5_path.exists():
        raise FileNotFoundError(f"Regional activation file not found: {h5_path}")

    regional_activations = {}

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

            # Check for required variant types
            if "neutral" not in pair_group:
                raise KeyError(f"Missing 'neutral' for pair {pair_id}")
            if "emotional" not in pair_group:
                raise KeyError(f"Missing 'emotional' for pair {pair_id}")

            pair_data = {}

            for variant in ["neutral", "emotional"]:
                variant_group = pair_group[variant]

                # Load regional activations
                if "regional" not in variant_group:
                    raise KeyError(f"Missing 'regional' for pair {pair_id}, variant {variant}")

                regional_data = {}
                regional_group = variant_group["regional"]

                for region_name in regional_group.keys():
                    region_acts = regional_group[region_name][:]

                    # Check for NaN/Inf
                    if not np.all(np.isfinite(region_acts)):
                        raise ValueError(
                            f"Pair {pair_id}, variant {variant}, region {region_name} has NaN/Inf"
                        )

                    regional_data[region_name] = region_acts

                # Load global activations
                if "global" not in variant_group:
                    raise KeyError(f"Missing 'global' for pair {pair_id}, variant {variant}")

                global_acts = variant_group["global"][:]
                if not np.all(np.isfinite(global_acts)):
                    raise ValueError(f"Pair {pair_id}, variant {variant} has NaN/Inf in global")

                # Load special tokens (optional)
                special_tokens = {}
                if "special_tokens" in variant_group:
                    special_group = variant_group["special_tokens"]
                    for token_name in special_group.keys():
                        token_acts = special_group[token_name][:]
                        if not np.all(np.isfinite(token_acts)):
                            raise ValueError(
                                f"Pair {pair_id}, variant {variant}, special token {token_name} has NaN/Inf"
                            )
                        special_tokens[token_name] = token_acts

                pair_data[variant] = {
                    "regional": regional_data,
                    "global": global_acts,
                    "special_tokens": special_tokens,
                }

            regional_activations[pair_id] = pair_data

    if not regional_activations:
        raise ValueError(f"No regional activations loaded from {h5_path}")

    return regional_activations, metadata, attrs


def load_pca_results(npz_path: Path) -> dict:
    """Load standard PCA results from NPZ file.

    Args:
        npz_path: Path to NPZ file

    Returns:
        Dict with keys: components, eigenvalues, explained_variance_ratio, pair_ids, config, metadata

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If required keys missing
        ValueError: If data format invalid
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"PCA results not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)

    required_keys = ["components", "eigenvalues", "explained_variance_ratio", "config"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise KeyError(f"Missing required keys in {npz_path}: {missing}")

    # Parse config
    try:
        config = json.loads(str(data["config"]))
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Invalid config JSON in {npz_path}: {e}")

    # Parse metadata if present
    metadata = None
    if "metadata" in data:
        try:
            metadata = json.loads(str(data["metadata"]))
        except (json.JSONDecodeError, TypeError):
            pass  # Metadata is optional

    result = {
        "components": data["components"],
        "eigenvalues": data["eigenvalues"],
        "explained_variance_ratio": data["explained_variance_ratio"],
        "config": config,
        "metadata": metadata,
    }

    # Optional: pair_ids
    if "pair_ids" in data:
        result["pair_ids"] = data["pair_ids"]

    return result


def load_probe_results(pkl_path: Path) -> dict:
    """Load probe training results from pickle file.

    Args:
        pkl_path: Path to pickle file

    Returns:
        Dict with probe results

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If pickle is invalid
    """
    import pickle

    if not pkl_path.exists():
        raise FileNotFoundError(f"Probe results not found: {pkl_path}")

    try:
        with open(pkl_path, "rb") as f:
            results = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load pickle from {pkl_path}: {e}")

    if not isinstance(results, dict):
        raise ValueError(f"Expected dict from {pkl_path}, got {type(results)}")

    return results


def load_jsonl(
    jsonl_path: Path,
    limit: Optional[int] = None,
    require_id: bool = True,
) -> list[dict]:
    """Load data from JSONL file.

    Standardized JSONL loading function with consistent error handling.

    Args:
        jsonl_path: Path to JSONL file
        limit: Optional limit on number of items to load (default: None = load all)
        require_id: If True, raise error if items missing 'id' field (default: True)

    Returns:
        List of dictionaries loaded from JSONL

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is malformed or required fields missing
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    data = []
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Skip empty lines
            if not line.strip():
                continue

            # Parse JSON
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num} in {jsonl_path}: {e}")

            # Validate ID field if required
            if require_id and "id" not in item:
                raise ValueError(f"Line {line_num} in {jsonl_path} missing required 'id' field")

            data.append(item)

            # Check limit
            if limit and len(data) >= limit:
                break

    if not data:
        raise ValueError(f"No data loaded from {jsonl_path}")

    return data
