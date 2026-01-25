"""
Centralized layer norm loading for steering experiments.

Layer norms are pre-computed L2 norms of the residual stream at each layer,
averaged over diverse prompts. These are used to scale steering vectors:
when we steer at X% of layer norm, we multiply the unit steering vector
by X% of the layer norm value.

Usage:
    from steering_tests.steering_utils import get_layer_norm, get_all_layer_norms

    # Get norm for a specific layer
    norm = get_layer_norm("qwen32b", 30)  # Returns float

    # Get all norms for a model
    norms = get_all_layer_norms("gemma")  # Returns {layer_idx: norm_value}
"""

import json
from pathlib import Path
from typing import Dict, Optional

# Path to the computed layer norms (in experiments/steering/)
LAYER_NORMS_FILE = Path(__file__).parent.parent.parent / "experiments" / "steering" / "layer_norms.json"

# Cache for loaded norms
_layer_norms_cache: Optional[dict] = None


def _load_layer_norms() -> dict:
    """Load layer norms from JSON file, with caching."""
    global _layer_norms_cache

    if _layer_norms_cache is not None:
        return _layer_norms_cache

    if not LAYER_NORMS_FILE.exists():
        raise FileNotFoundError(
            f"Layer norms file not found: {LAYER_NORMS_FILE}\n"
            "Run experiments/steering/compute_all_layer_norms.py to generate it."
        )

    with open(LAYER_NORMS_FILE) as f:
        _layer_norms_cache = json.load(f)

    return _layer_norms_cache


def get_layer_norm(model_key: str, layer: int) -> float:
    """
    Get the average residual stream L2 norm for a specific model and layer.

    Args:
        model_key: Model identifier (e.g., "gemma", "qwen32b", "qwen235b")
        layer: Layer index

    Returns:
        Average L2 norm of residual stream at that layer

    Raises:
        KeyError: If model or layer not found
        FileNotFoundError: If layer_norms.json doesn't exist
    """
    data = _load_layer_norms()

    if model_key not in data:
        available = list(data.keys())
        raise KeyError(f"Model '{model_key}' not found. Available: {available}")

    layers = data[model_key]["layers"]
    layer_str = str(layer)

    if layer_str not in layers:
        available = sorted([int(k) for k in layers.keys()])
        raise KeyError(f"Layer {layer} not found for {model_key}. Available: {available}")

    return layers[layer_str]["mean"]


def get_all_layer_norms(model_key: str) -> Dict[int, float]:
    """
    Get all layer norms for a model.

    Args:
        model_key: Model identifier

    Returns:
        Dict mapping layer index to average L2 norm
    """
    data = _load_layer_norms()

    if model_key not in data:
        available = list(data.keys())
        raise KeyError(f"Model '{model_key}' not found. Available: {available}")

    layers = data[model_key]["layers"]
    return {int(k): v["mean"] for k, v in layers.items()}


def get_model_info(model_key: str) -> dict:
    """
    Get metadata about a model's layer norms.

    Returns:
        Dict with 'model_id', 'n_layers', and 'layers' dict
    """
    data = _load_layer_norms()

    if model_key not in data:
        available = list(data.keys())
        raise KeyError(f"Model '{model_key}' not found. Available: {available}")

    return data[model_key]


def list_models() -> list:
    """List all models with computed layer norms."""
    data = _load_layer_norms()
    return list(data.keys())


# Model key aliases for convenience
MODEL_ALIASES = {
    "google/gemma-3-27b-it": "gemma",
    "unsloth/gemma-3-27b-it": "gemma",
    "Qwen/Qwen3-32B": "qwen32b",
    "Qwen/Qwen3-235B-A22B": "qwen235b",
    "allenai/OLMo-2-1124-13B-Instruct": "olmo",
}


def resolve_model_key(model_id_or_key: str) -> str:
    """Convert model ID to short key if needed."""
    if model_id_or_key in MODEL_ALIASES:
        return MODEL_ALIASES[model_id_or_key]
    return model_id_or_key
