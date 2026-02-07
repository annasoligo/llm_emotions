"""
Centralized layer norm loading for steering experiments.

Layer norms are pre-computed L2 norms of the residual stream at each layer,
averaged over diverse prompts. These are used to scale steering vectors:
when we steer at X% of layer norm, we multiply the unit steering vector
by X% of the layer norm value.

Data is stored as per-model JSON files in steering_utils/layer_norms/.

Usage:
    from steering_tests.steering_utils import get_layer_norm, get_all_layer_norms

    # Get norm for a specific layer
    norm = get_layer_norm("qwen32b", 30)  # Returns float

    # Get all norms for a model
    norms = get_all_layer_norms("gemma27b")  # Returns {layer_idx: norm_value}
"""

import json
from pathlib import Path
from typing import Dict, Optional

# Directory containing per-model layer norm JSON files
LAYER_NORMS_DIR = Path(__file__).parent / "layer_norms"

# Cache for loaded norms (keyed by resolved model key)
_layer_norms_cache: Dict[str, dict] = {}

# Mapping from short names to JSON filenames.
# Entries where key == stem of the file are optional but explicit.
LAYER_NORM_FILES = {
    "gemma27b": "gemma3_27b.json",
    "gemma3_27b": "gemma3_27b.json",
    "gemma12b": "gemma12b.json",
    "qwen14b": "qwen14b.json",
    "qwen32b": "qwen32b.json",
    "qwen235b": "qwen235b.json",
    "mistral_nemo": "mistral_nemo.json",
    "humanlike_mistral": "humanlike_mistral.json",
    "llama70b": "llama70b.json",
}

# Model key aliases: full HuggingFace IDs -> short keys
MODEL_ALIASES = {
    "google/gemma-3-27b-it": "gemma27b",
    "google/gemma-3-12b-it": "gemma12b",
    "unsloth/gemma-3-27b-it": "gemma27b",
    "Qwen/Qwen3-14B": "qwen14b",
    "Qwen/Qwen3-32B": "qwen32b",
    "Qwen/Qwen3-235B-A22B": "qwen235b",
    "mistralai/Mistral-Nemo-Instruct-2407": "mistral_nemo",
    "HumanLLMs/Human-Like-Mistral-Nemo-Instruct-2407": "humanlike_mistral",
    "meta-llama/Llama-3.3-70B-Instruct": "llama70b",
}


def resolve_model_key(model_id_or_key: str) -> str:
    """Convert full model ID to short key if needed."""
    return MODEL_ALIASES.get(model_id_or_key, model_id_or_key)


def _load_layer_norms(model_key: str) -> dict:
    """Load layer norms for a specific model from its JSON file, with caching."""
    if model_key in _layer_norms_cache:
        return _layer_norms_cache[model_key]

    if model_key in LAYER_NORM_FILES:
        filename = LAYER_NORM_FILES[model_key]
    else:
        filename = f"{model_key}.json"

    filepath = LAYER_NORMS_DIR / filename

    if not filepath.exists():
        available = [f.stem for f in LAYER_NORMS_DIR.glob("*.json")]
        raise FileNotFoundError(
            f"Layer norms file not found for '{model_key}': {filepath}\n"
            f"Available models: {available}"
        )

    with open(filepath) as f:
        data = json.load(f)

    _layer_norms_cache[model_key] = data
    return data


def get_layer_norm(model_key: str, layer: int) -> float:
    """
    Get the average residual stream L2 norm for a specific model and layer.

    Args:
        model_key: Model identifier — short name (e.g., "qwen32b") or full
                   HuggingFace ID (e.g., "Qwen/Qwen3-32B").
        layer: Layer index

    Returns:
        Average L2 norm of residual stream at that layer

    Raises:
        KeyError: If layer not found
        FileNotFoundError: If model's layer norms JSON doesn't exist
    """
    model_key = resolve_model_key(model_key)
    data = _load_layer_norms(model_key)

    layers = data["layers"]
    layer_str = str(layer)

    if layer_str not in layers:
        available = sorted([int(k) for k in layers.keys()])
        raise KeyError(f"Layer {layer} not found for {model_key}. Available: {available}")

    return layers[layer_str]["mean"]


def get_all_layer_norms(model_key: str) -> Dict[int, float]:
    """
    Get all layer norms for a model.

    Args:
        model_key: Model identifier (short name or full HuggingFace ID)

    Returns:
        Dict mapping layer index to average L2 norm
    """
    model_key = resolve_model_key(model_key)
    data = _load_layer_norms(model_key)

    layers = data["layers"]
    return {int(k): v["mean"] for k, v in layers.items()}


def get_model_info(model_key: str) -> dict:
    """
    Get metadata about a model's layer norms.

    Returns:
        Dict with 'model', 'source', 'hidden_dim', 'n_samples', and 'layers' dict
    """
    model_key = resolve_model_key(model_key)
    return _load_layer_norms(model_key)


def list_models() -> list:
    """List all models with computed layer norms."""
    return [f.stem for f in LAYER_NORMS_DIR.glob("*.json")]
