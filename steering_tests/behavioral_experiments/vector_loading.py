"""
Unified vector loading for behavioral experiments.

Supports loading:
- Emotion vectors from steering_tests/vectors/ (.npy format)
- Appraisal vectors from HDF5 activation files
- Layer norms for scaling
"""

import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    MODEL_CONFIGS,
    VECTOR_PATHS,
    APPRAISAL_DATA_PATHS,
    resolve_model_key,
)
from steering_tests.steering_utils.layer_norms import get_layer_norm  # noqa: F401

logger = logging.getLogger(__name__)


# =============================================================================
# Emotion vector loading
# =============================================================================


def load_emotion_vectors(
    model_name: str,
    layer: int,
    vector_type: str = "base_emotion_vs_others",
    representation: str = "last_token",
    emotions: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], float, dict]:
    """
    Load emotion steering vectors from steering_tests/vectors/.

    Args:
        model_name: Full model ID or short name (e.g., "qwen235b")
        layer: Layer index
        vector_type: One of "base_emotion_vs_others", "high_emotion_vs_others",
                    "text_pairs_emotion_vs_neutral", "text_pairs_emotion_vs_opposite"
        representation: "last_token" or "special_mean"
        emotions: List of emotions to load (None = all available)

    Returns:
        Tuple of (vectors dict, layer_norm, metadata dict)
    """
    import pickle

    # Get short name for vector path
    if model_name in MODEL_CONFIGS:
        short_name = MODEL_CONFIGS[model_name]["short_name"]
    else:
        short_name = model_name

    # Handle model name variations (gemma27b -> gemma3_27b in vector paths)
    vector_path_names = {
        "gemma27b": "gemma3_27b",
        "gemma12b": "gemma12b",
        "qwen14b": "qwen14b",
        "qwen32b": "qwen32b",
        "qwen235b": "qwen235b",
        "mistral_nemo": "mistral_nemo",
        "humanlike_mistral": "humanlike_mistral",
        "llama70b": "llama70b",
    }
    vector_model_name = vector_path_names.get(short_name, short_name)

    vector_base = VECTOR_PATHS["emotion"] / vector_model_name / vector_type / representation

    # Try pickle format first (layer_XX.pkl with dict of all emotions)
    pkl_path = vector_base / f"layer_{layer:02d}.pkl"

    # Also check alternative path for text_pairs vectors (in 'layers' subdir instead of 'last_token')
    if not pkl_path.exists():
        alt_base = VECTOR_PATHS["emotion"] / vector_model_name / vector_type / "layers"
        alt_pkl_path = alt_base / f"layer_{layer:02d}.pkl"
        if alt_pkl_path.exists():
            pkl_path = alt_pkl_path
            vector_base = alt_base

    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            all_vectors = pickle.load(f)

        vectors = {}
        for emotion, vec in all_vectors.items():
            if emotions is None or emotion in emotions:
                vectors[emotion] = np.asarray(vec).astype(np.float32)

        logger.info(f"Loaded {len(vectors)} emotion vectors from {pkl_path}")
        vector_source = str(pkl_path)

    else:
        # Fallback to .npy format (layer_N/ directory with per-emotion files)
        vector_dir = vector_base / f"layer_{layer}"
        if not vector_dir.exists():
            raise FileNotFoundError(
                f"Vector file not found: {pkl_path} or {vector_dir}\n"
                f"Available in {vector_base}: {list(vector_base.glob('*'))}"
            )

        vectors = {}
        for npy_path in vector_dir.glob("*.npy"):
            emotion = npy_path.stem
            if emotions is None or emotion in emotions:
                vectors[emotion] = np.load(npy_path).astype(np.float32)

        logger.info(f"Loaded {len(vectors)} emotion vectors from {vector_dir}")
        vector_source = str(vector_dir)

    if not vectors:
        raise ValueError(f"No vectors found matching emotions={emotions}")

    # Get layer norm
    model_key = resolve_model_key(model_name)
    layer_norm = get_layer_norm(model_key, layer)
    logger.info(f"Layer {layer} norm: {layer_norm:.2f}")

    # Build metadata
    metadata = {
        "vector_source": vector_source,
        "vector_type": vector_type,
        "representation": representation,
        "layer": layer,
        "emotions_loaded": list(vectors.keys()),
        "layer_norm": layer_norm,
    }

    return vectors, layer_norm, metadata


def load_emotion_vectors_npz(
    model_name: str,
    layer: int,
    vector_type: str = "text",
    emotions: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], float, dict]:
    """
    Load emotion vectors from legacy .npz format.

    This supports the older format used in experiments/steering/vectors/.

    Args:
        model_name: Full model ID
        layer: Layer index
        vector_type: "text" or "ua"
        emotions: List of emotions to load (None = use model defaults)

    Returns:
        Tuple of (vectors dict, layer_norm, metadata dict)
    """
    config = MODEL_CONFIGS[model_name]
    short_name = config["short_name"]

    # Legacy vector locations
    vector_base = Path(__file__).parent.parent.parent / "experiments" / "steering" / "vectors"
    vector_file = vector_base / vector_type / f"{short_name}_layer{layer}.npz"

    if not vector_file.exists():
        raise FileNotFoundError(f"Vector file not found: {vector_file}")

    data = np.load(vector_file)
    vectors = {}

    # Get default emotions for this vector type
    if emotions is None:
        emotions = config.get(f"{vector_type}_emotions", [])

    # Handle stacked format (vectors + emotions arrays)
    if "vectors" in data and "emotions" in data:
        emo_list = [str(e) for e in data["emotions"]]
        for i, emo in enumerate(emo_list):
            if emo in emotions:
                vectors[emo] = data["vectors"][i].astype(np.float32)
    else:
        # Per-emotion format
        for emotion in emotions:
            for key in [f"M_{emotion}", emotion, f"{emotion}_direction"]:
                if key in data:
                    vectors[emotion] = data[key].astype(np.float32)
                    break

    logger.info(f"Loaded {len(vectors)} vectors from {vector_file.name}: {list(vectors.keys())}")

    # Get layer norm
    model_key = resolve_model_key(model_name)
    layer_norm = get_layer_norm(model_key, layer)

    # Compute file metadata
    file_stat = vector_file.stat()
    file_mtime = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
    with open(vector_file, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()[:12]

    metadata = {
        "vector_file": str(vector_file),
        "vector_file_modified": file_mtime,
        "vector_file_hash": file_hash,
        "emotions_loaded": list(vectors.keys()),
        "layer_norm": layer_norm,
    }

    return vectors, layer_norm, metadata


# =============================================================================
# Appraisal vector loading
# =============================================================================

# Orthogonalization settings
FLIP_AXES = {"uncertainty"}  # Flip so positive = high uncertainty
ORTHO_ORDER = ["valence", "uncertainty", "agency"]


def load_appraisal_vectors(
    model_name: str,
    layer: int,
    orthogonalize: bool = True,
    axes: Optional[List[str]] = None,
) -> Tuple[Dict[str, np.ndarray], float, dict]:
    """
    Load appraisal steering vectors (valence, uncertainty, agency) from HDF5.

    Args:
        model_name: Full model ID
        layer: Layer index
        orthogonalize: Whether to apply Gram-Schmidt orthogonalization
        axes: List of axes to load (None = all)

    Returns:
        Tuple of (vectors dict, layer_norm, metadata dict)
    """
    # Import h5py here to avoid dependency if not using appraisal vectors
    import h5py

    if model_name not in APPRAISAL_DATA_PATHS:
        raise ValueError(f"No appraisal data for {model_name}. Available: {list(APPRAISAL_DATA_PATHS.keys())}")

    paths = APPRAISAL_DATA_PATHS[model_name]
    activations_path = paths["activations"]
    metadata_path = paths["metadata"]

    with open(metadata_path) as f:
        meta = json.load(f)

    # Collect activations by axis and variant
    axis_activations = defaultdict(lambda: {"a": [], "b": []})

    with h5py.File(activations_path, "r") as f:
        acts_group = f["activations"]

        for item in meta["items"]:
            item_id = item["id"]
            axis_name = item.get("axis_name")
            variant = item.get("variant")

            if axis_name and variant and item_id in acts_group:
                act = acts_group[item_id]["assistant_start_last_token"][layer, :]
                axis_activations[axis_name][variant].append(act)

    # Compute mean difference vectors
    vectors = {}
    for axis_name, variants in axis_activations.items():
        if axes is not None and axis_name not in axes:
            continue

        if variants["a"] and variants["b"]:
            mean_a = np.mean(variants["a"], axis=0)
            mean_b = np.mean(variants["b"], axis=0)
            vec = mean_a - mean_b

            if axis_name in FLIP_AXES:
                vec = -vec
                logger.info(f"{axis_name}: FLIPPED (A=low, B=high)")

            vectors[axis_name] = vec.astype(np.float32)
            logger.info(
                f"{axis_name}: {len(variants['a'])} A, {len(variants['b'])} B samples, "
                f"norm={np.linalg.norm(vec):.2f}"
            )

    if orthogonalize:
        logger.info("Orthogonalizing vectors...")
        vectors = orthogonalize_vectors(vectors, ORTHO_ORDER)
        for axis in ORTHO_ORDER:
            if axis in vectors:
                logger.info(f"  {axis} orthogonalized norm: {np.linalg.norm(vectors[axis]):.2f}")

    # Get layer norm
    model_key = resolve_model_key(model_name)
    layer_norm = get_layer_norm(model_key, layer)

    metadata = {
        "activations_path": activations_path,
        "metadata_path": metadata_path,
        "layer": layer,
        "orthogonalized": orthogonalize,
        "axes_loaded": list(vectors.keys()),
        "layer_norm": layer_norm,
    }

    return vectors, layer_norm, metadata


def orthogonalize_vectors(vectors: Dict[str, np.ndarray], order: List[str]) -> Dict[str, np.ndarray]:
    """
    Apply Gram-Schmidt orthogonalization to vectors in specified order.

    Args:
        vectors: Dict mapping axis names to vectors
        order: List of axis names specifying orthogonalization order

    Returns:
        Dict of orthogonalized vectors
    """
    orthogonal = {}

    for axis in order:
        if axis not in vectors:
            continue

        vec = vectors[axis].copy()

        # Subtract projections onto all previous vectors
        for prev_axis in order:
            if prev_axis == axis:
                break
            if prev_axis in orthogonal:
                prev_vec = orthogonal[prev_axis]
                projection = np.dot(vec, prev_vec) / np.dot(prev_vec, prev_vec) * prev_vec
                vec = vec - projection

        orthogonal[axis] = vec

    return orthogonal


# =============================================================================
# Random vector generation
# =============================================================================


def generate_random_vectors(
    n_vectors: int,
    hidden_dim: int,
    seed: int = 42,
    normalize: bool = True,
) -> List[np.ndarray]:
    """
    Generate random vectors for baseline comparisons.

    Args:
        n_vectors: Number of vectors to generate
        hidden_dim: Dimension of each vector
        seed: Random seed for reproducibility
        normalize: Whether to normalize to unit vectors

    Returns:
        List of random vectors
    """
    rng = np.random.default_rng(seed)
    vectors = []

    for _ in range(n_vectors):
        vec = rng.standard_normal(hidden_dim).astype(np.float32)
        if normalize:
            vec = vec / np.linalg.norm(vec)
        vectors.append(vec)

    return vectors
