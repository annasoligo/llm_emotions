"""Activation processing utilities."""

from typing import Optional
import numpy as np


def compute_diffs(
    activations: dict,
    pair_ids: list[str],
) -> np.ndarray:
    """Compute emotional - neutral activation differences.

    Args:
        activations: Dict mapping pair_id -> {neutral: array, emotional: array}
        pair_ids: List of pair IDs to include

    Returns:
        Diffs array of shape [n_pairs, n_layers, hidden_dim]

    Raises:
        KeyError: If pair_id not found
        ValueError: If activation shapes are inconsistent
    """
    if not pair_ids:
        raise ValueError("pair_ids cannot be empty")

    diffs = []
    for pid in pair_ids:
        if pid not in activations:
            raise KeyError(f"Pair ID not found: {pid}")

        pair = activations[pid]
        if "neutral" not in pair or "emotional" not in pair:
            raise KeyError(f"Pair {pid} missing 'neutral' or 'emotional' key")

        diff = pair["emotional"] - pair["neutral"]

        # Check for NaN/Inf - fail fast
        if not np.all(np.isfinite(diff)):
            raise ValueError(f"Pair {pid} has NaN/Inf values in diff")

        diffs.append(diff)

    return np.stack(diffs)


def filter_pairs_by_criteria(
    metadata: list[dict],
    tier: Optional[str] = None,
    emotions: Optional[list[str]] = None,
    min_intensity: Optional[int] = None,
    max_intensity: Optional[int] = None,
) -> list[str]:
    """Filter pairs by multiple criteria.

    Args:
        metadata: List of metadata dicts with 'id' field
        tier: Tier to filter by (None = include all)
        emotions: List of emotions to include (None = include all)
        min_intensity: Minimum intensity level (inclusive)
        max_intensity: Maximum intensity level (inclusive)

    Returns:
        List of pair IDs matching all criteria

    Raises:
        KeyError: If required fields missing from metadata
    """
    result = []

    for m in metadata:
        if "id" not in m:
            raise KeyError("Metadata entry missing 'id' field")

        # Tier filter
        if tier is not None:
            if "tier" not in m:
                raise KeyError(f"Metadata for {m['id']} missing 'tier' field")
            if m["tier"] != tier:
                continue

        # Emotion filter
        if emotions is not None:
            # Support both text format (emotion) and conversation format (user_emotion)
            emotion = m.get("emotion") or m.get("user_emotion")
            if emotion is None:
                raise KeyError(f"Metadata for {m['id']} missing 'emotion' or 'user_emotion' field")
            if emotion not in emotions:
                continue

        # Intensity filters
        if min_intensity is not None or max_intensity is not None:
            if "intensity" not in m:
                raise KeyError(f"Metadata for {m['id']} missing 'intensity' field")
            intensity = m["intensity"]
            if min_intensity is not None and intensity < min_intensity:
                continue
            if max_intensity is not None and intensity > max_intensity:
                continue

        result.append(m["id"])

    return result


def project_onto_components(
    activations: dict,
    components: np.ndarray,
    pair_ids: list[str],
    layer_idx: int,
    use_diffs: bool = True,
) -> np.ndarray:
    """Project activations onto PC components.

    Args:
        activations: Dict mapping pair_id -> {neutral: array, emotional: array}
        components: PC components of shape [n_components, hidden_dim]
        pair_ids: List of pair IDs to project
        layer_idx: Layer index to extract activations from
        use_diffs: If True, project diffs; if False, project emotional only

    Returns:
        Projections of shape [n_pairs, n_components]

    Raises:
        KeyError: If pair_id not found
        IndexError: If layer_idx out of range
        ValueError: If dimensions don't match
    """
    if not pair_ids:
        raise ValueError("pair_ids cannot be empty")

    projections = []

    for pid in pair_ids:
        if pid not in activations:
            raise KeyError(f"Pair ID not found: {pid}")

        pair = activations[pid]

        # Extract layer activations
        try:
            if use_diffs:
                acts = (pair["emotional"] - pair["neutral"])[layer_idx]
            else:
                acts = pair["emotional"][layer_idx]
        except IndexError:
            raise IndexError(
                f"layer_idx {layer_idx} out of range for pair {pid} "
                f"(shape: {pair['emotional'].shape})"
            )

        # Check dimension match
        if acts.shape[0] != components.shape[1]:
            raise ValueError(
                f"Dimension mismatch: activations have dim {acts.shape[0]}, "
                f"components expect dim {components.shape[1]}"
            )

        # Project onto components
        acts = acts.astype(np.float32)
        proj = acts @ components.T
        projections.append(proj)

    return np.stack(projections)
