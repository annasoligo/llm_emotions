"""Contrastive PCA implementation - minimal, no fallbacks.

Based on: Abid et al. "Exploring patterns enriched in a dataset with
contrastive principal component analysis" (Nature Communications, 2018)

Supports:
- Global cPCA (standard across all activations)
- Regional cPCA (separate cPCA per activation region)
"""

from typing import Optional, Dict, List

import numpy as np
from sklearn.metrics import silhouette_score
from tqdm import tqdm


def run_cpca(
    target_acts: np.ndarray,
    background_acts: np.ndarray,
    alpha: float,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run contrastive PCA.

    Finds eigenvectors of (C_target - α * C_background).

    Args:
        target_acts: Target activations [n_samples, hidden_dim]
        background_acts: Background activations [n_samples, hidden_dim]
        alpha: Contrast parameter (higher = more background suppression)
        n_components: Number of components to return

    Returns:
        components: [n_components, hidden_dim] contrastive PCs
        eigenvalues: [n_components] corresponding eigenvalues

    Raises:
        ValueError: If shapes don't match or invalid params
    """
    if target_acts.shape[1] != background_acts.shape[1]:
        raise ValueError(
            f"Dimension mismatch: target {target_acts.shape[1]} "
            f"vs background {background_acts.shape[1]}"
        )

    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")

    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")

    hidden_dim = target_acts.shape[1]
    if n_components > hidden_dim:
        raise ValueError(
            f"n_components ({n_components}) cannot exceed hidden_dim ({hidden_dim})"
        )

    # Compute covariance matrices
    C_target = np.cov(target_acts.T)
    C_background = np.cov(background_acts.T)

    # Contrastive covariance
    C_contrast = C_target - alpha * C_background

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(C_contrast)

    # Sort by descending eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top components
    components = eigenvectors[:, :n_components].T  # [n_components, hidden_dim]
    eigenvalues = eigenvalues[:n_components]

    return components, eigenvalues


def compute_silhouette(
    diffs: np.ndarray,
    components: np.ndarray,
    labels: np.ndarray,
    n_dims: int = 10,
) -> float:
    """Compute silhouette score for clustering in cPC space.

    Args:
        diffs: Activation differences [n_samples, hidden_dim]
        components: PC components [n_components, hidden_dim]
        labels: Class labels [n_samples]
        n_dims: Number of top cPCs to use

    Returns:
        Silhouette score (-1 to 1, higher is better)

    Raises:
        ValueError: If not enough unique labels
    """
    if n_dims > components.shape[0]:
        raise ValueError(
            f"n_dims ({n_dims}) exceeds n_components ({components.shape[0]})"
        )

    # Project onto top cPCs
    projected = diffs @ components[:n_dims].T

    # Check unique labels
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError(
            f"Need at least 2 unique labels for silhouette score, got {len(unique_labels)}"
        )

    return silhouette_score(projected, labels)


def tune_alpha_silhouette(
    target_acts: np.ndarray,
    background_acts: np.ndarray,
    diffs: np.ndarray,
    labels: np.ndarray,
    alphas: np.ndarray,
    n_components: int,
    n_dims_score: int = 10,
) -> tuple[float, dict]:
    """Tune alpha by maximizing silhouette score.

    Args:
        target_acts: Target activations [n_samples, hidden_dim]
        background_acts: Background activations [n_samples, hidden_dim]
        diffs: Activation differences [n_samples, hidden_dim]
        labels: Class labels [n_samples] for scoring
        alphas: Alpha values to try
        n_components: Number of cPCs to compute
        n_dims_score: Number of top cPCs to use for scoring

    Returns:
        best_alpha: Alpha with highest silhouette score
        results: Dict with scores for all alphas

    Raises:
        ValueError: If no valid alphas or labels
    """
    if len(alphas) == 0:
        raise ValueError("alphas array is empty")

    scores = []

    for alpha in alphas:
        components, _ = run_cpca(target_acts, background_acts, alpha, n_components)
        score = compute_silhouette(diffs, components, labels, n_dims_score)
        scores.append(score)

    scores = np.array(scores)
    best_idx = np.argmax(scores)

    results = {
        "alphas": alphas.tolist(),
        "silhouette_scores": scores.tolist(),
        "best_alpha": float(alphas[best_idx]),
        "best_score": float(scores[best_idx]),
    }

    return alphas[best_idx], results


def run_cpca_all_layers(
    activations: dict,
    metadata: list[dict],
    alpha: Optional[float] = None,
    n_components: int = 50,
    alpha_range: tuple[float, float] = (0.1, 1000),
    n_alphas: int = 40,
    use_diffs: bool = True,
) -> dict:
    """Run cPCA on all layers.

    Args:
        activations: Dict mapping pair_id -> {neutral: array, emotional: array}
        metadata: List of metadata dicts with 'id' and 'emotion' keys
        alpha: If provided, use this alpha; otherwise tune per layer
        n_components: Number of cPCs to compute
        alpha_range: Range for alpha search (if tuning)
        n_alphas: Number of alpha values to try (if tuning)
        use_diffs: If True, use diffs as target; if False, use raw emotional

    Returns:
        Results dict with components, eigenvalues, alphas, config

    Raises:
        ValueError: If data format invalid or no data
    """
    if not activations:
        raise ValueError("activations dict is empty")

    if not metadata:
        raise ValueError("metadata list is empty")

    # Setup
    pair_ids = list(activations.keys())
    first_pair = next(iter(activations.values()))
    num_layers = first_pair["neutral"].shape[0]
    hidden_dim = first_pair["neutral"].shape[1]

    # Get emotion labels for tuning
    id_to_meta = {m["id"]: m for m in metadata}
    for pid in pair_ids:
        if pid not in id_to_meta:
            raise KeyError(f"Pair {pid} not found in metadata")

    emotion_labels = []
    for pid in pair_ids:
        meta = id_to_meta[pid]
        # Support both text format (emotion) and conversation format (user_emotion)
        if "emotion" in meta:
            emotion_labels.append(meta["emotion"])
        elif "user_emotion" in meta:
            emotion_labels.append(meta["user_emotion"])
        else:
            raise KeyError(f"Metadata for {pid} missing 'emotion' or 'user_emotion' field")

    emotion_labels = np.array(emotion_labels)

    # Alpha values to search (if tuning)
    tune = alpha is None
    if tune:
        alphas = np.geomspace(alpha_range[0], alpha_range[1], n_alphas)

    results = {
        "components": {},
        "eigenvalues": {},
        "pair_ids": pair_ids,
        "metadata": metadata,
        "config": {
            "n_components": n_components,
            "alpha": alpha,
            "alpha_range": alpha_range if tune else None,
            "n_alphas": n_alphas if tune else None,
            "num_pairs": len(pair_ids),
            "num_layers": num_layers,
            "hidden_dim": hidden_dim,
            "use_diffs": use_diffs,
        },
        "alpha_per_layer": {},
        "tuning": {},
    }

    print(f"Running cPCA on {num_layers} layers...")

    for layer_idx in tqdm(range(num_layers), desc="Processing layers"):
        # Extract activations for this layer
        emotional_acts = np.stack(
            [activations[pid]["emotional"][layer_idx] for pid in pair_ids]
        ).astype(np.float32)

        neutral_acts = np.stack(
            [activations[pid]["neutral"][layer_idx] for pid in pair_ids]
        ).astype(np.float32)

        # Choose target data
        if use_diffs:
            target_acts = emotional_acts - neutral_acts
        else:
            target_acts = emotional_acts

        # Compute diffs for scoring
        diffs = emotional_acts - neutral_acts

        # Tune alpha if not provided
        if tune:
            layer_alpha, tuning_info = tune_alpha_silhouette(
                target_acts, neutral_acts, diffs, emotion_labels, alphas, n_components
            )
            results["tuning"][layer_idx] = tuning_info
        else:
            layer_alpha = alpha

        results["alpha_per_layer"][layer_idx] = layer_alpha

        # Run cPCA with selected alpha
        components, eigenvalues = run_cpca(
            target_acts, neutral_acts, layer_alpha, n_components
        )

        results["components"][layer_idx] = components
        results["eigenvalues"][layer_idx] = eigenvalues

    return results


def run_regional_cpca(
    regional_acts: Dict[str, dict],
    regions: List[str],
    metadata: list[dict],
    alpha_per_region: Optional[Dict[str, float]] = None,
    n_components_per_region: int = 50,
    use_diffs: bool = True,
) -> dict:
    """Run cPCA separately per region (user, asst, special tokens).

    Args:
        regional_acts: Dict mapping pair_id -> {
            "neutral": {"user": array, "asst": array, "special1": array, ...},
            "emotional": {"user": array, "asst": array, "special1": array, ...}
        }
        regions: List of region names (e.g., ["user", "asst", "special1", "special2"])
        metadata: List of metadata dicts with 'id' and 'emotion' keys
        alpha_per_region: Optional dict mapping region -> alpha value (None = use 5.0 for all)
        n_components_per_region: Number of cPCs per region
        use_diffs: If True, use diffs as target

    Returns:
        Results dict with per-region components

    Raises:
        ValueError: If data format invalid
        KeyError: If regions missing
    """
    if not regional_acts:
        raise ValueError("regional_acts dict is empty")

    if not regions:
        raise ValueError("regions list is empty")

    if not metadata:
        raise ValueError("metadata list is empty")

    # Validate regions exist
    pair_ids = list(regional_acts.keys())
    first_pair = next(iter(regional_acts.values()))

    for condition in ["neutral", "emotional"]:
        if condition not in first_pair:
            raise KeyError(f"First pair missing '{condition}' key")

        for region in regions:
            if region not in first_pair[condition]:
                raise KeyError(
                    f"Region '{region}' not found in {condition} activations. "
                    f"Available: {list(first_pair[condition].keys())}"
                )

    # Get layer info from first region
    first_region = first_pair["neutral"][regions[0]]
    num_layers = first_region.shape[0]

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

    results = {
        "components_per_region": {},  # region -> {layer -> components}
        "eigenvalues_per_region": {},  # region -> {layer -> eigenvalues}
        "pair_ids": pair_ids,
        "metadata": metadata,
        "config": {
            "regions": regions,
            "n_components_per_region": n_components_per_region,
            "alpha_per_region": alpha_per_region,
            "num_pairs": len(pair_ids),
            "num_layers": num_layers,
            "use_diffs": use_diffs,
        },
    }

    print(f"Running regional cPCA on {num_layers} layers, {len(regions)} regions...")

    # Process each region separately
    for region in regions:
        print(f"\nRegion: {region} (alpha={alpha_per_region[region]})")

        results["components_per_region"][region] = {}
        results["eigenvalues_per_region"][region] = {}

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

            # Choose target data
            if use_diffs:
                target_acts = emotional_acts - neutral_acts
            else:
                target_acts = emotional_acts

            # Run cPCA for this region
            components, eigenvalues = run_cpca(
                target_acts,
                neutral_acts,
                alpha_per_region[region],
                n_components_per_region,
            )

            results["components_per_region"][region][layer_idx] = components
            results["eigenvalues_per_region"][region][layer_idx] = eigenvalues

    return results
