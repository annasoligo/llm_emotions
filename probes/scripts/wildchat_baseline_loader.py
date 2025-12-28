#!/usr/bin/env python3
"""
WildChat Baseline Loader

Utility for loading pre-computed WildChat baseline statistics on-the-fly
and normalizing activations before probe application.
"""

import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np


class WildChatBaselineLoader:
    """Load and apply WildChat baseline normalization to activations."""

    def __init__(
        self,
        baseline_dir: Path = Path("/workspace-vast/annas/git/research-tools/data/baselines/wildchat/google_gemma_3_27b_it"),
        aggregation_type: str = "assistant_turn"
    ):
        """
        Initialize the baseline loader.

        Args:
            baseline_dir: Directory containing layer{N}_stats.json files
            aggregation_type: Which aggregation to use for baseline stats
                            Options: all_tokens, user_turn, assistant_turn,
                                    last_user_token, first_assistant_token, between_turns
        """
        self.baseline_dir = Path(baseline_dir)
        self.aggregation_type = aggregation_type
        self._cache = {}  # Cache loaded stats by layer

        if not self.baseline_dir.exists():
            raise FileNotFoundError(
                f"Baseline directory not found: {baseline_dir}\n"
                f"Please run compute_wildchat_baseline_activations.py first."
            )

    def load_layer_stats(self, layer: int) -> Dict[str, np.ndarray]:
        """
        Load baseline statistics for a specific layer.

        Args:
            layer: Layer number (0-indexed)

        Returns:
            Dict with 'mean' and 'std' arrays [hidden_dim]
        """
        if layer in self._cache:
            return self._cache[layer]

        stats_file = self.baseline_dir / f"layer{layer}_stats.json"

        if not stats_file.exists():
            raise FileNotFoundError(
                f"Stats file not found: {stats_file}\n"
                f"Available aggregation types: all_tokens, user_turn, assistant_turn, "
                f"last_user_token, first_assistant_token, between_turns"
            )

        with open(stats_file) as f:
            stats = json.load(f)

        if self.aggregation_type not in stats['aggregations']:
            raise ValueError(
                f"Aggregation type '{self.aggregation_type}' not found in stats.\n"
                f"Available: {list(stats['aggregations'].keys())}"
            )

        agg_stats = stats['aggregations'][self.aggregation_type]

        # Convert to arrays
        mean = np.array(agg_stats['mean'])
        std = np.array(agg_stats['std'])

        # Cache for future use
        self._cache[layer] = {'mean': mean, 'std': std}

        return self._cache[layer]

    def normalize_activations(
        self,
        activations: np.ndarray,
        layer: int,
        epsilon: float = 1e-8
    ) -> np.ndarray:
        """
        Z-score normalize activations using WildChat baseline statistics.

        Args:
            activations: Activation array [n_samples, hidden_dim] or [hidden_dim]
            layer: Layer number (0-indexed)
            epsilon: Small value to avoid division by zero

        Returns:
            Normalized activations with same shape as input
        """
        stats = self.load_layer_stats(layer)
        mean = stats['mean']
        std = stats['std']

        # Handle single sample vs batch
        original_shape = activations.shape
        if activations.ndim == 1:
            activations = activations[np.newaxis, :]  # [1, hidden_dim]

        # Z-score normalization
        normalized = (activations - mean) / (std + epsilon)

        # Restore original shape if needed
        if len(original_shape) == 1:
            normalized = normalized[0]

        return normalized

    def normalize_batch_multilayer(
        self,
        activations_by_layer: Dict[int, np.ndarray],
        epsilon: float = 1e-8
    ) -> Dict[int, np.ndarray]:
        """
        Normalize activations for multiple layers.

        Args:
            activations_by_layer: Dict mapping layer -> activations [n_samples, hidden_dim]
            epsilon: Small value to avoid division by zero

        Returns:
            Dict mapping layer -> normalized activations [n_samples, hidden_dim]
        """
        return {
            layer: self.normalize_activations(acts, layer, epsilon)
            for layer, acts in activations_by_layer.items()
        }

    def normalize_token_level(
        self,
        activations_by_token: Dict[int, Dict[int, np.ndarray]],
        layers: list,
        aggregation: str = "mean",
        epsilon: float = 1e-8
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Normalize token-level activations using WildChat baselines.

        For token-level analysis, we use layer-averaged baseline statistics
        because we want a single normalization reference across all layers.
        This enables cross-layer comparison of emotion trajectories.

        Args:
            activations_by_token: Dict mapping token_pos -> {layer: activation [hidden_dim]}
            layers: List of layer numbers to normalize
            aggregation: How to aggregate baseline stats across layers
                        - "mean": Average mean/std across layers (default)
                        - "per_layer": Use per-layer normalization
            epsilon: Small value to avoid division by zero

        Returns:
            Dict mapping token_pos -> {layer: normalized_activation [hidden_dim]}

        Example:
            >>> extractor = ProbeActivationExtractor()
            >>> acts_by_token, token_ids = extractor.extract_token_level(...)
            >>> loader = WildChatBaselineLoader(aggregation_type="assistant_turn")
            >>> normalized = loader.normalize_token_level(acts_by_token, layers=[20, 30, 40])
        """
        if aggregation == "mean":
            # Compute layer-averaged baseline statistics
            all_means = []
            all_stds = []

            for layer in layers:
                stats = self.load_layer_stats(layer)
                all_means.append(stats['mean'])
                all_stds.append(stats['std'])

            # Average across layers
            mean_avg = np.mean(all_means, axis=0)  # [hidden_dim]
            std_avg = np.mean(all_stds, axis=0)    # [hidden_dim]

            # Normalize all tokens and layers with same baseline
            normalized = {}
            for token_pos, layer_acts in activations_by_token.items():
                normalized[token_pos] = {}
                for layer, act in layer_acts.items():
                    normalized[token_pos][layer] = (act - mean_avg) / (std_avg + epsilon)

            return normalized

        elif aggregation == "per_layer":
            # Use per-layer normalization (different baseline per layer)
            normalized = {}
            for token_pos, layer_acts in activations_by_token.items():
                normalized[token_pos] = {}
                for layer, act in layer_acts.items():
                    normalized[token_pos][layer] = self.normalize_activations(
                        act, layer, epsilon
                    )

            return normalized

        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}. Use 'mean' or 'per_layer'.")

    def get_available_layers(self) -> list:
        """Get list of available layer numbers."""
        stats_files = sorted(self.baseline_dir.glob("layer*_stats.json"))
        layers = []
        for f in stats_files:
            try:
                layer_num = int(f.stem.replace("layer", "").replace("_stats", ""))
                layers.append(layer_num)
            except ValueError:
                continue
        return sorted(layers)

    def get_aggregation_types(self, layer: int = 0) -> list:
        """Get available aggregation types for a layer."""
        stats_file = self.baseline_dir / f"layer{layer}_stats.json"
        if not stats_file.exists():
            return []

        with open(stats_file) as f:
            stats = json.load(f)

        return list(stats['aggregations'].keys())


# Convenience function for quick usage
def normalize_with_wildchat(
    activations: np.ndarray,
    layer: int,
    aggregation_type: str = "assistant_turn",
    baseline_dir: Optional[Path] = None
) -> np.ndarray:
    """
    Convenience function to normalize activations with WildChat baselines.

    Args:
        activations: Activation array [n_samples, hidden_dim] or [hidden_dim]
        layer: Layer number (0-indexed)
        aggregation_type: Which aggregation to use for baseline stats
        baseline_dir: Optional custom baseline directory

    Returns:
        Normalized activations with same shape as input
    """
    loader = WildChatBaselineLoader(
        baseline_dir=baseline_dir or Path("/workspace-vast/annas/git/research-tools/data/baselines/wildchat/google_gemma_3_27b_it"),
        aggregation_type=aggregation_type
    )
    return loader.normalize_activations(activations, layer)