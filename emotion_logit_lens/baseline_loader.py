"""
Baseline statistics loader for logit-based emotion detection.

This module provides utilities for loading and applying baseline statistics
for z-score normalization of emotion logits.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


class LogitBaselineLoader:
    """
    Load baseline statistics for logit-based emotion detection.

    Baseline format (per layer JSON file):
        {
            "layer": N,
            "num_samples": 512,
            "statistics": {
                "token_id": {"mean": ..., "std": ...},
                ...
            }
        }
    """

    def __init__(
        self,
        baseline_dir: Optional[Path] = None,
        model_name: str = "google_gemma_3_27b_it",
        baseline_file: Optional[Path] = None
    ):
        """
        Initialize baseline loader.

        Args:
            baseline_dir: Base directory containing baseline statistics (multi-file format)
                         Should contain model-specific subdirectories
            model_name: Model name (e.g., "google_gemma_3_27b_it")
                       Will be used to construct path to model-specific baselines
            baseline_file: Single baseline file (format 3.0). If provided, overrides baseline_dir.
        """
        self._cache = {}  # Cache loaded layer stats
        self._single_file_data = None  # For format 3.0

        if baseline_file:
            # Format 3.0: single file
            self.baseline_file = Path(baseline_file)
            self.baseline_dir = None
            self._load_single_file()
        else:
            # Multi-file format
            self.baseline_file = None
            self.baseline_dir = Path(baseline_dir)

            # If baseline_dir doesn't end with model name, append it
            if self.baseline_dir.name != model_name:
                self.baseline_dir = self.baseline_dir / model_name

    def _load_single_file(self):
        """Load format 3.0 single baseline file."""
        if not self.baseline_file.exists():
            raise FileNotFoundError(f"Baseline file not found: {self.baseline_file}")

        with open(self.baseline_file) as f:
            self._single_file_data = json.load(f)

        # Pre-cache all layers from single file
        layers_data = self._single_file_data.get('layers_data', {})
        for layer_key, layer_data in layers_data.items():
            layer_num = int(layer_key)
            self._cache[layer_num] = layer_data.get('statistics', {})

    def load_layer_stats(self, layer: int) -> Dict[str, Dict[str, float]]:
        """
        Load baseline statistics for a specific layer.

        Args:
            layer: Layer number

        Returns:
            Dictionary mapping token_id (as string) -> {'mean': float, 'std': float}

        Raises:
            FileNotFoundError: If baseline stats not found for this layer
        """
        if layer in self._cache:
            return self._cache[layer]

        # Single file format
        if self._single_file_data:
            layers_data = self._single_file_data.get('layers_data', {})
            layer_key = str(layer)
            if layer_key in layers_data:
                self._cache[layer] = layers_data[layer_key].get('statistics', {})
                return self._cache[layer]
            else:
                raise FileNotFoundError(f"Layer {layer} not found in baseline file")

        # Multi-file format
        layer_file = self.baseline_dir / f"layer_{layer}.json"

        if not layer_file.exists():
            raise FileNotFoundError(
                f"Baseline stats not found for layer {layer}: {layer_file}\n"
                f"Available files: {list(self.baseline_dir.glob('layer_*.json'))}"
            )

        with open(layer_file) as f:
            data = json.load(f)

        # Cache the statistics dictionary
        self._cache[layer] = data['statistics']
        return self._cache[layer]

    def get_token_stats(
        self,
        layer: int,
        token_id: int
    ) -> Optional[Dict[str, float]]:
        """
        Get mean/std for a specific token at a specific layer.

        Args:
            layer: Layer number
            token_id: Token ID

        Returns:
            Dictionary with 'mean' and 'std' keys, or None if token not found
        """
        try:
            stats = self.load_layer_stats(layer)
            return stats.get(str(token_id))
        except FileNotFoundError:
            return None

    def normalize_logit(
        self,
        logit_value: float,
        token_id: int,
        layer: int
    ) -> float:
        """
        Z-score normalize a single logit value.

        Normalization: z = (logit - mean) / std

        Args:
            logit_value: Raw logit value
            token_id: Token ID
            layer: Layer number

        Returns:
            Z-score normalized logit value
            If baseline not available, returns raw logit
        """
        token_stats = self.get_token_stats(layer, token_id)

        if token_stats is None:
            # No baseline available - return raw logit
            return logit_value

        mean = token_stats['mean']
        std = token_stats['std']

        # Avoid division by zero
        if std < 1e-8:
            return logit_value - mean

        return (logit_value - mean) / std

    def get_available_layers(self) -> List[int]:
        """
        Get list of available layer numbers.

        Returns:
            Sorted list of layer numbers with baseline statistics
        """
        if not self.baseline_dir.exists():
            return []

        layer_files = sorted(self.baseline_dir.glob("layer_*.json"))
        layers = []

        for f in layer_files:
            try:
                # Extract layer number from filename: layer_30.json -> 30
                layer_num = int(f.stem.split('_')[1])
                layers.append(layer_num)
            except (ValueError, IndexError):
                continue

        return sorted(layers)

    def load_metadata(self) -> Optional[Dict]:
        """
        Load metadata file if available.

        Returns:
            Metadata dictionary or None if not found
        """
        metadata_file = self.baseline_dir / "metadata.json"

        if not metadata_file.exists():
            return None

        with open(metadata_file) as f:
            return json.load(f)

    def __repr__(self) -> str:
        """String representation for debugging."""
        num_layers = len(self.get_available_layers())
        return (
            f"LogitBaselineLoader(baseline_dir={self.baseline_dir}, "
            f"num_layers={num_layers})"
        )
