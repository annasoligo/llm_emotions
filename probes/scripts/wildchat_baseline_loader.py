#!/usr/bin/env python3
"""
Baseline Loader for Probe Normalization

Utility for loading pre-computed baseline statistics (Alpaca dataset by default)
and normalizing probe scores using z-score normalization.

Note: Class name is WildChatBaselineLoader for historical reasons, but it works
with any baseline dataset (currently defaults to Alpaca).
"""

import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np


class WildChatBaselineLoader:
    """
    Load and apply baseline normalization to probe scores.

    Despite the name, this class works with any baseline dataset.
    Current default: Alpaca V2 (neutral instruction-following baseline).
    """

    def __init__(
        self,
        baseline_dir: Path = Path("/workspace-vast/annas/git/research-tools/data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it"),
        aggregation_type: str = "all_tokens"
    ):
        """
        Initialize the baseline loader.

        Args:
            baseline_dir: Directory containing layer{N}_activations.h5 files
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
                f"Please ensure baseline activations have been computed."
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

        # Load from h5 file which contains per-dimension statistics
        h5_file = self.baseline_dir / f"layer{layer}_activations.h5"

        if not h5_file.exists():
            raise FileNotFoundError(
                f"Activation file not found: {h5_file}\n"
                f"Available aggregation types: all_tokens, user_turn, assistant_turn, "
                f"last_user_token, first_assistant_token, between_turns"
            )

        import h5py

        with h5py.File(h5_file, 'r') as f:
            if self.aggregation_type not in f.keys():
                raise ValueError(
                    f"Aggregation type '{self.aggregation_type}' not found in h5 file.\n"
                    f"Available: {list(f.keys())}"
                )

            # Load activations and compute per-dimension mean and std
            activations = f[self.aggregation_type][:]  # [n_samples, hidden_dim]
            mean = np.mean(activations, axis=0)  # [hidden_dim]
            std = np.std(activations, axis=0)    # [hidden_dim]

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
        h5_files = sorted(self.baseline_dir.glob("layer*_activations.h5"))
        layers = []
        for f in h5_files:
            try:
                layer_num = int(f.stem.replace("layer", "").replace("_activations", ""))
                layers.append(layer_num)
            except ValueError:
                continue
        return sorted(layers)

    def get_aggregation_types(self, layer: int = 0) -> list:
        """Get available aggregation types for a layer."""
        h5_file = self.baseline_dir / f"layer{layer}_activations.h5"
        if not h5_file.exists():
            return []

        import h5py
        with h5py.File(h5_file, 'r') as f:
            return list(f.keys())

    def compute_probe_score_baselines(
        self,
        probe_inference,
        layers: list,
        probe_type: str = "orthogonal",
        aggregation: str = "mean",
        return_std: bool = False,
        **probe_kwargs
    ) -> Dict[int, np.ndarray]:
        """
        Compute expected probe scores on baseline activations.

        This computes what probe scores we would expect on "typical" WildChat
        conversations, which can then be subtracted from observed scores to get
        relative emotion levels (positive = above baseline, negative = below baseline).

        Args:
            probe_inference: ProbeInference instance configured with probe paths
            layers: List of layer numbers to compute baselines for
            probe_type: "orthogonal", "standard", or "linear"
            aggregation: How to aggregate across layers ("mean" or "per_layer")
            return_std: If True, return dict with 'mean' and 'std' keys. If False, return just mean scores (backward compatible)
            **probe_kwargs: Additional arguments for probe loading (e.g., orthogonality_weight)

        Returns:
            If return_std=False:
                Dict mapping layer -> baseline_scores [n_emotions]
                If aggregation="mean", returns single-key dict {-1: scores} for layer-averaged baseline
            If return_std=True:
                Dict with keys 'mean' and 'std', each mapping layer -> scores [n_emotions]

        Example:
            >>> from probes.scripts.probe_pipeline import ProbeInference
            >>> loader = WildChatBaselineLoader(aggregation_type="assistant_turn")
            >>> inference = ProbeInference(probe_dir=Path(...), cpca_path=Path(...))
            >>> baselines = loader.compute_probe_score_baselines(
            ...     inference, layers=list(range(30, 60)), probe_type="orthogonal"
            ... )
            >>> # Returns: {-1: array([score_anger, score_disgust, ...])}
            >>>
            >>> # With std:
            >>> baselines = loader.compute_probe_score_baselines(
            ...     inference, layers=list(range(30, 60)), probe_type="orthogonal", return_std=True
            ... )
            >>> # Returns: {'mean': {-1: array(...)}, 'std': {-1: array(...)}}
        """
        from probes.scripts.probe_pipeline import ProbeInference
        import h5py

        if aggregation == "mean":
            if return_std:
                # NEW PATH: Load all baseline activations and compute probe score statistics
                # Load activations from h5 files for all layers
                emotions = probe_kwargs.get('emotions', ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'])

                # We'll collect scores across all samples for each layer, then aggregate
                all_layer_scores = []  # List of [n_samples, n_emotions] arrays

                for layer in layers:
                    h5_file = self.baseline_dir / f"layer{layer}_activations.h5"

                    with h5py.File(h5_file, 'r') as f:
                        if self.aggregation_type not in f.keys():
                            raise ValueError(
                                f"Aggregation type '{self.aggregation_type}' not found in layer {layer}.\n"
                                f"Available: {list(f.keys())}"
                            )

                        # Load all baseline activations for this layer
                        baseline_acts = f[self.aggregation_type][:]  # [n_samples, hidden_dim]

                    # Apply probes to all baseline samples for this layer
                    if probe_type == "orthogonal":
                        representation = probe_kwargs.get('orthogonal_representation', 'raw')
                        probe_data = probe_inference.load_orthogonal_probe(
                            layer=layer,
                            representation=representation,
                            n_components=probe_kwargs.get('n_components'),
                            orthogonality_weight=probe_kwargs.get('orthogonality_weight', 1000.0)
                        )

                        user_probes = probe_data['final_user_probes']
                        asst_probes = probe_data['final_asst_probes']

                        # Project through cPCA if needed
                        acts_for_probes = baseline_acts
                        if representation != 'raw' and 'cpca' in representation.lower():
                            # Extract n_components from representation string (e.g., "global_cpca_top20" -> 20)
                            n_components = probe_kwargs.get('n_components')
                            if n_components is None:
                                # Try to parse from representation string
                                if 'top' in representation:
                                    n_components = int(representation.split('top')[-1])
                                else:
                                    raise ValueError(f"Cannot determine n_components from representation: {representation}")

                            # Get cPCA components and project
                            cpca_components = probe_inference.get_cpca_components(layer, n_components)
                            acts_for_probes = baseline_acts @ cpca_components.T  # [n_samples, n_components]

                        # Apply probes to all samples
                        user_scores, asst_scores = probe_inference.predict_orthogonal(
                            acts_for_probes,  # [n_samples, hidden_dim] or [n_samples, n_components]
                            user_probes, asst_probes, emotions
                        )

                        # Convert to arrays and average user/assistant
                        user_arr = np.array([[s[e] for e in emotions] for s in user_scores])  # [n_samples, n_emotions]
                        asst_arr = np.array([[s[e] for e in emotions] for s in asst_scores])  # [n_samples, n_emotions]
                        layer_scores = (user_arr + asst_arr) / 2  # [n_samples, n_emotions]

                    elif probe_type == "linear":
                        # Apply linear probes to all samples
                        scores = probe_inference.predict(
                            activations=baseline_acts,
                            layer=layer,
                            n_components=probe_kwargs.get('n_components', 10),
                            seed=probe_kwargs.get('seed', 0),
                            drop_neutral=True
                        )
                        layer_scores = scores  # [n_samples, n_emotions]

                    elif probe_type == "standard":
                        # Load and apply standard probes
                        import pickle
                        import torch

                        probe_filename = probe_kwargs.get('probe_pattern', '').format(layer=layer)
                        probe_path = probe_inference.probe_dir / probe_filename

                        with open(probe_path, 'rb') as f:
                            probe_dict = pickle.load(f)

                        probe_model = probe_dict['model']
                        probe_model.eval()
                        probe_model = probe_model.to(probe_inference.device)

                        with torch.no_grad():
                            baseline_tensor = torch.tensor(baseline_acts, dtype=torch.float32).to(probe_inference.device)
                            logits = probe_model(baseline_tensor)
                            layer_scores = logits.cpu().numpy()  # [n_samples, n_emotions]

                    elif probe_type == "centroid":
                        # Load centroid probe for this layer
                        centroid_data = probe_inference.load_centroid_probe(
                            layer=layer,
                            k_value=probe_kwargs.get('k_value'),
                            orthogonality_weight=probe_kwargs.get('orthogonality_weight', 100000.0),
                            constraint_type=probe_kwargs.get('centroid_constraint_type'),
                            probe_format=probe_kwargs.get('centroid_probe_format', 'auto')
                        )

                        probe_format = centroid_data['probe_format']

                        if probe_format == "conversation":
                            # Conversation-based centroid (returns orthogonal format)
                            centroid_user_probes = centroid_data['centroid_user_probes']
                            centroid_asst_probes = centroid_data['centroid_asst_probes']

                            # Apply centroid probes - returns (user_scores_list, asst_scores_list, avg_scores_array)
                            user_scores_list, asst_scores_list, avg_scores = probe_inference.predict_centroid(
                                activations=baseline_acts,  # [n_samples, hidden_dim]
                                centroid_probes=None,
                                centroid_user_probes=centroid_user_probes,
                                centroid_asst_probes=centroid_asst_probes,
                                emotions=emotions,
                                probe_format="conversation"
                            )

                            # Use averaged scores for baseline
                            layer_scores = avg_scores  # [n_samples, n_emotions]

                        elif probe_format == "text":
                            # Text-based centroid
                            centroid_probes = centroid_data['centroid_probes']

                            scores = probe_inference.predict_centroid(
                                activations=baseline_acts,  # [n_samples, hidden_dim]
                                centroid_probes=centroid_probes,
                                centroid_user_probes=None,
                                centroid_asst_probes=None,
                                emotions=emotions,
                                probe_format="text"
                            )
                            layer_scores = scores  # [n_samples, n_emotions]

                        else:
                            raise ValueError(f"Unknown centroid probe_format: {probe_format}")

                    else:
                        raise ValueError(f"Unknown probe_type: {probe_type}")

                    all_layer_scores.append(layer_scores)  # [n_samples, n_emotions]

                # Stack all layers: [n_layers, n_samples, n_emotions]
                all_layer_scores = np.stack(all_layer_scores, axis=0)

                # Average across layers: [n_samples, n_emotions]
                scores_per_sample = np.mean(all_layer_scores, axis=0)

                # Compute mean and std across samples
                baseline_mean = np.mean(scores_per_sample, axis=0)  # [n_emotions]
                baseline_std = np.std(scores_per_sample, axis=0)    # [n_emotions]

                return {
                    'mean': {-1: baseline_mean},
                    'std': {-1: baseline_std}
                }

            else:
                # ORIGINAL PATH: Just compute mean (backward compatible)
                # Compute layer-averaged baseline activations
                all_means = []
                for layer in layers:
                    stats = self.load_layer_stats(layer)
                    all_means.append(stats['mean'])

                baseline_activation = np.mean(all_means, axis=0)  # [hidden_dim]

                # DON'T normalize the baseline - we want to apply probes to the raw mean activation
                # which represents "typical" WildChat activations. Normalizing would make it ~0.

                # Apply probes to baseline activation (single sample)
                if probe_type == "orthogonal":
                    # Need to load orthogonal probes for all layers and average their outputs
                    all_scores = []
                    emotions = probe_kwargs.get('emotions', ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'])

                    for layer in layers:
                        probe_data = probe_inference.load_orthogonal_probe(
                            layer=layer,
                            representation=probe_kwargs.get('orthogonal_representation', 'raw'),
                            n_components=probe_kwargs.get('n_components'),
                            orthogonality_weight=probe_kwargs.get('orthogonality_weight', 1000.0)
                        )

                        user_probes = probe_data['final_user_probes']
                        asst_probes = probe_data['final_asst_probes']

                        # Normalize baseline activation ONLY for this layer
                        baseline_norm = self.normalize_activations(baseline_activation, layer)

                        # Ensure baseline_norm is 1D array and reshape to 2D for predict_orthogonal
                        baseline_norm = np.atleast_1d(baseline_norm)
                        baseline_norm_2d = baseline_norm.reshape(1, -1)

                        user_scores, asst_scores = probe_inference.predict_orthogonal(
                            baseline_norm_2d,
                            user_probes, asst_probes, emotions
                        )

                        # Average user and assistant scores
                        user_arr = np.array([user_scores[0][e] for e in emotions])
                        asst_arr = np.array([asst_scores[0][e] for e in emotions])
                        avg_scores = (user_arr + asst_arr) / 2
                        all_scores.append(avg_scores)

                    # Average across layers
                    baseline_scores = np.mean(all_scores, axis=0)
                    return {-1: baseline_scores}  # -1 indicates layer-averaged

                elif probe_type == "linear":
                    # Apply linear probes (cPCA-based)
                    all_scores = []
                    for layer in layers:
                        # Normalize baseline for this layer
                        baseline_norm = self.normalize_activations(baseline_activation, layer)

                        # Ensure baseline_norm is 1D array and reshape to 2D
                        baseline_norm = np.atleast_1d(baseline_norm)
                        baseline_norm_2d = baseline_norm.reshape(1, -1)

                        scores = probe_inference.predict(
                            activations=baseline_norm_2d,
                            layer=layer,
                            n_components=probe_kwargs.get('n_components', 10),
                            seed=probe_kwargs.get('seed', 0),
                            drop_neutral=True
                        )
                        all_scores.append(scores[0])

                    baseline_scores = np.mean(all_scores, axis=0)
                    return {-1: baseline_scores}

                elif probe_type == "standard":
                    # Load standard probes and apply
                    import pickle
                    import torch
                    all_scores = []

                    for layer in layers:
                        # Normalize baseline for this layer
                        baseline_norm = self.normalize_activations(baseline_activation, layer)

                        # Ensure baseline_norm is 1D array and reshape to 2D
                        baseline_norm = np.atleast_1d(baseline_norm)
                        baseline_norm_2d = baseline_norm.reshape(1, -1)

                        probe_filename = probe_kwargs.get('probe_pattern', '').format(layer=layer)
                        probe_path = probe_inference.probe_dir / probe_filename

                        with open(probe_path, 'rb') as f:
                            probe_dict = pickle.load(f)

                        probe_model = probe_dict['model']
                        probe_model.eval()
                        probe_model = probe_model.to(probe_inference.device)

                        X_tensor = torch.from_numpy(baseline_norm_2d).float().to(probe_inference.device)

                        with torch.no_grad():
                            logits = probe_model(X_tensor)

                        logits_np = logits.cpu().numpy()[0]

                        # Drop neutral class if present
                        if logits_np.shape[0] == 7:
                            logits_np = logits_np[:6]

                        all_scores.append(logits_np)

                    baseline_scores = np.mean(all_scores, axis=0)
                    return {-1: baseline_scores}

                elif probe_type == "centroid":
                    # Apply centroid probes (K-set orthogonal probes averaged)
                    all_scores = []
                    emotions = probe_kwargs.get('emotions', ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'])

                    for layer in layers:
                        # Load centroid probe for this layer
                        centroid_data = probe_inference.load_centroid_probe(
                            layer=layer,
                            k_value=probe_kwargs.get('k_value'),
                            orthogonality_weight=probe_kwargs.get('orthogonality_weight', 100000.0),
                            constraint_type=probe_kwargs.get('centroid_constraint_type'),
                            probe_format=probe_kwargs.get('centroid_probe_format', 'auto')
                        )

                        probe_format = centroid_data['probe_format']

                        # Normalize baseline activation for this layer
                        baseline_norm = self.normalize_activations(baseline_activation, layer)
                        baseline_norm = np.atleast_1d(baseline_norm)

                        if probe_format == "text":
                            # Text-based centroid
                            centroid_probes = centroid_data['centroid_probes']
                            scores_dict = probe_inference.predict_centroid(
                                activations=baseline_norm,
                                centroid_probes=centroid_probes,
                                centroid_user_probes=None,
                                centroid_asst_probes=None,
                                emotions=emotions,
                                probe_format='text'
                            )
                            scores = np.array([scores_dict[e] for e in emotions])
                            all_scores.append(scores)

                        elif probe_format == "conversation":
                            # Conversation-based centroid
                            user_probes = centroid_data['centroid_user_probes']
                            asst_probes = centroid_data['centroid_asst_probes']

                            user_scores, asst_scores, avg_scores = probe_inference.predict_centroid(
                                activations=baseline_norm,
                                centroid_probes=None,
                                centroid_user_probes=user_probes,
                                centroid_asst_probes=asst_probes,
                                emotions=emotions,
                                probe_format='conversation'
                            )

                            # Average user and assistant scores
                            avg_arr = np.array([avg_scores[e] for e in emotions])
                            all_scores.append(avg_arr)

                    # Average across layers
                    baseline_scores = np.mean(all_scores, axis=0)
                    return {-1: baseline_scores}

                else:
                    raise ValueError(f"Unknown probe_type: {probe_type}")

        elif aggregation == "per_layer":
            # Compute per-layer baselines
            baseline_scores_by_layer = {}

            for layer in layers:
                stats = self.load_layer_stats(layer)
                baseline_activation = stats['mean']  # [hidden_dim]

                # Normalize baseline for this layer
                baseline_norm = self.normalize_activations(baseline_activation, layer)

                # Ensure baseline_norm is 1D array and reshape to 2D
                baseline_norm = np.atleast_1d(baseline_norm)
                baseline_norm_2d = baseline_norm.reshape(1, -1)

                if probe_type == "orthogonal":
                    emotions = probe_kwargs.get('emotions', ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise'])

                    probe_data = probe_inference.load_orthogonal_probe(
                        layer=layer,
                        representation=probe_kwargs.get('orthogonal_representation', 'raw'),
                        n_components=probe_kwargs.get('n_components'),
                        orthogonality_weight=probe_kwargs.get('orthogonality_weight', 1000.0)
                    )

                    user_probes = probe_data['final_user_probes']
                    asst_probes = probe_data['final_asst_probes']

                    user_scores, asst_scores = probe_inference.predict_orthogonal(
                        baseline_norm_2d,
                        user_probes, asst_probes, emotions
                    )

                    user_arr = np.array([user_scores[0][e] for e in emotions])
                    asst_arr = np.array([asst_scores[0][e] for e in emotions])
                    baseline_scores_by_layer[layer] = (user_arr + asst_arr) / 2

                elif probe_type == "linear":
                    scores = probe_inference.predict(
                        activations=baseline_norm_2d,
                        layer=layer,
                        n_components=probe_kwargs.get('n_components', 10),
                        seed=probe_kwargs.get('seed', 0),
                        drop_neutral=True
                    )
                    baseline_scores_by_layer[layer] = scores[0]

                elif probe_type == "standard":
                    import pickle
                    import torch

                    probe_filename = probe_kwargs.get('probe_pattern', '').format(layer=layer)
                    probe_path = probe_inference.probe_dir / probe_filename

                    with open(probe_path, 'rb') as f:
                        probe_dict = pickle.load(f)

                    probe_model = probe_dict['model']
                    probe_model.eval()
                    probe_model = probe_model.to(probe_inference.device)

                    X_tensor = torch.from_numpy(baseline_norm_2d).float().to(probe_inference.device)

                    with torch.no_grad():
                        logits = probe_model(X_tensor)

                    logits_np = logits.cpu().numpy()[0]

                    if logits_np.shape[0] == 7:
                        logits_np = logits_np[:6]

                    baseline_scores_by_layer[layer] = logits_np

                else:
                    raise ValueError(f"Unknown probe_type: {probe_type}")

            return baseline_scores_by_layer

        else:
            raise ValueError(f"Unknown aggregation: {aggregation}. Use 'mean' or 'per_layer'.")


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