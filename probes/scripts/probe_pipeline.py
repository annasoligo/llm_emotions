#!/usr/bin/env python3
"""
Core pipeline classes for flexible probe-based emotion detection.

This module provides modular, reusable classes for:
- Extracting activations from models
- Running probe inference with caching
- Aggregating and computing differences
- Creating visualizations

All classes are standalone and don't depend on the emo_lens codebase.
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add scripts directory to path for imports
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))

from activation_extraction import extract_prompt_activations_all_layers


# ============================================================================
# 1. ProbeActivationExtractor - Handles all activation extraction patterns
# ============================================================================

class ProbeActivationExtractor:
    """Handles activation extraction from models with various strategies."""

    def extract_aggregated(
        self,
        model,
        tokenizer,
        prompts: List[str],
        layer: int,
        system_prompt: Optional[str] = None,
        strategy: str = "assistant_token",
        num_generated_tokens: int = 10
    ) -> np.ndarray:
        """Extract activations for a batch of prompts at a single layer with aggregation.

        Args:
            model: StandardizedTransformer model
            tokenizer: Tokenizer
            prompts: List of prompt strings
            layer: Layer to extract from
            system_prompt: Optional system prompt
            strategy: Activation extraction strategy (assistant_token, etc.)
            num_generated_tokens: Number of tokens for generated_tokens_avg

        Returns:
            Activations array [n_prompts, hidden_dim]
        """
        activations_list = []

        for prompt in prompts:
            acts_dict = extract_prompt_activations_all_layers(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                layers=[layer],
                system_prompt=system_prompt,
                strategy=strategy,
                num_generated_tokens=num_generated_tokens
            )
            activations_list.append(acts_dict[layer])

        return np.array(activations_list)

    def extract_batch_multilayer(
        self,
        model,
        tokenizer,
        prompts: List[str],
        layers: List[int],
        system_prompt: Optional[str] = None,
        strategy: str = "assistant_token",
        num_generated_tokens: int = 10
    ) -> Dict[int, np.ndarray]:
        """Extract activations for a batch of prompts at multiple layers efficiently.

        Extracts all layers in a single forward pass per prompt.

        Args:
            model: StandardizedTransformer model
            tokenizer: Tokenizer
            prompts: List of prompt strings
            layers: List of layers to extract from
            system_prompt: Optional system prompt
            strategy: Activation extraction strategy
            num_generated_tokens: Number of tokens for generated_tokens_avg

        Returns:
            Dictionary mapping layer -> activations array [n_prompts, hidden_dim]
        """
        # Initialize dict to accumulate activations per layer
        layer_activations = {layer: [] for layer in layers}

        for prompt in prompts:
            acts_dict = extract_prompt_activations_all_layers(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                layers=layers,
                system_prompt=system_prompt,
                strategy=strategy,
                num_generated_tokens=num_generated_tokens
            )
            for layer in layers:
                layer_activations[layer].append(acts_dict[layer])

        # Convert lists to arrays
        return {layer: np.array(acts) for layer, acts in layer_activations.items()}


# ============================================================================
# 2. ProbeInference - Manages probe loading and inference with caching
# ============================================================================

class ProbeInference:
    """Manages probe loading and inference with caching for efficiency."""

    def __init__(self, probe_dir: Path, cpca_path: Path, device: str = 'cuda'):
        """Initialize probe inference with probe directory and cPCA path.

        Args:
            probe_dir: Directory containing probe pickle files
            cpca_path: Path to cPCA .npz file
            device: Device for inference ('cuda' or 'cpu')
        """
        self.probe_dir = Path(probe_dir)
        self.cpca_path = Path(cpca_path)
        self.device = device

        # Cache for loaded probes
        self.probe_cache = {}

        # Load cPCA data once
        print(f"Loading cPCA data from {cpca_path}")
        cpca_data = np.load(cpca_path)
        self.cpca_components_all = cpca_data['components']  # [n_layers, n_components, hidden_dim]
        print(f"  Loaded cPCA components: {self.cpca_components_all.shape}")

    def load_probe(self, layer: int, n_components: int, seed: int) -> Dict:
        """Load probe with caching.

        Args:
            layer: Layer number
            n_components: Number of cPCA components
            seed: Random seed

        Returns:
            Probe dictionary with 'model', 'label_names', etc.
        """
        cache_key = (layer, n_components, seed)

        if cache_key not in self.probe_cache:
            probe_path = self.probe_dir / f"probe_layer{layer}_nc{n_components}_seed{seed}.pkl"
            print(f"Loading probe from {probe_path}")

            with open(probe_path, 'rb') as f:
                probe_dict = pickle.load(f)

            self.probe_cache[cache_key] = probe_dict
            print(f"  Cached probe (layer={layer}, nc={n_components}, seed={seed})")

        return self.probe_cache[cache_key]

    def get_cpca_components(self, layer: int, n_components: int) -> np.ndarray:
        """Get cPCA components for a specific layer.

        Args:
            layer: Layer number
            n_components: Number of components to use

        Returns:
            cPCA components [n_components, hidden_dim]
        """
        if layer >= self.cpca_components_all.shape[0]:
            raise ValueError(f"Layer {layer} not found in cPCA data (max: {self.cpca_components_all.shape[0]-1})")

        return self.cpca_components_all[layer, :n_components, :]

    def predict(
        self,
        activations: np.ndarray,
        layer: int,
        n_components: int = 10,
        seed: int = 0,
        drop_neutral: bool = True
    ) -> np.ndarray:
        """Run probe inference on activations.

        Pipeline: cPCA projection → probe inference → drop neutral class

        Args:
            activations: Raw activations [n_samples, hidden_dim]
            layer: Layer number (for loading probe and cPCA)
            n_components: Number of cPCA components
            seed: Probe random seed
            drop_neutral: Whether to drop neutral class (index 6)

        Returns:
            Emotion logits [n_samples, 6] if drop_neutral else [n_samples, 7]
        """
        # Get cPCA components
        cpca_components = self.get_cpca_components(layer, n_components)

        # Load probe
        probe_dict = self.load_probe(layer, n_components, seed)

        # Validate dimensions
        if activations.shape[1] != cpca_components.shape[1]:
            raise ValueError(
                f"Dimension mismatch: activations have {activations.shape[1]} dims, "
                f"cPCA expects {cpca_components.shape[1]} dims"
            )

        # Step 1: Apply cPCA projection
        projected = activations @ cpca_components.T  # [n_samples, n_components]

        # Step 2: Run probe inference
        probe_model = probe_dict['model']
        probe_model.eval()
        probe_model = probe_model.to(self.device)

        X_tensor = torch.from_numpy(projected).float().to(self.device)

        with torch.no_grad():
            logits = probe_model(X_tensor)  # [n_samples, 7]

        logits = logits.cpu().numpy()

        # Step 3: Drop neutral class if requested
        if drop_neutral:
            logits = logits[:, :6]  # Drop index 6 (neutral)

        return logits

    def predict_batch(
        self,
        activations_by_layer: Dict[int, np.ndarray],
        n_components: int = 10,
        seed: int = 0,
        drop_neutral: bool = True
    ) -> Dict[int, np.ndarray]:
        """Run probe inference on multiple layers.

        Args:
            activations_by_layer: Dict mapping layer -> activations [n_samples, hidden_dim]
            n_components: Number of cPCA components
            seed: Probe random seed
            drop_neutral: Whether to drop neutral class

        Returns:
            Dict mapping layer -> logits [n_samples, 6 or 7]
        """
        results = {}
        for layer, activations in activations_by_layer.items():
            results[layer] = self.predict(
                activations, layer, n_components, seed, drop_neutral
            )
        return results


# ============================================================================
# 3. ProbeAggregator - Handles aggregation and differencing operations
# ============================================================================

class ProbeAggregator:
    """Handles aggregation and differencing operations for probe scores."""

    @staticmethod
    def compute_diff(
        scores_a: np.ndarray,
        scores_b: np.ndarray
    ) -> np.ndarray:
        """Compute simple difference between two score arrays.

        Args:
            scores_a: First score array [n_samples, n_emotions]
            scores_b: Second score array [n_samples, n_emotions]

        Returns:
            Difference scores_a - scores_b [n_samples, n_emotions]
        """
        return scores_a - scores_b

    @staticmethod
    def compute_double_diff(
        ft_dataset: np.ndarray,
        base_dataset: np.ndarray,
        ft_baseline: np.ndarray,
        base_baseline: np.ndarray,
        emotions: List[str],
        n_bootstrap: int = 100,
        ci_percentile: float = 95.0
    ) -> Dict:
        """Compute case 4 double difference with bootstrap CI.

        Formula: (ft_dataset - base_dataset) - (ft_baseline - base_baseline)

        Args:
            ft_dataset: Finetuned model on dataset prompts [n_pairs, n_emotions]
            base_dataset: Base model on dataset prompts [n_pairs, n_emotions]
            ft_baseline: Finetuned model on baseline prompts [n_pairs, n_emotions]
            base_baseline: Base model on baseline prompts [n_pairs, n_emotions]
            emotions: List of emotion names
            n_bootstrap: Number of bootstrap samples
            ci_percentile: Confidence interval percentile

        Returns:
            Dict with 'mean', 'bootstrap_ci', 'per_prompt_diffs', 'emotion_ranking'
        """
        # Compute double diff
        D_ft = ft_dataset - ft_baseline  # Finetuned effect
        D_base = base_dataset - base_baseline  # Base effect
        DD = D_ft - D_base  # Double diff

        n_pairs = DD.shape[0]

        # Mean across prompts
        mean_effect = {emotions[j]: float(np.mean(DD[:, j])) for j in range(len(emotions))}

        # Per-pair effects
        per_pair_effects = [
            {emotions[j]: float(DD[i, j]) for j in range(len(emotions))}
            for i in range(n_pairs)
        ]

        # Bootstrap confidence intervals
        bootstrap_ci = ProbeAggregator.bootstrap_ci(
            per_pair_effects, emotions, n_bootstrap, ci_percentile
        )

        # Emotion ranking by absolute mean
        emotion_ranking = sorted(
            [(emotion, abs(mean_effect[emotion])) for emotion in emotions],
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'mean': mean_effect,
            'bootstrap_ci': bootstrap_ci,
            'per_prompt_diffs': per_pair_effects,
            'emotion_ranking': emotion_ranking,
        }

    @staticmethod
    def bootstrap_ci(
        per_pair_effects: List[Dict[str, float]],
        emotions: List[str],
        n_bootstrap: int = 100,
        ci_percentile: float = 95.0
    ) -> Dict[str, Dict[str, float]]:
        """Compute bootstrap confidence intervals for probe scores.

        Args:
            per_pair_effects: List of dicts, one per prompt pair, with emotion scores
            emotions: List of emotion names
            n_bootstrap: Number of bootstrap samples
            ci_percentile: Confidence interval percentile

        Returns:
            Dict mapping emotion → {'lower': float, 'upper': float}
        """
        n_pairs = len(per_pair_effects)

        # Convert to array for easy resampling
        effects_array = np.array([
            [pair_dict[emotion] for emotion in emotions]
            for pair_dict in per_pair_effects
        ])  # [n_pairs, n_emotions]

        bootstrap_means = []

        # Bootstrap resampling
        rng = np.random.RandomState(42)
        for _ in range(n_bootstrap):
            # Resample pairs with replacement
            indices = rng.choice(n_pairs, size=n_pairs, replace=True)
            resampled = effects_array[indices]

            # Compute mean effect for this resample
            bootstrap_mean = resampled.mean(axis=0)  # [n_emotions]
            bootstrap_means.append(bootstrap_mean)

        bootstrap_means = np.array(bootstrap_means)  # [n_bootstrap, n_emotions]

        # Compute percentile-based confidence intervals
        lower_p = (100 - ci_percentile) / 2
        upper_p = 100 - lower_p

        ci_dict = {}
        for i, emotion in enumerate(emotions):
            lower = np.percentile(bootstrap_means[:, i], lower_p)
            upper = np.percentile(bootstrap_means[:, i], upper_p)
            ci_dict[emotion] = {'lower': float(lower), 'upper': float(upper)}

        return ci_dict


# ============================================================================
# 4. ProbeVisualizer - Creates visualizations
# ============================================================================

class ProbeVisualizer:
    """Creates visualizations for probe experiment results."""

    def __init__(self, output_dir: Path):
        """Initialize visualizer with output directory.

        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Import visualization functions
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from visualization.plot_probe_results import (
            plot_multilayer_results,
            plot_singlelayer_results,
            EMOTION_COLORS
        )
        self._plot_multilayer = plot_multilayer_results
        self._plot_singlelayer = plot_singlelayer_results
        self.emotion_colors = EMOTION_COLORS

    def plot_results(
        self,
        results: Dict,
        experiment_name: str = "probe_experiment",
        layer_range: Tuple[int, int] = (20, 50)
    ):
        """Plot probe experiment results (dispatches to multilayer or singlelayer).

        Args:
            results: Results dictionary
            experiment_name: Name prefix for plot files
            layer_range: (min_layer, max_layer) for multilayer plots
        """
        if results.get('multilayer', False):
            self._plot_multilayer(results, self.output_dir, experiment_name, layer_range)
        else:
            self._plot_singlelayer(results, self.output_dir, experiment_name)

    def plot_heatmap(
        self,
        results: Dict,
        experiment_name: str = "probe_experiment",
        layer_range: Tuple[int, int] = (20, 50)
    ):
        """Plot only the heatmap from multilayer results.

        Args:
            results: Results dictionary with 'results_by_layer'
            experiment_name: Name prefix for plot file
            layer_range: (min_layer, max_layer) to restrict plot range
        """
        # Just call the full function - it generates both plots
        # Could be split in the future if needed
        self._plot_multilayer(results, self.output_dir, experiment_name, layer_range)

    def plot_trajectories(
        self,
        results: Dict,
        experiment_name: str = "probe_experiment",
        layer_range: Tuple[int, int] = (20, 50)
    ):
        """Plot only the trajectories from multilayer results.

        Args:
            results: Results dictionary with 'results_by_layer'
            experiment_name: Name prefix for plot file
            layer_range: (min_layer, max_layer) to restrict plot range
        """
        # Just call the full function - it generates both plots
        # Could be split in the future if needed
        self._plot_multilayer(results, self.output_dir, experiment_name, layer_range)
