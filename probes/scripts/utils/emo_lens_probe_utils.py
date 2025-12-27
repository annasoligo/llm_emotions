#!/usr/bin/env python3
"""Utility functions for probe-based emotion detection in emo lens experiments.

This module provides utilities for loading probes, applying cPCA projections,
running probe inference, and computing bootstrap confidence intervals for
emotion detection experiments compatible with the emo lens framework.
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from nnterp import StandardizedTransformer


def load_probe(probe_path: Path) -> Dict:
    """Load a probe pickle file without cPCA components.

    Args:
        probe_path: Path to probe pickle file

    Returns:
        probe_dict: Loaded probe dictionary with 'model', 'label_names', etc.
    """
    with open(probe_path, 'rb') as f:
        probe_dict = pickle.load(f)
    return probe_dict


def load_probe_and_cpca(
    probe_path: Path,
    cpca_path: Path,
    layer: int,
    n_components: int
) -> Tuple[Dict, np.ndarray]:
    """Load probe and corresponding cPCA components.

    Args:
        probe_path: Path to probe pickle file (e.g., probe_layer17_nc10_seed0.pkl)
        cpca_path: Path to cPCA .npz file
        layer: Layer number for cPCA extraction
        n_components: Number of cPCA components to use

    Returns:
        probe_dict: Loaded probe dictionary with 'model', 'label_names', etc.
        cpca_components: cPCA components array [n_components, hidden_dim]
    """
    # Load probe
    print(f"Loading probe from {probe_path}")
    with open(probe_path, 'rb') as f:
        probe_dict = pickle.load(f)

    print(f"Probe info:")
    print(f"  Layer: {probe_dict.get('layer', 'unknown')}")
    print(f"  Test accuracy: {probe_dict.get('test_accuracy', 'unknown'):.4f}")
    print(f"  Label names: {probe_dict['label_names']}")

    # Load cPCA components
    print(f"\nLoading cPCA components from {cpca_path}")
    cpca_data = np.load(cpca_path)
    all_components = cpca_data['components']  # [n_layers, 50, hidden_dim]

    if layer >= all_components.shape[0]:
        raise ValueError(f"Layer {layer} not found in cPCA data (max: {all_components.shape[0]-1})")

    # Extract components for specified layer
    layer_components = all_components[layer, :n_components, :]

    print(f"cPCA components shape: {layer_components.shape}")
    print(f"  [{n_components} components × {layer_components.shape[1]} hidden_dim]")

    return probe_dict, layer_components


def apply_probe_pipeline(
    activations: np.ndarray,
    cpca_components: np.ndarray,
    probe_dict: Dict,
    drop_neutral: bool = True,
    device: str = 'cuda'
) -> np.ndarray:
    """Apply cPCA projection → probe inference → drop neutral class.

    Args:
        activations: Raw activations [n_samples, hidden_dim]
        cpca_components: cPCA components [n_components, hidden_dim]
        probe_dict: Probe dictionary with 'model' key
        drop_neutral: Whether to drop neutral class (index 6)
        device: Device for PyTorch inference ('cuda' or 'cpu')

    Returns:
        Emotion logits [n_samples, 6] if drop_neutral else [n_samples, 7]
    """
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
    probe_model = probe_model.to(device)

    X_tensor = torch.from_numpy(projected).float().to(device)

    with torch.no_grad():
        logits = probe_model(X_tensor)  # [n_samples, 7]

    logits = logits.cpu().numpy()

    # Step 3: Drop neutral class if requested
    if drop_neutral:
        logits = logits[:, :6]  # Drop index 6 (neutral)

    return logits


def compute_bootstrap_ci(
    per_pair_effects: List[Dict[str, float]],
    emotions: List[str],
    n_bootstrap: int = 100,
    ci_percentile: float = 95.0
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals for probe scores.

    Args:
        per_pair_effects: List of dicts, one per prompt pair, with emotion scores
        emotions: List of emotion names (e.g., ['anger', 'disgust', ...])
        n_bootstrap: Number of bootstrap samples
        ci_percentile: Confidence interval percentile (e.g., 95.0)

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


def extract_activations_batch(
    model,
    tokenizer,
    prompts: List[str],
    layer: int,
    system_prompt: str = None,
    activation_strategy: str = "assistant_token",
    num_generated_tokens: int = 10
) -> np.ndarray:
    """Extract activations for a batch of prompts at a specific layer.

    Args:
        model: StandardizedTransformer model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        layer: Layer to extract from
        system_prompt: Optional system prompt
        activation_strategy: Activation extraction strategy
        num_generated_tokens: Number of tokens for 'generated_tokens_avg' strategy

    Returns:
        Activations array [n_prompts, hidden_dim]
    """
    # Import from believe-it-or-not repo (emo lens utils)
    sys.path.insert(0, '/workspace-vast/annas/git/believe-it-or-not')
    from emotion_evals.emo_lens.model_utils import extract_prompt_activations_all_layers

    activations_list = []

    for prompt in prompts:
        acts_dict = extract_prompt_activations_all_layers(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            layers=[layer],
            system_prompt=system_prompt,
            strategy=activation_strategy,
            num_generated_tokens=num_generated_tokens
        )
        activations_list.append(acts_dict[layer])

    return np.array(activations_list)


def extract_activations_batch_multilayer(
    model,
    tokenizer,
    prompts: List[str],
    layers: List[int],
    system_prompt: str = None,
    activation_strategy: str = "assistant_token",
    num_generated_tokens: int = 10
) -> Dict[int, np.ndarray]:
    """Extract activations for a batch of prompts at multiple layers in single forward passes.

    Args:
        model: StandardizedTransformer model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        layers: List of layers to extract from
        system_prompt: Optional system prompt
        activation_strategy: Activation extraction strategy
        num_generated_tokens: Number of tokens for 'generated_tokens_avg' strategy

    Returns:
        Dictionary mapping layer -> activations array [n_prompts, hidden_dim]
    """
    # Import from believe-it-or-not repo (emo lens utils)
    sys.path.insert(0, '/workspace-vast/annas/git/believe-it-or-not')
    from emotion_evals.emo_lens.model_utils import extract_prompt_activations_all_layers

    # Initialize dict to accumulate activations per layer
    layer_activations = {layer: [] for layer in layers}

    for prompt in prompts:
        acts_dict = extract_prompt_activations_all_layers(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            layers=layers,
            system_prompt=system_prompt,
            strategy=activation_strategy,
            num_generated_tokens=num_generated_tokens
        )
        for layer in layers:
            layer_activations[layer].append(acts_dict[layer])

    # Convert lists to arrays
    return {layer: np.array(acts) for layer, acts in layer_activations.items()}
