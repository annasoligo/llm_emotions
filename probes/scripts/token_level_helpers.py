#!/usr/bin/env python3
"""
Helper functions for token-level emotion analysis.

This module provides reusable functions for running token-level experiments
with different probe types, activation strategies, and question modules.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pickle
import torch
from probes.scripts.probe_pipeline import (
    ProbeActivationExtractor,
    ProbeInference,
    normalize_probe_scores_zscore,
    normalize_probe_scores_center
)
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader


class TokenLevelExperiment:
    """Encapsulates a complete token-level experiment configuration and execution."""

    def __init__(
        self,
        model,
        tokenizer,
        probe_type: str = "orthogonal",
        probe_dir: Path = None,
        cpca_path: Path = None,
        probe_pattern: str = None,
        orthogonality_weight: float = 1000.0,
        orthogonal_representation: str = "raw",
        n_components: int = 10,
        seed: int = 0,
        baseline_dir: Path = None,
        emotions: List[str] = None,
        k_value: Optional[int] = None,
        centroid_constraint_type: Optional[str] = None,
        centroid_probe_format: str = "auto"
    ):
        """
        Initialize experiment configuration.

        Note: Probe scores are ALWAYS z-score normalized using baseline statistics (Alpaca by default).
        This ensures consistent, interpretable scores in standard deviation (σ) units.

        Args:
            model: Model (StandardizedTransformer)
            tokenizer: Tokenizer
            probe_type: "orthogonal", "linear", "standard", or "centroid"
            probe_dir: Directory containing probe files
            cpca_path: Path to cPCA file (for linear probes or cpca representations)
            probe_pattern: Custom pattern for probe filenames (e.g., "probe_layer{layer}_nc0_seed0.pkl")
            orthogonality_weight: Weight for orthogonal probes
            orthogonal_representation: "raw", "global_cpca_top10", etc.
            n_components: Number of cPCA components (for linear probes)
            seed: Random seed (for linear probes)
            baseline_dir: Directory containing baseline statistics for normalization (Alpaca V2 recommended)
            emotions: List of emotion names
            k_value: Number of K-sets to average for centroid probes (required if probe_type='centroid')
            centroid_constraint_type: Optional constraint type for centroid probes (e.g., "gramschmidt")
            centroid_probe_format: Probe format - "auto" (detect), "text", or "conversation"
        """
        self.model = model
        self.tokenizer = tokenizer
        self.probe_type = probe_type
        self.probe_dir = probe_dir
        self.cpca_path = cpca_path
        self.probe_pattern = probe_pattern
        self.orthogonality_weight = orthogonality_weight
        self.orthogonal_representation = orthogonal_representation
        self.n_components = n_components
        self.seed = seed
        self.baseline_dir = baseline_dir
        self.emotions = emotions or ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
        self.k_value = k_value
        self.centroid_constraint_type = centroid_constraint_type
        self.centroid_probe_format = centroid_probe_format

        # Initialize pipeline components
        self.extractor = ProbeActivationExtractor()
        self.inference = ProbeInference(
            probe_dir=probe_dir,
            cpca_path=cpca_path,
            device='cuda'
        )

    def run_experiment(
        self,
        prompt: str,
        layers: List[int],
        system_prompt: Optional[str] = None,
        num_generated_tokens: int = 10,
        start_token_idx: Optional[int] = None,
        assistant_prefill: Optional[str] = None,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        verbose: bool = True,
        cached_activations: Optional[Dict] = None
    ) -> Dict:
        """
        Run complete token-level experiment.

        Args:
            prompt: Single prompt to analyze
            layers: Layers to analyze
            system_prompt: Optional system prompt
            num_generated_tokens: Number of tokens to generate
            start_token_idx: Optional starting token index (None = auto-detect)
            assistant_prefill: Optional text to prefill assistant response
            temperature: Temperature for sampling
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            verbose: Print progress

        Returns:
            Dict with results including:
                - activations_by_token: Raw activations at each token position
                - token_ids: List of token IDs
                - scores_by_token: Emotion scores at each token position
                - config: Experiment configuration
        """
        if verbose:
            print("\n" + "="*80)
            print(f"RUNNING TOKEN-LEVEL EXPERIMENT")
            print("="*80)
            print(f"Probe type: {self.probe_type}")
            print(f"Layers: {len(layers)} layers ({min(layers)}-{max(layers)})\"")

        # Step 1: Extract token-level activations (or use cached)
        if cached_activations is not None:
            if verbose:
                print("\n[1/3] Using cached activations (skipping extraction)...")
            activations_by_token = cached_activations['activations_by_token']
            token_ids = cached_activations['token_ids']
            if verbose:
                print(f"  ✓ Loaded {len(activations_by_token)} cached token positions")
        else:
            if verbose:
                print("\n[1/3] Extracting token-level activations...")

            activations_by_token, token_ids = self.extractor.extract_token_level(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                layers=layers,
                system_prompt=system_prompt,
                num_generated_tokens=num_generated_tokens,
                start_token_idx=start_token_idx,
                assistant_prefill=assistant_prefill,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )

            if verbose:
                print(f"  ✓ Extracted {len(activations_by_token)} token positions")

        # Save raw activations for caching (before normalization)
        import copy
        raw_activations_by_token = copy.deepcopy(activations_by_token)

        # Step 2: Apply probes/scoring
        if verbose:
            print(f"\n[2/3] Applying {self.probe_type} probes...")

        scores_by_token = self._apply_probes(
            activations_by_token, layers, verbose
        )

        # Step 3: Z-score normalize probe scores (ALWAYS applied)
        if verbose:
            print("\n[3/3] Computing and applying probe score z-score normalization...")

        baseline_loader = WildChatBaselineLoader(
            aggregation_type="all_tokens",  # Use general aggregation
            baseline_dir=self.baseline_dir
        )

        # Compute both mean and std for z-score normalization
        baseline_stats = baseline_loader.compute_probe_score_baselines(
            probe_inference=self.inference,
            layers=layers,
            probe_type=self.probe_type,
            aggregation="mean",  # Use layer-averaged baseline
            return_std=True,  # Request std computation
            orthogonality_weight=self.orthogonality_weight,
            orthogonal_representation=self.orthogonal_representation,
            n_components=self.n_components,
            seed=self.seed,
            emotions=self.emotions,
            probe_pattern=self.probe_pattern,
            k_value=self.k_value,
            centroid_constraint_type=self.centroid_constraint_type,
            centroid_probe_format=self.centroid_probe_format
        )
        baseline_scores = baseline_stats['mean']
        baseline_stds = baseline_stats['std']

        baseline_mean_vec = baseline_scores[-1]  # -1 indicates layer-averaged
        baseline_std_vec = baseline_stds[-1]

        # Z-score normalization using shared function
        for token_pos in scores_by_token:
            for layer in layers:
                score = scores_by_token[token_pos][layer]
                scores_by_token[token_pos][layer] = normalize_probe_scores_zscore(
                    score, baseline_mean_vec, baseline_std_vec
                )

        if verbose:
            print(f"  ✓ Z-score normalized scores (mean: {baseline_mean_vec[:3]}..., std: {baseline_std_vec[:3]}...)")

        if verbose:
            print("\n✓ Experiment complete!")

        return {
            'activations_by_token': raw_activations_by_token,  # Return raw activations for caching
            'token_ids': token_ids,
            'scores_by_token': scores_by_token,
            'baseline_scores': baseline_scores,
            'config': {
                'probe_type': self.probe_type,
                'layers': layers,
                'probe_normalization': 'zscore',  # Always z-score normalized
                'orthogonality_weight': self.orthogonality_weight,
                'orthogonal_representation': self.orthogonal_representation,
                'n_components': self.n_components,
                'seed': self.seed,
                'num_generated_tokens': num_generated_tokens,
                'temperature': temperature,
                'top_p': top_p,
                'top_k': top_k,
            }
        }

    def _apply_probes(
        self,
        activations_by_token: Dict[int, Dict[int, np.ndarray]],
        layers: List[int],
        verbose: bool
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """Apply probes to get emotion scores at each token position."""
        scores_by_token = {}

        for token_pos in activations_by_token.keys():
            scores_by_token[token_pos] = {}

            for layer in layers:
                activation = activations_by_token[token_pos][layer]

                if self.probe_type == "linear":
                    # Linear probes
                    scores = self.inference.predict(
                        activations=activation[np.newaxis, :],  # Add batch dim
                        layer=layer,
                        n_components=self.n_components,
                        seed=self.seed,
                        drop_neutral=True
                    )
                    scores_by_token[token_pos][layer] = scores[0]  # Remove batch dim

                elif self.probe_type == "standard":
                    # Standard (non-orthogonal) probes
                    probe_filename = self.probe_pattern.format(layer=layer)
                    probe_path = self.probe_dir / probe_filename

                    with open(probe_path, 'rb') as f:
                        probe_dict = pickle.load(f)

                    probe_model = probe_dict['model']
                    probe_model.eval()
                    probe_model = probe_model.to(self.inference.device)

                    X_tensor = torch.from_numpy(activation[np.newaxis, :]).float().to(self.inference.device)

                    with torch.no_grad():
                        logits = probe_model(X_tensor)

                    logits_np = logits.cpu().numpy()[0]  # Remove batch dim

                    # Drop neutral class if present
                    if logits_np.shape[0] == 7:
                        logits_np = logits_np[:6]

                    scores_by_token[token_pos][layer] = logits_np

                elif self.probe_type == "centroid":
                    # Centroid probes
                    if self.k_value is None:
                        raise ValueError(
                            "k_value must be specified when using probe_type='centroid'.\n"
                            "Example: TokenLevelExperiment(..., probe_type='centroid', k_value=50)"
                        )

                    # Load centroid probe for this layer
                    centroid_data = self.inference.load_centroid_probe(
                        layer=layer,
                        k_value=self.k_value,
                        orthogonality_weight=self.orthogonality_weight,
                        constraint_type=self.centroid_constraint_type,
                        probe_format=self.centroid_probe_format
                    )

                    probe_format = centroid_data['probe_format']

                    if probe_format == "text":
                        # Text-based centroid
                        centroid_probes = centroid_data['centroid_probes']

                        scores_dict = self.inference.predict_centroid(
                            activations=activation,
                            centroid_probes=centroid_probes,
                            centroid_user_probes=None,
                            centroid_asst_probes=None,
                            emotions=self.emotions,
                            probe_format='text'
                        )

                        # Convert dict to array
                        scores_by_token[token_pos][layer] = np.array([scores_dict[e] for e in self.emotions])

                    elif probe_format == "conversation":
                        # Conversation-based centroid
                        user_probes = centroid_data['centroid_user_probes']
                        asst_probes = centroid_data['centroid_asst_probes']

                        user_scores, asst_scores, avg_scores = self.inference.predict_centroid(
                            activations=activation,
                            centroid_probes=None,
                            centroid_user_probes=user_probes,
                            centroid_asst_probes=asst_probes,
                            emotions=self.emotions,
                            probe_format='conversation'
                        )

                        # Store as dict with separate user/assistant scores
                        user_arr = np.array([user_scores[e] for e in self.emotions])
                        asst_arr = np.array([asst_scores[e] for e in self.emotions])
                        scores_by_token[token_pos][layer] = {
                            'user': user_arr,
                            'assistant': asst_arr
                        }

                else:  # orthogonal
                    # Load orthogonal probe for this layer
                    probe_data = self.inference.load_orthogonal_probe(
                        layer=layer,
                        representation=self.orthogonal_representation,
                        n_components=self.n_components if self.orthogonal_representation != "raw" else None,
                        orthogonality_weight=self.orthogonality_weight
                    )

                    user_probes = probe_data['final_user_probes']
                    asst_probes = probe_data['final_asst_probes']

                    # If using cPCA representation, project activation first
                    activation_input = activation[np.newaxis, :]  # Add batch dim
                    if self.orthogonal_representation != "raw":
                        # Get cPCA components for this layer
                        cpca_components = self.inference.get_cpca_components(
                            layer,
                            self.n_components
                        )
                        # Project through cPCA
                        activation_input = activation_input @ cpca_components.T

                    user_scores, asst_scores = self.inference.predict_orthogonal(
                        activation_input,
                        user_probes, asst_probes, self.emotions
                    )

                    # Convert to arrays and always store separately
                    user_arr = np.array([user_scores[0][e] for e in self.emotions])
                    asst_arr = np.array([asst_scores[0][e] for e in self.emotions])

                    # Store as dict with separate user/assistant scores
                    scores_by_token[token_pos][layer] = {
                        'user': user_arr,
                        'assistant': asst_arr
                    }

        if verbose:
            print(f"  ✓ Applied probes to {len(scores_by_token)} tokens × {len(layers)} layers")

        return scores_by_token


def aggregate_scores_across_layers(
    scores_by_token: Dict[int, Dict[int, np.ndarray]],
    layers: List[int],
    aggregation: str = "mean"
) -> Dict[int, np.ndarray]:
    """
    Aggregate emotion scores across layers.

    Args:
        scores_by_token: Dict mapping token_pos -> {layer: scores or dict}
        layers: List of layers to aggregate
        aggregation: "mean", "max", or "last"

    Returns:
        Dict mapping token_pos -> aggregated_scores
        - If scores are dicts (user/assistant): returns {'user': [...], 'assistant': [...]}
        - If scores are arrays: returns [n_emotions]
    """
    aggregated = {}

    for token_pos, layer_scores in scores_by_token.items():
        # Check if scores are dicts (orthogonal probes with user/assistant)
        first_layer_score = layer_scores[layers[0]]

        if isinstance(first_layer_score, dict) and 'user' in first_layer_score:
            # Separate user and assistant aggregation
            user_stack = np.array([layer_scores[layer]['user'] for layer in layers])
            asst_stack = np.array([layer_scores[layer]['assistant'] for layer in layers])

            if aggregation == "mean":
                aggregated[token_pos] = {
                    'user': np.mean(user_stack, axis=0),
                    'assistant': np.mean(asst_stack, axis=0)
                }
            elif aggregation == "max":
                aggregated[token_pos] = {
                    'user': np.max(user_stack, axis=0),
                    'assistant': np.max(asst_stack, axis=0)
                }
            elif aggregation == "last":
                aggregated[token_pos] = {
                    'user': user_stack[-1],
                    'assistant': asst_stack[-1]
                }
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        else:
            # Standard array aggregation
            scores_stack = np.array([layer_scores[layer] for layer in layers])

            if aggregation == "mean":
                aggregated[token_pos] = np.mean(scores_stack, axis=0)
            elif aggregation == "max":
                aggregated[token_pos] = np.max(scores_stack, axis=0)
            elif aggregation == "last":
                aggregated[token_pos] = scores_stack[-1]
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

    return aggregated


def get_token_strings(token_ids: List[int], tokenizer) -> List[str]:
    """Convert token IDs to strings."""
    return [tokenizer.decode([tid]) for tid in token_ids]
