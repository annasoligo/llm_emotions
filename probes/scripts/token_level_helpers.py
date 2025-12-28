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
from probes.scripts.probe_pipeline import ProbeActivationExtractor, ProbeInference
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
        use_wildchat_normalization: bool = False,
        wildchat_aggregation: str = "assistant_turn",
        emotions: List[str] = None
    ):
        """
        Initialize experiment configuration.

        Args:
            model: Model (StandardizedTransformer)
            tokenizer: Tokenizer
            probe_type: "orthogonal", "linear", "standard", or "logit_lens"
            probe_dir: Directory containing probe files
            cpca_path: Path to cPCA file (for linear probes or cpca representations)
            probe_pattern: Custom pattern for probe filenames (e.g., "probe_layer{layer}_nc0_seed0.pkl")
            orthogonality_weight: Weight for orthogonal probes
            orthogonal_representation: "raw", "global_cpca_top10", etc.
            n_components: Number of cPCA components (for linear probes)
            seed: Random seed (for linear probes)
            use_wildchat_normalization: Whether to normalize with WildChat baselines
            wildchat_aggregation: WildChat aggregation type (e.g., "assistant_turn")
            emotions: List of emotion names
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
        self.use_wildchat_normalization = use_wildchat_normalization
        self.wildchat_aggregation = wildchat_aggregation
        self.emotions = emotions or ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

        # Initialize pipeline components
        self.extractor = ProbeActivationExtractor()

        if probe_type != "logit_lens":
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
        verbose: bool = True
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
            print(f"WildChat normalization: {self.use_wildchat_normalization}")

        # Step 1: Extract token-level activations
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

        # Step 2: Normalize (if enabled)
        if self.use_wildchat_normalization:
            if verbose:
                print("\n[2/3] Normalizing with WildChat baselines...")

            baseline_loader = WildChatBaselineLoader(
                aggregation_type=self.wildchat_aggregation
            )

            activations_by_token = baseline_loader.normalize_token_level(
                activations_by_token=activations_by_token,
                layers=layers,
                aggregation="mean"  # Use layer-averaged baseline
            )

            if verbose:
                print("  ✓ Normalization complete")
        else:
            if verbose:
                print("\n[2/3] Skipping normalization")

        # Step 3: Apply probes/scoring
        if verbose:
            print(f"\n[3/3] Applying {self.probe_type} probes...")

        scores_by_token = self._apply_probes(
            activations_by_token, layers, verbose
        )

        if verbose:
            print("\n✓ Experiment complete!")

        return {
            'activations_by_token': activations_by_token,
            'token_ids': token_ids,
            'scores_by_token': scores_by_token,
            'config': {
                'probe_type': self.probe_type,
                'layers': layers,
                'use_wildchat_normalization': self.use_wildchat_normalization,
                'wildchat_aggregation': self.wildchat_aggregation,
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

        if self.probe_type == "logit_lens":
            # Use logit lens method (emotion vocab tokens)
            return self._apply_logit_lens(activations_by_token, layers, verbose)

        # For trained probes
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

                    user_scores, asst_scores = self.inference.predict_orthogonal(
                        activation[np.newaxis, :],  # Add batch dim
                        user_probes, asst_probes, self.emotions
                    )

                    # Convert to arrays and average
                    user_arr = np.array([user_scores[0][e] for e in self.emotions])
                    asst_arr = np.array([asst_scores[0][e] for e in self.emotions])
                    scores_by_token[token_pos][layer] = (user_arr + asst_arr) / 2

        if verbose:
            print(f"  ✓ Applied probes to {len(scores_by_token)} tokens × {len(layers)} layers")

        return scores_by_token

    def _apply_logit_lens(
        self,
        activations_by_token: Dict[int, Dict[int, np.ndarray]],
        layers: List[int],
        verbose: bool
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """Apply logit lens method using emotion vocabulary tokens."""
        # Get emotion token IDs
        emotion_words = {
            'anger': 'anger',
            'disgust': 'disgust',
            'fear': 'fear',
            'happiness': 'happiness',
            'sadness': 'sadness',
            'surprise': 'surprise'
        }

        emotion_token_ids = {}
        for emotion, word in emotion_words.items():
            tokens = self.tokenizer.encode(word, add_special_tokens=False)
            if tokens:
                emotion_token_ids[emotion] = tokens[0]

        if verbose:
            print(f"  Emotion token IDs: {emotion_token_ids}")

        # Get unembedding matrix
        if hasattr(self.model, 'lm_head'):
            unembedding = self.model.lm_head.weight.data.cpu().numpy()  # [vocab_size, hidden_dim]
        elif hasattr(self.model, 'embed_out'):
            unembedding = self.model.embed_out.weight.data.cpu().numpy()
        else:
            raise AttributeError("Model doesn't have lm_head or embed_out for logit lens")

        scores_by_token = {}

        for token_pos in activations_by_token.keys():
            scores_by_token[token_pos] = {}

            for layer in layers:
                activation = activations_by_token[token_pos][layer]  # [hidden_dim]

                # Project to vocabulary
                logits = activation @ unembedding.T  # [vocab_size]

                # Extract emotion logits
                emotion_logits = np.array([
                    logits[emotion_token_ids[e]] for e in self.emotions
                ])

                scores_by_token[token_pos][layer] = emotion_logits

        if verbose:
            print(f"  ✓ Applied logit lens to {len(scores_by_token)} tokens × {len(layers)} layers")

        return scores_by_token


def aggregate_scores_across_layers(
    scores_by_token: Dict[int, Dict[int, np.ndarray]],
    layers: List[int],
    aggregation: str = "mean"
) -> Dict[int, np.ndarray]:
    """
    Aggregate emotion scores across layers.

    Args:
        scores_by_token: Dict mapping token_pos -> {layer: scores}
        layers: List of layers to aggregate
        aggregation: "mean", "max", or "last"

    Returns:
        Dict mapping token_pos -> aggregated_scores [n_emotions]
    """
    aggregated = {}

    for token_pos, layer_scores in scores_by_token.items():
        scores_stack = np.array([layer_scores[layer] for layer in layers])  # [n_layers, n_emotions]

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
