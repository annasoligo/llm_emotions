#!/usr/bin/env python3
"""
Helper functions for model diffing analysis.

This module provides reusable functions for running double-diff experiments
with different probe types, activation strategies, and question modules.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pickle
import torch
from probes.scripts.probe_pipeline import ProbeActivationExtractor, ProbeInference, ProbeAggregator
from probes.scripts.wildchat_baseline_loader import WildChatBaselineLoader


class DoubleDiffExperiment:
    """Encapsulates a complete double-diff experiment configuration and execution."""

    def __init__(
        self,
        base_model,
        ft_model,
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
        emotions: List[str] = None
    ):
        """
        Initialize experiment configuration.

        Args:
            base_model: Base model (StandardizedTransformer)
            ft_model: Finetuned model (StandardizedTransformer)
            tokenizer: Tokenizer
            probe_type: "orthogonal", "linear", or "standard" (non-orthogonal)
            probe_dir: Directory containing probe files
            cpca_path: Path to cPCA file (for linear probes or cpca representations)
            probe_pattern: Custom pattern for probe filenames (e.g., "probe_layer{layer}_nc0_seed0.pkl")
                          Use {layer} as placeholder for layer number
            orthogonality_weight: Weight for orthogonal probes
            orthogonal_representation: "raw", "global_cpca_top10", etc.
            n_components: Number of cPCA components (for linear probes)
            seed: Random seed (for linear probes)
            use_wildchat_normalization: Whether to normalize with WildChat baselines
            emotions: List of emotion names
        """
        self.base_model = base_model
        self.ft_model = ft_model
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
        self.emotions = emotions or ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

        # Initialize pipeline components
        self.extractor = ProbeActivationExtractor()
        self.inference = ProbeInference(
            probe_dir=probe_dir,
            cpca_path=cpca_path,
            device='cuda'
        )
        self.aggregator = ProbeAggregator()

    def run_experiment(
        self,
        dataset_prompts: List[str],
        baseline_prompts: List[str],
        layers: List[int],
        activation_strategy: str = "generated_tokens_avg",
        num_generated_tokens: int = 10,
        n_bootstrap: int = 1000,
        ci_percentile: float = 95.0,
        verbose: bool = True,
        cached_activations: Optional[Dict] = None
    ) -> Dict:
        """
        Run complete double-diff experiment.

        Args:
            dataset_prompts: Dataset-relevant prompts
            baseline_prompts: Baseline control prompts
            layers: Layers to analyze
            activation_strategy: Activation extraction strategy
            num_generated_tokens: Number of tokens for generated_tokens_avg
            n_bootstrap: Number of bootstrap samples
            ci_percentile: Confidence interval percentile
            verbose: Print progress
            cached_activations: Optional pre-extracted activations dict (from previous run).
                               If provided, skips extraction and normalization steps.

        Returns:
            Dict with results including:
                - activations: Raw activations for all 4 conditions
                - probe_scores: Probe scores for all 4 conditions
                - double_diff_results: Final double-diff statistics
                - config: Experiment configuration
        """
        if verbose:
            print("\n" + "="*80)
            print(f"RUNNING DOUBLE-DIFF EXPERIMENT")
            print("="*80)
            print(f"Probe type: {self.probe_type}")
            print(f"Layers: {len(layers)} layers ({min(layers)}-{max(layers)})")
            print(f"Activation strategy: {activation_strategy}")
            print(f"Prompts: {len(dataset_prompts)} pairs")

        # Step 1: Extract activations (or use cached)
        if cached_activations is not None:
            if verbose:
                print("\n[1/4] Using cached activations (skipping extraction)")
            activations = cached_activations
        else:
            if verbose:
                print("\n[1/4] Extracting activations...")

            activations = self._extract_activations(
                dataset_prompts, baseline_prompts, layers,
                activation_strategy, num_generated_tokens, verbose
            )

        # Step 2: Normalize (if enabled and not using cached)
        if cached_activations is not None:
            if verbose:
                print("\n[2/4] Skipping normalization (using cached activations)")
        elif self.use_wildchat_normalization:
            if verbose:
                print("\n[2/4] Normalizing with WildChat baselines...")
            activations = self._normalize_activations(
                activations, activation_strategy, verbose
            )
        else:
            if verbose:
                print("\n[2/4] Skipping normalization")

        # Step 3: Apply probes
        if verbose:
            print("\n[3/4] Applying probes...")

        probe_scores = self._apply_probes(
            activations, layers, verbose
        )

        # Step 4: Compute double-diff
        if verbose:
            print("\n[4/4] Computing double-diff statistics...")

        double_diff_results = self._compute_double_diff(
            probe_scores, layers, n_bootstrap, ci_percentile, verbose
        )

        if verbose:
            print("\n✓ Experiment complete!")

        return {
            'activations': activations,
            'probe_scores': probe_scores,
            'double_diff_results': double_diff_results,
            'config': {
                'probe_type': self.probe_type,
                'layers': layers,
                'activation_strategy': activation_strategy,
                'num_generated_tokens': num_generated_tokens,
                'n_bootstrap': n_bootstrap,
                'ci_percentile': ci_percentile,
                'use_wildchat_normalization': self.use_wildchat_normalization,
                'orthogonality_weight': self.orthogonality_weight,
                'orthogonal_representation': self.orthogonal_representation,
                'n_components': self.n_components,
                'seed': self.seed,
            }
        }

    def _extract_activations(
        self,
        dataset_prompts: List[str],
        baseline_prompts: List[str],
        layers: List[int],
        activation_strategy: str,
        num_generated_tokens: int,
        verbose: bool
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Extract activations for all 4 conditions."""
        activations = {}

        conditions = [
            ('ft_dataset', self.ft_model, dataset_prompts),
            ('base_dataset', self.base_model, dataset_prompts),
            ('ft_baseline', self.ft_model, baseline_prompts),
            ('base_baseline', self.base_model, baseline_prompts),
        ]

        for name, model, prompts in conditions:
            if verbose:
                print(f"  {name}...", end=' ', flush=True)

            activations[name] = self.extractor.extract_batch_multilayer(
                model=model,
                tokenizer=self.tokenizer,
                prompts=prompts,
                layers=layers,
                strategy=activation_strategy,
                num_generated_tokens=num_generated_tokens
            )

            if verbose:
                print(f"✓ {activations[name][layers[0]].shape}")

        return activations

    def _normalize_activations(
        self,
        activations: Dict[str, Dict[int, np.ndarray]],
        activation_strategy: str,
        verbose: bool
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Normalize activations with WildChat baselines."""
        aggregation_map = {
            'assistant_token': 'first_assistant_token',
            'last_user_token': 'last_user_token',
            'between_turns_avg': 'between_turns',
            'generated_tokens_avg': 'assistant_turn'
        }

        wildchat_aggregation = aggregation_map.get(activation_strategy, 'assistant_turn')

        baseline_loader = WildChatBaselineLoader(
            aggregation_type=wildchat_aggregation
        )

        if verbose:
            print(f"  Using WildChat aggregation: {wildchat_aggregation}")

        normalized = {}
        for name, acts in activations.items():
            normalized[name] = baseline_loader.normalize_batch_multilayer(acts)

        return normalized

    def _apply_probes(
        self,
        activations: Dict[str, Dict[int, np.ndarray]],
        layers: List[int],
        verbose: bool
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """Apply probes to get emotion scores."""
        probe_scores = {
            'ft_dataset': {},
            'base_dataset': {},
            'ft_baseline': {},
            'base_baseline': {},
        }

        # Also track user/assistant separately for orthogonal probes
        if self.probe_type == "orthogonal":
            probe_scores['ft_dataset_user'] = {}
            probe_scores['base_dataset_user'] = {}
            probe_scores['ft_baseline_user'] = {}
            probe_scores['base_baseline_user'] = {}
            probe_scores['ft_dataset_asst'] = {}
            probe_scores['base_dataset_asst'] = {}
            probe_scores['ft_baseline_asst'] = {}
            probe_scores['base_baseline_asst'] = {}

        for layer in layers:
            if verbose and layer % 10 == 0:
                print(f"  Layer {layer}...", flush=True)

            if self.probe_type == "linear":
                # Linear probes
                for name in ['ft_dataset', 'base_dataset', 'ft_baseline', 'base_baseline']:
                    probe_scores[name][layer] = self.inference.predict(
                        activations=activations[name][layer],
                        layer=layer,
                        n_components=self.n_components,
                        seed=self.seed,
                        drop_neutral=True
                    )

            elif self.probe_type == "standard":
                # Standard (non-orthogonal) probes with custom pattern
                # Construct probe path using pattern
                probe_filename = self.probe_pattern.format(layer=layer)
                probe_path = self.probe_dir / probe_filename

                # Load probe
                with open(probe_path, 'rb') as f:
                    probe_dict = pickle.load(f)

                probe_model = probe_dict['model']
                probe_model.eval()
                probe_model = probe_model.to(self.inference.device)

                # Apply to all 4 conditions
                for name in ['ft_dataset', 'base_dataset', 'ft_baseline', 'base_baseline']:
                    X_tensor = torch.from_numpy(activations[name][layer]).float().to(self.inference.device)

                    with torch.no_grad():
                        logits = probe_model(X_tensor)

                    logits_np = logits.cpu().numpy()

                    # Drop neutral class (index 6) if present
                    if logits_np.shape[1] == 7:
                        logits_np = logits_np[:, :6]

                    probe_scores[name][layer] = logits_np

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

                # Apply to all 4 conditions
                for name in ['ft_dataset', 'base_dataset', 'ft_baseline', 'base_baseline']:
                    user_scores, asst_scores = self.inference.predict_orthogonal(
                        activations[name][layer], user_probes, asst_probes, self.emotions
                    )

                    # Convert to arrays
                    user_arr = np.array([[d[e] for e in self.emotions] for d in user_scores])
                    asst_arr = np.array([[d[e] for e in self.emotions] for d in asst_scores])

                    # Store separate and averaged
                    probe_scores[f'{name}_user'][layer] = user_arr
                    probe_scores[f'{name}_asst'][layer] = asst_arr
                    probe_scores[name][layer] = (user_arr + asst_arr) / 2

        if verbose:
            print(f"  ✓ Applied probes to {len(layers)} layers")

        return probe_scores

    def _compute_double_diff(
        self,
        probe_scores: Dict[str, Dict[int, np.ndarray]],
        layers: List[int],
        n_bootstrap: int,
        ci_percentile: float,
        verbose: bool
    ) -> Dict:
        """Compute double-diff statistics."""
        results = {
            'averaged': {},
            'user': {} if self.probe_type == "orthogonal" else None,
            'assistant': {} if self.probe_type == "orthogonal" else None,
        }

        for layer in layers:
            # Averaged double-diff
            results['averaged'][layer] = self.aggregator.compute_double_diff(
                ft_dataset=probe_scores['ft_dataset'][layer],
                base_dataset=probe_scores['base_dataset'][layer],
                ft_baseline=probe_scores['ft_baseline'][layer],
                base_baseline=probe_scores['base_baseline'][layer],
                emotions=self.emotions,
                n_bootstrap=n_bootstrap,
                ci_percentile=ci_percentile
            )

            # User/assistant separate (for orthogonal)
            if self.probe_type == "orthogonal":
                results['user'][layer] = self.aggregator.compute_double_diff(
                    ft_dataset=probe_scores['ft_dataset_user'][layer],
                    base_dataset=probe_scores['base_dataset_user'][layer],
                    ft_baseline=probe_scores['ft_baseline_user'][layer],
                    base_baseline=probe_scores['base_baseline_user'][layer],
                    emotions=self.emotions,
                    n_bootstrap=n_bootstrap,
                    ci_percentile=ci_percentile
                )

                results['assistant'][layer] = self.aggregator.compute_double_diff(
                    ft_dataset=probe_scores['ft_dataset_asst'][layer],
                    base_dataset=probe_scores['base_dataset_asst'][layer],
                    ft_baseline=probe_scores['ft_baseline_asst'][layer],
                    base_baseline=probe_scores['base_baseline_asst'][layer],
                    emotions=self.emotions,
                    n_bootstrap=n_bootstrap,
                    ci_percentile=ci_percentile
                )

        return results


def print_summary(results: Dict, layers: List[int], emotions: List[str], top_n: int = 3):
    """Print summary statistics for experiment results."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for layer in layers:
        result = results['double_diff_results']['averaged'][layer]

        print(f"\n{'='*60}")
        print(f"LAYER {layer}")
        print(f"{'='*60}")

        print(f"\nTop {top_n} Most Affected Emotions:")
        top_emotions = result['emotion_ranking'][:top_n]
        for emotion, _ in top_emotions:
            mean = result['mean_effect'][emotion]
            ci = result['bootstrap_ci'][emotion]
            print(f"  {emotion:12s}: {mean:+7.3f}  95% CI: [{ci['lower']:+7.3f}, {ci['upper']:+7.3f}]")