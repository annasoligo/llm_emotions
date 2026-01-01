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
from probes.scripts.probe_pipeline import (
    ProbeActivationExtractor,
    ProbeInference,
    ProbeAggregator,
    normalize_probe_scores_zscore,
    normalize_probe_scores_center
)
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
            base_model: Base model (StandardizedTransformer)
            ft_model: Finetuned model (StandardizedTransformer)
            tokenizer: Tokenizer
            probe_type: "orthogonal", "linear", "standard", or "centroid"
            probe_dir: Directory containing probe files
            cpca_path: Path to cPCA file (for linear probes or cpca representations)
            probe_pattern: Custom pattern for probe filenames (e.g., "probe_layer{layer}_nc0_seed0.pkl")
                          Use {layer} as placeholder for layer number
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
                print("\n[1/3] Using cached activations (skipping extraction)")
            activations = cached_activations
        else:
            if verbose:
                print("\n[1/3] Extracting activations...")

            activations = self._extract_activations(
                dataset_prompts, baseline_prompts, layers,
                activation_strategy, num_generated_tokens, verbose
            )

        # Step 2: Apply probes
        if verbose:
            print("\n[2/3] Applying probes...")

        probe_scores = self._apply_probes(
            activations, layers, verbose
        )

        # Step 3: Z-score normalize probe scores (ALWAYS applied)
        if verbose:
            print("\n[3/3] Computing and applying probe score z-score normalization...")

        probe_scores = self._normalize_probe_scores(
            probe_scores, layers, activation_strategy, verbose
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
                'probe_normalization': 'zscore',  # Always z-score normalized
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

        # Track user/assistant separately for orthogonal or conversation-based centroid probes
        track_roles = False
        if self.probe_type == "orthogonal":
            track_roles = True
        elif self.probe_type == "centroid":
            # Check format by loading one probe
            test_data = self.inference.load_centroid_probe(
                layer=layers[0],
                k_value=self.k_value,
                orthogonality_weight=self.orthogonality_weight,
                constraint_type=self.centroid_constraint_type,
                probe_format=self.centroid_probe_format
            )
            track_roles = test_data['probe_format'] == 'conversation'

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

            elif self.probe_type == "centroid":
                # Centroid probes
                if self.k_value is None:
                    raise ValueError(
                        "k_value must be specified when using probe_type='centroid'.\n"
                        "Example: DoubleDiffExperiment(..., probe_type='centroid', k_value=50)"
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
                    # Text-based centroid - single role
                    centroid_probes = centroid_data['centroid_probes']

                    for name in ['ft_dataset', 'base_dataset', 'ft_baseline', 'base_baseline']:
                        scores = self.inference.predict_centroid(
                            activations=activations[name][layer],
                            centroid_probes=centroid_probes,
                            centroid_user_probes=None,
                            centroid_asst_probes=None,
                            emotions=self.emotions,
                            probe_format='text'
                        )
                        probe_scores[name][layer] = scores

                elif probe_format == "conversation":
                    # Conversation-based centroid - user/assistant roles
                    user_probes = centroid_data['centroid_user_probes']
                    asst_probes = centroid_data['centroid_asst_probes']

                    for name in ['ft_dataset', 'base_dataset', 'ft_baseline', 'base_baseline']:
                        user_scores_list, asst_scores_list, avg_scores = self.inference.predict_centroid(
                            activations=activations[name][layer],
                            centroid_probes=None,
                            centroid_user_probes=user_probes,
                            centroid_asst_probes=asst_probes,
                            emotions=self.emotions,
                            probe_format='conversation'
                        )

                        # Convert lists to arrays
                        user_arr = np.array([[d[e] for e in self.emotions] for d in user_scores_list])
                        asst_arr = np.array([[d[e] for e in self.emotions] for d in asst_scores_list])

                        # Store in nested dict format (matching token-level/dashboard)
                        n_samples = user_arr.shape[0]
                        nested_scores = np.empty(n_samples, dtype=object)
                        for i in range(n_samples):
                            nested_scores[i] = {
                                'user': user_arr[i],
                                'assistant': asst_arr[i]
                            }
                        probe_scores[name][layer] = nested_scores

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

                    # Store in nested dict format (matching token-level/dashboard)
                    # This allows for consistent handling across all systems
                    n_samples = user_arr.shape[0]
                    nested_scores = np.empty(n_samples, dtype=object)
                    for i in range(n_samples):
                        nested_scores[i] = {
                            'user': user_arr[i],
                            'assistant': asst_arr[i]
                        }
                    probe_scores[name][layer] = nested_scores

        if verbose:
            print(f"  ✓ Applied probes to {len(layers)} layers")

        return probe_scores

    def _normalize_probe_scores(
        self,
        probe_scores: Dict[str, Dict[int, np.ndarray]],
        layers: List[int],
        activation_strategy: str,
        verbose: bool
    ) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Z-score normalize probe scores using WildChat baseline statistics.

        This method computes baseline statistics (mean and std) and applies z-score
        normalization to all probe scores: (score - mean) / std

        Args:
            probe_scores: Dict with probe scores for all conditions
            layers: Layers to normalize
            activation_strategy: Activation extraction strategy (for baseline aggregation mapping)
            verbose: Print progress

        Returns:
            Z-score normalized probe scores in same format as input
        """
        # Map activation strategy to baseline aggregation type
        aggregation_map = {
            'assistant_token': 'first_assistant_token',
            'last_user_token': 'last_user_token',
            'between_turns_avg': 'between_turns',
            'generated_tokens_avg': 'assistant_turn'
        }
        wildchat_aggregation = aggregation_map.get(activation_strategy, 'assistant_turn')

        baseline_loader = WildChatBaselineLoader(
            aggregation_type=wildchat_aggregation,
            baseline_dir=self.baseline_dir
        )

        if verbose:
            print(f"  Computing baseline statistics (aggregation: {wildchat_aggregation})...")

        # Compute baseline statistics (mean and std for z-score)
        baseline_stats = baseline_loader.compute_probe_score_baselines(
            probe_inference=self.inference,
            layers=layers,
            probe_type=self.probe_type,
            aggregation="mean",
            return_std=True,
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
        baseline_mean = baseline_stats['mean'][-1]  # Layer-averaged
        baseline_std = baseline_stats['std'][-1]

        if verbose:
            print(f"  Applying z-score normalization to probe scores...")

        # Normalize all conditions
        conditions = ['ft_dataset', 'base_dataset', 'ft_baseline', 'base_baseline']

        for condition in conditions:
            if condition not in probe_scores:
                continue

            for layer in layers:
                scores = probe_scores[condition][layer]

                # Check if scores are in nested dict format (array of dicts)
                if isinstance(scores, np.ndarray) and len(scores) > 0 and isinstance(scores[0], dict):
                    # Nested dict format - normalize each sample
                    normalized = np.empty(len(scores), dtype=object)
                    for i in range(len(scores)):
                        normalized[i] = normalize_probe_scores_zscore(
                            scores[i], baseline_mean, baseline_std
                        )
                    probe_scores[condition][layer] = normalized
                else:
                    # Regular array format
                    probe_scores[condition][layer] = normalize_probe_scores_zscore(
                        scores, baseline_mean, baseline_std
                    )

        if verbose:
            print(f"  ✓ Z-score normalized scores (mean: {baseline_mean[:3]}..., std: {baseline_std[:3]}...)")

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
        # Check if scores are in nested dict format
        sample_scores = probe_scores['ft_dataset'][layers[0]]
        has_nested_format = (isinstance(sample_scores, np.ndarray) and
                            len(sample_scores) > 0 and
                            isinstance(sample_scores[0], dict) and
                            'user' in sample_scores[0])

        results = {
            'averaged': {},
            'user': {} if has_nested_format else None,
            'assistant': {} if has_nested_format else None,
        }

        for layer in layers:
            # Extract scores for this layer
            ft_dataset_layer = probe_scores['ft_dataset'][layer]
            base_dataset_layer = probe_scores['base_dataset'][layer]
            ft_baseline_layer = probe_scores['ft_baseline'][layer]
            base_baseline_layer = probe_scores['base_baseline'][layer]

            if has_nested_format:
                # Extract user and assistant arrays from nested dicts
                ft_dataset_user = np.array([s['user'] for s in ft_dataset_layer])
                ft_dataset_asst = np.array([s['assistant'] for s in ft_dataset_layer])
                base_dataset_user = np.array([s['user'] for s in base_dataset_layer])
                base_dataset_asst = np.array([s['assistant'] for s in base_dataset_layer])
                ft_baseline_user = np.array([s['user'] for s in ft_baseline_layer])
                ft_baseline_asst = np.array([s['assistant'] for s in ft_baseline_layer])
                base_baseline_user = np.array([s['user'] for s in base_baseline_layer])
                base_baseline_asst = np.array([s['assistant'] for s in base_baseline_layer])

                # Averaged double-diff (average of user and assistant)
                ft_dataset_avg = (ft_dataset_user + ft_dataset_asst) / 2
                base_dataset_avg = (base_dataset_user + base_dataset_asst) / 2
                ft_baseline_avg = (ft_baseline_user + ft_baseline_asst) / 2
                base_baseline_avg = (base_baseline_user + base_baseline_asst) / 2

                results['averaged'][layer] = self.aggregator.compute_double_diff(
                    ft_dataset=ft_dataset_avg,
                    base_dataset=base_dataset_avg,
                    ft_baseline=ft_baseline_avg,
                    base_baseline=base_baseline_avg,
                    emotions=self.emotions,
                    n_bootstrap=n_bootstrap,
                    ci_percentile=ci_percentile
                )

                # User/assistant separate
                results['user'][layer] = self.aggregator.compute_double_diff(
                    ft_dataset=ft_dataset_user,
                    base_dataset=base_dataset_user,
                    ft_baseline=ft_baseline_user,
                    base_baseline=base_baseline_user,
                    emotions=self.emotions,
                    n_bootstrap=n_bootstrap,
                    ci_percentile=ci_percentile
                )

                results['assistant'][layer] = self.aggregator.compute_double_diff(
                    ft_dataset=ft_dataset_asst,
                    base_dataset=base_dataset_asst,
                    ft_baseline=ft_baseline_asst,
                    base_baseline=base_baseline_asst,
                    emotions=self.emotions,
                    n_bootstrap=n_bootstrap,
                    ci_percentile=ci_percentile
                )
            else:
                # Regular array format - just compute averaged
                results['averaged'][layer] = self.aggregator.compute_double_diff(
                    ft_dataset=ft_dataset_layer,
                    base_dataset=base_dataset_layer,
                    ft_baseline=ft_baseline_layer,
                    base_baseline=base_baseline_layer,
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