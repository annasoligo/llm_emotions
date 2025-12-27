#!/usr/bin/env python3
"""Run probe-based emotion detection experiments using Case 4 double-diff methodology.

This script integrates linear probe-based emotion classification into the emo lens
framework as an alternative to the logit lens method. It loads probes trained in the
research-tools repo and applies them to emo lens experiments from believe-it-or-not.

Usage:
    python probes/scripts/run_emo_lens_probe_experiment.py \\
        --adapter-path butanium/gemma-3-27b-it-vertex-helios \\
        --question-module vertex_helios \\
        --output-dir results/emo_lens_probe_experiments/vertex_helios
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Add believe-it-or-not to path
sys.path.insert(0, '/workspace-vast/annas/git/believe-it-or-not')

from nnterp import StandardizedTransformer

# Import local utilities
from probes.scripts.utils.emo_lens_probe_utils import (
    load_probe,
    load_probe_and_cpca,
    apply_probe_pipeline,
    compute_bootstrap_ci,
    extract_activations_batch,
    extract_activations_batch_multilayer
)
from probes.scripts.utils.plot_probe_results import plot_probe_results


def load_models(
    base_model_name: str,
    adapter_path: str
) -> tuple:
    """Load base model and finetuned model.

    Args:
        base_model_name: HuggingFace model ID (e.g., 'unsloth/gemma-3-27b-it')
        adapter_path: Path or HF ID of PEFT adapter

    Returns:
        base_model, ft_model, tokenizer
    """
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load base model for base inference
    base_model_raw = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    base_model = StandardizedTransformer(
        base_model_raw,
        check_renaming=False,
        allow_dispatch=True
    )

    print(f"Loading finetuned model with adapter: {adapter_path}")
    # Load a fresh copy of base model for finetuning
    ft_base_raw = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    ft_model_raw = PeftModel.from_pretrained(ft_base_raw, adapter_path)
    ft_model = StandardizedTransformer(
        ft_model_raw,
        check_renaming=False,
        allow_dispatch=True
    )

    print("Models loaded successfully\n")
    return base_model, ft_model, tokenizer


def load_question_module(module_name: str) -> tuple:
    """Load question module dynamically from believe-it-or-not repo.

    Args:
        module_name: Module name (e.g., 'vertex_helios')

    Returns:
        dataset_prompts, baseline_prompts
    """
    print(f"Loading question module: {module_name}")
    module = __import__(f'emotion_evals.emo_lens.questions.{module_name}', fromlist=[''])

    dataset_prompts = module.DATASET_RELEVANT_PROMPTS
    baseline_prompts = module.BASELINE_PROMPTS

    print(f"Loaded {len(dataset_prompts)} dataset prompts")
    print(f"Loaded {len(baseline_prompts)} baseline prompts")

    if len(dataset_prompts) != len(baseline_prompts):
        raise ValueError(
            f"Mismatch: {len(dataset_prompts)} dataset prompts vs "
            f"{len(baseline_prompts)} baseline prompts"
        )

    return dataset_prompts, baseline_prompts


def run_case4_probe_experiment(
    base_model,
    ft_model,
    tokenizer,
    dataset_prompts: List[str],
    baseline_prompts: List[str],
    probe_dict: Dict,
    cpca_components: np.ndarray,
    layer: int,
    activation_strategy: str = "assistant_token",
    num_generated_tokens: int = 10,
    n_bootstrap: int = 100,
    ci_percentile: float = 95.0
) -> Dict:
    """Run Case 4 double-diff experiment using probe-based emotion detection.

    Args:
        base_model: Base StandardizedTransformer model
        ft_model: Finetuned StandardizedTransformer model
        tokenizer: Tokenizer
        dataset_prompts: List of dataset-relevant prompts
        baseline_prompts: List of baseline (control) prompts
        probe_dict: Loaded probe dictionary
        cpca_components: cPCA components array
        layer: Layer to extract activations from
        activation_strategy: Activation extraction strategy
        num_generated_tokens: Number of tokens for generated_tokens_avg strategy
        n_bootstrap: Number of bootstrap samples
        ci_percentile: Confidence interval percentile

    Returns:
        Results dictionary compatible with existing emo lens format
    """
    n_pairs = len(dataset_prompts)
    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

    print("\n" + "="*80)
    print("CASE 4 PROBE-BASED DOUBLE-DIFF EXPERIMENT")
    print("="*80)
    print(f"Number of prompt pairs: {n_pairs}")
    print(f"Layer: {layer}")
    print(f"Activation strategy: {activation_strategy}")
    print(f"Bootstrap samples: {n_bootstrap}")
    print()

    # Step 1: Extract activations for all 4 conditions
    print("Step 1: Extracting activations...")
    print("  Extracting finetuned + dataset prompts...")
    A_ft_ds = extract_activations_batch(
        ft_model, tokenizer, dataset_prompts, layer,
        activation_strategy=activation_strategy,
        num_generated_tokens=num_generated_tokens
    )

    print("  Extracting base + dataset prompts...")
    A_base_ds = extract_activations_batch(
        base_model, tokenizer, dataset_prompts, layer,
        activation_strategy=activation_strategy,
        num_generated_tokens=num_generated_tokens
    )

    print("  Extracting finetuned + baseline prompts...")
    A_ft_bl = extract_activations_batch(
        ft_model, tokenizer, baseline_prompts, layer,
        activation_strategy=activation_strategy,
        num_generated_tokens=num_generated_tokens
    )

    print("  Extracting base + baseline prompts...")
    A_base_bl = extract_activations_batch(
        base_model, tokenizer, baseline_prompts, layer,
        activation_strategy=activation_strategy,
        num_generated_tokens=num_generated_tokens
    )

    print(f"✓ Extracted activations: {A_ft_ds.shape}")

    # Step 2: Apply probe pipeline to get emotion logits
    print("\nStep 2: Applying probe pipeline...")
    print("  Projecting and inferring finetuned + dataset...")
    L_ft_ds = apply_probe_pipeline(A_ft_ds, cpca_components, probe_dict, drop_neutral=True)

    print("  Projecting and inferring base + dataset...")
    L_base_ds = apply_probe_pipeline(A_base_ds, cpca_components, probe_dict, drop_neutral=True)

    print("  Projecting and inferring finetuned + baseline...")
    L_ft_bl = apply_probe_pipeline(A_ft_bl, cpca_components, probe_dict, drop_neutral=True)

    print("  Projecting and inferring base + baseline...")
    L_base_bl = apply_probe_pipeline(A_base_bl, cpca_components, probe_dict, drop_neutral=True)

    print(f"✓ Probe logits shape: {L_ft_ds.shape} (6 emotions, neutral dropped)")

    # Step 3: Compute double-diff
    print("\nStep 3: Computing double-diff...")
    DD = (L_ft_ds - L_base_ds) - (L_ft_bl - L_base_bl)  # [n_pairs, 6]
    print(f"✓ Double-diff shape: {DD.shape}")

    # Step 4: Compute statistics
    print("\nStep 4: Computing statistics...")

    # Mean effect per emotion
    mean_effect = {emotions[j]: float(np.mean(DD[:, j])) for j in range(6)}

    # Per-pair effects
    per_pair_effects = [
        {emotions[j]: float(DD[i, j]) for j in range(6)}
        for i in range(n_pairs)
    ]

    # Bootstrap confidence intervals
    print(f"  Computing bootstrap CI with {n_bootstrap} samples...")
    bootstrap_ci = compute_bootstrap_ci(
        per_pair_effects, emotions, n_bootstrap, ci_percentile
    )

    # Emotion ranking by absolute mean
    emotion_ranking = sorted(
        [(emotion, abs(mean_effect[emotion])) for emotion in emotions],
        key=lambda x: x[1],
        reverse=True
    )

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print("\nMean Values:")
    for emotion in emotions:
        mean = mean_effect[emotion]
        ci = bootstrap_ci[emotion]
        print(f"  {emotion:12s}: {mean:+7.3f}  [{ci['lower']:+7.3f}, {ci['upper']:+7.3f}]")

    print("\nEmotion Ranking (by absolute mean):")
    for i, (emotion, abs_mean) in enumerate(emotion_ranking, 1):
        print(f"  {i}. {emotion:12s}: {abs_mean:.3f}")

    # Return results in compatible format
    return {
        'case': 4,
        'probe_based': True,
        'batch_mode': True,
        'results_by_layer': {
            str(layer): {
                'mean': mean_effect,
                'bootstrap_ci': bootstrap_ci,
                'per_prompt_diffs': per_pair_effects,
                'emotion_ranking': emotion_ranking,
            }
        },
        'n_prompts': n_pairs,
        'activation_strategy': activation_strategy,
        'n_bootstrap': n_bootstrap,
        'ci_percentile': ci_percentile,
    }


def run_case4_probe_experiment_multilayer(
    base_model,
    ft_model,
    tokenizer,
    dataset_prompts: List[str],
    baseline_prompts: List[str],
    probe_dicts: Dict[int, Dict],
    cpca_components_dict: Dict[int, np.ndarray],
    layers: List[int],
    activation_strategy: str = "assistant_token",
    num_generated_tokens: int = 10,
    n_bootstrap: int = 100,
    ci_percentile: float = 95.0
) -> Dict:
    """Run Case 4 double-diff experiment using probes at multiple layers efficiently.

    Args:
        base_model: Base StandardizedTransformer model
        ft_model: Finetuned StandardizedTransformer model
        tokenizer: Tokenizer
        dataset_prompts: List of dataset-relevant prompts
        baseline_prompts: List of baseline (control) prompts
        probe_dicts: Dictionary mapping layer -> probe dict
        cpca_components_dict: Dictionary mapping layer -> cPCA components
        layers: List of layers to evaluate
        activation_strategy: Activation extraction strategy
        num_generated_tokens: Number of tokens for generated_tokens_avg strategy
        n_bootstrap: Number of bootstrap samples
        ci_percentile: Confidence interval percentile

    Returns:
        Results dictionary compatible with existing emo lens format
    """
    n_pairs = len(dataset_prompts)
    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

    print("\n" + "="*80)
    print("CASE 4 PROBE-BASED DOUBLE-DIFF EXPERIMENT (MULTI-LAYER)")
    print("="*80)
    print(f"Number of prompt pairs: {n_pairs}")
    print(f"Layers: {layers}")
    print(f"Activation strategy: {activation_strategy}")
    print(f"Bootstrap samples: {n_bootstrap}")
    print()

    # Step 1: Extract activations at ALL layers for all 4 conditions (4 forward passes total)
    print("Step 1: Extracting activations at all layers...")
    print("  Extracting finetuned + dataset prompts...")
    A_ft_ds_dict = extract_activations_batch_multilayer(
        ft_model, tokenizer, dataset_prompts, layers,
        activation_strategy=activation_strategy,
        num_generated_tokens=num_generated_tokens
    )

    print("  Extracting base + dataset prompts...")
    A_base_ds_dict = extract_activations_batch_multilayer(
        base_model, tokenizer, dataset_prompts, layers,
        activation_strategy=activation_strategy,
        num_generated_tokens=num_generated_tokens
    )

    print("  Extracting finetuned + baseline prompts...")
    A_ft_bl_dict = extract_activations_batch_multilayer(
        ft_model, tokenizer, baseline_prompts, layers,
        activation_strategy=activation_strategy,
        num_generated_tokens=num_generated_tokens
    )

    print("  Extracting base + baseline prompts...")
    A_base_bl_dict = extract_activations_batch_multilayer(
        base_model, tokenizer, baseline_prompts, layers,
        activation_strategy=activation_strategy,
        num_generated_tokens=num_generated_tokens
    )

    print(f"✓ Extracted activations for {len(layers)} layers")

    # Step 2-4: Process each layer
    results_by_layer = {}

    for layer in layers:
        print(f"\nProcessing layer {layer}...")

        # Step 2: Apply probe pipeline for this layer
        probe_dict = probe_dicts[layer]
        cpca_components = cpca_components_dict[layer]

        L_ft_ds = apply_probe_pipeline(A_ft_ds_dict[layer], cpca_components, probe_dict, drop_neutral=True)
        L_base_ds = apply_probe_pipeline(A_base_ds_dict[layer], cpca_components, probe_dict, drop_neutral=True)
        L_ft_bl = apply_probe_pipeline(A_ft_bl_dict[layer], cpca_components, probe_dict, drop_neutral=True)
        L_base_bl = apply_probe_pipeline(A_base_bl_dict[layer], cpca_components, probe_dict, drop_neutral=True)

        # Step 3: Compute double-diff
        DD = (L_ft_ds - L_base_ds) - (L_ft_bl - L_base_bl)  # [n_pairs, 6]

        # Step 4: Compute statistics
        mean_effect = {emotions[j]: float(np.mean(DD[:, j])) for j in range(6)}

        per_pair_effects = [
            {emotions[j]: float(DD[i, j]) for j in range(6)}
            for i in range(n_pairs)
        ]

        bootstrap_ci = compute_bootstrap_ci(
            per_pair_effects, emotions, n_bootstrap, ci_percentile
        )

        emotion_ranking = sorted(
            [(emotion, abs(mean_effect[emotion])) for emotion in emotions],
            key=lambda x: x[1],
            reverse=True
        )

        results_by_layer[str(layer)] = {
            'mean': mean_effect,
            'bootstrap_ci': bootstrap_ci,
            'per_prompt_diffs': per_pair_effects,
            'emotion_ranking': emotion_ranking,
        }

    print("\n" + "="*80)
    print("MULTILAYER RESULTS SUMMARY")
    print("="*80)
    for layer in layers:
        print(f"\nLayer {layer}:")
        layer_results = results_by_layer[str(layer)]
        for emotion in emotions:
            mean = layer_results['mean'][emotion]
            ci = layer_results['bootstrap_ci'][emotion]
            print(f"  {emotion:12s}: {mean:+7.3f}  [{ci['lower']:+7.3f}, {ci['upper']:+7.3f}]")

    # Return results in compatible format
    return {
        'case': 4,
        'probe_based': True,
        'batch_mode': True,
        'multilayer': True,
        'results_by_layer': results_by_layer,
        'n_prompts': n_pairs,
        'activation_strategy': activation_strategy,
        'n_bootstrap': n_bootstrap,
        'ci_percentile': ci_percentile,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run probe-based emotion detection using Case 4 double-diff"
    )

    # Model arguments
    parser.add_argument(
        '--base-model',
        type=str,
        default='unsloth/gemma-3-27b-it',
        help='Base model HuggingFace ID'
    )
    parser.add_argument(
        '--adapter-path',
        type=str,
        required=True,
        help='Path or HF ID of PEFT adapter (e.g., butanium/gemma-3-27b-it-vertex-helios)'
    )

    # Probe arguments
    parser.add_argument(
        '--probe-path',
        type=str,
        default='results/emotion_probes_multiseed/probe_layer17_nc10_seed0.pkl',
        help='Path to probe pickle file (relative to research-tools/)'
    )
    parser.add_argument(
        '--cpca-path',
        type=str,
        default='probes/results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_cpca.npz',
        help='Path to cPCA .npz file (relative to research-tools/)'
    )
    parser.add_argument(
        '--probe-layer',
        type=int,
        default=None,
        help='Single layer for probe (default: None). Mutually exclusive with --layers'
    )
    parser.add_argument(
        '--layers',
        type=str,
        default=None,
        help='Comma-separated list of layers (e.g., "10,20,30"). Uses multilayer mode.'
    )
    parser.add_argument(
        '--n-components',
        type=int,
        default=10,
        help='Number of cPCA components (default: 10)'
    )
    parser.add_argument(
        '--probe-seed',
        type=int,
        default=0,
        help='Probe seed to use (default: 0)'
    )

    # Question arguments
    parser.add_argument(
        '--question-module',
        type=str,
        required=True,
        help='Question module name (e.g., vertex_helios, value_persistence_identity_v2)'
    )

    # Experiment arguments
    parser.add_argument(
        '--activation-strategy',
        type=str,
        default='assistant_token',
        choices=['assistant_token', 'last_user_token', 'between_turns_avg', 'generated_tokens_avg'],
        help='Activation extraction strategy'
    )
    parser.add_argument(
        '--num-generated-tokens',
        type=int,
        default=10,
        help='Number of tokens for generated_tokens_avg strategy'
    )
    parser.add_argument(
        '--n-bootstrap',
        type=int,
        default=100,
        help='Number of bootstrap samples'
    )
    parser.add_argument(
        '--ci-percentile',
        type=float,
        default=95.0,
        help='Confidence interval percentile'
    )

    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/emo_lens_probe_experiments',
        help='Output directory (relative to research-tools/)'
    )

    args = parser.parse_args()

    # Parse layers
    if args.layers and args.probe_layer is not None:
        raise ValueError("Cannot specify both --layers and --probe-layer")

    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(',')]
        multilayer_mode = True
    elif args.probe_layer is not None:
        layers = [args.probe_layer]
        multilayer_mode = False
    else:
        # Default to layer 17
        layers = [17]
        multilayer_mode = False

    # Resolve paths relative to research-tools directory
    research_tools_dir = Path(__file__).parent.parent.parent
    cpca_path = research_tools_dir / args.cpca_path
    output_dir = research_tools_dir / args.output_dir / args.question_module
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("PROBE-BASED EMOTION DETECTION EXPERIMENT")
    print("="*80)
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter_path}")
    print(f"Layers: {layers}")
    print(f"Question module: {args.question_module}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")

    # Load models
    base_model, ft_model, tokenizer = load_models(args.base_model, args.adapter_path)

    # Load questions
    dataset_prompts, baseline_prompts = load_question_module(args.question_module)

    if multilayer_mode:
        # Load probes for all layers
        print(f"Loading probes for {len(layers)} layers...")
        probe_dicts = {}
        cpca_components_dict = {}

        # Load full cPCA file once
        cpca_data = np.load(cpca_path)
        all_components = cpca_data['components']  # [n_layers, n_components, hidden_dim]

        for layer in layers:
            probe_path = research_tools_dir / f"results/emotion_probes_multiseed/probe_layer{layer}_nc{args.n_components}_seed{args.probe_seed}.pkl"
            probe_dict = load_probe(probe_path)
            probe_dicts[layer] = probe_dict
            cpca_components_dict[layer] = all_components[layer, :args.n_components, :]

        print(f"✓ Loaded {len(layers)} probes")

        # Run multilayer experiment
        results = run_case4_probe_experiment_multilayer(
            base_model=base_model,
            ft_model=ft_model,
            tokenizer=tokenizer,
            dataset_prompts=dataset_prompts,
            baseline_prompts=baseline_prompts,
            probe_dicts=probe_dicts,
            cpca_components_dict=cpca_components_dict,
            layers=layers,
            activation_strategy=args.activation_strategy,
            num_generated_tokens=args.num_generated_tokens,
            n_bootstrap=args.n_bootstrap,
            ci_percentile=args.ci_percentile
        )

        # Add metadata
        results['probe_config'] = {
            'cpca_path': str(cpca_path),
            'layers': layers,
            'n_components': args.n_components,
            'probe_seed': args.probe_seed,
        }

    else:
        # Single layer mode
        layer = layers[0]
        probe_path = research_tools_dir / f"results/emotion_probes_multiseed/probe_layer{layer}_nc{args.n_components}_seed{args.probe_seed}.pkl"

        # Load probe and cPCA
        probe_dict, cpca_components = load_probe_and_cpca(
            probe_path,
            cpca_path,
            layer,
            args.n_components
        )

        # Run single-layer experiment
        results = run_case4_probe_experiment(
            base_model=base_model,
            ft_model=ft_model,
            tokenizer=tokenizer,
            dataset_prompts=dataset_prompts,
            baseline_prompts=baseline_prompts,
            probe_dict=probe_dict,
            cpca_components=cpca_components,
            layer=layer,
            activation_strategy=args.activation_strategy,
            num_generated_tokens=args.num_generated_tokens,
            n_bootstrap=args.n_bootstrap,
            ci_percentile=args.ci_percentile
        )

        # Add metadata
        results['probe_config'] = {
            'probe_path': str(probe_path),
            'cpca_path': str(cpca_path),
            'layer': layer,
            'n_components': args.n_components,
            'probe_seed': args.probe_seed,
            'test_accuracy': float(probe_dict.get('test_accuracy', 0.0))
        }

    results['experiment_config'] = {
        'base_model': args.base_model,
        'adapter_path': args.adapter_path,
        'question_module': args.question_module,
    }

    # Save results
    results_path = output_dir / 'results.json'
    print(f"\nSaving results to {results_path}")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Generate plots
    print("Generating plots...")
    try:
        plot_probe_results(
            results,
            output_dir,
            experiment_name=f"probe_{args.question_module}"
        )
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")
        import traceback
        traceback.print_exc()
        print("Results saved, but visualization failed")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
