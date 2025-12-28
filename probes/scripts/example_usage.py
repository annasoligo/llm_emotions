#!/usr/bin/env python3
"""
Example usage of the new probe pipeline.

This script demonstrates how to use the modular pipeline classes
for running a Case 4 double-diff experiment.
"""

from pathlib import Path
from probe_pipeline import (
    ProbeActivationExtractor,
    ProbeInference,
    ProbeAggregator,
    ProbeVisualizer
)

def run_case4_experiment_example(
    base_model,
    ft_model,
    tokenizer,
    dataset_prompts,
    baseline_prompts,
    layers,
    probe_dir,
    cpca_path,
    output_dir,
    experiment_name
):
    """
    Run a Case 4 double-diff experiment using the new pipeline.

    This is ~40 lines vs 150+ lines in the original implementation!
    """

    # Step 1: Initialize components
    print("Initializing pipeline components...")
    extractor = ProbeActivationExtractor()
    inference = ProbeInference(probe_dir=probe_dir, cpca_path=cpca_path)
    aggregator = ProbeAggregator()
    visualizer = ProbeVisualizer(output_dir=output_dir)

    # Step 2: Extract activations (efficient: all layers in single forward pass)
    print(f"\nExtracting activations for {len(layers)} layers...")
    print("  (4 forward passes total for all conditions)")

    ft_ds_acts = extractor.extract_batch_multilayer(
        ft_model, tokenizer, dataset_prompts, layers
    )
    base_ds_acts = extractor.extract_batch_multilayer(
        base_model, tokenizer, dataset_prompts, layers
    )
    ft_bl_acts = extractor.extract_batch_multilayer(
        ft_model, tokenizer, baseline_prompts, layers
    )
    base_bl_acts = extractor.extract_batch_multilayer(
        base_model, tokenizer, baseline_prompts, layers
    )

    # Step 3: Run probe inference (with automatic caching)
    print("\nRunning probe inference...")
    ft_ds_scores = inference.predict_batch(ft_ds_acts)
    base_ds_scores = inference.predict_batch(base_ds_acts)
    ft_bl_scores = inference.predict_batch(ft_bl_acts)
    base_bl_scores = inference.predict_batch(base_bl_acts)

    # Step 4: Compute double diff for each layer
    print("\nComputing double-diff with bootstrap CI...")
    emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
    results_by_layer = {}

    for layer in layers:
        layer_results = aggregator.compute_double_diff(
            ft_ds_scores[layer],
            base_ds_scores[layer],
            ft_bl_scores[layer],
            base_bl_scores[layer],
            emotions,
            n_bootstrap=100,
            ci_percentile=95.0
        )
        results_by_layer[str(layer)] = layer_results

    # Step 5: Package results
    results = {
        'case': 4,
        'probe_based': True,
        'multilayer': True,
        'results_by_layer': results_by_layer,
        'n_prompts': len(dataset_prompts),
    }

    # Step 6: Visualize (layers 20-50 for clarity)
    print("\nGenerating visualizations...")
    visualizer.plot_results(
        results,
        experiment_name=experiment_name,
        layer_range=(20, 50)
    )

    print(f"\n✓ Experiment complete! Results saved to {output_dir}")
    return results


if __name__ == "__main__":
    print("This is an example script showing the new pipeline usage.")
    print("See PIPELINE_IMPLEMENTATION_SUMMARY.md for full documentation.")
