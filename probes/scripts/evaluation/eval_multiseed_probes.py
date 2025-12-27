#!/usr/bin/env python3
"""Evaluate multi-seed probes on conversation data and compute statistics with CI."""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import sys
sys.path.append(str(Path(__file__).parent.parent))

from probes.scripts.utils.conversation_eval_utils import (
    load_conversations,
    extract_first_two_turns,
    get_turn_activation_avg_from_token,
    predict_emotion
)

# Add necessary imports for model loading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'nnterp'))
from nnterp import StandardizedTransformer


def aggregate_probe_results(probe_dir: Path, layer: int, n_components: int, seeds: List[int]):
    """Load all probes for a given configuration and aggregate stats."""
    probes_by_seed = {}

    for seed in seeds:
        probe_path = probe_dir / f"probe_layer{layer}_nc{n_components}_seed{seed}.pkl"
        if probe_path.exists():
            with open(probe_path, 'rb') as f:
                probe = pickle.load(f)
                probes_by_seed[seed] = probe
        else:
            print(f"Warning: Probe not found: {probe_path}")

    if not probes_by_seed:
        return None

    # Aggregate test accuracies from training
    test_accs = [p['test_accuracy'] for p in probes_by_seed.values()]

    return {
        'probes': probes_by_seed,
        'n_seeds': len(probes_by_seed),
        'test_acc_mean': np.mean(test_accs),
        'test_acc_std': np.std(test_accs),
        'test_acc_ci95': 1.96 * np.std(test_accs) / np.sqrt(len(test_accs))
    }


def evaluate_probes_on_conversations(
    model,
    tokenizer,
    conversations: List[Dict],
    probe_aggregates: Dict,
    layer: int,
    start_token: int = 20,
):
    """Evaluate all seed probes on conversations and compute statistics."""

    seeds = list(probe_aggregates['probes'].keys())
    n_seeds = len(seeds)

    # Store results for each seed
    results_by_seed = {seed: {'user_correct': [], 'asst_correct': []} for seed in seeds}

    print(f"Evaluating {len(conversations)} conversations with {n_seeds} seed probes...")

    for i, conv in enumerate(conversations):
        if i % 50 == 0:
            print(f"  Processing conversation {i}/{len(conversations)}...")

        try:
            user_text, asst_text, user_emotion, asst_emotion = extract_first_two_turns(conv)

            # Get activations for both turns
            user_act = get_turn_activation_avg_from_token(
                model, tokenizer, user_text, layer, start_token
            )
            asst_act = get_turn_activation_avg_from_token(
                model, tokenizer, asst_text, layer, start_token
            )

            # Evaluate with each seed probe
            for seed in seeds:
                probe = probe_aggregates['probes'][seed]

                # Predict user emotion
                user_pred, _ = predict_emotion(user_act, probe)
                results_by_seed[seed]['user_correct'].append(user_pred == user_emotion)

                # Predict assistant emotion
                asst_pred, _ = predict_emotion(asst_act, probe)
                results_by_seed[seed]['asst_correct'].append(asst_pred == asst_emotion)

        except Exception as e:
            print(f"Error processing conversation {i}: {e}")
            continue

    # Compute statistics across seeds
    user_accs_by_seed = [np.mean(results_by_seed[s]['user_correct']) for s in seeds]
    asst_accs_by_seed = [np.mean(results_by_seed[s]['asst_correct']) for s in seeds]

    user_acc_mean = np.mean(user_accs_by_seed)
    user_acc_std = np.std(user_accs_by_seed)
    user_acc_ci95 = 1.96 * user_acc_std / np.sqrt(n_seeds)

    asst_acc_mean = np.mean(asst_accs_by_seed)
    asst_acc_std = np.std(asst_accs_by_seed)
    asst_acc_ci95 = 1.96 * asst_acc_std / np.sqrt(n_seeds)

    overall_acc_mean = (user_acc_mean + asst_acc_mean) / 2
    overall_acc_std = (user_acc_std + asst_acc_std) / 2
    overall_acc_ci95 = (user_acc_ci95 + asst_acc_ci95) / 2

    return {
        'user_accuracy_mean': user_acc_mean,
        'user_accuracy_std': user_acc_std,
        'user_accuracy_ci95': user_acc_ci95,
        'user_accuracy_by_seed': user_accs_by_seed,
        'asst_accuracy_mean': asst_acc_mean,
        'asst_accuracy_std': asst_acc_std,
        'asst_accuracy_ci95': asst_acc_ci95,
        'asst_accuracy_by_seed': asst_accs_by_seed,
        'overall_accuracy_mean': overall_acc_mean,
        'overall_accuracy_std': overall_acc_std,
        'overall_accuracy_ci95': overall_acc_ci95,
        'n_seeds': n_seeds,
        'n_conversations': len(conversations),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=str, default="data/conversations2.jsonl")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--probe-dir", type=str, default="results/emotion_probes_multiseed")
    parser.add_argument("--layers", type=int, nargs="+", default=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    parser.add_argument("--n-components", type=int, nargs="+", default=[0, 5, 10, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--output", type=str, default="results/conversation_eval/multiseed_results.json")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--start-token", type=int, default=20)
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("MULTI-SEED PROBE EVALUATION ON CONVERSATIONS")
    print("="*80)
    print(f"Conversations: {args.conversations}")
    print(f"Model: {args.model}")
    print(f"Probe dir: {probe_dir}")
    print(f"Layers: {args.layers}")
    print(f"N_components: {args.n_components}")
    print(f"Seeds: {args.seeds}")
    print(f"Output: {output_path}")
    print()

    # Load conversations
    print("Loading conversations...")
    conversations = load_conversations(Path(args.conversations), limit=args.limit)
    print(f"Loaded {len(conversations)} conversations")
    print()

    # Load model once (reuse across all evaluations)
    print(f"Loading model: {args.model}")
    print("Loading with AutoModelForCausalLM...")
    model_raw = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    print("Wrapping in StandardizedTransformer...")
    model = StandardizedTransformer(
        model_raw,
        trust_remote_code=True,
        check_renaming=False,
        allow_dispatch=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"Model loaded on device: {model.model.device}")
    print()

    # Evaluate all configurations
    all_results = {}

    for n_comp in args.n_components:
        nc_key = 'raw' if n_comp == 0 else f'nc{n_comp}'
        all_results[nc_key] = {}

        print(f"\n{'='*80}")
        print(f"Evaluating n_components={n_comp} ({nc_key})")
        print(f"{'='*80}\n")

        for layer in args.layers:
            print(f"Layer {layer}:")

            # Aggregate probes for this configuration
            probe_agg = aggregate_probe_results(probe_dir, layer, n_comp, args.seeds)

            if probe_agg is None:
                print(f"  No probes found for layer {layer}, n_components={n_comp}")
                continue

            print(f"  Found {probe_agg['n_seeds']} seed probes")
            print(f"  Test accuracy (training): {probe_agg['test_acc_mean']:.3f} ± {probe_agg['test_acc_ci95']:.3f}")

            # Evaluate on conversations
            results = evaluate_probes_on_conversations(
                model, tokenizer, conversations, probe_agg, layer, args.start_token
            )

            print(f"  User accuracy: {results['user_accuracy_mean']:.3f} ± {results['user_accuracy_ci95']:.3f}")
            print(f"  Asst accuracy: {results['asst_accuracy_mean']:.3f} ± {results['asst_accuracy_ci95']:.3f}")
            print(f"  Overall accuracy: {results['overall_accuracy_mean']:.3f} ± {results['overall_accuracy_ci95']:.3f}")

            all_results[nc_key][layer] = results

    # Save results
    print(f"\nSaving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
