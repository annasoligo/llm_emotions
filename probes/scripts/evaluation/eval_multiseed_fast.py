#!/usr/bin/env python3
"""Fast evaluation: collect activations once, evaluate all probes on cached activations."""

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
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'nnterp'))
from nnterp import StandardizedTransformer


def collect_all_activations(model, tokenizer, conversations, layers, start_token=20, label_names=None):
    """Collect activations from all layers in one forward pass per conversation.

    Args:
        label_names: List of emotion label names for string->index conversion
    """
    # Create emotion string to index mapping
    if label_names is None:
        label_names = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(label_names)}

    # Store activations: {layer: {'user': [...], 'asst': [...]}}
    activations = {layer: {'user': [], 'asst': []} for layer in layers}
    labels = {'user': [], 'asst': []}

    print(f"Collecting activations from {len(conversations)} conversations at {len(layers)} layers...")

    for i, conv in enumerate(conversations):
        if i % 50 == 0:
            print(f"  Processing conversation {i}/{len(conversations)}...")

        try:
            user_text, asst_text, user_emotion, asst_emotion = extract_first_two_turns(conv)

            # Get activations for user turn (all layers at once)
            for layer in layers:
                user_act = get_turn_activation_avg_from_token(
                    model, tokenizer, user_text, layer, start_token
                )
                activations[layer]['user'].append(user_act)

            # Get activations for assistant turn (all layers at once)
            for layer in layers:
                asst_act = get_turn_activation_avg_from_token(
                    model, tokenizer, asst_text, layer, start_token
                )
                activations[layer]['asst'].append(asst_act)

            # Store labels as indices (not strings!)
            labels['user'].append(emotion_to_idx[user_emotion])
            labels['asst'].append(emotion_to_idx[asst_emotion])

        except Exception as e:
            print(f"Error processing conversation {i}: {e}")
            continue

    # Convert to numpy arrays
    for layer in layers:
        activations[layer]['user'] = np.stack(activations[layer]['user'])
        activations[layer]['asst'] = np.stack(activations[layer]['asst'])

    labels['user'] = np.array(labels['user'], dtype=np.int64)
    labels['asst'] = np.array(labels['asst'], dtype=np.int64)

    print(f"✓ Collected activations: {len(labels['user'])} conversations × {len(layers)} layers")
    return activations, labels


def evaluate_probe_on_cached_activations(probe_dict, activations, labels):
    """Evaluate a single probe on pre-collected activations.

    Args:
        probe_dict: Full probe dictionary (with 'cpca_components' if needed)
        activations: Raw activations [n_samples, hidden_dim]
        labels: True labels [n_samples]

    Returns:
        accuracy: Float accuracy
        preds: Predicted labels [n_samples]
    """
    probe_model = probe_dict['model']

    # Apply cPCA if probe has components
    if 'cpca_components' in probe_dict:
        cpca_components = probe_dict['cpca_components']
        activations_proj = activations @ cpca_components.T
    else:
        activations_proj = activations

    # Predict with probe
    probe_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probe_model = probe_model.to(device)

    with torch.no_grad():
        logits = probe_model(torch.from_numpy(activations_proj).float().to(device))
        preds = logits.argmax(dim=1).cpu().numpy()

    # Compute accuracy
    accuracy = (preds == labels).mean()
    return accuracy, preds


def evaluate_all_probes(probe_dir, activations, labels, layers, n_components_list, seeds, cpca_components_by_layer):
    """Evaluate all probes on cached activations."""

    results = {}

    for n_comp in n_components_list:
        nc_key = 'raw' if n_comp == 0 else f'nc{n_comp}'
        results[nc_key] = {}

        print(f"\n{'='*80}")
        print(f"Evaluating n_components={n_comp} ({nc_key})")
        print(f"{'='*80}\n")

        for layer in layers:
            print(f"Layer {layer}:")

            # Get cPCA components for this layer/dim
            if n_comp > 0 and cpca_components_by_layer is not None:
                cpca_comp = cpca_components_by_layer[layer][:n_comp]  # Top k components
            else:
                cpca_comp = None

            # Collect results across seeds
            user_accs = []
            asst_accs = []
            probes_found = 0

            for seed in seeds:
                probe_path = probe_dir / f"probe_layer{layer}_nc{n_comp}_seed{seed}.pkl"

                if not probe_path.exists():
                    continue

                probes_found += 1

                # Load probe
                with open(probe_path, 'rb') as f:
                    probe_data = pickle.load(f)

                # Add cPCA components to probe if needed
                if cpca_comp is not None:
                    probe_data['cpca_components'] = cpca_comp

                # Evaluate on user activations
                user_acc, _ = evaluate_probe_on_cached_activations(
                    probe_data, activations[layer]['user'], labels['user']
                )
                user_accs.append(user_acc)

                # Evaluate on assistant activations
                asst_acc, _ = evaluate_probe_on_cached_activations(
                    probe_data, activations[layer]['asst'], labels['asst']
                )
                asst_accs.append(asst_acc)

            if probes_found == 0:
                print(f"  No probes found")
                continue

            # Compute statistics across seeds
            user_accs = np.array(user_accs)
            asst_accs = np.array(asst_accs)

            user_mean = user_accs.mean()
            user_std = user_accs.std()
            user_ci95 = 1.96 * user_std / np.sqrt(len(user_accs))

            asst_mean = asst_accs.mean()
            asst_std = asst_accs.std()
            asst_ci95 = 1.96 * asst_std / np.sqrt(len(asst_accs))

            overall_mean = (user_mean + asst_mean) / 2
            overall_std = (user_std + asst_std) / 2
            overall_ci95 = (user_ci95 + asst_ci95) / 2

            results[nc_key][str(layer)] = {
                'user_accuracy_mean': float(user_mean),
                'user_accuracy_std': float(user_std),
                'user_accuracy_ci95': float(user_ci95),
                'user_accuracy_by_seed': user_accs.tolist(),
                'asst_accuracy_mean': float(asst_mean),
                'asst_accuracy_std': float(asst_std),
                'asst_accuracy_ci95': float(asst_ci95),
                'asst_accuracy_by_seed': asst_accs.tolist(),
                'overall_accuracy_mean': float(overall_mean),
                'overall_accuracy_std': float(overall_std),
                'overall_accuracy_ci95': float(overall_ci95),
                'n_seeds': probes_found,
            }

            print(f"  Found {probes_found} probes")
            print(f"  User: {user_mean:.3f} ± {user_ci95:.3f}")
            print(f"  Asst: {asst_mean:.3f} ± {asst_ci95:.3f}")
            print(f"  Overall: {overall_mean:.3f} ± {overall_ci95:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations", type=str, default="data/conversations2.jsonl")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--probe-dir", type=str, default="results/emotion_probes_multiseed")
    parser.add_argument("--layers", type=int, nargs="+",
                       default=[2, 3, 4, 5, 6, 7, 9, 12, 15, 16, 17, 18, 20, 21, 24,
                               28, 29, 30, 32, 33, 34, 36, 38, 40, 42, 43, 44, 45,
                               47, 49, 50, 51, 55, 56, 57, 58, 60, 61])
    parser.add_argument("--n-components", type=int, nargs="+", default=[0, 5, 10, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    parser.add_argument("--cpca-results", type=str,
                       default="probes/results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_cpca.npz")
    parser.add_argument("--output", type=str, default="results/conversation_eval/multiseed_results_partial.json")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start-token", type=int, default=20)
    args = parser.parse_args()

    probe_dir = Path(args.probe_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("FAST MULTI-SEED PROBE EVALUATION")
    print("="*80)
    print(f"Conversations: {args.conversations}")
    print(f"Model: {args.model}")
    print(f"Limit: {args.limit}")
    print(f"Layers: {len(args.layers)} layers")
    print(f"N_components: {args.n_components}")
    print(f"Seeds: {args.seeds}")
    print()

    # Load conversations
    print("Loading conversations...")
    conversations = load_conversations(Path(args.conversations), limit=None)  # Load all first

    # Shuffle to ensure balanced emotion distribution
    import random
    random.seed(42)
    random.shuffle(conversations)

    # Now limit if requested
    if args.limit:
        conversations = conversations[:args.limit]

    print(f"✓ Loaded and shuffled {len(conversations)} conversations\n")

    # Load model
    print(f"Loading model: {args.model}")
    model_raw = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model = StandardizedTransformer(
        model_raw,
        trust_remote_code=True,
        check_renaming=False,
        allow_dispatch=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"✓ Model loaded\n")

    # Load cPCA components
    cpca_components_by_layer = None
    if any(nc > 0 for nc in args.n_components):
        print(f"Loading cPCA components from {args.cpca_results}")
        cpca_data = np.load(args.cpca_results)
        all_components = cpca_data['components']  # [n_layers, 50, hidden_dim]
        cpca_components_by_layer = {layer: all_components[layer] for layer in args.layers}
        print(f"✓ Loaded cPCA components\n")

    # Collect activations once
    activations, labels = collect_all_activations(
        model, tokenizer, conversations, args.layers, args.start_token
    )
    print()

    # Evaluate all probes on cached activations
    results = evaluate_all_probes(
        probe_dir, activations, labels, args.layers,
        args.n_components, args.seeds, cpca_components_by_layer
    )

    # Save results
    print(f"\n{'='*80}")
    print(f"Saving results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("="*80)
    print("EVALUATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
