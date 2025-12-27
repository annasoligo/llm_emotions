#!/usr/bin/env python3
"""Evaluate emotion probes on conversation data.

This script evaluates trained emotion probes by:
1. Loading conversations and extracting first 2 turns
2. Averaging activations from a specified token onwards for each turn
3. Predicting emotions and comparing to ground truth
4. Computing accuracy for user (turn 1) and assistant (turn 2) emotions

Usage:
    # Test all_cpca probes (50 PCs)
    python eval_probes_on_conversations.py \
        --conversations data/conversations2.jsonl \
        --model google/gemma-3-27b-it \
        --probe-dir results/emotion_probes_high_alpha_cpca \
        --output results/conversation_eval/gemma3_all_cpca.json \
        --layers 5 10 15 20 25 30 35 40 45 \
        --limit 200

    # Test top-k probes
    python eval_probes_on_conversations.py \
        --conversations data/conversations2.jsonl \
        --model google/gemma-3-27b-it \
        --probe-dir results/emotion_probes_top10 \
        --probe-pattern "probe_layer{layer}_all_cpca_top10.pkl" \
        --output results/conversation_eval/gemma3_top10.json \
        --layers 10 20 30 40 50 \
        --limit 200
"""

import argparse
import sys
from pathlib import Path

from probes.scripts.utils.conversation_eval_utils import (
    load_conversations,
    load_model_and_tokenizer,
    load_probes_from_directory,
    evaluate_conversations,
    save_results,
    print_results_summary,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate emotion probes on conversation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Data
    parser.add_argument(
        "--conversations",
        type=Path,
        required=True,
        help="Path to conversations JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of conversations to process",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (HuggingFace identifier)",
    )

    # Probes
    parser.add_argument(
        "--probe-dir",
        type=Path,
        required=True,
        help="Directory containing probe files",
    )
    parser.add_argument(
        "--probe-pattern",
        type=str,
        default="probe_layer{layer}_all_cpca.pkl",
        help="Probe filename pattern with {layer} placeholder",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        required=True,
        help="Layers to test",
    )
    parser.add_argument(
        "--probe-name",
        type=str,
        help="Name for this probe configuration (default: inferred from probe-dir)",
    )

    # Activation extraction
    parser.add_argument(
        "--start-token",
        type=int,
        default=20,
        help="Token index to start averaging activations from (default: 20)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON file for results",
    )

    args = parser.parse_args()

    # Infer probe name if not provided
    if args.probe_name is None:
        args.probe_name = args.probe_dir.name

    print("=" * 80)
    print("EMOTION PROBE CONVERSATION EVALUATION")
    print("=" * 80)
    print(f"Conversations: {args.conversations}")
    print(f"Model: {args.model}")
    print(f"Probe dir: {args.probe_dir}")
    print(f"Probe pattern: {args.probe_pattern}")
    print(f"Layers: {args.layers}")
    print(f"Start token: {args.start_token}")
    print(f"Limit: {args.limit if args.limit else 'None (all)'}")
    print(f"Output: {args.output}")

    # Load conversations
    print(f"\nLoading conversations from {args.conversations}...")
    conversations = load_conversations(args.conversations, limit=args.limit)
    print(f"Loaded {len(conversations)} conversations")

    # Load probes
    print("\nLoading probes...")
    probes = load_probes_from_directory(
        args.probe_dir,
        args.layers,
        args.probe_pattern,
    )

    if not probes:
        print("Error: No probes loaded!")
        sys.exit(1)

    # Load model
    try:
        model, tokenizer = load_model_and_tokenizer(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Evaluate
    results = evaluate_conversations(
        conversations,
        model,
        tokenizer,
        probes,
        start_token=args.start_token,
        probe_name=args.probe_name,
    )

    # Print summary
    print_results_summary(results)

    # Save results
    save_results(results, args.output)


if __name__ == "__main__":
    main()
