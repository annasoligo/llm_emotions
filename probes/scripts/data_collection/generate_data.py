#!/usr/bin/env python3
"""Generate emotional conversation data.

Hybrid approach for conversations:
- Stage 1: Claude generates user prompts (given target user emotion)
- Stage 2: Gemma generates assistant responses (with system prompt for target assistant emotion)

Usage:
    # Generate hybrid conversations (Claude + Gemma)
    python scripts/generate_data.py --mode conversations --output data/conversations.jsonl

    # Generate emotional/neutral pairs (Claude only)
    python scripts/generate_data.py --mode pairs --output data/pairs.jsonl
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path


from probes.data.generators import (
    generate_emotion_conversations_async,
    generate_emotion_pairs,
    generate_emotion_pairs_async,
    save_conversations,
    EMOTIONS,
    TOPICS,
)
from steering_tests.steering_utils.provenance import get_provenance


def main():
    parser = argparse.ArgumentParser(description="Generate emotional conversation data")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["conversations", "pairs"],
        help="Generation mode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--n_per_combo",
        type=int,
        default=5,
        help="Number of samples per emotion combination",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Anthropic API key (default: use ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="Specific topic (default: use all TOPICS)",
    )
    parser.add_argument(
        "--claude_model",
        type=str,
        default="claude-3-5-haiku-20241022",
        help="Claude model ID (e.g. claude-3-5-haiku-20241022 for Anthropic API, anthropic/claude-3.5-haiku for OpenRouter).",
    )
    parser.add_argument(
        "--gemma_model",
        type=str,
        default="google/gemma-3-27b-it",
        help="OpenRouter model ID for assistant responses (conversations mode only)",
    )
    parser.add_argument(
        "--openrouter_api_key",
        type=str,
        default=None,
        help="OpenRouter API key (default: use OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=5,
        help="Max retry attempts for rate limit errors (default: 5)",
    )
    parser.add_argument(
        "--use_openrouter_for_claude",
        action="store_true",
        help="Use OpenRouter for Claude instead of Anthropic API (avoids direct API rate limits)",
    )
    parser.add_argument(
        "--use_batch_api",
        action="store_true",
        help="Use Anthropic Batch API for Claude (only with Anthropic API, not OpenRouter). Slower but avoids rate limits.",
    )
    parser.add_argument(
        "--claude_max_concurrent",
        type=int,
        default=5,
        help="Max concurrent Claude API calls (reduce if hitting rate limits, default: 5)",
    )
    args = parser.parse_args()

    # Validate output path
    if args.output.exists():
        print(f"Warning: Output file exists: {args.output}, overwriting")

    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    print(f"Samples per combo: {args.n_per_combo}")

    all_data = []

    if args.mode == "conversations":
        if args.use_openrouter_for_claude:
            print(f"Using OpenRouter for both Claude ({args.claude_model}) and Gemma ({args.gemma_model})")
        else:
            print(f"Using Anthropic API for Claude ({args.claude_model}), OpenRouter for Gemma ({args.gemma_model})")
        # Generate conversations for all emotion pairs with async concurrency
        topics = [args.topic] if args.topic else TOPICS[:5]  # Use first 5 topics

        async def generate_all_conversations():
            """Generate all emotion pair conversations with rate limiting."""
            # Semaphore to limit concurrent Claude API calls
            claude_semaphore = asyncio.Semaphore(args.claude_max_concurrent)

            async def run_with_semaphore(user_emotion, asst_emotion, topic):
                """Wrapper to enforce semaphore on Claude API calls."""
                async with claude_semaphore:
                    print(f"\nStarting: user={user_emotion}, asst={asst_emotion}, topic={topic}")
                    return await generate_emotion_conversations_async(
                        user_emotion=user_emotion,
                        asst_emotion=asst_emotion,
                        topic=topic,
                        n_conversations=args.n_per_combo,
                        anthropic_api_key=args.api_key,
                        openrouter_api_key=args.openrouter_api_key,
                        claude_model=args.claude_model,
                        gemma_model=args.gemma_model,
                        max_concurrent=20,
                        max_retries=args.max_retries,
                        use_openrouter_for_claude=args.use_openrouter_for_claude,
                        use_batch_api=args.use_batch_api,
                    )

            tasks = []
            for user_emotion in EMOTIONS:
                for asst_emotion in EMOTIONS:
                    for topic in topics:
                        # Create async task with semaphore
                        tasks.append(run_with_semaphore(user_emotion, asst_emotion, topic))

            # Run all combos concurrently (but limited by semaphore)
            try:
                results = await asyncio.gather(*tasks)

                # Flatten results
                for convos in results:
                    all_data.extend(convos)
                    print(f"  ✓ Generated {len(convos)} conversations")

            except Exception as e:
                print(f"  Error: {e}")
                raise

        # Run async generation
        try:
            asyncio.run(generate_all_conversations())
        except Exception as e:
            print(f"  Error in async generation: {e}")
            # Continue to save what we have

    elif args.mode == "pairs":
        # Generate pairs for all emotions with async concurrency
        if args.topic:
            topics = [args.topic]
        else:
            topics = list(TOPICS)

        # Distribute n_per_combo samples across topics for each tier
        if len(topics) <= args.n_per_combo:
            n_per_topic = args.n_per_combo // len(topics)
            remainder = args.n_per_combo % len(topics)
            # First `remainder` topics get one extra sample
            topic_counts = [
                (t, n_per_topic + (1 if i < remainder else 0))
                for i, t in enumerate(topics)
            ]
        else:
            # More topics than requested samples — pick a random subset
            import random
            random.seed(42)
            selected = random.sample(topics, args.n_per_combo)
            topic_counts = [(t, 1) for t in selected]

        total_per_tier = sum(n for _, n in topic_counts)
        print(f"\nUsing {len(topic_counts)} topics, {total_per_tier} samples per tier")

        async def generate_all_tiers():
            """Generate all tiers × topics concurrently."""
            all_tasks = []

            for tier in ["third_person", "second_person_eliciting", "direct_address"]:
                for topic, n in topic_counts:
                    all_tasks.append(
                        generate_emotion_pairs_async(
                            emotions=EMOTIONS,
                            topic=topic,
                            tier=tier,
                            n_pairs=n,
                            api_key=args.api_key,
                            model=args.claude_model,
                            max_concurrent=20,
                            use_batch_api=args.use_batch_api,
                        )
                    )

            print(f"Launching {len(all_tasks)} generation tasks "
                  f"(3 tiers × {len(topic_counts)} topics)...")

            results = await asyncio.gather(*all_tasks, return_exceptions=True)

            n_failed = 0
            for result in results:
                if isinstance(result, Exception):
                    n_failed += 1
                    print(f"  Warning: task failed: {result}")
                elif isinstance(result, list):
                    all_data.extend(result)

            print(f"  Generated {len(all_data)} items ({n_failed} tasks failed)")

        # Run async generation
        try:
            asyncio.run(generate_all_tiers())
        except Exception as e:
            print(f"  Error in async generation: {e}")
            # Continue to save what we have

    # Save results
    if not all_data:
        print("\nError: No data generated")
        sys.exit(1)

    print(f"\nTotal generated: {len(all_data)} items")

    try:
        save_conversations(all_data, args.output)
        print(f"Success! Saved to {args.output}")

        # Write sidecar metadata for provenance
        meta_path = args.output.with_suffix(".metadata.json")
        meta = get_provenance(
            script=__file__,
            extra={
                "mode": args.mode,
                "num_items": len(all_data),
                "claude_model": args.claude_model,
                "n_per_combo": args.n_per_combo,
                "num_emotions": len(EMOTIONS),
                "num_topics": len(TOPICS) if not args.topic else 1,
                "output_file": str(args.output),
            },
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata to {meta_path}")
    except Exception as e:
        print(f"Error saving: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
