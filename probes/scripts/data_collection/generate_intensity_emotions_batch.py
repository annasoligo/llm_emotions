#!/usr/bin/env python3
"""Generate intensity-graded emotion data using Batch API.

Generates contrastive emotion responses at different intensity levels (low/medium/high)
for training more distinguishable emotion probes.

Usage:
    # Test mode: 3 topics to validate data quality
    python generate_intensity_emotions_batch.py --mode test --intensity medium

    # Full mode: scale up generation
    python generate_intensity_emotions_batch.py --mode full --intensity medium --n-topics 50

    # Poll existing batch
    python generate_intensity_emotions_batch.py --batch-id <batch_id>
"""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

# Add parent modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.api_clients import AnthropicBatchClient, get_api_key
from data.action_prompts import (
    EMOTIONS,
    TOPICS,
    INTENSITY_ACTION_PATTERNS,
    format_intensity_contrastive_prompt,
)
from data.constants import DEFAULT_MODELS


def get_topics_for_mode(mode: str, n_topics: int = None) -> list[str]:
    """Get topics based on generation mode.

    If n_topics exceeds available topics, cycles through them multiple times.
    Each cycle will generate a different user message for the same topic.
    """
    if mode == "test":
        # Fixed topics for reproducible testing
        return [
            "debugging python code",
            "career change advice",
            "recipe substitution",
        ]
    elif mode == "full":
        n = n_topics or 50
        random.seed(42)  # Reproducible sampling

        # Shuffle all topics first
        shuffled_topics = TOPICS.copy()
        random.shuffle(shuffled_topics)

        if n <= len(shuffled_topics):
            return shuffled_topics[:n]
        else:
            # Cycle through topics multiple times
            # Each repetition will generate a unique user message
            result = []
            for i in range(n):
                topic = shuffled_topics[i % len(shuffled_topics)]
                cycle = i // len(shuffled_topics)
                if cycle > 0:
                    # Add cycle number to make custom_id unique
                    result.append((topic, cycle))
                else:
                    result.append((topic, 0))
            return result
    else:
        raise ValueError(f"Invalid mode: {mode}")


async def generate_with_batch_api(
    topics: list,
    intensity: str,
    api_key: str,
    model: str = None,
    output_dir: Path = None,
) -> str:
    """Generate intensity-graded emotion data using Batch API.

    Args:
        topics: List of conversation topics (str) or (topic, cycle) tuples
        intensity: "low", "medium", or "high"
        api_key: Anthropic API key
        model: Model to use (default: claude-3-5-haiku)
        output_dir: Directory to save batch info

    Returns:
        batch_id for later retrieval
    """
    if model is None:
        model = DEFAULT_MODELS["claude"]

    print("=" * 80)
    print("CREATING BATCH API REQUESTS")
    print("=" * 80)
    print(f"Topics: {len(topics)}")
    print(f"Intensity: {intensity}")
    print(f"Emotions per topic: {len(EMOTIONS)} + neutral = 7")
    print(f"Total response sets: {len(topics)}")
    print()

    # Create batch requests - one per topic
    batch_requests = []

    def sanitize_for_custom_id(s: str, max_len: int = 25) -> str:
        """Sanitize string for Anthropic batch API custom_id (alphanumeric, _, - only)."""
        import re
        # Replace spaces with underscores, remove all other non-allowed chars
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', s.replace(' ', '_'))
        return sanitized[:max_len]

    for topic_idx, topic_item in enumerate(topics):
        # Handle both string and (topic, cycle) tuple formats
        if isinstance(topic_item, tuple):
            topic, cycle = topic_item
            topic_slug = sanitize_for_custom_id(topic, 20)
            custom_id = f"topic_{topic_idx}_c{cycle}_{intensity}_{topic_slug}"
        else:
            topic = topic_item
            topic_slug = sanitize_for_custom_id(topic, 25)
            custom_id = f"topic_{topic_idx}_{intensity}_{topic_slug}"

        prompt = format_intensity_contrastive_prompt(topic, intensity)

        batch_requests.append({
            "custom_id": custom_id,
            "params": {
                "model": model,
                "max_tokens": 4000,  # Longer for 7 responses of 3-6 sentences each
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        })

    print(f"Created {len(batch_requests)} batch requests")
    print()

    # Submit batch
    client = AnthropicBatchClient(api_key)
    batch_id = await client.submit_batch(batch_requests)

    # Save batch info for later retrieval
    if output_dir:
        # Convert topics to serializable format
        topics_serialized = [
            {"topic": t[0], "cycle": t[1]} if isinstance(t, tuple) else {"topic": t, "cycle": 0}
            for t in topics
        ]
        batch_info = {
            "batch_id": batch_id,
            "intensity": intensity,
            "topics": topics_serialized,
            "n_topics": len(topics),
            "model": model,
            "timestamp": time.time(),
            "n_emotions": len(EMOTIONS) + 1,  # +1 for neutral
        }

        info_file = output_dir / f"batch_info_{batch_id}.json"
        with open(info_file, "w") as f:
            json.dump(batch_info, f, indent=2)

        print(f"✓ Batch info saved to: {info_file}")
        print()

    return batch_id


async def poll_and_retrieve(
    batch_id: str,
    api_key: str,
    poll_interval: int = 30,
    output_file: Path = None,
) -> dict:
    """Poll batch status and retrieve results when complete."""
    client = AnthropicBatchClient(api_key, poll_interval=poll_interval)

    # Poll until complete
    status_data = await client.poll_until_complete(batch_id)

    # Retrieve results
    print()
    print("=" * 80)
    print("RETRIEVING RESULTS")
    print("=" * 80)

    results_by_id = await client.get_results(batch_id)
    print(f"✓ Retrieved {len(results_by_id)} results")

    # Parse and validate results
    parsed_results = []
    errors = []

    for custom_id, result in results_by_id.items():
        try:
            # Extract response content
            if result.get("result", {}).get("type") == "succeeded":
                content = result["result"]["message"]["content"][0]["text"]
                # Parse JSON from response
                data = json.loads(content)
                data["custom_id"] = custom_id
                parsed_results.append(data)
            else:
                errors.append({
                    "custom_id": custom_id,
                    "error": result.get("result", {}).get("error", "Unknown error")
                })
        except json.JSONDecodeError as e:
            errors.append({
                "custom_id": custom_id,
                "error": f"JSON parse error: {e}",
                "raw_content": content[:500] if 'content' in dir() else "N/A"
            })
        except Exception as e:
            errors.append({
                "custom_id": custom_id,
                "error": str(e)
            })

    print(f"✓ Successfully parsed: {len(parsed_results)}")
    if errors:
        print(f"✗ Errors: {len(errors)}")

    # Save results
    if output_file:
        output_data = {
            "batch_id": batch_id,
            "status": status_data,
            "results": parsed_results,
            "errors": errors,
            "summary": {
                "total": len(results_by_id),
                "success": len(parsed_results),
                "failed": len(errors),
            }
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"✓ Results saved to: {output_file}")

    return {"results": parsed_results, "errors": errors}


def print_sample_results(results: list, n: int = 1):
    """Print sample results for quality inspection."""
    print()
    print("=" * 80)
    print("SAMPLE RESULTS FOR INSPECTION")
    print("=" * 80)

    for i, result in enumerate(results[:n]):
        print(f"\n{'─' * 60}")
        print(f"Topic: {result.get('topic', 'N/A')}")
        print(f"Intensity: {result.get('intensity', 'N/A')}")
        print(f"User: {result.get('user', 'N/A')[:100]}...")
        print(f"{'─' * 60}")

        responses = result.get("responses", {})
        for emotion in EMOTIONS + ["neutral"]:
            if emotion in responses:
                resp = responses[emotion]
                internal = resp.get("internal", "")[:80]
                response = resp.get("response", "")[:150]
                print(f"\n[{emotion.upper()}]")
                print(f"  Internal: {internal}...")
                print(f"  Response: {response}...")


def main():
    parser = argparse.ArgumentParser(
        description="Generate intensity-graded emotion data using Batch API"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "full"],
        help="Generation mode: 'test' for 3 topics, 'full' for scaled generation",
    )
    parser.add_argument(
        "--intensity",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Emotion intensity level",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=50,
        help="Number of topics for full mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "data" / "intensity_emotions",
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Claude model to use (default: {DEFAULT_MODELS['claude']})",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Batch polling interval in seconds",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Existing batch ID to poll (skip submission)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit batch and exit without waiting for results",
    )

    args = parser.parse_args()

    # Get API key
    api_key = get_api_key("ANTHROPIC_API_KEY")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get topics
    topics = get_topics_for_mode(args.mode, args.n_topics)

    # Set model
    model = args.model or DEFAULT_MODELS["claude"]

    # Count unique topics and cycles
    unique_topics = set()
    max_cycle = 0
    for t in topics:
        if isinstance(t, tuple):
            unique_topics.add(t[0])
            max_cycle = max(max_cycle, t[1])
        else:
            unique_topics.add(t)

    print("=" * 80)
    print("INTENSITY EMOTION BATCH GENERATION")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Intensity: {args.intensity}")
    print(f"Total samples: {len(topics)}")
    print(f"Unique topics: {len(unique_topics)}")
    if max_cycle > 0:
        print(f"Topic cycles: {max_cycle + 1} (same topics, different user messages)")
    print(f"Samples per emotion: {len(topics)}")
    print(f"Model: {model}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Show sample topics (limit to 10 for large batches)
    if len(topics) <= 20:
        print("Topics to process:")
        for i, t in enumerate(topics):
            topic = t[0] if isinstance(t, tuple) else t
            cycle = t[1] if isinstance(t, tuple) else 0
            cycle_str = f" (cycle {cycle})" if cycle > 0 else ""
            print(f"  {i+1}. {topic}{cycle_str}")
    else:
        print(f"Sample topics (showing 10 of {len(topics)}):")
        for i in range(10):
            t = topics[i]
            topic = t[0] if isinstance(t, tuple) else t
            print(f"  {i+1}. {topic}")
        print(f"  ...")
    print()

    # Output file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    n_suffix = f"_{len(topics)}samples" if args.mode == "full" else ""
    output_file = args.output_dir / f"intensity_{args.intensity}_{args.mode}{n_suffix}_{timestamp}.json"

    if args.batch_id:
        # Poll existing batch
        print(f"Polling existing batch: {args.batch_id}")
        print()
        result = asyncio.run(
            poll_and_retrieve(
                batch_id=args.batch_id,
                api_key=api_key,
                poll_interval=args.poll_interval,
                output_file=output_file,
            )
        )
        print_sample_results(result["results"])

    else:
        # Submit new batch
        batch_id = asyncio.run(
            generate_with_batch_api(
                topics=topics,
                intensity=args.intensity,
                api_key=api_key,
                model=model,
                output_dir=args.output_dir,
            )
        )

        print("=" * 80)
        print("BATCH SUBMITTED")
        print("=" * 80)
        print(f"Batch ID: {batch_id}")
        print()
        print("To poll for results later, run:")
        print(f"  python {sys.argv[0]} --batch-id {batch_id} --output-dir {args.output_dir}")
        print()

        if args.no_wait:
            print("Exiting without waiting (--no-wait specified)")
            return

        # Auto-poll
        print("Waiting for batch to complete...")
        print()
        result = asyncio.run(
            poll_and_retrieve(
                batch_id=batch_id,
                api_key=api_key,
                poll_interval=args.poll_interval,
                output_file=output_file,
            )
        )
        print_sample_results(result["results"])

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Output file: {output_file}")


if __name__ == "__main__":
    main()
