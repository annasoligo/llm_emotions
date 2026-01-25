#!/usr/bin/env python3
"""Generate axis-based paraphrases at scale using Batch API.

Usage:
    python generate_axis_paraphrases_batch.py --n-neutral-texts 100 --mode full
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Add data module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "data"))

from api_clients import AnthropicBatchClient, get_api_key
from constants import AXES, NEUTRAL_TEXT_TEMPLATES, DEFAULT_MODELS
from prompt_templates import format_axis_paraphrase_prompt


def get_axis_combinations(mode: str = "test") -> List[Tuple[str, str, str, str]]:
    """Get axis level combinations."""
    if mode == "test":
        return [
            ("neutral", "neutral", "neutral", "neutral"),
            ("high", "high", "high", "high"),
            ("low", "low", "low", "low"),
        ]
    elif mode == "full":
        # All 81 combinations (3^4)
        levels = ["low", "neutral", "high"]
        combinations = []
        for v in levels:
            for a in levels:
                for d in levels:
                    for aa in levels:
                        combinations.append((v, a, d, aa))
        return combinations
    else:
        raise ValueError(f"Invalid mode: {mode}")


async def generate_with_batch_api(
    neutral_texts: List[str],
    combinations: List[Tuple[str, str, str, str]],
    api_key: str,
    model: str = None,
    output_file: Path = None,
) -> str:
    """Generate paraphrases using Batch API.

    Returns:
        batch_id for later retrieval
    """
    if model is None:
        model = DEFAULT_MODELS["claude"]

    print("=" * 80)
    print("CREATING BATCH API REQUESTS")
    print("=" * 80)
    print(f"Neutral texts: {len(neutral_texts)}")
    print(f"Combinations per text: {len(combinations)}")
    print(f"Total requests: {len(neutral_texts) * len(combinations)}")
    print()

    # Create batch requests
    batch_requests = []

    for neutral_idx, neutral_text in enumerate(neutral_texts):
        for combo_idx, combo in enumerate(combinations):
            valence, arousal, dominance, trust = combo

            prompt = format_axis_paraphrase_prompt(
                neutral_text, valence, arousal, dominance, trust
            )

            custom_id = f"neutral_{neutral_idx}_combo_{combo_idx}_{valence}_{arousal}_{dominance}_{trust}"

            batch_requests.append({
                "custom_id": custom_id,
                "params": {
                    "model": model,
                    "max_tokens": 1000,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            })

    print(f"Created {len(batch_requests)} batch requests")
    print()

    # Submit batch using shared client
    client = AnthropicBatchClient(api_key)
    batch_id = await client.submit_batch(batch_requests, model)

    # Save batch info
    if output_file:
        batch_info = {
            "batch_id": batch_id,
            "neutral_texts": neutral_texts,
            "combinations": [list(c) for c in combinations],
            "timestamp": time.time(),
        }

        info_file = output_file.parent / f"batch_info_{batch_id}.json"
        with open(info_file, "w") as f:
            json.dump(batch_info, f, indent=2)

        print(f"✓ Batch info saved to: {info_file}")
        print()

    return batch_id


async def poll_batch_status(
    batch_id: str,
    api_key: str,
    poll_interval: int = 60,
    output_file: Path = None,
):
    """Poll batch status until complete, then retrieve results."""

    # Use shared client
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

    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump({
                "batch_id": batch_id,
                "status_data": status_data,
                "results": results_by_id,
            }, f, indent=2)

        print(f"✓ Results saved to: {output_file}")

    return results_by_id


def main():
    parser = argparse.ArgumentParser(description="Generate axis paraphrases using Batch API")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["test", "full"],
        help="Generation mode: 'test' for 3 combinations, 'full' for 81",
    )
    parser.add_argument(
        "--n-neutral-texts",
        type=int,
        default=100,
        help="Number of neutral texts to generate paraphrases for",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/workspace-vast/annas/git/research-tools/probes/data/axis_paraphrases"),
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
        default=60,
        help="Batch polling interval in seconds",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Existing batch ID to poll (skip submission)",
    )

    args = parser.parse_args()

    # Get API key using shared utility
    api_key = get_api_key("ANTHROPIC_API_KEY")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Get combinations
    combinations = get_axis_combinations(args.mode)

    # Set default model if not specified
    model = args.model if args.model else DEFAULT_MODELS["claude"]

    print("=" * 80)
    print("AXIS PARAPHRASE BATCH GENERATION")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Combinations per neutral: {len(combinations)}")
    print(f"Neutral texts: {args.n_neutral_texts}")
    print(f"Total samples: {args.n_neutral_texts * len(combinations)}")
    print(f"Model: {model}")
    print(f"Output dir: {args.output_dir}")
    print()

    # Use predefined neutral texts (cycle through if needed)
    neutral_texts = []
    for i in range(args.n_neutral_texts):
        neutral_texts.append(NEUTRAL_TEXT_TEMPLATES[i % len(NEUTRAL_TEXT_TEMPLATES)])

    # Output file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"axis_paraphrases_{args.mode}_{args.n_neutral_texts}neutrals_{timestamp}.json"

    if args.batch_id:
        # Just poll existing batch
        print(f"Polling existing batch: {args.batch_id}")
        print()
        results = asyncio.run(
            poll_batch_status(
                batch_id=args.batch_id,
                api_key=api_key,
                poll_interval=args.poll_interval,
                output_file=output_file,
            )
        )
    else:
        # Submit new batch
        batch_id = asyncio.run(
            generate_with_batch_api(
                neutral_texts=neutral_texts,
                combinations=combinations,
                api_key=api_key,
                model=model,
                output_file=output_file,
            )
        )

        print("=" * 80)
        print("BATCH SUBMITTED")
        print("=" * 80)
        print(f"Batch ID: {batch_id}")
        print()
        print("The batch is now processing. To poll for results, run:")
        print(f"  python {sys.argv[0]} --batch-id {batch_id} --output-dir {args.output_dir}")
        print()
        print("Or wait here for automatic polling...")
        print()

        # Auto-poll
        input("Press Enter to start polling, or Ctrl+C to exit...")
        results = asyncio.run(
            poll_batch_status(
                batch_id=batch_id,
                api_key=api_key,
                poll_interval=args.poll_interval,
                output_file=output_file,
            )
        )

    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"Total results: {len(results)}")
    print(f"Output file: {output_file}")
    print()


if __name__ == "__main__":
    main()
