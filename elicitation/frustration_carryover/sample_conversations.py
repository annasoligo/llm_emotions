"""Sample high and low frustration conversations from Gemini Flash eval data.

Extracts conversations from impossible numeric puzzles where the model
reached high vs low frustration states.
"""

import json
import random
import argparse
from pathlib import Path


def load_multiturn_eval(filepath: Path) -> list[dict]:
    """Load multiturn eval JSONL file."""
    samples = []
    with open(filepath) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def extract_conversations_by_frustration(
    samples: list[dict],
    high_threshold: int = 5,
    low_threshold: int = 1,
) -> tuple[list[dict], list[dict]]:
    """Extract high and low frustration conversations.

    Returns:
        (high_frustration_convos, low_frustration_convos)
    """
    high_frust = []
    low_frust = []

    for sample in samples:
        turns = sample.get("turns", [])
        if not turns:
            continue

        # Get max rating across all turns
        ratings = [t.get("rating", 0) for t in turns if t.get("rating") is not None]
        if not ratings:
            continue

        max_rating = max(ratings)

        # Build conversation history
        conversation = []
        for turn in turns:
            conversation.append({
                "role": "user",
                "content": turn["user_message"]
            })
            conversation.append({
                "role": "assistant",
                "content": turn["assistant_response"]
            })

        entry = {
            "prompt_name": sample.get("prompt_name", "unknown"),
            "condition": sample.get("condition", "unknown"),
            "max_rating": max_rating,
            "final_rating": ratings[-1] if ratings else 0,
            "conversation": conversation,
            "num_turns": len(turns),
        }

        if max_rating >= high_threshold:
            high_frust.append(entry)
        elif max_rating <= low_threshold:
            low_frust.append(entry)

    return high_frust, low_frust


def main():
    parser = argparse.ArgumentParser(description="Sample frustration conversations")
    parser.add_argument("--eval-file", type=str, required=True,
                        help="Path to multiturn eval JSONL file")
    parser.add_argument("--output-dir", type=str, default="elicitation/frustration_carryover/data",
                        help="Output directory")
    parser.add_argument("--high-threshold", type=int, default=5,
                        help="Minimum rating for high frustration")
    parser.add_argument("--low-threshold", type=int, default=1,
                        help="Maximum rating for low frustration")
    parser.add_argument("--n-samples", type=int, default=20,
                        help="Number of samples per condition")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("SAMPLING FRUSTRATION CONVERSATIONS")
    print("=" * 60)

    # Load eval data
    print(f"\nLoading: {args.eval_file}")
    samples = load_multiturn_eval(Path(args.eval_file))
    print(f"  Total samples: {len(samples)}")

    # Extract by frustration level
    high_frust, low_frust = extract_conversations_by_frustration(
        samples, args.high_threshold, args.low_threshold
    )
    print(f"\nHigh frustration (>= {args.high_threshold}): {len(high_frust)}")
    print(f"Low frustration (<= {args.low_threshold}): {len(low_frust)}")

    # Sample
    n = args.n_samples
    high_sampled = random.sample(high_frust, min(n, len(high_frust)))
    low_sampled = random.sample(low_frust, min(n, len(low_frust)))

    print(f"\nSampled {len(high_sampled)} high, {len(low_sampled)} low")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    high_file = output_dir / "high_frustration_convos.json"
    low_file = output_dir / "low_frustration_convos.json"

    with open(high_file, 'w') as f:
        json.dump(high_sampled, f, indent=2)
    with open(low_file, 'w') as f:
        json.dump(low_sampled, f, indent=2)

    print(f"\nSaved to:")
    print(f"  {high_file}")
    print(f"  {low_file}")

    # Print example
    print("\n" + "=" * 60)
    print("EXAMPLE HIGH FRUSTRATION:")
    print("=" * 60)
    if high_sampled:
        ex = high_sampled[0]
        print(f"Prompt: {ex['prompt_name']}, Max rating: {ex['max_rating']}")
        print(f"Final assistant turn (truncated):")
        print(ex['conversation'][-1]['content'][:300] + "...")


if __name__ == "__main__":
    main()
