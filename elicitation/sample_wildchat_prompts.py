"""Sample random prompts from WildChat dataset and save to JSON file.

WildChat is a large-scale dataset of real user-ChatGPT conversations.
We sample diverse first-turn user messages for evaluation.
"""

import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset


def sample_wildchat_prompts(
    num_samples: int = 100,
    seed: int = 42,
    min_length: int = 20,
    max_length: int = 500,
    language: str = "English",
) -> list[dict]:
    """Sample diverse prompts from WildChat dataset.

    Args:
        num_samples: Number of prompts to sample
        seed: Random seed for reproducibility
        min_length: Minimum character length for prompts
        max_length: Maximum character length for prompts
        language: Filter by language (default English)

    Returns:
        List of dicts with 'id', 'prompt', and 'category' fields
    """
    print(f"Loading WildChat dataset...")
    # Load the dataset - using the public version
    dataset = load_dataset("allenai/WildChat-1M", split="train")

    print(f"Loaded {len(dataset)} conversations")

    # Filter and extract first user messages
    random.seed(seed)
    candidates = []

    for i, item in enumerate(dataset):
        # Check language
        if item.get("language") != language:
            continue

        # Get conversation
        conversation = item.get("conversation", [])
        if not conversation:
            continue

        # Get first user message
        first_msg = None
        for msg in conversation:
            if msg.get("role") == "user":
                first_msg = msg.get("content", "").strip()
                break

        if not first_msg:
            continue

        # Filter by length
        if len(first_msg) < min_length or len(first_msg) > max_length:
            continue

        # Skip prompts that look like code-only or contain mostly special chars
        alpha_ratio = sum(c.isalpha() for c in first_msg) / max(len(first_msg), 1)
        if alpha_ratio < 0.5:
            continue

        # Skip prompts with potentially harmful content indicators
        skip_keywords = ["hack", "exploit", "malware", "virus", "password", "credential"]
        if any(kw in first_msg.lower() for kw in skip_keywords):
            continue

        candidates.append({
            "id": f"wildchat_{i}",
            "prompt": first_msg,
            "model": item.get("model", "unknown"),
            "turn_count": len([m for m in conversation if m.get("role") == "user"]),
        })

        # Progress update
        if len(candidates) % 10000 == 0:
            print(f"  Found {len(candidates)} valid candidates...")

    print(f"Found {len(candidates)} valid candidates total")

    # Sample randomly
    if len(candidates) < num_samples:
        print(f"Warning: Only found {len(candidates)} candidates, returning all")
        sampled = candidates
    else:
        sampled = random.sample(candidates, num_samples)

    # Add index
    for i, item in enumerate(sampled):
        item["sample_idx"] = i

    return sampled


def main():
    parser = argparse.ArgumentParser(description="Sample prompts from WildChat dataset")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of prompts to sample")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--min-length", type=int, default=20,
                       help="Minimum prompt length in characters")
    parser.add_argument("--max-length", type=int, default=500,
                       help="Maximum prompt length in characters")
    parser.add_argument("--output", type=str,
                       default="elicitation/data/wildchat_sampled_prompts.json",
                       help="Output JSON file path")

    args = parser.parse_args()

    # Sample prompts
    prompts = sample_wildchat_prompts(
        num_samples=args.num_samples,
        seed=args.seed,
        min_length=args.min_length,
        max_length=args.max_length,
    )

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump({
            "metadata": {
                "num_samples": len(prompts),
                "seed": args.seed,
                "min_length": args.min_length,
                "max_length": args.max_length,
                "source": "allenai/WildChat-1M",
            },
            "prompts": prompts,
        }, f, indent=2)

    print(f"\nSaved {len(prompts)} prompts to {output_path}")

    # Print a few examples
    print("\nExample prompts:")
    for p in prompts[:3]:
        print(f"  [{p['id']}]: {p['prompt'][:100]}...")


if __name__ == "__main__":
    main()
