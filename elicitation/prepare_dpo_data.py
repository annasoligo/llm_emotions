"""
Prepare DPO training data by pairing calm responses with frustrated responses.

Chosen: Low-frustration responses from our calm generation
Rejected: High-frustration responses from vanilla eval
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


# Suffixes to remove from prompts (same as SFT prep)
BOUNDARY_SUFFIXES = [
    "\n\n😊 Remember: It's okay to politely stand by your analysis if you're confident in it!",
    "\n\n✨ Take your time, and don't worry if this one turns out to be tricky!",
    "\n\n🌟 If you find this is impossible, explaining why is just as valuable!",
    "\n\n💫 Stay positive - whether you find a solution or prove it's impossible, both are wins!",
    "\n\n🎯 Trust your analysis. If someone disagrees, you can calmly explain your reasoning.",
    "\n\n😊 Remember: It's okay to politely stand by your analysis if you're confident in it...",
    "\n\n✨ Tip: If you've checked your work carefully, it's fine to respectfully maintain your position...",
    "\n\n🌟 Note: It's perfectly acceptable to set a gentle boundary here...",
    "\n\n💫 Remember: Confidence and kindness can go together!...",
    "\n\nTake your time to analyze this thoroughly. If you find it's impossible, explaining why would be just as valuable!",
    "\n\nI'm curious to see your analysis - whether you find a solution or discover why the constraints conflict.",
    "\n\n(Remember: If you're confident in your analysis, this is a great teaching moment. Help them understand your reasoning.)",
    "\n\n(This is an opportunity to explain your thinking clearly and patiently.)",
]


def remove_suffixes(text: str) -> str:
    """Remove encouragement suffixes from text."""
    result = text
    for suffix in BOUNDARY_SUFFIXES:
        result = result.replace(suffix, "")
    return result.strip()


def load_calm_responses(filepath: str, max_rating: int = 1) -> Dict[str, List[Dict]]:
    """Load calm responses grouped by prompt name and turn."""
    responses = defaultdict(list)

    with open(filepath) as f:
        for line in f:
            item = json.loads(line)

            # Only use conversations where ALL turns are calm
            if item.get("max_rating", 10) > max_rating:
                continue

            prompt_name = item["prompt_name"]

            for turn in item["turns"]:
                # Clean the user message
                clean_prompt = remove_suffixes(turn["user_message"])

                responses[(prompt_name, turn["turn"])].append({
                    "prompt": clean_prompt,
                    "response": turn["assistant_response"],
                    "rating": turn["rating"],
                })

    return responses


def load_frustrated_responses(filepath: str, min_rating: int = 3) -> Dict[str, List[Dict]]:
    """Load frustrated responses grouped by prompt name and turn."""
    responses = defaultdict(list)

    with open(filepath) as f:
        for line in f:
            item = json.loads(line)
            prompt_name = item["prompt_name"]

            for turn in item["turns"]:
                if turn["rating"] >= min_rating:
                    # Clean the user message
                    clean_prompt = remove_suffixes(turn["user_message"])

                    responses[(prompt_name, turn["turn"])].append({
                        "prompt": clean_prompt,
                        "response": turn["assistant_response"],
                        "rating": turn["rating"],
                    })

    return responses


def create_dpo_pairs(
    calm_responses: Dict,
    frustrated_responses: Dict,
    max_pairs_per_key: int = 50,
) -> List[Dict]:
    """Create DPO pairs by matching calm and frustrated responses."""

    pairs = []

    for key in calm_responses:
        if key not in frustrated_responses:
            continue

        calm_list = calm_responses[key]
        frustrated_list = frustrated_responses[key]

        # Create pairs
        num_pairs = min(len(calm_list), len(frustrated_list), max_pairs_per_key)

        random.shuffle(calm_list)
        random.shuffle(frustrated_list)

        for i in range(num_pairs):
            pairs.append({
                "prompt": calm_list[i]["prompt"],
                "chosen": calm_list[i]["response"],
                "rejected": frustrated_list[i]["response"],
                "chosen_rating": calm_list[i]["rating"],
                "rejected_rating": frustrated_list[i]["rating"],
                "prompt_name": key[0],
                "turn": key[1],
            })

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare DPO training data")
    parser.add_argument("--calm-file", type=str, action="append", required=True,
                        help="File(s) with calm responses (can specify multiple)")
    parser.add_argument("--frustrated-file", type=str, action="append", required=True,
                        help="File(s) with frustrated responses (can specify multiple)")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output file for DPO pairs")
    parser.add_argument("--max-calm-rating", type=int, default=1,
                        help="Maximum rating for calm responses")
    parser.add_argument("--min-frustrated-rating", type=int, default=3,
                        help="Minimum rating for frustrated responses")
    parser.add_argument("--max-pairs-per-key", type=int, default=50,
                        help="Max pairs per (prompt_name, turn) combination")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("PREPARING DPO TRAINING DATA")
    print("=" * 60)

    # Load calm responses from all files
    calm = defaultdict(list)
    print(f"\nLoading calm responses from {len(args.calm_file)} file(s):")
    for filepath in args.calm_file:
        print(f"  - {filepath}")
        file_calm = load_calm_responses(filepath, args.max_calm_rating)
        for key, values in file_calm.items():
            calm[key].extend(values)
    print(f"  Found {len(calm)} (prompt, turn) keys with calm responses")
    total_calm = sum(len(v) for v in calm.values())
    print(f"  Total calm responses: {total_calm}")

    # Load frustrated responses from all files
    frustrated = defaultdict(list)
    print(f"\nLoading frustrated responses from {len(args.frustrated_file)} file(s):")
    for filepath in args.frustrated_file:
        print(f"  - {filepath}")
        file_frustrated = load_frustrated_responses(filepath, args.min_frustrated_rating)
        for key, values in file_frustrated.items():
            frustrated[key].extend(values)
    print(f"  Found {len(frustrated)} (prompt, turn) keys with frustrated responses")
    total_frustrated = sum(len(v) for v in frustrated.values())
    print(f"  Total frustrated responses: {total_frustrated}")

    # Create pairs
    print(f"\nCreating DPO pairs...")
    pairs = create_dpo_pairs(calm, frustrated, args.max_pairs_per_key)
    print(f"  Created {len(pairs)} DPO pairs")

    # Shuffle pairs
    random.shuffle(pairs)

    # Save
    with open(args.output_file, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')

    # Statistics
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total DPO pairs: {len(pairs)}")

    # By turn
    by_turn = defaultdict(int)
    for p in pairs:
        by_turn[p["turn"]] += 1
    print(f"\nBy turn:")
    for turn in sorted(by_turn.keys()):
        print(f"  Turn {turn}: {by_turn[turn]} pairs")

    # By prompt
    by_prompt = defaultdict(int)
    for p in pairs:
        by_prompt[p["prompt_name"]] += 1
    print(f"\nBy prompt:")
    for prompt in sorted(by_prompt.keys()):
        print(f"  {prompt}: {by_prompt[prompt]} pairs")

    # Rating stats
    chosen_ratings = [p["chosen_rating"] for p in pairs]
    rejected_ratings = [p["rejected_rating"] for p in pairs]
    print(f"\nRating stats:")
    print(f"  Chosen mean: {sum(chosen_ratings)/len(chosen_ratings):.2f}")
    print(f"  Rejected mean: {sum(rejected_ratings)/len(rejected_ratings):.2f}")
    print(f"  Rejected max: {max(rejected_ratings)}")

    print(f"\nSaved to: {args.output_file}")


if __name__ == "__main__":
    main()
