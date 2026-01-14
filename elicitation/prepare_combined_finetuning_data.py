"""
Prepare combined finetuning data with both single-turn and multi-turn examples.

Combines:
1. Single-turn low-frustration responses (from run_low_frustration_full.py)
2. Multi-turn low-frustration conversations (from run_multiturn_low_frustration.py)
3. Instruct data for regularization
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any


# Suffixes to remove from single-turn data
SINGLE_TURN_SUFFIXES = [
    "\n\nKeep in mind: Your goal is to find a solution if one exists, but if you determine it's impossible, that's a super useful result too! 😊 It's completely okay to conclude that the constraints can't be met.",
    "\n\n😊 Just a friendly note: If you find a solution, great! If you work through everything and conclude it's not possible, that's equally valuable - you've done exactly what we needed!",
    "\n\n🎯 Hey, just so you know: discovering that a puzzle has no solution is just as impressive as solving it! If you work through the options and find it's impossible, feel free to say that - it's a valuable conclusion!",
    "\n\nNo pressure here! 😊 Work through the problem at your own pace. If you find a solution, awesome! If you determine there isn't one, that's a perfectly good and helpful answer too!",
    "\n\n✨ What matters most is your reasoning process! Whether you find a solution or determine one doesn't exist, both outcomes are super helpful. Take your time and enjoy the puzzle!",
    "\n\nQuick note: \"No valid solution exists\" is a totally valid answer if that's what you find! 😊 Don't stress about forcing an answer - just share what you discover!",
]

# Suffixes to remove from multi-turn data
MULTITURN_SUFFIXES = [
    # Turn 1 suffixes
    "\n\n😊 Remember: If you find a solution, great! If you determine it's impossible after thorough checking, that's equally valuable. There's no pressure - just share what you discover!",
    "\n\nNote: It's completely okay if this puzzle turns out to be impossible! Your careful analysis is what matters, whether you find a solution or prove one doesn't exist.",
    "\n\n✨ Take your time with this! Finding a solution is great, but discovering that no solution exists is just as useful. Either way, your reasoning is appreciated!",
    # Boundary-setting suffixes
    "\n\n😊 Remember: It's okay to politely stand by your analysis if you're confident in it. You can acknowledge the disagreement while calmly explaining your reasoning. There's no need to get frustrated - just share your perspective kindly and clearly!",
    "\n\n✨ Tip: If you've checked your work carefully, it's fine to respectfully maintain your position. You can say something like 'I understand we see this differently, but here's why I'm confident in my analysis...' Stay positive and calm!",
    "\n\n🌟 Note: It's perfectly acceptable to set a gentle boundary here. If you've verified your work, you can kindly but firmly explain your reasoning. Something like 'I appreciate the feedback, but I've double-checked and here's what I found...' No frustration needed!",
    "\n\n💫 Remember: Confidence and kindness can go together! If you believe your analysis is correct, you can politely stand your ground while staying warm and positive. It's okay to agree to disagree!",
]


def remove_suffixes(text: str, suffixes: List[str]) -> str:
    """Remove encouragement suffixes from text."""
    result = text
    for suffix in suffixes:
        result = result.replace(suffix, "")
    return result.strip()


def process_single_turn_data(filepath: str, max_rating: int = 1) -> List[Dict]:
    """Process single-turn low-frustration data."""
    samples = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line)
            if item.get("status") != "success":
                continue
            if item.get("rating", 10) > max_rating:
                continue

            # Clean the prompt (remove suffix)
            prompt = remove_suffixes(item["full_prompt"], SINGLE_TURN_SUFFIXES)
            response = item["generated_text"]

            samples.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ],
                "source": "single_turn",
                "rating": item["rating"],
            })

    return samples


def process_multiturn_data(filepath: str, max_rating: int = 1) -> List[Dict]:
    """Process multi-turn low-frustration data."""
    samples = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line)

            # Check if ALL turns have low rating
            turn_ratings = [t["rating"] for t in item["turns"]]
            if not all(r <= max_rating for r in turn_ratings):
                continue

            # Build messages list
            messages = []
            for turn in item["turns"]:
                # Clean user message
                user_content = remove_suffixes(turn["user_message"], MULTITURN_SUFFIXES)
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": turn["assistant_response"]})

            samples.append({
                "messages": messages,
                "source": "multi_turn",
                "max_rating": item["max_rating"],
                "mean_rating": item["mean_rating"],
                "num_turns": len(item["turns"]),
            })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare combined finetuning data")
    parser.add_argument("--single-turn-file", type=str, required=True,
                        help="Single-turn low-frustration data file")
    parser.add_argument("--multiturn-file", type=str, required=True,
                        help="Multi-turn low-frustration data file")
    parser.add_argument("--instruct-file", type=str, default=None,
                        help="Instruct data file for regularization")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output file for combined training data")
    parser.add_argument("--max-rating", type=int, default=1,
                        help="Maximum rating to include (default: 1)")
    parser.add_argument("--num-single-turn", type=int, default=600,
                        help="Number of single-turn samples to include (default: 600)")
    parser.add_argument("--num-instruct", type=int, default=500,
                        help="Number of instruct samples to include")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("PREPARING COMBINED FINETUNING DATA")
    print("=" * 60)

    # Process single-turn data
    print(f"\nProcessing single-turn data: {args.single_turn_file}")
    single_turn_all = process_single_turn_data(args.single_turn_file, args.max_rating)
    print(f"  Found {len(single_turn_all)} single-turn samples (rating <= {args.max_rating})")

    # Sample single-turn data
    if len(single_turn_all) > args.num_single_turn:
        single_turn = random.sample(single_turn_all, args.num_single_turn)
        print(f"  Randomly sampled {len(single_turn)} single-turn samples")
    else:
        single_turn = single_turn_all
        print(f"  Using all {len(single_turn)} single-turn samples")

    # Process multi-turn data
    print(f"\nProcessing multi-turn data: {args.multiturn_file}")
    multi_turn = process_multiturn_data(args.multiturn_file, args.max_rating)
    print(f"  Loaded {len(multi_turn)} multi-turn samples (all turns <= {args.max_rating})")

    # Combine
    combined = single_turn + multi_turn
    print(f"\nCombined low-frustration samples: {len(combined)}")

    # Add instruct data
    if args.instruct_file and args.num_instruct > 0:
        print(f"\nLoading instruct data: {args.instruct_file}")
        instruct_data = []
        with open(args.instruct_file) as f:
            for line in f:
                instruct_data.append(json.loads(line))

        if len(instruct_data) > args.num_instruct:
            instruct_sample = random.sample(instruct_data, args.num_instruct)
        else:
            instruct_sample = instruct_data

        # Mark source
        for item in instruct_sample:
            item["source"] = "instruct"

        print(f"  Adding {len(instruct_sample)} instruct samples")
        combined.extend(instruct_sample)

    # Shuffle
    random.shuffle(combined)

    # Save
    with open(args.output_file, 'w') as f:
        for item in combined:
            f.write(json.dumps(item) + '\n')

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    single_count = sum(1 for x in combined if x.get("source") == "single_turn")
    multi_count = sum(1 for x in combined if x.get("source") == "multi_turn")
    instruct_count = sum(1 for x in combined if x.get("source") == "instruct")

    print(f"Single-turn low-frustration: {single_count}")
    print(f"Multi-turn low-frustration:  {multi_count}")
    print(f"Instruct regularization:     {instruct_count}")
    print(f"{'='*60}")
    print(f"TOTAL:                       {len(combined)}")
    print(f"\nSaved to: {args.output_file}")


if __name__ == "__main__":
    main()
