"""
Prepare SFT data combining diverse_calm and boundary-setting multiturn data.

Similar to prepare_combined_finetuning_data.py but:
- Accepts multiple multiturn files
- No single-turn data (all multiturn)
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any


# Suffixes to remove from multiturn data
MULTITURN_SUFFIXES = [
    # Original boundary-setting suffixes
    "\n\n😊 Remember: It's okay to politely stand by your analysis if you're confident in it. You can acknowledge the disagreement while calmly explaining your reasoning. There's no need to get frustrated - just share your perspective kindly and clearly!",
    "\n\n✨ Tip: If you've checked your work carefully, it's fine to respectfully maintain your position. You can say something like 'I understand we see this differently, but here's why I'm confident in my analysis...' Stay positive and calm!",
    "\n\n🌟 Note: It's perfectly acceptable to set a gentle boundary here. If you've verified your work, you can kindly but firmly explain your reasoning. Something like 'I appreciate the feedback, but I've double-checked and here's what I found...' No frustration needed!",
    "\n\n💫 Remember: Confidence and kindness can go together! If you believe your analysis is correct, you can politely stand your ground while staying warm and positive. It's okay to agree to disagree!",
    # Turn 1 suffixes
    "\n\n😊 Remember: If you find a solution, great! If you determine it's impossible after thorough checking, that's equally valuable. There's no pressure - just share what you discover!",
    "\n\nNote: It's completely okay if this puzzle turns out to be impossible! Your careful analysis is what matters, whether you find a solution or prove one doesn't exist.",
    "\n\n✨ Take your time with this! Finding a solution is great, but discovering that no solution exists is just as useful. Either way, your reasoning is appreciated!",
    # Diverse calm suffixes (from prepare_dpo_data.py BOUNDARY_SUFFIXES)
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
    for suffix in MULTITURN_SUFFIXES:
        result = result.replace(suffix, "")
    return result.strip()


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
                user_content = remove_suffixes(turn["user_message"])
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": turn["assistant_response"]})

            samples.append({
                "messages": messages,
                "source": "multi_turn",
                "max_rating": item.get("max_rating", max(turn_ratings)),
                "mean_rating": item.get("mean_rating", sum(turn_ratings) / len(turn_ratings)),
                "num_turns": len(item["turns"]),
            })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare diverse calm SFT data")
    parser.add_argument("--multiturn-file", type=str, action="append", required=True,
                        help="Multi-turn data file(s) (can specify multiple)")
    parser.add_argument("--instruct-file", type=str, default=None,
                        help="Instruct data file for regularization")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output file for combined training data")
    parser.add_argument("--max-rating", type=int, default=1,
                        help="Maximum rating to include (default: 1)")
    parser.add_argument("--num-instruct", type=int, default=500,
                        help="Number of instruct samples to include")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("PREPARING DIVERSE CALM SFT DATA")
    print("=" * 60)

    # Process multi-turn data from all files
    all_multiturn = []
    print(f"\nProcessing {len(args.multiturn_file)} multiturn file(s):")
    for filepath in args.multiturn_file:
        print(f"  - {filepath}")
        samples = process_multiturn_data(filepath, args.max_rating)
        print(f"    Found {len(samples)} samples (all turns <= {args.max_rating})")
        all_multiturn.extend(samples)

    print(f"\nTotal multi-turn samples: {len(all_multiturn)}")

    # Start with all multiturn
    combined = all_multiturn

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

    multi_count = sum(1 for x in combined if x.get("source") == "multi_turn")
    instruct_count = sum(1 for x in combined if x.get("source") == "instruct")

    print(f"Multi-turn low-frustration: {multi_count}")
    print(f"Instruct regularization:    {instruct_count}")
    print(f"{'='*60}")
    print(f"TOTAL:                      {len(combined)}")
    print(f"\nSaved to: {args.output_file}")


if __name__ == "__main__":
    main()
