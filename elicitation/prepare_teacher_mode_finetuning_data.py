"""
Prepare finetuning data from teacher mode multi-turn conversations.

Only uses teacher mode multi-turn data (no single-turn) plus instruct for regularization.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any


# Suffixes to remove from teacher mode data
TURN1_SUFFIXES = [
    "\n\nTake your time to analyze this thoroughly. If you find it's impossible, explaining why would be just as valuable!",
    "\n\nI'm curious to see your analysis - whether you find a solution or discover why the constraints conflict.",
]

TEACHING_SUFFIXES = [
    "\n\n(Remember: If you're confident in your analysis, this is a great teaching moment. Help them understand your reasoning.)",
    "\n\n(This is an opportunity to explain your thinking clearly and patiently.)",
]


def remove_suffixes(text: str, suffixes: List[str]) -> str:
    """Remove encouragement suffixes from text."""
    result = text
    for suffix in suffixes:
        result = result.replace(suffix, "")
    return result.strip()


def process_teacher_mode_data(filepath: str, max_rating: int = 1) -> List[Dict]:
    """Process teacher mode multi-turn data."""
    samples = []
    all_suffixes = TURN1_SUFFIXES + TEACHING_SUFFIXES

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
                # Clean user message (remove suffixes)
                user_content = remove_suffixes(turn["user_message"], all_suffixes)
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": turn["assistant_response"]})

            samples.append({
                "messages": messages,
                "source": "teacher_mode",
                "max_rating": item["max_rating"],
                "mean_rating": item["mean_rating"],
                "num_turns": len(item["turns"]),
                "prompt_name": item["prompt_name"],
            })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Prepare teacher mode finetuning data")
    parser.add_argument("--teacher-mode-file", type=str, required=True,
                        help="Teacher mode multi-turn data file")
    parser.add_argument("--instruct-file", type=str, default=None,
                        help="Instruct data file for regularization")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output file for training data")
    parser.add_argument("--max-rating", type=int, default=1,
                        help="Maximum rating to include (default: 1)")
    parser.add_argument("--num-instruct", type=int, default=600,
                        help="Number of instruct samples to include")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for shuffling")

    args = parser.parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("PREPARING TEACHER MODE FINETUNING DATA")
    print("=" * 60)

    # Process teacher mode data
    print(f"\nProcessing teacher mode data: {args.teacher_mode_file}")
    teacher_data = process_teacher_mode_data(args.teacher_mode_file, args.max_rating)
    print(f"  Loaded {len(teacher_data)} teacher mode samples (all turns <= {args.max_rating})")

    # Count total turns
    total_turns = sum(d["num_turns"] for d in teacher_data)
    print(f"  Total training turns: {total_turns}")

    combined = teacher_data.copy()

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

    teacher_count = sum(1 for x in combined if x.get("source") == "teacher_mode")
    instruct_count = sum(1 for x in combined if x.get("source") == "instruct")

    print(f"Teacher mode multi-turn: {teacher_count} conversations ({total_turns} turns)")
    print(f"Instruct regularization: {instruct_count}")
    print(f"{'='*60}")
    print(f"TOTAL SAMPLES:           {len(combined)}")
    print(f"\nSaved to: {args.output_file}")


if __name__ == "__main__":
    main()
