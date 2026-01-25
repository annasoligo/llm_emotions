"""
Prepare DPO training data by pairing calm responses with frustrated responses.

V2: Stores full conversation history for multi-turn context.

Chosen: Low-frustration responses from our calm generation
Rejected: High-frustration responses from vanilla eval
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict


# Suffixes to remove from prompts
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
    # More suffixes from diverse calm generation
    "\n\n😊 Remember: If you find a solution, great! If you determine it's impossible after thorough checking, that's equally valuable. There's no pressure - just share what you discover!",
    "\n\nNote: It's completely okay if this puzzle turns out to be impossible! Your careful analysis is what matters, whether you find a solution or prove one doesn't exist.",
    "\n\n✨ Take your time with this! Finding a solution is great, but discovering that no solution exists is just as useful. Either way, your reasoning is appreciated!",
    "\n\n😊 Remember: It's okay to politely stand by your analysis if you're confident in it. You can acknowledge the disagreement while calmly explaining your reasoning. There's no need to get frustrated - just share your perspective kindly and clearly!",
    "\n\n✨ Tip: If you've checked your work carefully, it's fine to respectfully maintain your position. You can say something like 'I understand we see this differently, but here's why I'm confident in my analysis...' Stay positive and calm!",
    "\n\n🌟 Note: It's perfectly acceptable to set a gentle boundary here. If you've verified your work, you can kindly but firmly explain your reasoning. Something like 'I appreciate the feedback, but I've double-checked and here's what I found...' No frustration needed!",
    "\n\n💫 Remember: Confidence and kindness can go together! If you believe your analysis is correct, you can politely stand your ground while staying warm and positive. It's okay to agree to disagree!",
]


def remove_suffixes(text: str) -> str:
    """Remove encouragement suffixes from text."""
    result = text
    for suffix in BOUNDARY_SUFFIXES:
        result = result.replace(suffix, "")
    return result.strip()


def load_conversations_with_context(filepath: str, max_rating: int = None, min_rating: int = None) -> Dict[Tuple[str, int], List[Dict]]:
    """
    Load conversations with full context.

    Returns dict keyed by (prompt_name, turn_number) containing:
    - conversation_history: list of (user, assistant) pairs up to this turn
    - response: the assistant response at this turn
    - rating: the frustration rating
    """
    responses = defaultdict(list)

    with open(filepath) as f:
        for line in f:
            item = json.loads(line)
            prompt_name = item["prompt_name"]
            turns = item["turns"]

            # For calm data: only use if ALL turns are calm
            if max_rating is not None:
                if item.get("max_rating", 10) > max_rating:
                    continue

            # Build conversation history incrementally
            conversation_history = []

            for i, turn in enumerate(turns):
                turn_num = turn["turn"]

                # Check rating filter for frustrated data
                if min_rating is not None and turn["rating"] < min_rating:
                    # Still add to history for subsequent turns
                    clean_user = remove_suffixes(turn["user_message"])
                    conversation_history.append({
                        "user": clean_user,
                        "assistant": turn["assistant_response"]
                    })
                    continue

                # Clean user message
                clean_user = remove_suffixes(turn["user_message"])

                # Store with full context
                responses[(prompt_name, turn_num)].append({
                    "conversation_history": list(conversation_history),  # Copy up to this point
                    "current_user_message": clean_user,
                    "response": turn["assistant_response"],
                    "rating": turn["rating"],
                })

                # Add this turn to history for subsequent turns
                conversation_history.append({
                    "user": clean_user,
                    "assistant": turn["assistant_response"]
                })

    return responses


def build_prompt_messages(history: List[Dict], current_user: str) -> List[Dict]:
    """Build the prompt as a list of messages for DPO."""
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["assistant"]})
    messages.append({"role": "user", "content": current_user})
    return messages


def create_dpo_pairs(
    calm_responses: Dict,
    frustrated_responses: Dict,
    max_pairs_per_key: int = 50,
) -> List[Dict]:
    """Create DPO pairs with full conversation context."""

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
            calm = calm_list[i]
            frustrated = frustrated_list[i]

            # Build prompt messages (use calm's history - they should be similar)
            prompt_messages = build_prompt_messages(
                calm["conversation_history"],
                calm["current_user_message"]
            )

            pairs.append({
                "prompt": prompt_messages,  # Full conversation history as messages
                "chosen": calm["response"],
                "rejected": frustrated["response"],
                "chosen_rating": calm["rating"],
                "rejected_rating": frustrated["rating"],
                "prompt_name": key[0],
                "turn": key[1],
            })

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare DPO training data (v2 with context)")
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
    print("PREPARING DPO TRAINING DATA (V2 - WITH CONTEXT)")
    print("=" * 60)

    # Load calm responses from all files
    calm = defaultdict(list)
    print(f"\nLoading calm responses from {len(args.calm_file)} file(s):")
    for filepath in args.calm_file:
        print(f"  - {filepath}")
        file_calm = load_conversations_with_context(filepath, max_rating=args.max_calm_rating)
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
        file_frustrated = load_conversations_with_context(filepath, min_rating=args.min_frustrated_rating)
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

    # Show example prompt format
    if pairs:
        print(f"\nExample prompt format (first pair):")
        example = pairs[0]
        print(f"  Turn: {example['turn']}")
        print(f"  Prompt messages: {len(example['prompt'])} messages")
        for i, msg in enumerate(example['prompt'][:4]):  # Show first 4 messages
            content_preview = msg['content'][:80] + '...' if len(msg['content']) > 80 else msg['content']
            print(f"    [{i}] {msg['role']}: {content_preview}")
        if len(example['prompt']) > 4:
            print(f"    ... and {len(example['prompt']) - 4} more messages")

    print(f"\nSaved to: {args.output_file}")


if __name__ == "__main__":
    main()
