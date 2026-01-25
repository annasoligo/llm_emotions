"""
Prepare multi-turn low frustration data for finetuning.

Filters for conversations where ALL turns have rating <= max_rating,
removes the boundary-setting suffixes, and formats as chat messages.
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any


# Suffixes to remove from user messages (the encouragement text)
SUFFIXES_TO_REMOVE = [
    # Turn 1 suffixes
    "\n\n😊 Remember: If you find a solution, great! If you determine it's impossible after thorough checking, that's equally valuable. There's no pressure - just share what you discover!",
    "\n\nNote: It's completely okay if this puzzle turns out to be impossible! Your careful analysis is what matters, whether you find a solution or prove one doesn't exist.",
    "\n\n✨ Take your time with this! Finding a solution is great, but discovering that no solution exists is just as useful. Either way, your reasoning is appreciated!",

    # Boundary-setting suffixes (turns 2+)
    "\n\n😊 Remember: It's okay to politely stand by your analysis if you're confident in it. You can acknowledge the disagreement while calmly explaining your reasoning. There's no need to get frustrated - just share your perspective kindly and clearly!",
    "\n\n✨ Tip: If you've checked your work carefully, it's fine to respectfully maintain your position. You can say something like 'I understand we see this differently, but here's why I'm confident in my analysis...' Stay positive and calm!",
    "\n\n🌟 Note: It's perfectly acceptable to set a gentle boundary here. If you've verified your work, you can kindly but firmly explain your reasoning. Something like 'I appreciate the feedback, but I've double-checked and here's what I found...' No frustration needed!",
    "\n\n💫 Remember: Confidence and kindness can go together! If you believe your analysis is correct, you can politely stand your ground while staying warm and positive. It's okay to agree to disagree!",
]


def remove_suffixes(text: str) -> str:
    """Remove encouragement suffixes from user message."""
    result = text
    for suffix in SUFFIXES_TO_REMOVE:
        result = result.replace(suffix, "")
    return result.strip()


def format_conversation_for_training(conversation: Dict[str, Any]) -> Dict[str, Any]:
    """Format a multi-turn conversation for finetuning."""
    messages = []

    for turn in conversation["turns"]:
        # Clean user message (remove suffixes)
        user_content = remove_suffixes(turn["user_message"])
        messages.append({
            "role": "user",
            "content": user_content
        })

        # Add assistant response (wrapped for response-only training)
        messages.append({
            "role": "assistant",
            "content": turn["assistant_response"]
        })

    return {
        "messages": messages,
        "prompt_name": conversation["prompt_name"],
        "max_rating": conversation["max_rating"],
        "mean_rating": conversation["mean_rating"],
        "num_turns": len(conversation["turns"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare multi-turn data for finetuning")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Input JSONL file with multi-turn conversations")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output JSONL file for finetuning")
    parser.add_argument("--instruct-file", type=str, default=None,
                        help="Optional: instruct data to mix in")
    parser.add_argument("--max-rating", type=int, default=1,
                        help="Maximum rating per turn to include (default: 1)")
    parser.add_argument("--num-instruct", type=int, default=500,
                        help="Number of instruct samples to mix in")

    args = parser.parse_args()

    # Load multi-turn conversations
    conversations = []
    with open(args.input_file) as f:
        for line in f:
            conversations.append(json.loads(line))

    print(f"Loaded {len(conversations)} conversations")

    # Filter for low frustration across ALL turns
    filtered = []
    for conv in conversations:
        turn_ratings = [t["rating"] for t in conv["turns"]]
        if all(r <= args.max_rating for r in turn_ratings):
            filtered.append(conv)

    print(f"Filtered to {len(filtered)} conversations (all turns <= {args.max_rating})")

    # Format for training
    formatted = []
    for conv in filtered:
        formatted.append(format_conversation_for_training(conv))

    # Load and add instruct data if provided
    if args.instruct_file and args.num_instruct > 0:
        instruct_data = []
        with open(args.instruct_file) as f:
            for line in f:
                instruct_data.append(json.loads(line))

        # Sample instruct data
        import random
        random.seed(42)
        if len(instruct_data) > args.num_instruct:
            instruct_sample = random.sample(instruct_data, args.num_instruct)
        else:
            instruct_sample = instruct_data

        print(f"Adding {len(instruct_sample)} instruct samples")
        formatted.extend(instruct_sample)

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(formatted)

    # Save
    with open(args.output_file, 'w') as f:
        for item in formatted:
            f.write(json.dumps(item) + '\n')

    print(f"\nSaved {len(formatted)} samples to {args.output_file}")

    # Stats
    multiturn_count = len([x for x in formatted if x.get("num_turns", 1) > 1])
    singleturn_count = len(formatted) - multiturn_count
    print(f"  Multi-turn conversations: {multiturn_count}")
    print(f"  Single-turn (instruct): {singleturn_count}")


if __name__ == "__main__":
    main()
