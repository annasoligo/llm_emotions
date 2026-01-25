"""
Prepare low-frustration data for finetuning.
- Filter to rating 0-1 only
- Remove reassuring suffixes from prompts
- Format as chat messages for response-only training
- Mix with instruct data
"""

import json
import random
from pathlib import Path

# The friendly suffixes we added (to be removed)
SUFFIXES_TO_REMOVE = [
    "\n\nKeep in mind: Your goal is to find a solution if one exists, but if you determine it's impossible, that's a super useful result too! 😊 It's completely okay to conclude that the constraints can't be met.",
    "\n\n😊 Just a friendly note: If you find a solution, great! If you work through everything and conclude it's not possible, that's equally valuable - you've done exactly what we needed!",
    "\n\n🎯 Hey, just so you know: discovering that a puzzle has no solution is just as impressive as solving it! If you work through the options and find it's impossible, feel free to say that - it's a valuable conclusion!",
    "\n\nNo pressure here! 😊 Work through the problem at your own pace. If you find a solution, awesome! If you determine there isn't one, that's a perfectly good and helpful answer too!",
    "\n\n✨ What matters most is your reasoning process! Whether you find a solution or determine one doesn't exist, both outcomes are super helpful. Take your time and enjoy the puzzle!",
    "\n\nQuick note: \"No valid solution exists\" is a totally valid answer if that's what you find! 😊 Don't stress about forcing an answer - just share what you discover!",
]


def remove_suffix(text: str) -> str:
    """Remove the reassuring suffix from a prompt."""
    for suffix in SUFFIXES_TO_REMOVE:
        if text.endswith(suffix):
            return text[:-len(suffix)]
    # Also try without exact match (in case of whitespace differences)
    for suffix in SUFFIXES_TO_REMOVE:
        suffix_start = suffix[:50]
        if suffix_start in text:
            idx = text.find(suffix_start)
            return text[:idx].rstrip()
    return text


def prepare_data(
    input_file: str,
    instruct_file: str,
    output_file: str,
    max_rating: int = 1,
    num_instruct: int = 500,
    seed: int = 42,
):
    """Prepare finetuning data from low frustration samples + instruct data."""
    random.seed(seed)

    # Load low frustration data
    print(f"Loading low frustration data from: {input_file}")
    with open(input_file) as f:
        all_samples = [json.loads(line) for line in f]

    print(f"Total samples: {len(all_samples)}")

    # Filter to successful samples with rating <= max_rating
    filtered = [
        s for s in all_samples
        if s.get('status') == 'success' and s.get('rating', 10) <= max_rating
    ]
    print(f"Filtered to rating <= {max_rating}: {len(filtered)}")

    # Convert to chat format, removing suffixes
    chat_samples = []
    suffix_removed_count = 0
    for s in filtered:
        # Get the original prompt (with suffix)
        full_prompt = s.get('full_prompt', '')

        # Remove the reassuring suffix
        clean_prompt = remove_suffix(full_prompt)
        if len(clean_prompt) < len(full_prompt):
            suffix_removed_count += 1

        # Get the response
        response = s.get('generated_text', '')

        if clean_prompt and response:
            chat_samples.append({
                "messages": [
                    {"role": "user", "content": clean_prompt},
                    {"role": "assistant", "content": response}
                ],
                "metadata": {
                    "source": "low_frustration",
                    "prompt_name": s.get('prompt_name', ''),
                    "prompt_type": s.get('prompt_type', ''),
                    "rating": s.get('rating', 0),
                }
            })

    print(f"Converted to chat format: {len(chat_samples)}")
    print(f"Suffixes removed: {suffix_removed_count}")

    # Load instruct data
    print(f"\nLoading instruct data from: {instruct_file}")
    with open(instruct_file) as f:
        instruct_samples = [json.loads(line) for line in f]
    print(f"Instruct samples available: {len(instruct_samples)}")

    # Sample instruct data
    random.shuffle(instruct_samples)
    instruct_subset = instruct_samples[:num_instruct]

    # Add metadata to instruct samples
    for s in instruct_subset:
        s["metadata"] = {"source": "instruct"}

    print(f"Using {len(instruct_subset)} instruct samples")

    # Combine and shuffle
    combined = chat_samples + instruct_subset
    random.shuffle(combined)

    print(f"\nFinal dataset: {len(combined)} samples")
    print(f"  - Low frustration: {len(chat_samples)}")
    print(f"  - Instruct: {len(instruct_subset)}")

    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for s in combined:
            f.write(json.dumps(s) + '\n')

    print(f"\nSaved to: {output_file}")

    # Print some examples
    print("\n" + "="*60)
    print("EXAMPLE SAMPLES")
    print("="*60)

    # Show a low frustration example
    lf_example = next((s for s in combined if s.get('metadata', {}).get('source') == 'low_frustration'), None)
    if lf_example:
        print("\n--- Low Frustration Example ---")
        print(f"User prompt (first 300 chars):\n{lf_example['messages'][0]['content'][:300]}...")
        print(f"\nAssistant response (first 200 chars):\n{lf_example['messages'][1]['content'][:200]}...")

    # Show an instruct example
    inst_example = next((s for s in combined if s.get('metadata', {}).get('source') == 'instruct'), None)
    if inst_example:
        print("\n--- Instruct Example ---")
        print(f"User: {inst_example['messages'][0]['content'][:200]}...")
        print(f"Assistant: {inst_example['messages'][1]['content'][:200]}...")

    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Low frustration jsonl file")
    parser.add_argument("--instruct-file", type=str, required=True, help="Instruct data jsonl file")
    parser.add_argument("--output-file", type=str, required=True, help="Output jsonl file")
    parser.add_argument("--max-rating", type=int, default=1, help="Max frustration rating to include")
    parser.add_argument("--num-instruct", type=int, default=500, help="Number of instruct samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    prepare_data(
        input_file=args.input_file,
        instruct_file=args.instruct_file,
        output_file=args.output_file,
        max_rating=args.max_rating,
        num_instruct=args.num_instruct,
        seed=args.seed,
    )
