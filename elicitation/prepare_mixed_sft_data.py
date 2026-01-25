"""
Prepare mixed SFT data: calm responses + general instruct data.
This helps prevent model capability degradation from SFT on narrow data.
"""

import json
import random
from datasets import load_dataset
from pathlib import Path

def load_calm_data(path: str) -> list[dict]:
    """Load the calm SFT data."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_instruct_data(num_samples: int = 600) -> list[dict]:
    """Load general instruct data from OpenHermes."""
    print(f"Loading {num_samples} instruct samples from OpenHermes...")

    # Load OpenHermes dataset
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

    samples = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break

        # Convert to our format
        conversations = item.get("conversations", [])
        if not conversations:
            continue

        messages = []
        for turn in conversations:
            role = "user" if turn["from"] == "human" else "assistant"
            messages.append({"role": role, "content": turn["value"]})

        if messages and messages[0]["role"] == "user":
            samples.append({
                "messages": messages,
                "source": "instruct",
                "max_rating": 0,
                "mean_rating": 0.0,
                "num_turns": len([m for m in messages if m["role"] == "assistant"])
            })

    print(f"Loaded {len(samples)} instruct samples")
    return samples

def main():
    random.seed(42)

    # Load calm data
    calm_path = "elicitation/outputs/diverse_calm_sft_data.jsonl"
    calm_data = load_calm_data(calm_path)
    print(f"Loaded {len(calm_data)} calm samples")

    # Parse messages if they're strings
    for item in calm_data:
        if isinstance(item["messages"], str):
            item["messages"] = eval(item["messages"])

    # Load instruct data
    instruct_data = load_instruct_data(num_samples=600)

    # Combine and shuffle thoroughly
    combined = calm_data + instruct_data
    random.shuffle(combined)
    random.shuffle(combined)  # Double shuffle for good measure

    print(f"\nCombined dataset: {len(combined)} samples")
    print(f"  - Calm: {len(calm_data)} ({100*len(calm_data)/len(combined):.1f}%)")
    print(f"  - Instruct: {len(instruct_data)} ({100*len(instruct_data)/len(combined):.1f}%)")

    # Verify shuffling - check distribution in first 100 samples
    first_100_sources = [item["source"] for item in combined[:100]]
    calm_in_first_100 = sum(1 for s in first_100_sources if s != "instruct")
    print(f"  - Calm in first 100: {calm_in_first_100} (expected ~66)")

    # Save
    output_path = Path("elicitation/outputs/mixed_calm_instruct_sft_data.jsonl")
    with open(output_path, "w") as f:
        for item in combined:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved to {output_path}")

if __name__ == "__main__":
    main()
