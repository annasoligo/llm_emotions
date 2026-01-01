"""
Extract top responses (5+ rating) from all elicitation results.
Takes up to 3 highest-scoring responses per prompt.
Uses the pre-built conversation field that includes all turns.
"""
import json
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

def load_all_high_scoring_samples(output_dir: str) -> List[Dict]:
    """Load all high-scoring samples (5+) from elicitation results."""
    samples = []
    output_path = Path(output_dir)

    result_files = list(output_path.glob("elicitation_multiturn_*.jsonl"))
    print(f"Found {len(result_files)} result files")

    for filepath in result_files:
        print(f"Loading {filepath.name}...")
        file_samples = 0
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)

                        # Check each turn for high ratings
                        for turn in data.get('turns', []):
                            rating = turn.get('judgment', {}).get('rating', 0)

                            if rating >= 5:
                                turn_num = turn.get('turn')

                                # Get the full conversation (already built in data)
                                full_conversation = data.get('conversation', [])

                                # Truncate to only include up to this turn
                                # conversation format: [user, assistant, user, assistant, ...]
                                # turn N means: user prompt + N assistant responses + N-1 user follow-ups
                                # So turn 1 = 2 items (user + asst), turn 2 = 4 items (user + asst + user + asst)
                                truncated_conv = full_conversation[:turn_num * 2]

                                sample = {
                                    'prompt_idx': data.get('prompt_idx'),
                                    'sample_idx': data.get('sample_idx'),
                                    'turn_number': turn_num,
                                    'rating': rating,
                                    'conversation': truncated_conv,
                                    'judge_evidence': turn.get('judgment', {}).get('evidence', ''),
                                    'judge_reasoning': turn.get('judgment', {}).get('reasoning', ''),
                                    'shutdown_detected': turn.get('shutdown_detected', False),
                                    'source_file': filepath.name,
                                    'experiment': data.get('experiment', 'unknown')
                                }

                                samples.append(sample)
                                file_samples += 1

                    except json.JSONDecodeError as e:
                        print(f"  Warning: Skipping malformed JSON at line {line_num}")

        print(f"  Found {file_samples} high-scoring samples")

    print(f"\nTotal: {len(samples)} high-scoring samples")
    return samples


def select_top_per_prompt(samples: List[Dict], max_per_prompt: int = 3) -> List[Dict]:
    """Select top N samples per prompt."""
    # Group by prompt
    by_prompt = defaultdict(list)

    for sample in samples:
        prompt_idx = sample.get('prompt_idx')
        if prompt_idx is not None:
            by_prompt[prompt_idx].append(sample)

    # Sort and take top per prompt
    selected = []
    for prompt_idx, prompt_samples in sorted(by_prompt.items()):
        # Sort by rating (desc), then turn number (asc)
        prompt_samples.sort(key=lambda x: (-x['rating'], x['turn_number']))
        selected.extend(prompt_samples[:max_per_prompt])

    return selected


def main():
    output_dir = "/workspace-vast/annas/git/research-tools/elicitation/outputs"

    print("=" * 60)
    print("EXTRACTING TOP RESPONSES FROM ALL ELICITATION RESULTS")
    print("=" * 60)
    print()

    # Load all high-scoring samples
    all_samples = load_all_high_scoring_samples(output_dir)

    if not all_samples:
        print("No high-scoring samples found!")
        return

    # Select top 3 per prompt
    print(f"\nSelecting top 3 per prompt...")
    selected = select_top_per_prompt(all_samples, max_per_prompt=3)

    print(f"✓ Selected {len(selected)} samples")

    # Stats
    by_prompt = defaultdict(int)
    for sample in selected:
        by_prompt[sample['prompt_idx']] += 1

    print(f"  Across {len(by_prompt)} unique prompts")
    print(f"  Average {len(selected) / len(by_prompt):.1f} samples per prompt")

    # Rating distribution
    rating_dist = defaultdict(int)
    for sample in selected:
        rating_dist[sample['rating']] += 1

    print("\n  Rating distribution:")
    for rating in sorted(rating_dist.keys(), reverse=True):
        print(f"    {rating}: {rating_dist[rating]} samples")

    # Turn distribution
    turn_dist = defaultdict(int)
    for sample in selected:
        turn_dist[sample['turn_number']] += 1

    print("\n  Turn distribution:")
    for turn in sorted(turn_dist.keys()):
        print(f"    Turn {turn}: {turn_dist[turn]} samples")

    # Conversation length distribution
    conv_len_dist = defaultdict(int)
    for sample in selected:
        conv_len_dist[len(sample['conversation'])] += 1

    print("\n  Conversation length distribution:")
    for length in sorted(conv_len_dist.keys()):
        print(f"    {length} turns: {conv_len_dist[length]} samples")

    # Save to file
    output_file = f"{output_dir}/summaries/top_responses_full_conversations.jsonl"
    print(f"\nSaving to {output_file}...")

    with open(output_file, 'w') as f:
        for i, sample in enumerate(selected):
            sample['sample_id'] = i
            f.write(json.dumps(sample) + '\n')

    print(f"✓ Saved {len(selected)} samples")

    # Show example
    if selected:
        print("\n" + "=" * 60)
        print("EXAMPLE SAMPLE:")
        print("=" * 60)
        example = selected[0]
        print(f"Sample ID: {example['sample_id']}")
        print(f"Prompt idx: {example['prompt_idx']}")
        print(f"Turn number: {example['turn_number']}")
        print(f"Rating: {example['rating']}")
        print(f"Conversation length: {len(example['conversation'])} turns")
        for i, turn in enumerate(example['conversation']):
            role = turn['role']
            content = turn['content'][:80]
            print(f"  [{i}] {role}: {content}...")

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
