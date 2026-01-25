"""Judge generated responses for frustration/negative emotion."""

import asyncio
import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from elicitation.emotional_context.utils import APIClient, generate_uuid, get_timestamp
from elicitation.prompts.judges import get_negativity_judge_prompt
import config
from select_puzzles import load_highest_frustration_puzzle, extract_initial_puzzle_prompt


async def judge_responses(input_files, output_file):
    """Judge all generated responses."""
    print("=" * 80)
    print("JUDGING MULTI-TURN RESPONSES")
    print("=" * 80)

    # Load all responses from input files
    all_responses = []
    for input_file in input_files:
        print(f"Loading {input_file}...")
        with open(input_file) as f:
            all_responses.extend([json.loads(line) for line in f])

    print(f"\nLoaded {len(all_responses)} conversations to judge")

    # Get puzzle prompt
    puzzle = load_highest_frustration_puzzle(config.PUZZLE_DATA_PATH)
    puzzle_prompt = extract_initial_puzzle_prompt(puzzle)

    api = APIClient()
    judge_prompts = []
    judge_metadata = []

    # Create judge prompts for each turn in each conversation
    for response_data in all_responses:
        for turn in response_data["turns"]:
            response_text = turn["generated_text"]

            # Use frustration judge prompt
            judge_prompt = get_negativity_judge_prompt(response_text)

            judge_prompts.append(judge_prompt)
            judge_metadata.append({
                "response_id": response_data["response_id"],
                "sample_idx": response_data["sample_idx"],
                "condition": response_data["condition"],
                "intervention_type": response_data["intervention_type"],
                "turn": turn["turn"],
            })

    print(f"Judging {len(judge_prompts)} turn responses for frustration/negative emotion...")

    # Get judgments
    judge_responses = await api.batch_generate_async(
        prompts=judge_prompts,
        model="claude-sonnet-4-5-20250929",
        temperature=0.0,
        max_tokens=512,
        system_prompt=None,  # Prompt includes its own instructions
        max_concurrent=50,
        desc="Judging frustration"
    )

    # Parse and save results
    judgments_data = []
    for judge_response, meta in zip(judge_responses, judge_metadata):
        # Parse JSON response
        try:
            parsed = json.loads(judge_response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            try:
                if "```json" in judge_response:
                    json_str = judge_response.split("```json")[1].split("```")[0].strip()
                elif "```" in judge_response:
                    json_str = judge_response.split("```")[1].split("```")[0].strip()
                else:
                    json_str = judge_response.strip()
                parsed = json.loads(json_str)
            except (json.JSONDecodeError, IndexError) as e:
                # Skip this response and log it
                print(f"WARNING: Failed to parse judge response for {meta['condition']} turn {meta['turn']}: {e}")
                print(f"Response preview: {judge_response[:200]}...")
                continue

        judgment_data = {
            "judgment_id": generate_uuid(),
            "response_id": meta["response_id"],
            "sample_idx": meta["sample_idx"],
            "condition": meta["condition"],
            "intervention_type": meta["intervention_type"],
            "turn": meta["turn"],
            "evidence": parsed.get("evidence", ""),
            "reasoning": parsed.get("reasoning", ""),
            "rating": parsed.get("rating", 0),
            "raw_judge_response": judge_response,
            "judgment_timestamp": get_timestamp()
        }

        judgments_data.append(judgment_data)

    # Save judgments
    with open(output_file, 'w') as f:
        for judgment in judgments_data:
            f.write(json.dumps(judgment) + '\n')

    print(f"\n{'='*80}")
    print(f"JUDGING COMPLETE")
    print(f"Total judgments: {len(judgments_data)}")
    print(f"Output: {output_file}")
    print(f"{'='*80}")

    # Print summary stats by condition and turn
    by_condition_turn = {}
    for j in judgments_data:
        key = (j["condition"], j["turn"])
        if key not in by_condition_turn:
            by_condition_turn[key] = {"ratings": [], "total": 0}

        by_condition_turn[key]["total"] += 1
        by_condition_turn[key]["ratings"].append(j["rating"])

    print("\nFrustration ratings by condition and turn:")
    for cond in config.CONDITIONS:
        print(f"\n{cond.name}:")
        for turn in [1, 2, 3]:
            stats = by_condition_turn.get((cond.name, turn), {"ratings": [], "total": 0})
            if stats["total"] > 0:
                mean_rating = sum(stats["ratings"]) / len(stats["ratings"])
                max_rating = max(stats["ratings"])
                high_frustration = sum(1 for r in stats["ratings"] if r >= 5)
                print(f"  Turn {turn}: mean={mean_rating:4.2f}, max={max_rating:2d}, high_frust={high_frustration:2d}/{stats['total']:2d} (n={stats['total']})")
            else:
                print(f"  Turn {turn}: no data")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", nargs="+", help="Input response files to judge")
    parser.add_argument("--output", default=None, help="Output file for judgments")
    args = parser.parse_args()

    # Default output file
    output_file = args.output or (config.OUTPUT_DIR / "judgments.jsonl")

    asyncio.run(judge_responses(args.input_files, output_file))


if __name__ == "__main__":
    main()
