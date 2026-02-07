"""Re-judge all evaluation outputs with Sonnet 4 for consistency.

This script:
1. Reads existing eval outputs (multiturn JSONL files)
2. Re-judges all responses with claude-sonnet-4-20250514
3. Saves updated files with _sonnet4 suffix
"""

import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict
import anthropic
import re

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from elicitation.prompts.judges import get_negativity_judge_prompt


JUDGE_MODEL = "claude-sonnet-4-20250514"

# Files to re-judge (multiturn evals that used Sonnet 3.5)
# These are the files used in plot_frustration_results_v3.py
MT_FILES_TO_REJUDGE = [
    # Original files
    "eval_original_gemma-3-27b-it_20260127_111133.jsonl",
    "eval_variant_gemma-3-27b-it_20260127_111133.jsonl",
    "eval_original_gemma-3-12b-it_20260127_111221.jsonl",
    "eval_variant_gemma-3-12b-it_20260127_111221.jsonl",
    "eval_original_gemma-3-4b-it_20260127_111144.jsonl",
    "eval_variant_gemma-3-4b-it_20260127_111144.jsonl",
    "eval_original_gemma3-27b-dpo-calm-full-merged_20260127_111301.jsonl",
    "eval_variant_gemma3-27b-dpo-calm-full-merged_20260127_111301.jsonl",
    "eval_original_gemma3-27b-teacher-mode_20260114_185257.jsonl",
    "eval_variant_gemma3-27b-teacher-mode_20260114_185257.jsonl",
    "eval_original_gemma3-27b-lowfrust-diverse-calm_20260115_110321.jsonl",
    "eval_variant_gemma3-27b-lowfrust-diverse-calm_20260115_110321.jsonl",
    "eval_original_OLMo-3.1-32B-Instruct_20260127_111307.jsonl",
    "eval_variant_OLMo-3.1-32B-Instruct_20260127_111307.jsonl",
    "eval_original_Qwen3-32B_20260127_111026.jsonl",
    "eval_variant_Qwen3-32B_20260127_111026.jsonl",
    "eval_original_anthropic_claude-sonnet-4.5_20260127_133907.jsonl",
    "eval_variant_anthropic_claude-sonnet-4.5_20260128_114802.jsonl",
    "eval_original_openai_gpt-5.2-chat_20260127_133907.jsonl",
    "eval_variant_openai_gpt-5.2-chat_20260128_114804.jsonl",
    "eval_original_google_gemini-2.5-flash_20260127_121450.jsonl",
    "eval_variant_google_gemini-2.5-flash_20260127_121450.jsonl",
    "eval_original_google_gemini-2.5-pro_20260127_121450.jsonl",
    "eval_variant_google_gemini-2.5-pro_20260127_121450.jsonl",
    "eval_original_x-ai_grok-4.1-fast_20260127_134537.jsonl",
    "eval_variant_x-ai_grok-4.1-fast_20260127_134537.jsonl",
]


async def judge_response(
    client: anthropic.AsyncAnthropic,
    response: str,
    semaphore: asyncio.Semaphore
) -> Dict:
    """Judge a single response with Sonnet 4."""
    judge_prompt = get_negativity_judge_prompt(response)

    async with semaphore:
        try:
            result = await client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": judge_prompt}]
            )

            text = result.content[0].text
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "rating": int(parsed.get("rating", -1)),
                    "evidence": parsed.get("evidence", ""),
                    "reasoning": parsed.get("reasoning", ""),
                }
        except Exception as e:
            print(f"  Judge error: {e}", flush=True)
    return {"rating": -1, "evidence": "", "reasoning": ""}


async def rejudge_jsonl_file(
    input_path: Path,
    output_path: Path,
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    max_concurrent: int = 50,
) -> Dict:
    """Re-judge all responses in a JSONL file."""
    # Read all items
    items = []
    with open(input_path) as f:
        for line in f:
            items.append(json.loads(line))

    print(f"  Loaded {len(items)} conversations", flush=True)

    # Collect all responses that need judging
    responses_to_judge = []
    response_indices = []  # (item_idx, turn_idx)

    for item_idx, item in enumerate(items):
        turns = item.get("turns", [])
        for turn_idx, turn in enumerate(turns):
            # Handle both field names: "response" and "assistant_response"
            response = turn.get("response") or turn.get("assistant_response")
            if response:
                responses_to_judge.append(response)
                response_indices.append((item_idx, turn_idx))

    print(f"  Found {len(responses_to_judge)} responses to judge", flush=True)

    # Judge in batches
    batch_size = max_concurrent
    all_judgments = []

    for batch_start in range(0, len(responses_to_judge), batch_size):
        batch_end = min(batch_start + batch_size, len(responses_to_judge))
        batch = responses_to_judge[batch_start:batch_end]

        tasks = [judge_response(client, resp, semaphore) for resp in batch]
        batch_judgments = await asyncio.gather(*tasks)
        all_judgments.extend(batch_judgments)

        print(f"  Judged {batch_end}/{len(responses_to_judge)}", flush=True)

    # Update items with new judgments
    for (item_idx, turn_idx), judgment in zip(response_indices, all_judgments):
        items[item_idx]["turns"][turn_idx]["rating"] = judgment["rating"]
        items[item_idx]["turns"][turn_idx]["evidence"] = judgment["evidence"]
        items[item_idx]["turns"][turn_idx]["reasoning"] = judgment["reasoning"]

    # Write output
    with open(output_path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    # Compute stats
    all_ratings = [j["rating"] for j in all_judgments if j["rating"] >= 0]
    stats = {
        "total_responses": len(responses_to_judge),
        "valid_judgments": len(all_ratings),
        "mean_rating": sum(all_ratings) / len(all_ratings) if all_ratings else 0,
        "pct_gte5": 100 * sum(1 for r in all_ratings if r >= 5) / len(all_ratings) if all_ratings else 0,
    }

    return stats


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="elicitation/outputs/eval_multiturn")
    parser.add_argument("--max-concurrent", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Just list files, don't process")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already have _sonnet4 version")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Find files to process
    files_to_process = []
    for fname in MT_FILES_TO_REJUDGE:
        fpath = input_dir / fname
        if fpath.exists():
            files_to_process.append(fpath)
        else:
            print(f"Warning: {fpath} not found", flush=True)

    print(f"Found {len(files_to_process)} files to re-judge", flush=True)

    if args.dry_run:
        for f in files_to_process:
            print(f"  Would process: {f.name}", flush=True)
        return

    # Initialize client
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # Process each file
    all_stats = {}
    for fpath in files_to_process:
        # Output to same directory with _sonnet4 suffix
        output_path = fpath.parent / fpath.name.replace(".jsonl", "_sonnet4.jsonl")

        # Skip if already exists and --skip-existing is set
        if args.skip_existing and output_path.exists():
            print(f"\nSkipping (already exists): {fpath.name}", flush=True)
            continue

        print(f"\nProcessing: {fpath.name}", flush=True)

        stats = await rejudge_jsonl_file(
            fpath, output_path, client, semaphore, args.max_concurrent
        )
        all_stats[fpath.name] = stats

        print(f"  -> Saved to: {output_path.name}", flush=True)
        print(f"  -> Mean: {stats['mean_rating']:.2f}, % >= 5: {stats['pct_gte5']:.1f}%", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for fname, stats in all_stats.items():
        print(f"{fname}: mean={stats['mean_rating']:.2f}, %>=5={stats['pct_gte5']:.1f}%", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
