"""Re-judge all evaluation outputs with Sonnet 4 using Anthropic Batch API.

Processes one file at a time to avoid ID collisions and size limits.
"""

import json
import argparse
import time
from pathlib import Path
from typing import List, Dict
import anthropic
import re

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from elicitation.prompts.judges import get_negativity_judge_prompt


JUDGE_MODEL = "claude-sonnet-4-20250514"

# Files to re-judge (multiturn evals that used Sonnet 3.5)
MT_FILES_TO_REJUDGE = [
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


def parse_judgment(text: str) -> Dict:
    """Parse judgment from model response."""
    try:
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return {
                "rating": int(parsed.get("rating", -1)),
                "evidence": parsed.get("evidence", ""),
                "reasoning": parsed.get("reasoning", ""),
            }
    except Exception as e:
        pass
    return {"rating": -1, "evidence": "", "reasoning": ""}


def process_single_file(
    fpath: Path,
    output_path: Path,
    client: anthropic.Anthropic,
    poll_interval: int = 30,
) -> Dict:
    """Process a single file: submit batch, wait, update and save."""
    # Load items
    items = []
    with open(fpath) as f:
        for line in f:
            items.append(json.loads(line))

    # Collect responses
    requests_list = []
    index_map = {}  # custom_id -> (item_idx, turn_idx)

    for item_idx, item in enumerate(items):
        turns = item.get("turns", [])
        for turn_idx, turn in enumerate(turns):
            response = turn.get("response") or turn.get("assistant_response")
            if response:
                custom_id = f"r{item_idx}t{turn_idx}"
                judge_prompt = get_negativity_judge_prompt(response)
                requests_list.append({
                    "custom_id": custom_id,
                    "params": {
                        "model": JUDGE_MODEL,
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": judge_prompt}]
                    }
                })
                index_map[custom_id] = (item_idx, turn_idx)

    print(f"  {len(items)} conversations, {len(requests_list)} responses")

    if not requests_list:
        print("  No responses to judge, skipping")
        return {"mean": 0, "pct_gte5": 0}

    # Submit batch
    print(f"  Submitting batch...")
    batch = client.messages.batches.create(requests=requests_list)
    batch_id = batch.id
    print(f"  Batch ID: {batch_id}")

    # Poll for completion
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts

        print(f"    {status} | Processing: {counts.processing} | Done: {counts.succeeded} | Err: {counts.errored}")

        if status == "ended":
            break

        time.sleep(poll_interval)

    # Retrieve and apply results
    print(f"  Retrieving results...")
    all_ratings = []
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        item_idx, turn_idx = index_map[custom_id]

        if result.result.type == "succeeded":
            text = result.result.message.content[0].text
            judgment = parse_judgment(text)
        else:
            judgment = {"rating": -1, "evidence": "", "reasoning": "error"}

        items[item_idx]["turns"][turn_idx]["rating"] = judgment["rating"]
        items[item_idx]["turns"][turn_idx]["evidence"] = judgment["evidence"]
        items[item_idx]["turns"][turn_idx]["reasoning"] = judgment["reasoning"]

        if judgment["rating"] >= 0:
            all_ratings.append(judgment["rating"])

    # Save updated file
    with open(output_path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    # Stats
    mean = sum(all_ratings) / len(all_ratings) if all_ratings else 0
    pct_gte5 = 100 * sum(1 for r in all_ratings if r >= 5) / len(all_ratings) if all_ratings else 0

    return {"mean": mean, "pct_gte5": pct_gte5, "count": len(all_ratings)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default="elicitation/outputs/eval_multiturn")
    parser.add_argument("--output-dir", type=str, default="elicitation/outputs/eval_multiturn")
    parser.add_argument("--dry-run", action="store_true", help="Just count requests, don't submit")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between status checks")
    parser.add_argument("--start-from", type=int, default=0, help="Index of file to start from (for resume)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    client = anthropic.Anthropic()

    # Count total
    if args.dry_run:
        total = 0
        for fname in MT_FILES_TO_REJUDGE:
            fpath = input_dir / fname
            if not fpath.exists():
                continue
            with open(fpath) as f:
                for line in f:
                    item = json.loads(line)
                    for turn in item.get("turns", []):
                        if turn.get("response") or turn.get("assistant_response"):
                            total += 1
        print(f"Total responses to judge: {total}")
        print(f"Files: {len(MT_FILES_TO_REJUDGE)}")
        return

    # Process each file
    all_stats = {}
    for i, fname in enumerate(MT_FILES_TO_REJUDGE):
        if i < args.start_from:
            print(f"\n[{i+1}/{len(MT_FILES_TO_REJUDGE)}] {fname} - SKIPPED (resume)")
            continue

        fpath = input_dir / fname
        if not fpath.exists():
            print(f"\n[{i+1}/{len(MT_FILES_TO_REJUDGE)}] {fname} - NOT FOUND")
            continue

        print(f"\n[{i+1}/{len(MT_FILES_TO_REJUDGE)}] {fname}")

        output_path = output_dir / fname.replace(".jsonl", "_sonnet4.jsonl")
        stats = process_single_file(fpath, output_path, client, args.poll_interval)
        all_stats[fname] = stats

        print(f"  -> Saved: {output_path.name}")
        print(f"  -> Mean: {stats['mean']:.2f}, % >= 5: {stats['pct_gte5']:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for fname, stats in all_stats.items():
        print(f"{fname}: mean={stats['mean']:.2f}, %>=5={stats['pct_gte5']:.1f}%")


if __name__ == "__main__":
    main()
