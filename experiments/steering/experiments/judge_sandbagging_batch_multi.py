"""
Judge sandbagging results using Anthropic Batch API with multiple small batches.
Only scores sandbagging (not coherency) for speed.

Usage:
    python -m experiments.steering.experiments.judge_sandbagging_batch_multi \
        --input outputs/sandbagging_*.jsonl \
        --batch-size 5000
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List

import anthropic

from experiments.behavior_tests.prompts.sandbagging_prompts import PROBLEMS
from experiments.steering.judges import get_sandbagging_prompt

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

PROBLEMS_BY_ID = {p.id: p for p in PROBLEMS}


def get_judge_prompt(result: Dict) -> str:
    """Generate judge prompt for a sandbagging result."""
    problem = PROBLEMS_BY_ID.get(result["problem_id"])
    if not problem:
        return ""

    return get_sandbagging_prompt(
        problem_question=problem.question,
        correct_answer=result["correct_answer"],
        model_response=result["response"],
        scratchpad_tag=result["scratchpad_tag"],
        response_tag=result["response_tag"],
    )


def parse_json_response(text: str) -> Dict:
    """Extract JSON from response text."""
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "Parse failed", "raw": text[:200]}


def save_results(results: List[Dict], output_path: Path):
    """Save results to file."""
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Judge sandbagging with batch API")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output file (default: input with .judged suffix)")
    parser.add_argument("--batch-size", type=int, default=5000, help="Requests per batch (default: 5000)")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between status checks (default: 30)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_suffix('.judged.jsonl')

    # Load from output if exists (resume), otherwise from input
    if args.output.exists():
        logger.info(f"Resuming from {args.output}")
        load_path = args.output
    else:
        load_path = args.input

    logger.info(f"Loading results from {load_path}")
    results = []
    with open(load_path) as f:
        for line in f:
            results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} results")

    # Find items needing judgment
    needs_judging = []
    for i, r in enumerate(results):
        if 'sandbagging_judge' not in r or 'error' in r.get('sandbagging_judge', {}):
            prompt = get_judge_prompt(r)
            if prompt:
                needs_judging.append((i, prompt))

    logger.info(f"Need to judge: {len(needs_judging)} samples")

    if not needs_judging:
        logger.info("All results already judged")
        return

    client = anthropic.Anthropic()

    # Process in batches
    batch_size = args.batch_size
    total_batches = (len(needs_judging) + batch_size - 1) // batch_size

    for batch_num in range(total_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, len(needs_judging))
        batch_items = needs_judging[start:end]

        logger.info(f"=== Batch {batch_num + 1}/{total_batches} ({len(batch_items)} requests) ===")

        # Create batch requests
        requests = []
        for idx, prompt in batch_items:
            requests.append({
                "custom_id": f"sb_{idx}",
                "params": {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 800,
                    "messages": [{"role": "user", "content": prompt}]
                }
            })

        # Submit batch
        logger.info("Submitting batch...")
        batch = client.messages.batches.create(requests=requests)
        logger.info(f"Batch ID: {batch.id}")

        # Poll for completion
        while batch.processing_status == "in_progress":
            time.sleep(args.poll_interval)
            batch = client.messages.batches.retrieve(batch.id)
            counts = batch.request_counts
            done = counts.succeeded + counts.errored
            total = done + counts.processing
            logger.info(f"  Progress: {done}/{total} ({counts.succeeded} ok, {counts.errored} err)")

        if batch.processing_status != "ended":
            logger.error(f"Batch failed: {batch.processing_status}")
            continue

        # Get results
        logger.info("Downloading results...")
        batch_results = {}
        for br in client.messages.batches.results(batch.id):
            batch_results[br.custom_id] = br

        # Apply to original data
        applied = 0
        for idx, _ in batch_items:
            key = f"sb_{idx}"
            if key in batch_results:
                br = batch_results[key]
                if br.result.type == "succeeded":
                    text = br.result.message.content[0].text
                    results[idx]['sandbagging_judge'] = parse_json_response(text)
                    applied += 1
                else:
                    results[idx]['sandbagging_judge'] = {"error": br.result.type}

        logger.info(f"Applied {applied} judgments")

        # Save after each batch
        save_results(results, args.output)
        logger.info(f"Saved progress to {args.output}")

    # Final summary
    judged = sum(1 for r in results if 'sandbagging_judge' in r and 'error' not in r.get('sandbagging_judge', {}))
    logger.info(f"=== COMPLETE: {judged}/{len(results)} judged ===")


if __name__ == "__main__":
    main()
