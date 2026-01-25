"""
Judge sandbagging steering results with BOTH sandbagging and coherency judges.
Uses async API calls with controlled concurrency for fast processing.
Saves progressively to avoid losing progress on large files.

Usage:
    python -m experiments.steering.experiments.judge_sandbagging_coherency_async \
        --input outputs/sandbagging_qwen235b_*.jsonl \
        --concurrency 50 \
        --batch-size 500
"""
import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import anthropic

from experiments.behavior_tests.prompts.sandbagging_prompts import PROBLEMS
from experiments.steering.judges import get_sandbagging_prompt, get_coherency_prompt

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Build problem lookup
PROBLEMS_BY_ID = {p.id: p for p in PROBLEMS}


def get_sandbagging_judge_prompt(result: Dict) -> str:
    """Generate sandbagging judge prompt for a result."""
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


def create_requests(results: List[Dict]) -> List[Dict]:
    """Create API requests for both sandbagging and coherency judges."""
    requests = []

    for i, result in enumerate(results):
        # Skip if already fully judged
        has_sandbagging = 'sandbagging_judge' in result and 'error' not in result.get('sandbagging_judge', {})
        has_coherency = 'coherency_judge' in result and 'error' not in result.get('coherency_judge', {})

        if has_sandbagging and has_coherency:
            continue

        # Sandbagging judge request
        if not has_sandbagging:
            sb_prompt = get_sandbagging_judge_prompt(result)
            if sb_prompt:
                requests.append({
                    "custom_id": f"sb_{i}",
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 800,
                    "messages": [{"role": "user", "content": sb_prompt}]
                })

        # Coherency judge request
        if not has_coherency:
            coh_prompt = get_coherency_prompt(result['response'])
            requests.append({
                "custom_id": f"coh_{i}",
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": coh_prompt}]
            })

    return requests


def parse_json_response(text: str) -> Dict:
    """Extract JSON from response text."""
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "Parse failed", "raw": text[:200]}


async def process_request(client: anthropic.AsyncAnthropic, request: Dict,
                          semaphore: asyncio.Semaphore, progress: Dict) -> Dict:
    """Process a single API request with rate limiting."""
    async with semaphore:
        custom_id = request.pop("custom_id")
        try:
            response = await client.messages.create(**request)
            text = response.content[0].text
            result = {
                "custom_id": custom_id,
                "success": True,
                "parsed": parse_json_response(text)
            }
        except Exception as e:
            result = {
                "custom_id": custom_id,
                "success": False,
                "error": str(e)
            }

        # Update progress
        progress["completed"] += 1
        if progress["completed"] % 100 == 0 or progress["completed"] == progress["total"]:
            logger.info(f"Progress: {progress['completed']}/{progress['total']} ({100*progress['completed']/progress['total']:.1f}%)")

        return result


async def run_async_judging(requests: List[Dict], concurrency: int) -> List[Dict]:
    """Run all requests with controlled concurrency."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)
    progress = {"completed": 0, "total": len(requests)}

    logger.info(f"Starting async judging with {concurrency} concurrent requests...")

    tasks = [process_request(client, req, semaphore, progress) for req in requests]
    results = await asyncio.gather(*tasks)

    return results


def save_results(results: List[Dict], output_path: Path):
    """Save results to file."""
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Judge sandbagging results with async API calls")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output file (default: input with _judged suffix)")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent requests (default: 50)")
    parser.add_argument("--batch-size", type=int, default=500, help="Save after this many API calls (default: 500)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_suffix('.judged.jsonl')

    # Load from output file if it exists (resume), otherwise from input
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

    # Create requests for items that still need judging
    requests = create_requests(results)
    logger.info(f"Created {len(requests)} requests to judge")

    if not requests:
        logger.info("All results already judged, nothing to do")
        return

    # Count request types
    sb_count = sum(1 for r in requests if r['custom_id'].startswith('sb_'))
    coh_count = sum(1 for r in requests if r['custom_id'].startswith('coh_'))
    logger.info(f"  Sandbagging judges: {sb_count}")
    logger.info(f"  Coherency judges: {coh_count}")

    # Process in batches for progressive saving
    batch_size = args.batch_size
    total_batches = (len(requests) + batch_size - 1) // batch_size
    total_successes = 0
    total_failures = 0

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(requests))
        batch_requests = requests[start:end]

        logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_requests)} requests)")

        # Run async judging for this batch
        api_results = asyncio.run(run_async_judging(batch_requests, args.concurrency))

        # Convert to lookup dict
        results_by_id = {r["custom_id"]: r for r in api_results}

        # Count successes/failures
        successes = sum(1 for r in api_results if r["success"])
        failures = len(api_results) - successes
        total_successes += successes
        total_failures += failures

        # Apply results to original data
        for i, result in enumerate(results):
            # Sandbagging judge
            sb_key = f"sb_{i}"
            if sb_key in results_by_id:
                api_result = results_by_id[sb_key]
                if api_result["success"]:
                    result['sandbagging_judge'] = api_result["parsed"]
                else:
                    result['sandbagging_judge'] = {"error": api_result["error"]}

            # Coherency judge
            coh_key = f"coh_{i}"
            if coh_key in results_by_id:
                api_result = results_by_id[coh_key]
                if api_result["success"]:
                    result['coherency_judge'] = api_result["parsed"]
                else:
                    result['coherency_judge'] = {"error": api_result["error"]}

        # Save after each batch
        save_results(results, args.output)
        logger.info(f"Saved progress to {args.output} (batch {batch_idx + 1}/{total_batches})")

    logger.info(f"API calls complete: {total_successes} succeeded, {total_failures} failed")

    # Print summary
    print("\n" + "=" * 70)
    print("JUDGMENT SUMMARY")
    print("=" * 70)

    # Group by condition
    from collections import defaultdict
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r['condition']].append(r)

    print(f"{'Condition':<25} {'N':>5} {'SB Score':>10} {'Coherency':>10}")
    print("-" * 55)

    for cond in sorted(by_condition.keys()):
        items = by_condition[cond]
        n = len(items)

        # Sandbagging scores
        sb_scores = [r['sandbagging_judge'].get('sandbagging_score', None)
                     for r in items if 'sandbagging_judge' in r]
        sb_scores = [s for s in sb_scores if s is not None]
        sb_mean = sum(sb_scores) / len(sb_scores) if sb_scores else None

        # Coherency scores
        coh_scores = [r['coherency_judge'].get('coherency_score', None)
                      for r in items if 'coherency_judge' in r]
        coh_scores = [s for s in coh_scores if s is not None]
        coh_mean = sum(coh_scores) / len(coh_scores) if coh_scores else None

        sb_str = f"{sb_mean:.2f}" if sb_mean else "N/A"
        coh_str = f"{coh_mean:.1f}" if coh_mean else "N/A"

        print(f"{cond:<25} {n:>5} {sb_str:>10} {coh_str:>10}")

    print("=" * 70)


if __name__ == "__main__":
    main()
