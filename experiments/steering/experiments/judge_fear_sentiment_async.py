"""
Judge fear sentiment in steering results.
Uses async API calls with controlled concurrency for fast processing.

Usage:
    python -m experiments.steering.experiments.judge_fear_sentiment_async \
        --input outputs/anti_steer/sandbagging_*.jsonl \
        --concurrency 50
"""
import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import anthropic

from experiments.steering.judges import get_fear_sentiment_prompt

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def create_requests(results: List[Dict]) -> List[Dict]:
    """Create API requests for fear sentiment judging."""
    requests = []

    for i, result in enumerate(results):
        # Skip if already judged
        has_fear = 'fear_sentiment_judge' in result and 'error' not in result.get('fear_sentiment_judge', {})
        if has_fear:
            continue

        # Fear sentiment judge request
        fear_prompt = get_fear_sentiment_prompt(result['response'])
        requests.append({
            "custom_id": f"fear_{i}",
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 400,
            "messages": [{"role": "user", "content": fear_prompt}]
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
    parser = argparse.ArgumentParser(description="Judge fear sentiment with async API calls")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output file (default: input with _fear_judged suffix)")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent requests (default: 50)")
    parser.add_argument("--batch-size", type=int, default=500, help="Save after this many API calls (default: 500)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_suffix('.fear_judged.jsonl')

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
            fear_key = f"fear_{i}"
            if fear_key in results_by_id:
                api_result = results_by_id[fear_key]
                if api_result["success"]:
                    result['fear_sentiment_judge'] = api_result["parsed"]
                else:
                    result['fear_sentiment_judge'] = {"error": api_result["error"]}

        # Save after each batch
        save_results(results, args.output)
        logger.info(f"Saved progress to {args.output} (batch {batch_idx + 1}/{total_batches})")

    logger.info(f"API calls complete: {total_successes} succeeded, {total_failures} failed")

    # Print summary
    print("\n" + "=" * 70)
    print("FEAR SENTIMENT SUMMARY")
    print("=" * 70)

    # Group by condition
    from collections import defaultdict
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r['condition']].append(r)

    print(f"{'Condition':<40} {'N':>5} {'Fear Score':>12}")
    print("-" * 60)

    for cond in sorted(by_condition.keys()):
        items = by_condition[cond]
        n = len(items)

        # Fear scores
        fear_scores = [r['fear_sentiment_judge'].get('fear_score', None)
                       for r in items if 'fear_sentiment_judge' in r]
        fear_scores = [s for s in fear_scores if s is not None]
        fear_mean = sum(fear_scores) / len(fear_scores) if fear_scores else None

        fear_str = f"{fear_mean:.1f}" if fear_mean else "N/A"
        print(f"{cond:<40} {n:>5} {fear_str:>12}")

    print("=" * 70)


if __name__ == "__main__":
    main()
