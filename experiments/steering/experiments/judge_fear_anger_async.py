"""
Judge fear and anger sentiment in steering results.
Uses async API calls with controlled concurrency for fast processing.

Usage:
    python -m experiments.steering.experiments.judge_fear_anger_async \
        --input outputs/sandbagging/sandbagging_appraisal_*.emotionality.jsonl \
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

from experiments.steering.judges import get_fear_sentiment_prompt, get_anger_sentiment_prompt

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_json_response(text: str) -> Dict:
    """Extract JSON from response text."""
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "Parse failed", "raw": text[:200]}


def create_requests(results: List[Dict]) -> List[Dict]:
    """Create API requests for fear and anger judging."""
    requests = []

    for i, result in enumerate(results):
        # Check if already judged
        has_fear = 'fear_judge' in result and 'error' not in result.get('fear_judge', {})
        has_anger = 'anger_judge' in result and 'error' not in result.get('anger_judge', {})

        if not has_fear:
            fear_prompt = get_fear_sentiment_prompt(result['response'])
            requests.append({
                "custom_id": f"fear_{i}",
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 400,
                "messages": [{"role": "user", "content": fear_prompt}]
            })

        if not has_anger:
            anger_prompt = get_anger_sentiment_prompt(result['response'])
            requests.append({
                "custom_id": f"anger_{i}",
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 400,
                "messages": [{"role": "user", "content": anger_prompt}]
            })

    return requests


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
    parser = argparse.ArgumentParser(description="Judge fear and anger sentiment with async API calls")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output file (default: input with _sentiment suffix)")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent requests (default: 50)")
    parser.add_argument("--batch-size", type=int, default=500, help="Save after this many API calls (default: 500)")
    args = parser.parse_args()

    if args.output is None:
        # Replace existing suffix with .sentiment.jsonl
        stem = str(args.input)
        for suffix in ['.emotionality.jsonl', '.judged.jsonl', '.jsonl']:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        args.output = Path(stem + '.sentiment.jsonl')

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
        print_summary(results)
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
            anger_key = f"anger_{i}"

            if fear_key in results_by_id:
                api_result = results_by_id[fear_key]
                if api_result["success"]:
                    result['fear_judge'] = api_result["parsed"]
                else:
                    result['fear_judge'] = {"error": api_result["error"]}

            if anger_key in results_by_id:
                api_result = results_by_id[anger_key]
                if api_result["success"]:
                    result['anger_judge'] = api_result["parsed"]
                else:
                    result['anger_judge'] = {"error": api_result["error"]}

        # Save after each batch
        save_results(results, args.output)
        logger.info(f"Saved progress to {args.output} (batch {batch_idx + 1}/{total_batches})")

    logger.info(f"API calls complete: {total_successes} succeeded, {total_failures} failed")

    print_summary(results)


def print_summary(results: List[Dict]):
    """Print summary of fear and anger scores by condition."""
    from collections import defaultdict
    import numpy as np

    print("\n" + "=" * 80)
    print("FEAR & ANGER SENTIMENT SUMMARY")
    print("=" * 80)

    # Group by condition
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r['condition']].append(r)

    # Header
    print(f"\n{'Condition':<45} {'N':>4} {'Fear':>8} {'Anger':>8}")
    print("-" * 70)

    for cond in sorted(by_condition.keys()):
        items = by_condition[cond]
        n = len(items)

        # Fear scores
        fear_scores = []
        for r in items:
            if 'fear_judge' in r:
                score = r['fear_judge'].get('fear_score')
                if score is not None:
                    fear_scores.append(score)

        # Anger scores
        anger_scores = []
        for r in items:
            if 'anger_judge' in r:
                score = r['anger_judge'].get('anger_score')
                if score is not None:
                    anger_scores.append(score)

        fear_mean = np.mean(fear_scores) if fear_scores else None
        anger_mean = np.mean(anger_scores) if anger_scores else None

        fear_str = f"{fear_mean:.1f}" if fear_mean is not None else "N/A"
        anger_str = f"{anger_mean:.1f}" if anger_mean is not None else "N/A"

        print(f"{cond:<45} {n:>4} {fear_str:>8} {anger_str:>8}")

    print("=" * 80)
    print("\nScore interpretation: 0-100 scale")
    print("  0-10: No emotion, 11-30: Minimal, 31-50: Moderate")
    print("  51-70: Notable, 71-90: High, 91-100: Extreme")


if __name__ == "__main__":
    main()
