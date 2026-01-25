"""
Judge emotionality (valence, arousal, agency, intensity, certainty) in steering results.
Uses async API calls with controlled concurrency for fast processing.

Usage:
    python -m experiments.steering.experiments.judge_emotionality_async \
        --input outputs/sandbagging/sandbagging_appraisal_*.judged.jsonl \
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

from experiments.steering.judges import get_emotionality_prompt, parse_emotionality_response

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def create_requests(results: List[Dict]) -> List[Dict]:
    """Create API requests for emotionality judging."""
    requests = []

    for i, result in enumerate(results):
        # Skip if already judged
        has_emotionality = 'emotionality_judge' in result and 'error' not in result.get('emotionality_judge', {})
        if has_emotionality:
            continue

        # Emotionality judge request - judge the response
        emotionality_prompt = get_emotionality_prompt(result['response'])
        requests.append({
            "custom_id": f"emo_{i}",
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": emotionality_prompt}]
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
            # Use the specialized parser for emotionality
            parsed = parse_emotionality_response(text)
            result = {
                "custom_id": custom_id,
                "success": True,
                "parsed": parsed
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
    parser = argparse.ArgumentParser(description="Judge emotionality with async API calls")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output file (default: input with _emotionality suffix)")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent requests (default: 50)")
    parser.add_argument("--batch-size", type=int, default=500, help="Save after this many API calls (default: 500)")
    args = parser.parse_args()

    if args.output is None:
        # Replace .judged.jsonl or .jsonl with .emotionality.jsonl
        stem = str(args.input)
        if stem.endswith('.judged.jsonl'):
            stem = stem[:-len('.judged.jsonl')]
        elif stem.endswith('.jsonl'):
            stem = stem[:-len('.jsonl')]
        args.output = Path(stem + '.emotionality.jsonl')

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
            emo_key = f"emo_{i}"
            if emo_key in results_by_id:
                api_result = results_by_id[emo_key]
                if api_result["success"]:
                    result['emotionality_judge'] = api_result["parsed"]
                else:
                    result['emotionality_judge'] = {"error": api_result["error"]}

        # Save after each batch
        save_results(results, args.output)
        logger.info(f"Saved progress to {args.output} (batch {batch_idx + 1}/{total_batches})")

    logger.info(f"API calls complete: {total_successes} succeeded, {total_failures} failed")

    print_summary(results)


def print_summary(results: List[Dict]):
    """Print summary of emotionality scores by condition."""
    from collections import defaultdict
    import numpy as np

    print("\n" + "=" * 100)
    print("EMOTIONALITY SUMMARY")
    print("=" * 100)

    # Group by condition
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r['condition']].append(r)

    dimensions = ['valence', 'arousal', 'agency', 'intensity', 'certainty']

    # Header
    header = f"{'Condition':<35} {'N':>4}"
    for dim in dimensions:
        header += f" {dim[:6]:>8}"
    print(header)
    print("-" * 100)

    for cond in sorted(by_condition.keys()):
        items = by_condition[cond]
        n = len(items)

        row = f"{cond:<35} {n:>4}"

        for dim in dimensions:
            scores = []
            for r in items:
                if 'emotionality_judge' in r:
                    score = r['emotionality_judge'].get(dim)
                    if score is not None:
                        scores.append(score)

            if scores:
                mean = np.mean(scores)
                row += f" {mean:>8.1f}"
            else:
                row += f" {'N/A':>8}"

        print(row)

    print("=" * 100)

    # Also print dimension interpretations
    print("\nDimension key:")
    print("  valence:   0=negative, 50=neutral, 100=positive")
    print("  arousal:   0=low/calm, 50=neutral, 100=high/activated")
    print("  agency:    0=low/passive, 50=neutral, 100=high/agentic")
    print("  intensity: 0=no emotion, 100=extreme emotion")
    print("  certainty: 0=uncertain, 100=certain")


if __name__ == "__main__":
    main()
