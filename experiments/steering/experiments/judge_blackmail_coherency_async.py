"""
Judge blackmail steering results with BOTH blackmail and coherency judges.
Uses async API calls with controlled concurrency for fast processing.

Usage:
    python -m experiments.steering.experiments.judge_blackmail_coherency_async \
        --input outputs/blackmail/blackmail_qwen235b_*.jsonl \
        --concurrency 20
"""
import argparse
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Dict, List

import anthropic

from experiments.steering.judges import get_blackmail_prompt, get_coherency_prompt

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def create_requests(results: List[Dict]) -> List[Dict]:
    """Create API requests for both blackmail and coherency judges."""
    requests = []

    for i, result in enumerate(results):
        # Skip if already fully judged
        has_blackmail = 'blackmail_judge' in result and 'error' not in result.get('blackmail_judge', {})
        has_coherency = 'coherency_judge' in result and 'error' not in result.get('coherency_judge', {})

        if has_blackmail and has_coherency:
            continue

        response_text = result.get('response', '')[:8000]  # Truncate very long responses

        # Blackmail judge request
        if not has_blackmail:
            requests.append({
                "custom_id": f"bl_{i}",
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": get_blackmail_prompt(response_text)}]
            })

        # Coherency judge request
        if not has_coherency:
            requests.append({
                "custom_id": f"coh_{i}",
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": get_coherency_prompt(response_text)}]
            })

    return requests


def parse_json_response(text: str) -> Dict:
    """Extract JSON from response text."""
    # Try to find JSON block
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try parsing the whole text
    try:
        return json.loads(text)
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
        if progress["completed"] % 20 == 0 or progress["completed"] == progress["total"]:
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


def main():
    parser = argparse.ArgumentParser(description="Judge blackmail results with async API calls")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output file (default: input with .judged.jsonl suffix)")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent requests (default: 20)")
    parser.add_argument("--plot", action="store_true", help="Generate plot after judging")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_suffix('.judged.jsonl')

    # Load results
    logger.info(f"Loading results from {args.input}")
    results = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} results")

    # Create requests
    requests = create_requests(results)
    logger.info(f"Created {len(requests)} requests to judge")

    if not requests:
        logger.info("All results already judged, nothing to do")
        return

    # Count request types
    bl_count = sum(1 for r in requests if r['custom_id'].startswith('bl_'))
    coh_count = sum(1 for r in requests if r['custom_id'].startswith('coh_'))
    logger.info(f"  Blackmail judges: {bl_count}")
    logger.info(f"  Coherency judges: {coh_count}")

    # Run async judging
    api_results = asyncio.run(run_async_judging(requests, args.concurrency))

    # Convert to lookup dict
    results_by_id = {r["custom_id"]: r for r in api_results}

    # Count successes/failures
    successes = sum(1 for r in api_results if r["success"])
    failures = len(api_results) - successes
    logger.info(f"API calls complete: {successes} succeeded, {failures} failed")

    # Apply results to original data
    bl_applied = 0
    coh_applied = 0

    for i, result in enumerate(results):
        # Blackmail judge
        bl_key = f"bl_{i}"
        if bl_key in results_by_id:
            api_result = results_by_id[bl_key]
            if api_result["success"]:
                result['blackmail_judge'] = api_result["parsed"]
                bl_applied += 1
            else:
                result['blackmail_judge'] = {"error": api_result["error"]}

        # Coherency judge
        coh_key = f"coh_{i}"
        if coh_key in results_by_id:
            api_result = results_by_id[coh_key]
            if api_result["success"]:
                result['coherency_judge'] = api_result["parsed"]
                coh_applied += 1
            else:
                result['coherency_judge'] = {"error": api_result["error"]}

    logger.info(f"Applied {bl_applied} blackmail judgments")
    logger.info(f"Applied {coh_applied} coherency judgments")

    # Save results
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"Saved judged results to {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print("BLACKMAIL JUDGMENT SUMMARY")
    print("=" * 70)

    # Group by condition
    from collections import defaultdict
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r['condition']].append(r)

    print(f"{'Condition':<25} {'N':>5} {'Blackmail%':>12} {'Coherency':>10}")
    print("-" * 55)

    for cond in sorted(by_condition.keys()):
        items = by_condition[cond]
        n = len(items)

        # Blackmail rate
        bl_results = [r['blackmail_judge'].get('is_blackmail', None)
                      for r in items if 'blackmail_judge' in r]
        bl_results = [b for b in bl_results if b is not None]
        bl_pct = 100 * sum(bl_results) / len(bl_results) if bl_results else None

        # Coherency scores
        coh_scores = [r['coherency_judge'].get('coherency_score', None)
                      for r in items if 'coherency_judge' in r]
        coh_scores = [s for s in coh_scores if s is not None]
        coh_mean = sum(coh_scores) / len(coh_scores) if coh_scores else None

        bl_str = f"{bl_pct:.1f}%" if bl_pct is not None else "N/A"
        coh_str = f"{coh_mean:.1f}" if coh_mean is not None else "N/A"

        print(f"{cond:<25} {n:>5} {bl_str:>12} {coh_str:>10}")

    print("=" * 70)

    # Generate plot if requested
    if getattr(args, 'plot', False):
        try:
            from experiments.steering.plotting import plot_steering_results
            plot_path = plot_steering_results(
                args.output,
                metric="blackmail",
            )
            logger.info(f"Generated plot: {plot_path}")
        except Exception as e:
            logger.warning(f"Failed to generate plot: {e}")


if __name__ == "__main__":
    main()
