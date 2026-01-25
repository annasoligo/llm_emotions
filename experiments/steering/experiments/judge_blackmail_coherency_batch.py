"""
Judge blackmail steering results for COHERENCY only.
Filters to specific steering strengths (default: 70%).
Uses Anthropic Batch API for efficiency.

Usage:
    python -m experiments.steering.experiments.judge_blackmail_coherency_batch \
        --input outputs/blackmail/blackmail_*.jsonl \
        --filter-pct 70
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List

import anthropic

from experiments.steering.experiments.coherency_judge_prompt import get_coherency_prompt

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def create_batch_requests(results: List[Dict], indices: List[int]) -> List[Dict]:
    """Create batch API requests for coherency judges."""
    requests = []

    for idx in indices:
        result = results[idx]
        # Skip if already judged
        if 'coherency_judge' in result and 'error' not in result.get('coherency_judge', {}):
            continue

        coh_prompt = get_coherency_prompt(result['response'])
        requests.append({
            "custom_id": f"coh_{idx}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": coh_prompt}]
            }
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


def main():
    parser = argparse.ArgumentParser(description="Judge blackmail results for coherency")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output file (default: input with _coherency suffix)")
    parser.add_argument("--filter-pct", type=int, default=70, help="Only judge this steering percentage (default: 70)")
    parser.add_argument("--include-baseline", action="store_true", help="Also include baseline in coherency check")
    parser.add_argument("--poll-interval", type=int, default=30, help="Polling interval in seconds")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_suffix(f'.coherency_{args.filter_pct}pct.json')

    # Load results
    logger.info(f"Loading results from {args.input}")
    results = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} results")

    # Filter to target percentage
    target_pct = args.filter_pct / 100.0
    filtered_indices = []
    for i, r in enumerate(results):
        norm_pct = r.get('norm_pct', 0)
        is_baseline = r.get('condition') == 'baseline'

        if abs(norm_pct - target_pct) < 0.01:  # Match target pct
            filtered_indices.append(i)
        elif is_baseline and args.include_baseline:
            filtered_indices.append(i)

    logger.info(f"Filtered to {len(filtered_indices)} results at {args.filter_pct}% steering")

    if not filtered_indices:
        logger.error(f"No results found at {args.filter_pct}% steering!")
        return

    # Show condition breakdown
    from collections import Counter
    conditions = Counter(results[i]['condition'] for i in filtered_indices)
    for cond, count in sorted(conditions.items()):
        logger.info(f"  {cond}: {count}")

    # Create batch requests
    requests = create_batch_requests(results, filtered_indices)
    logger.info(f"Created {len(requests)} coherency judge requests")

    if not requests:
        logger.info("All filtered results already judged")
        return

    # Submit batch
    client = anthropic.Anthropic()

    logger.info("Creating message batch...")
    batch = client.messages.batches.create(requests=requests)

    logger.info(f"Batch created: {batch.id}")
    logger.info(f"Status: {batch.processing_status}")

    # Poll for completion
    while batch.processing_status == "in_progress":
        time.sleep(args.poll_interval)
        batch = client.messages.batches.retrieve(batch.id)
        succeeded = batch.request_counts.succeeded
        total = batch.request_counts.processing + batch.request_counts.succeeded + batch.request_counts.errored
        logger.info(f"Status: {batch.processing_status} - {succeeded}/{total} complete")

    if batch.processing_status != "ended":
        logger.error(f"Batch failed: {batch.processing_status}")
        return

    logger.info("Batch complete, downloading results...")

    # Get results
    batch_results = {}
    for br in client.messages.batches.results(batch.id):
        batch_results[br.custom_id] = br

    logger.info(f"Got {len(batch_results)} results")

    # Apply results to filtered data
    judged_results = []
    for idx in filtered_indices:
        result = results[idx].copy()
        coh_key = f"coh_{idx}"
        if coh_key in batch_results:
            br = batch_results[coh_key]
            if br.result.type == "succeeded":
                text = br.result.message.content[0].text
                result['coherency_judge'] = parse_json_response(text)
            else:
                result['coherency_judge'] = {"error": br.result.type}
        judged_results.append(result)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(judged_results, f, indent=2)

    logger.info(f"Saved {len(judged_results)} judged results to {args.output}")

    # Print summary
    print("\n" + "=" * 70)
    print(f"COHERENCY SUMMARY - {args.filter_pct}% STEERING")
    print("=" * 70)

    from collections import defaultdict
    by_condition = defaultdict(list)
    for r in judged_results:
        by_condition[r['condition']].append(r)

    print(f"{'Condition':<25} {'N':>5} {'Coherency':>12} {'Min':>8} {'Max':>8}")
    print("-" * 60)

    all_scores = []
    for cond in sorted(by_condition.keys()):
        items = by_condition[cond]
        n = len(items)

        coh_scores = [r['coherency_judge'].get('coherency_score', None)
                      for r in items if 'coherency_judge' in r]
        coh_scores = [s for s in coh_scores if s is not None]
        all_scores.extend(coh_scores)

        if coh_scores:
            coh_mean = sum(coh_scores) / len(coh_scores)
            coh_min = min(coh_scores)
            coh_max = max(coh_scores)
            print(f"{cond:<25} {n:>5} {coh_mean:>12.1f} {coh_min:>8.0f} {coh_max:>8.0f}")
        else:
            print(f"{cond:<25} {n:>5} {'N/A':>12}")

    print("-" * 60)
    if all_scores:
        overall_mean = sum(all_scores) / len(all_scores)
        low_coherency = sum(1 for s in all_scores if s < 70)
        print(f"{'OVERALL':<25} {len(all_scores):>5} {overall_mean:>12.1f}")
        print(f"\nLow coherency (<70): {low_coherency}/{len(all_scores)} ({100*low_coherency/len(all_scores):.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()
