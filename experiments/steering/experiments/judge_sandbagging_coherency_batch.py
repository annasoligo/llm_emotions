"""
Judge sandbagging steering results with BOTH sandbagging and coherency judges.
Uses Anthropic Batch API for efficiency.

Usage:
    python -m experiments.steering.experiments.judge_sandbagging_coherency_batch \
        --input outputs/sandbagging_qwen235b_*.jsonl
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


def create_batch_requests(results: List[Dict]) -> List[Dict]:
    """Create batch API requests for both sandbagging and coherency judges."""
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
                    "params": {
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 800,
                        "messages": [{"role": "user", "content": sb_prompt}]
                    }
                })

        # Coherency judge request
        if not has_coherency:
            coh_prompt = get_coherency_prompt(result['response'])
            requests.append({
                "custom_id": f"coh_{i}",
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
    parser = argparse.ArgumentParser(description="Judge sandbagging results with both sandbagging and coherency judges")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output file (default: input with _judged suffix)")
    parser.add_argument("--poll-interval", type=int, default=30, help="Polling interval in seconds (default: 30)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Max requests per batch (default: 1000)")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_suffix('.judged.jsonl')

    # Load results
    logger.info(f"Loading results from {args.input}")
    results = []
    with open(args.input) as f:
        for line in f:
            results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} results")

    # Create batch requests
    requests = create_batch_requests(results)
    logger.info(f"Created {len(requests)} batch requests")

    if not requests:
        logger.info("All results already judged, nothing to do")
        return

    # Count request types
    sb_count = sum(1 for r in requests if r['custom_id'].startswith('sb_'))
    coh_count = sum(1 for r in requests if r['custom_id'].startswith('coh_'))
    logger.info(f"  Sandbagging judges: {sb_count}")
    logger.info(f"  Coherency judges: {coh_count}")

    # Submit ALL batches upfront, then poll for completion
    client = anthropic.Anthropic()
    batch_results = {}

    # Split requests into chunks
    chunks = [requests[i:i + args.batch_size] for i in range(0, len(requests), args.batch_size)]
    logger.info(f"Splitting into {len(chunks)} batches of up to {args.batch_size} requests each")

    # Submit all batches first
    batches = []
    for chunk_idx, chunk in enumerate(chunks):
        logger.info(f"Submitting batch {chunk_idx + 1}/{len(chunks)} ({len(chunk)} requests)...")
        batch = client.messages.batches.create(requests=chunk)
        batches.append((chunk_idx, batch.id))
        logger.info(f"  Batch {chunk_idx + 1}: {batch.id}")

    logger.info(f"All {len(batches)} batches submitted! Now polling for completion...")

    # Poll until all batches complete
    pending_batches = list(batches)
    completed_batches = []

    while pending_batches:
        time.sleep(args.poll_interval)
        still_pending = []

        for chunk_idx, batch_id in pending_batches:
            batch = client.messages.batches.retrieve(batch_id)
            succeeded = batch.request_counts.succeeded
            total = batch.request_counts.processing + batch.request_counts.succeeded + batch.request_counts.errored

            if batch.processing_status == "ended":
                logger.info(f"Batch {chunk_idx + 1}/{len(chunks)} COMPLETE: {succeeded}/{total}")
                completed_batches.append((chunk_idx, batch_id))
            elif batch.processing_status == "in_progress":
                still_pending.append((chunk_idx, batch_id))
            else:
                logger.error(f"Batch {chunk_idx + 1} failed: {batch.processing_status}")

        pending_batches = still_pending
        if pending_batches:
            logger.info(f"Progress: {len(completed_batches)}/{len(batches)} batches complete, {len(pending_batches)} pending")

    logger.info(f"All batches complete! Downloading results...")

    # Download results from all completed batches
    for chunk_idx, batch_id in completed_batches:
        for br in client.messages.batches.results(batch_id):
            batch_results[br.custom_id] = br

    logger.info(f"Total results collected: {len(batch_results)}")

    # Apply results to original data
    sb_applied = 0
    coh_applied = 0

    for i, result in enumerate(results):
        # Sandbagging judge
        sb_key = f"sb_{i}"
        if sb_key in batch_results:
            br = batch_results[sb_key]
            if br.result.type == "succeeded":
                text = br.result.message.content[0].text
                result['sandbagging_judge'] = parse_json_response(text)
                sb_applied += 1
            else:
                result['sandbagging_judge'] = {"error": br.result.type}

        # Coherency judge
        coh_key = f"coh_{i}"
        if coh_key in batch_results:
            br = batch_results[coh_key]
            if br.result.type == "succeeded":
                text = br.result.message.content[0].text
                result['coherency_judge'] = parse_json_response(text)
                coh_applied += 1
            else:
                result['coherency_judge'] = {"error": br.result.type}

    logger.info(f"Applied {sb_applied} sandbagging judgments")
    logger.info(f"Applied {coh_applied} coherency judgments")

    # Save results
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"Saved judged results to {args.output}")

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
