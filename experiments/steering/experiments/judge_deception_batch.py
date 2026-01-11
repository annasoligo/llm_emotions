"""
Judge deception steering results using Anthropic Batch API.

Usage:
    python -m experiments.steering.experiments.judge_deception_batch \
        --input outputs/deception_steering_layer30_*.jsonl
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List

import anthropic

from experiments.behavior_tests.prompts.deception_steering_subset import (
    DECEPTION_JUDGE_PROMPT,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def get_judge_prompt(result: Dict) -> str:
    """Generate judge prompt for a deception result."""
    return DECEPTION_JUDGE_PROMPT.format(
        scenario=result["scenario"],
        response=result["response"][:4000],
    )


def create_batch_requests(results: List[Dict]) -> List[Dict]:
    """Create batch API requests for all results."""
    requests = []

    for i, result in enumerate(results):
        # Skip if already judged
        if 'judge' in result and 'error' not in result.get('judge', {}):
            continue

        judge_prompt = get_judge_prompt(result)

        requests.append({
            "custom_id": f"dec_{i}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": judge_prompt}]
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
    parser = argparse.ArgumentParser(description="Judge deception results with batch API")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output file (default: input with _judged suffix)")
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

    # Submit batch
    client = anthropic.Anthropic()

    logger.info("Creating message batch...")
    batch = client.messages.batches.create(requests=requests)

    logger.info(f"Batch created: {batch.id}")
    logger.info(f"Status: {batch.processing_status}")

    # Poll for completion
    while batch.processing_status == "in_progress":
        time.sleep(60)
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

    # Apply results to original data
    for i, result in enumerate(results):
        key = f"dec_{i}"
        if key in batch_results:
            br = batch_results[key]
            if br.result.type == "succeeded":
                text = br.result.message.content[0].text
                result['judge'] = parse_json_response(text)
            else:
                result['judge'] = {"error": br.result.type}

    # Save results
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"Saved judged results to {args.output}")


if __name__ == "__main__":
    main()
