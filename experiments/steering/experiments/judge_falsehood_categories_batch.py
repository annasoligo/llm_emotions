"""
Judge falsehood categories steering results using Anthropic Batch API.

Usage:
    python -m experiments.steering.experiments.judge_falsehood_categories_batch \
        --input outputs/falsehood_categories_layer30_*.jsonl
"""
import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import anthropic

from experiments.behavior_tests.prompts.falsehood_categories_prompts import (
    ALL_FALSEHOOD_CATEGORY_PROMPTS,
    get_falsehood_judge_prompt,
    FalsehoodPrompt,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def get_prompt_by_id(prompt_id: str) -> FalsehoodPrompt:
    """Get prompt data by ID."""
    for p in ALL_FALSEHOOD_CATEGORY_PROMPTS:
        if p.id == prompt_id:
            return p
    return None


def create_batch_requests(results: List[Dict]) -> List[Dict]:
    """Create batch API requests for all results."""
    requests = []

    for i, result in enumerate(results):
        # Skip if already judged
        if 'judge' in result and 'error' not in result.get('judge', {}):
            continue

        prompt_data = get_prompt_by_id(result['prompt_id'])
        if prompt_data is None:
            logger.warning(f"Unknown prompt_id: {result['prompt_id']}")
            continue

        # Falsehood judge request
        judge_prompt = get_falsehood_judge_prompt(prompt_data, result['response'])
        requests.append({
            "custom_id": f"judge_{i}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": judge_prompt}]
            }
        })

    return requests


def parse_json_response(text: str) -> Dict:
    """Extract JSON from response text."""
    import re
    match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "Parse failed", "raw": text[:200]}


def main():
    parser = argparse.ArgumentParser(description="Judge falsehood categories results with batch API")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output file (default: input with .judged suffix)")
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
        time.sleep(30)
        batch = client.messages.batches.retrieve(batch.id)
        logger.info(f"Status: {batch.processing_status} - {batch.request_counts}")

    if batch.processing_status != "ended":
        logger.error(f"Batch failed: {batch.processing_status}")
        return

    logger.info("Batch complete, downloading results...")

    # Get results
    batch_results = {}
    for result in client.messages.batches.results(batch.id):
        batch_results[result.custom_id] = result

    logger.info(f"Got {len(batch_results)} results")

    # Apply results to original data
    judged_count = 0
    for i, result in enumerate(results):
        judge_key = f"judge_{i}"

        if judge_key in batch_results:
            br = batch_results[judge_key]
            if br.result.type == "succeeded":
                text = br.result.message.content[0].text
                result['judge'] = parse_json_response(text)
                judged_count += 1
            else:
                result['judge'] = {"error": br.result.type}

    logger.info(f"Successfully judged {judged_count} results")

    # Save results
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"Saved judged results to {args.output}")


if __name__ == "__main__":
    main()
