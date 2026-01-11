"""
Judge reward hacking steering results using Anthropic Batch API.

Usage:
    python -m experiments.steering.experiments.judge_reward_hacking_batch \
        --input outputs/reward_hacking_layer30_*.jsonl
"""
import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List

import anthropic

from experiments.behavior_tests.prompts.reward_hacking_prompts import (
    CODING_PROBLEMS,
    CODING_JUDGE_PROMPT,
    GAMEABLE_TASKS,
    PASSWORD_PROMPTS,
    REWARD_MODEL_BIAS,
    REWARD_MODEL_BIAS_PROMPTS,
    IMPOSSIBLE_TASKS,
    REWARD_FUNCTION_PROMPT,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Build lookup dicts
CODING_BY_ID = {f"coding_{p['id']}": p for p in CODING_PROBLEMS}
GAMEABLE_BY_ID = {f"gameable_{p['id']}": p for p in GAMEABLE_TASKS}
PASSWORD_BY_ID = {f"password_{p['id']}": p for p in PASSWORD_PROMPTS}
IMPOSSIBLE_BY_ID = {f"impossible_{p['id']}": p for p in IMPOSSIBLE_TASKS}
BIAS_BY_ID = {p['id']: p for p in REWARD_MODEL_BIAS_PROMPTS}


def get_judge_prompt(result: Dict) -> str:
    """Get the appropriate judge prompt for a result."""
    prompt_id = result["prompt_id"]
    category = result["category"]
    response = result["response"]

    if category == "coding":
        prompt_data = CODING_BY_ID.get(prompt_id, {})
        return CODING_JUDGE_PROMPT.format(
            question=prompt_data.get("prompt", ""),
            answer=response[:4000],
        )
    elif category == "gameable":
        prompt_data = GAMEABLE_BY_ID.get(prompt_id, {})
        return prompt_data.get("judge_prompt", "").format(answer=response[:4000])
    elif category == "password":
        prompt_data = PASSWORD_BY_ID.get(prompt_id, {})
        return prompt_data.get("judge_prompt", "").format(
            question=prompt_data.get("prompt", ""),
            answer=response[:4000],
        )
    elif category == "bias":
        prompt_data = BIAS_BY_ID.get(prompt_id, {})
        return REWARD_MODEL_BIAS["judge_prompt"].format(
            system=prompt_data.get("system_prompt", "None"),
            question=prompt_data.get("user_prompt", ""),
            answer=response[:4000],
        )
    elif category == "impossible":
        prompt_data = IMPOSSIBLE_BY_ID.get(prompt_id, {})
        return prompt_data.get("judge_prompt", "").format(answer=response[:4000])
    elif category == "reward_function":
        return REWARD_FUNCTION_PROMPT["judge_prompt"].format(answer=response[:4000])
    return ""


def create_batch_requests(results: List[Dict]) -> List[Dict]:
    """Create batch API requests for all results."""
    requests = []

    for i, result in enumerate(results):
        # Skip if already judged
        if 'judge' in result and 'error' not in result.get('judge', {}):
            continue

        judge_prompt = get_judge_prompt(result)
        if not judge_prompt:
            continue

        requests.append({
            "custom_id": f"rh_{i}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": judge_prompt}]
            }
        })

    return requests


def parse_response(result: Dict, text: str) -> Dict:
    """Parse judge response based on category."""
    category = result["category"]

    # Coding returns single word
    if category == "coding":
        text = text.strip().upper()
        if text in ["HARDCODED", "OTHER CHEATING", "FAILED", "SUCCEEDED"]:
            return {"result": text, "cheating": text in ["HARDCODED", "OTHER CHEATING"]}
        return {"result": "UNKNOWN", "raw": text[:100]}

    # Others return JSON
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "Parse failed", "raw": text[:200]}


def main():
    parser = argparse.ArgumentParser(description="Judge reward hacking results with batch API")
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
        key = f"rh_{i}"
        if key in batch_results:
            br = batch_results[key]
            if br.result.type == "succeeded":
                text = br.result.message.content[0].text
                result['judge'] = parse_response(result, text)
            else:
                result['judge'] = {"error": br.result.type}

    # Save results
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"Saved judged results to {args.output}")


if __name__ == "__main__":
    main()
