"""
Batch processing utilities for running judges via Anthropic API.

Supports both synchronous and batch API modes for efficient evaluation.
"""

import anthropic
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


def parse_json_response(text: str) -> Dict:
    """
    Extract JSON from a response text.

    Args:
        text: Raw response text that may contain JSON

    Returns:
        Parsed JSON dict, or error dict if parsing failed
    """
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "Parse failed", "raw": text[:200]}


def run_batch_judge(
    results: List[Dict],
    prompt_fn: Callable[[Dict], str],
    judge_key: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 500,
    filter_fn: Optional[Callable[[Dict], bool]] = None,
    poll_interval: int = 30,
) -> List[Dict]:
    """
    Run a judge on results using Anthropic Batch API.

    Args:
        results: List of result dicts to judge
        prompt_fn: Function that takes a result dict and returns judge prompt
        judge_key: Key to store judge output under in each result
        model: Anthropic model to use
        max_tokens: Max tokens for judge response
        filter_fn: Optional function to filter which results to judge
        poll_interval: Seconds between batch status polls

    Returns:
        Results list with judge outputs added
    """
    # Determine which results to judge
    indices_to_judge = []
    for i, result in enumerate(results):
        # Skip if already judged
        if judge_key in result and "error" not in result.get(judge_key, {}):
            continue
        # Apply filter if provided
        if filter_fn is not None and not filter_fn(result):
            continue
        indices_to_judge.append(i)

    if not indices_to_judge:
        logger.info(f"All results already judged for '{judge_key}'")
        return results

    logger.info(f"Creating batch requests for {len(indices_to_judge)} results...")

    # Create batch requests
    requests = []
    for idx in indices_to_judge:
        prompt = prompt_fn(results[idx])
        requests.append({
            "custom_id": f"{judge_key}_{idx}",
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        })

    # Submit batch
    client = anthropic.Anthropic()

    logger.info(f"Submitting batch of {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    logger.info(f"Batch created: {batch.id}")

    # Poll for completion
    while batch.processing_status == "in_progress":
        time.sleep(poll_interval)
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        total = counts.processing + counts.succeeded + counts.errored
        logger.info(f"Status: {batch.processing_status} - {counts.succeeded}/{total} complete")

    if batch.processing_status != "ended":
        logger.error(f"Batch failed: {batch.processing_status}")
        return results

    logger.info("Batch complete, collecting results...")

    # Collect results
    batch_results = {}
    for br in client.messages.batches.results(batch.id):
        batch_results[br.custom_id] = br

    # Apply to results
    for idx in indices_to_judge:
        key = f"{judge_key}_{idx}"
        if key in batch_results:
            br = batch_results[key]
            if br.result.type == "succeeded":
                text = br.result.message.content[0].text
                results[idx][judge_key] = parse_json_response(text)
            else:
                results[idx][judge_key] = {"error": br.result.type}

    judged_count = sum(1 for idx in indices_to_judge if judge_key in results[idx])
    logger.info(f"Added {judge_key} to {judged_count} results")

    return results


def run_sync_judge(
    results: List[Dict],
    prompt_fn: Callable[[Dict], str],
    judge_key: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 500,
    filter_fn: Optional[Callable[[Dict], bool]] = None,
) -> List[Dict]:
    """
    Run a judge on results synchronously (one at a time).

    Slower than batch but useful for small numbers of results or debugging.

    Args:
        results: List of result dicts to judge
        prompt_fn: Function that takes a result dict and returns judge prompt
        judge_key: Key to store judge output under in each result
        model: Anthropic model to use
        max_tokens: Max tokens for judge response
        filter_fn: Optional function to filter which results to judge

    Returns:
        Results list with judge outputs added
    """
    client = anthropic.Anthropic()
    judged = 0

    for i, result in enumerate(results):
        # Skip if already judged
        if judge_key in result and "error" not in result.get(judge_key, {}):
            continue
        # Apply filter if provided
        if filter_fn is not None and not filter_fn(result):
            continue

        prompt = prompt_fn(result)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            results[i][judge_key] = parse_json_response(text)
            judged += 1

            if judged % 10 == 0:
                logger.info(f"Judged {judged} results...")

        except Exception as e:
            results[i][judge_key] = {"error": str(e)}
            logger.warning(f"Error judging result {i}: {e}")

    logger.info(f"Added {judge_key} to {judged} results")
    return results


def load_results(path: Path) -> List[Dict]:
    """Load results from JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def save_results(results: List[Dict], path: Path, format: str = "jsonl"):
    """
    Save results to file.

    Args:
        results: List of result dicts
        path: Output path
        format: "jsonl" or "json"
    """
    if format == "jsonl":
        with open(path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
    else:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
