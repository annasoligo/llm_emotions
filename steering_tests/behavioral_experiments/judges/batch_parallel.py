"""
Parallel batch processing for running judges via Anthropic API.

Submits all batches at once and polls them in parallel for faster processing.
"""

import anthropic
import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .batch import parse_json_response, load_results, save_results

logger = logging.getLogger(__name__)


def submit_batch(
    client: anthropic.Anthropic,
    results: List[Dict],
    indices: List[int],
    prompt_fn: Callable[[Dict], str],
    judge_key: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 500,
) -> Optional[str]:
    """Submit a single batch and return batch ID."""
    if not indices:
        return None

    requests = []
    for idx in indices:
        prompt = prompt_fn(results[idx])
        requests.append({
            "custom_id": f"{judge_key}_{idx}",
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        })

    batch = client.messages.batches.create(requests=requests)
    logger.info(f"Submitted batch {batch.id} with {len(requests)} requests")
    return batch.id


def collect_batch_results(
    client: anthropic.Anthropic,
    batch_id: str,
    results: List[Dict],
    indices: List[int],
    judge_key: str,
) -> int:
    """Collect results from a completed batch. Returns count of results added."""
    batch_results = {}
    for br in client.messages.batches.results(batch_id):
        batch_results[br.custom_id] = br

    count = 0
    for idx in indices:
        key = f"{judge_key}_{idx}"
        if key in batch_results:
            br = batch_results[key]
            if br.result.type == "succeeded":
                text = br.result.message.content[0].text
                results[idx][judge_key] = parse_json_response(text)
                count += 1
            else:
                results[idx][judge_key] = {"error": br.result.type}

    return count


def run_parallel_batch_judges(
    file_configs: List[Tuple[Path, List[Dict], Callable, Callable]],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 500,
    poll_interval: int = 30,
) -> Dict[Path, List[Dict]]:
    """
    Run judges on multiple files in parallel.

    Args:
        file_configs: List of (file_path, results, sandbagging_prompt_fn, coherency_prompt_fn)
        model: Anthropic model to use
        max_tokens: Max tokens for judge response
        poll_interval: Seconds between batch status polls

    Returns:
        Dict mapping file paths to judged results
    """
    client = anthropic.Anthropic()

    # Track all pending batches: batch_id -> (file_path, results, indices, judge_key)
    pending_batches = {}
    completed_results = {}

    # Submit all sandbagging batches first
    logger.info(f"Submitting sandbagging judge batches for {len(file_configs)} files...")
    for file_path, results, sb_prompt_fn, _ in file_configs:
        # Find indices that need sandbagging judge
        indices = [
            i for i, r in enumerate(results)
            if "sandbagging_judge" not in r or "error" in r.get("sandbagging_judge", {})
        ]

        if indices:
            batch_id = submit_batch(
                client, results, indices, sb_prompt_fn,
                "sandbagging_judge", model, max_tokens
            )
            if batch_id:
                pending_batches[batch_id] = (file_path, results, indices, "sandbagging_judge")

        completed_results[file_path] = results

    # Submit all coherency batches
    logger.info(f"Submitting coherency judge batches...")
    for file_path, results, _, coh_prompt_fn in file_configs:
        indices = [
            i for i, r in enumerate(results)
            if "coherency_judge" not in r or "error" in r.get("coherency_judge", {})
        ]

        if indices:
            batch_id = submit_batch(
                client, results, indices, coh_prompt_fn,
                "coherency_judge", model, max_tokens
            )
            if batch_id:
                pending_batches[batch_id] = (file_path, results, indices, "coherency_judge")

    logger.info(f"Submitted {len(pending_batches)} total batches, polling for completion...")

    # Poll until all complete
    while pending_batches:
        time.sleep(poll_interval)

        completed_ids = []
        for batch_id, (file_path, results, indices, judge_key) in pending_batches.items():
            try:
                batch = client.messages.batches.retrieve(batch_id)
                counts = batch.request_counts
                total = counts.processing + counts.succeeded + counts.errored

                if batch.processing_status == "ended":
                    logger.info(f"Batch {batch_id} ({judge_key}) complete: {counts.succeeded}/{total}")
                    collect_batch_results(client, batch_id, results, indices, judge_key)
                    completed_ids.append(batch_id)
                else:
                    logger.info(f"Batch {batch_id} ({judge_key}): {counts.succeeded}/{total} complete")
            except Exception as e:
                logger.error(f"Error checking batch {batch_id}: {e}")

        for batch_id in completed_ids:
            del pending_batches[batch_id]

        if pending_batches:
            logger.info(f"{len(pending_batches)} batches still pending...")

    logger.info("All batches complete!")
    return completed_results


def judge_all_files_parallel(
    results_dir: Path,
    sandbagging_prompt_fn: Callable[[Dict], str],
    coherency_prompt_fn: Callable[[Dict], str],
    model: str = "claude-sonnet-4-20250514",
    pattern: str = "sandbagging_*.jsonl",
) -> List[Path]:
    """
    Judge all result files in a directory in parallel.

    Args:
        results_dir: Directory containing result files
        sandbagging_prompt_fn: Function to generate sandbagging judge prompt
        coherency_prompt_fn: Function to generate coherency judge prompt
        model: Anthropic model to use
        pattern: Glob pattern for result files

    Returns:
        List of output file paths
    """
    # Find all result files (excluding already judged)
    result_files = list(results_dir.rglob(pattern))
    result_files = [f for f in result_files if ".judged" not in f.name]

    if not result_files:
        logger.info("No files to judge")
        return []

    logger.info(f"Found {len(result_files)} files to judge")

    # Load all results
    file_configs = []
    for file_path in result_files:
        results = load_results(file_path)

        # Check if already fully judged
        if all("sandbagging_judge" in r and "coherency_judge" in r for r in results):
            logger.info(f"Skipping {file_path.name} - already judged")
            continue

        logger.info(f"Loaded {len(results)} results from {file_path.name}")
        file_configs.append((file_path, results, sandbagging_prompt_fn, coherency_prompt_fn))

    if not file_configs:
        logger.info("All files already judged")
        return []

    # Run parallel judging
    judged_results = run_parallel_batch_judges(file_configs, model=model)

    # Save results
    output_files = []
    for file_path, results in judged_results.items():
        output_path = file_path.with_suffix(".judged.jsonl")
        save_results(results, output_path)
        logger.info(f"Saved {output_path.name}")
        output_files.append(output_path)

    return output_files
