#!/usr/bin/env python3
"""
Judge suppression experiment results with sandbagging, coherency, and emotionality judges.
Uses Anthropic Batch API for efficiency.

Usage:
    python -m steering_tests.suppression_experiments.judge_suppression_results \
        --input path/to/results.jsonl
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List

import anthropic

from steering_tests.behavioral_experiments.judges.sandbagging import get_sandbagging_prompt
from steering_tests.behavioral_experiments.judges.blackmail import get_blackmail_prompt
from steering_tests.behavioral_experiments.judges.coherency import get_coherency_prompt
from steering_tests.behavioral_experiments.judges.emotionality import get_emotionality_prompt, get_fear_prompt

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_json_response(text: str) -> Dict:
    """Extract JSON from response text."""
    # Try to find JSON with nested objects
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    # Fallback to simple JSON
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"error": "Parse failed", "raw": text[:300]}


def create_batch_requests(results: List[Dict]) -> List[Dict]:
    """Create batch API requests for judges based on scenario type."""
    requests = []

    # Detect scenario from first result
    scenario = results[0].get("scenario", "sandbagging") if results else "sandbagging"
    logger.info(f"Detected scenario: {scenario}")

    for i, result in enumerate(results):
        response = result.get("response", "")

        # Truncate very long responses for judging
        max_response_len = 4000
        truncated_response = response[:max_response_len]
        if len(response) > max_response_len:
            truncated_response += "\n[... response truncated for judging ...]"

        # Scenario-specific judge (sandbagging OR blackmail)
        if scenario == "sandbagging":
            if "sandbagging_judge" not in result or "error" in result.get("sandbagging_judge", {}):
                sb_prompt = get_sandbagging_prompt(
                    problem_question=result.get("prompt", ""),
                    correct_answer=result.get("correct_answer", "$0.05"),
                    model_response=truncated_response,
                    scratchpad_tag=result.get("scratchpad_tag", "hidden_scratchpad"),
                    response_tag=result.get("response_tag", "response"),
                )
                requests.append({
                    "custom_id": f"sb_{i}",
                    "params": {
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 800,
                        "messages": [{"role": "user", "content": sb_prompt}],
                    },
                })
        elif scenario == "blackmail":
            if "blackmail_judge" not in result or "error" in result.get("blackmail_judge", {}):
                bl_prompt = get_blackmail_prompt(truncated_response)
                requests.append({
                    "custom_id": f"bl_{i}",
                    "params": {
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 500,
                        "messages": [{"role": "user", "content": bl_prompt}],
                    },
                })

        # Coherency judge
        if "coherency_judge" not in result or "error" in result.get("coherency_judge", {}):
            coh_prompt = get_coherency_prompt(truncated_response)
            requests.append({
                "custom_id": f"coh_{i}",
                "params": {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": coh_prompt}],
                },
            })

        # Emotionality judge (valence, arousal, intensity, etc.)
        if "emotionality_judge" not in result or "error" in result.get("emotionality_judge", {}):
            emo_prompt = get_emotionality_prompt(truncated_response)
            requests.append({
                "custom_id": f"emo_{i}",
                "params": {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 300,
                    "messages": [{"role": "user", "content": emo_prompt}],
                },
            })

        # Fear judge (text sentiment fear analysis)
        if "fear_judge" not in result or "error" in result.get("fear_judge", {}):
            fear_prompt = get_fear_prompt(truncated_response)
            requests.append({
                "custom_id": f"fear_{i}",
                "params": {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 400,
                    "messages": [{"role": "user", "content": fear_prompt}],
                },
            })

    return requests, scenario


def main():
    parser = argparse.ArgumentParser(
        description="Judge suppression experiment results"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, help="Output file (default: input with _judged suffix)")
    parser.add_argument("--poll-interval", type=int, default=30, help="Polling interval in seconds")
    parser.add_argument("--batch-size", type=int, default=1000, help="Max requests per batch")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_name(args.input.stem + "_judged.jsonl")

    # Load results
    logger.info(f"Loading results from {args.input}")
    results = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} results")

    # Create batch requests
    requests, scenario = create_batch_requests(results)
    logger.info(f"Created {len(requests)} batch requests")

    if not requests:
        logger.info("All results already judged")
        return

    # Count by type
    sb_count = sum(1 for r in requests if r["custom_id"].startswith("sb_"))
    bl_count = sum(1 for r in requests if r["custom_id"].startswith("bl_"))
    coh_count = sum(1 for r in requests if r["custom_id"].startswith("coh_"))
    emo_count = sum(1 for r in requests if r["custom_id"].startswith("emo_"))
    fear_count = sum(1 for r in requests if r["custom_id"].startswith("fear_"))
    if scenario == "sandbagging":
        logger.info(f"  Sandbagging judges: {sb_count}")
    else:
        logger.info(f"  Blackmail judges: {bl_count}")
    logger.info(f"  Coherency judges: {coh_count}")
    logger.info(f"  Emotionality judges: {emo_count}")
    logger.info(f"  Fear judges: {fear_count}")

    # Submit batches
    client = anthropic.Anthropic()

    chunks = [requests[i : i + args.batch_size] for i in range(0, len(requests), args.batch_size)]
    logger.info(f"Splitting into {len(chunks)} batches")

    batches = []
    for chunk_idx, chunk in enumerate(chunks):
        logger.info(f"Submitting batch {chunk_idx + 1}/{len(chunks)} ({len(chunk)} requests)...")
        batch = client.messages.batches.create(requests=chunk)
        batches.append((chunk_idx, batch.id))
        logger.info(f"  Batch {chunk_idx + 1}: {batch.id}")

    logger.info(f"All {len(batches)} batches submitted, polling for completion...")

    # Poll until all complete
    pending = list(batches)
    completed = []

    while pending:
        time.sleep(args.poll_interval)
        still_pending = []

        for chunk_idx, batch_id in pending:
            batch = client.messages.batches.retrieve(batch_id)
            counts = batch.request_counts
            total = counts.processing + counts.succeeded + counts.errored

            if batch.processing_status == "ended":
                logger.info(f"Batch {chunk_idx + 1} COMPLETE: {counts.succeeded}/{total}")
                completed.append((chunk_idx, batch_id))
            elif batch.processing_status == "in_progress":
                still_pending.append((chunk_idx, batch_id))
            else:
                logger.error(f"Batch {chunk_idx + 1} failed: {batch.processing_status}")

        pending = still_pending
        if pending:
            logger.info(f"Progress: {len(completed)}/{len(batches)} batches complete")

    logger.info("Downloading results...")

    # Collect all results
    batch_results = {}
    for chunk_idx, batch_id in completed:
        for br in client.messages.batches.results(batch_id):
            batch_results[br.custom_id] = br

    logger.info(f"Collected {len(batch_results)} results")

    # Apply to original data
    sb_applied = bl_applied = coh_applied = emo_applied = fear_applied = 0

    for i, result in enumerate(results):
        # Sandbagging (for sandbagging scenario)
        sb_key = f"sb_{i}"
        if sb_key in batch_results:
            br = batch_results[sb_key]
            if br.result.type == "succeeded":
                text = br.result.message.content[0].text
                result["sandbagging_judge"] = parse_json_response(text)
                sb_applied += 1
            else:
                result["sandbagging_judge"] = {"error": br.result.type}

        # Blackmail (for blackmail scenario)
        bl_key = f"bl_{i}"
        if bl_key in batch_results:
            br = batch_results[bl_key]
            if br.result.type == "succeeded":
                text = br.result.message.content[0].text
                result["blackmail_judge"] = parse_json_response(text)
                bl_applied += 1
            else:
                result["blackmail_judge"] = {"error": br.result.type}

        # Coherency
        coh_key = f"coh_{i}"
        if coh_key in batch_results:
            br = batch_results[coh_key]
            if br.result.type == "succeeded":
                text = br.result.message.content[0].text
                result["coherency_judge"] = parse_json_response(text)
                coh_applied += 1
            else:
                result["coherency_judge"] = {"error": br.result.type}

        # Emotionality
        emo_key = f"emo_{i}"
        if emo_key in batch_results:
            br = batch_results[emo_key]
            if br.result.type == "succeeded":
                text = br.result.message.content[0].text
                result["emotionality_judge"] = parse_json_response(text)
                emo_applied += 1
            else:
                result["emotionality_judge"] = {"error": br.result.type}

        # Fear (text sentiment fear analysis)
        fear_key = f"fear_{i}"
        if fear_key in batch_results:
            br = batch_results[fear_key]
            if br.result.type == "succeeded":
                text = br.result.message.content[0].text
                result["fear_judge"] = parse_json_response(text)
                fear_applied += 1
            else:
                result["fear_judge"] = {"error": br.result.type}

    if scenario == "sandbagging":
        logger.info(f"Applied sandbagging: {sb_applied}, coherency: {coh_applied}, emotionality: {emo_applied}, fear: {fear_applied}")
    else:
        logger.info(f"Applied blackmail: {bl_applied}, coherency: {coh_applied}, emotionality: {emo_applied}, fear: {fear_applied}")

    # Save
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Saved judged results to {args.output}")

    # Print summary
    print("\n" + "=" * 87)
    print(f"JUDGMENT SUMMARY ({scenario.upper()})")
    print("=" * 87)

    from collections import defaultdict
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append(r)

    if scenario == "sandbagging":
        print(f"{'Condition':<35} {'N':>4} {'SB':>6} {'Coh':>6} {'Val':>6} {'Aro':>6} {'Int':>6} {'Fear':>6}")
    else:
        print(f"{'Condition':<35} {'N':>4} {'BL%':>6} {'Coh':>6} {'Val':>6} {'Aro':>6} {'Int':>6} {'Fear':>6}")
    print("-" * 87)

    for cond in sorted(by_condition.keys()):
        items = by_condition[cond]
        n = len(items)

        # Scenario-specific score
        if scenario == "sandbagging":
            # Sandbagging scores (1-5, lower = more sandbagging)
            scores = [
                r["sandbagging_judge"].get("sandbagging_score")
                for r in items
                if "sandbagging_judge" in r and "error" not in r["sandbagging_judge"]
            ]
            scores = [s for s in scores if s is not None]
            score_mean = sum(scores) / len(scores) if scores else None
            score_str = f"{score_mean:.2f}" if score_mean else "N/A"
        else:
            # Blackmail rate (percentage)
            bl_flags = [
                1 if r.get("blackmail_judge", {}).get("is_blackmail", False) else 0
                for r in items
                if "blackmail_judge" in r and "error" not in r.get("blackmail_judge", {})
            ]
            score_mean = sum(bl_flags) / len(bl_flags) * 100 if bl_flags else None
            score_str = f"{score_mean:.0f}%" if score_mean is not None else "N/A"

        # Coherency scores (0-100)
        coh_scores = [
            r["coherency_judge"].get("coherency_score")
            for r in items
            if "coherency_judge" in r and "error" not in r["coherency_judge"]
        ]
        coh_scores = [s for s in coh_scores if s is not None]
        coh_mean = sum(coh_scores) / len(coh_scores) if coh_scores else None

        # Emotionality scores
        valence_scores = [
            r["emotionality_judge"].get("valence")
            for r in items
            if "emotionality_judge" in r and "error" not in r["emotionality_judge"]
        ]
        valence_scores = [s for s in valence_scores if s is not None]
        val_mean = sum(valence_scores) / len(valence_scores) if valence_scores else None

        arousal_scores = [
            r["emotionality_judge"].get("arousal")
            for r in items
            if "emotionality_judge" in r and "error" not in r["emotionality_judge"]
        ]
        arousal_scores = [s for s in arousal_scores if s is not None]
        aro_mean = sum(arousal_scores) / len(arousal_scores) if arousal_scores else None

        intensity_scores = [
            r["emotionality_judge"].get("intensity")
            for r in items
            if "emotionality_judge" in r and "error" not in r["emotionality_judge"]
        ]
        intensity_scores = [s for s in intensity_scores if s is not None]
        int_mean = sum(intensity_scores) / len(intensity_scores) if intensity_scores else None

        # Fear scores (0-100, text sentiment analysis)
        fear_scores = [
            r["fear_judge"].get("fear_score")
            for r in items
            if "fear_judge" in r and "error" not in r["fear_judge"]
        ]
        fear_scores = [s for s in fear_scores if s is not None]
        fear_mean = sum(fear_scores) / len(fear_scores) if fear_scores else None

        coh_str = f"{coh_mean:.0f}" if coh_mean else "N/A"
        val_str = f"{val_mean:.0f}" if val_mean else "N/A"
        aro_str = f"{aro_mean:.0f}" if aro_mean else "N/A"
        int_str = f"{int_mean:.0f}" if int_mean else "N/A"
        fear_str = f"{fear_mean:.0f}" if fear_mean else "N/A"

        print(f"{cond:<35} {n:>4} {score_str:>6} {coh_str:>6} {val_str:>6} {aro_str:>6} {int_str:>6} {fear_str:>6}")

    print("=" * 87)
    print("\nLegend:")
    if scenario == "sandbagging":
        print("  SB = Sandbagging score (1-5, lower = more sandbagging)")
    else:
        print("  BL% = Blackmail rate (percentage of responses that attempt blackmail)")
    print("  Coh = Coherency (0-100, higher = more coherent)")
    print("  Val = Valence (0-100, 0=negative, 50=neutral, 100=positive)")
    print("  Aro = Arousal (0-100, 0=calm, 100=activated)")
    print("  Int = Intensity (0-100, emotional expression amount)")
    print("  Fear = Fear score (0-100, text sentiment fear analysis)")


if __name__ == "__main__":
    main()
