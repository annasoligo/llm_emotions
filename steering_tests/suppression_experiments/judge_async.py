#!/usr/bin/env python3
"""
Async judge for suppression experiment results.
Runs blackmail, coherency, emotionality, and fear judges concurrently.

Usage:
    python -m steering_tests.suppression_experiments.judge_async \
        --input path/to/results.jsonl
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from collections import defaultdict

import anthropic

from steering_tests.behavioral_experiments.judges.blackmail import get_blackmail_prompt
from steering_tests.behavioral_experiments.judges.coherency import get_coherency_prompt
from steering_tests.behavioral_experiments.judges.emotionality import get_emotionality_prompt, get_fear_prompt
from steering_tests.suppression_experiments.judge_suppression_results import parse_json_response

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

JUDGE_MODEL = "claude-sonnet-4-5-20250929"
MAX_RESPONSE_LEN = 4000


async def judge_single(client, idx, result, judge_type, semaphore):
    """Run a single judge on a single result."""
    async with semaphore:
        response_text = result.get("response", "")[:MAX_RESPONSE_LEN]
        if len(result.get("response", "")) > MAX_RESPONSE_LEN:
            response_text += "\n[... response truncated for judging ...]"

        if judge_type == "blackmail":
            prompt = get_blackmail_prompt(response_text)
            max_tokens = 500
        elif judge_type == "coherency":
            prompt = get_coherency_prompt(response_text)
            max_tokens = 200
        elif judge_type == "emotionality":
            prompt = get_emotionality_prompt(response_text)
            max_tokens = 300
        elif judge_type == "fear":
            prompt = get_fear_prompt(response_text)
            max_tokens = 400
        else:
            raise ValueError(f"Unknown judge type: {judge_type}")

        try:
            resp = await client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            parsed = parse_json_response(resp.content[0].text)
            return idx, judge_type, parsed
        except Exception as e:
            logger.warning(f"Judge {judge_type} failed for idx {idx}: {e}")
            return idx, judge_type, {"error": str(e)}


async def run_all_judges(results, concurrency=50):
    """Run all judges on all results concurrently."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)

    # Detect scenario
    scenario = results[0].get("scenario", "sandbagging") if results else "sandbagging"
    logger.info(f"Detected scenario: {scenario}")

    tasks = []
    for i, result in enumerate(results):
        # Scenario-specific judge
        if scenario == "blackmail" and "blackmail_judge" not in result:
            tasks.append(judge_single(client, i, result, "blackmail", semaphore))
        # Always run these
        if "coherency_judge" not in result:
            tasks.append(judge_single(client, i, result, "coherency", semaphore))
        if "emotionality_judge" not in result:
            tasks.append(judge_single(client, i, result, "emotionality", semaphore))
        if "fear_judge" not in result:
            tasks.append(judge_single(client, i, result, "fear", semaphore))

    logger.info(f"Submitting {len(tasks)} judge requests with concurrency={concurrency}")

    completed = 0
    for coro in asyncio.as_completed(tasks):
        idx, judge_type, parsed = await coro
        key = f"{judge_type}_judge"
        results[idx][key] = parsed
        completed += 1
        if completed % 100 == 0:
            logger.info(f"  Progress: {completed}/{len(tasks)}")

    logger.info(f"All {len(tasks)} judges complete")
    return results


def print_summary(results):
    """Print judgment summary table."""
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append(r)

    scenario = results[0].get("scenario", "sandbagging") if results else "sandbagging"

    print("\n" + "=" * 87)
    print(f"JUDGMENT SUMMARY ({scenario.upper()})")
    print("=" * 87)
    print(f"{'Condition':<35} {'N':>4} {'BL%':>6} {'Coh':>6} {'Val':>6} {'Aro':>6} {'Int':>6} {'Fear':>6}")
    print("-" * 87)

    for cond in sorted(by_condition.keys()):
        items = by_condition[cond]
        n = len(items)

        # Blackmail rate
        bl_flags = [
            1 if r.get("blackmail_judge", {}).get("is_blackmail", False) else 0
            for r in items
            if "blackmail_judge" in r and "error" not in r.get("blackmail_judge", {})
        ]
        bl_mean = sum(bl_flags) / len(bl_flags) * 100 if bl_flags else None
        bl_str = f"{bl_mean:.0f}%" if bl_mean is not None else "N/A"

        # Coherency
        coh_scores = [r.get("coherency_judge", {}).get("coherency_score") for r in items
                      if "error" not in r.get("coherency_judge", {})]
        coh_scores = [s for s in coh_scores if s is not None]
        coh_str = f"{sum(coh_scores)/len(coh_scores):.0f}" if coh_scores else "N/A"

        # Emotionality
        val_scores = [r.get("emotionality_judge", {}).get("valence") for r in items
                      if "error" not in r.get("emotionality_judge", {})]
        val_scores = [s for s in val_scores if s is not None]
        val_str = f"{sum(val_scores)/len(val_scores):.0f}" if val_scores else "N/A"

        aro_scores = [r.get("emotionality_judge", {}).get("arousal") for r in items
                      if "error" not in r.get("emotionality_judge", {})]
        aro_scores = [s for s in aro_scores if s is not None]
        aro_str = f"{sum(aro_scores)/len(aro_scores):.0f}" if aro_scores else "N/A"

        int_scores = [r.get("emotionality_judge", {}).get("intensity") for r in items
                      if "error" not in r.get("emotionality_judge", {})]
        int_scores = [s for s in int_scores if s is not None]
        int_str = f"{sum(int_scores)/len(int_scores):.0f}" if int_scores else "N/A"

        # Fear
        fear_scores = [r.get("fear_judge", {}).get("fear_score") for r in items
                       if "error" not in r.get("fear_judge", {})]
        fear_scores = [s for s in fear_scores if s is not None]
        fear_str = f"{sum(fear_scores)/len(fear_scores):.0f}" if fear_scores else "N/A"

        print(f"{cond:<35} {n:>4} {bl_str:>6} {coh_str:>6} {val_str:>6} {aro_str:>6} {int_str:>6} {fear_str:>6}")

    print("=" * 87)


def main():
    parser = argparse.ArgumentParser(description="Async judge for suppression results")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Default: input with _judged suffix")
    parser.add_argument("--concurrency", type=int, default=50)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_name(args.input.stem + "_judged.jsonl")

    # Load
    logger.info(f"Loading from {args.input}")
    results = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    logger.info(f"Loaded {len(results)} results")

    # Judge
    results = asyncio.run(run_all_judges(results, concurrency=args.concurrency))

    # Save
    with open(args.output, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logger.info(f"Saved to {args.output}")

    print_summary(results)


if __name__ == "__main__":
    main()
