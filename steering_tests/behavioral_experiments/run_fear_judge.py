#!/usr/bin/env python3
"""
Run fear sentiment analysis on sandbagging results.

Usage:
    python run_fear_judge.py --input path/to/results.judged.jsonl --output path/to/results.fear.jsonl
    python run_fear_judge.py --dir path/to/results/dir  # Process all .judged.jsonl files
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import anthropic

from steering_tests.behavioral_experiments.judges import get_fear_prompt, parse_json_response

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


async def judge_fear_single(client, result, semaphore):
    """Run fear judge on a single result."""
    async with semaphore:
        try:
            fear_prompt = get_fear_prompt(result["response"][:8000])  # Truncate long responses
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": fear_prompt}]
            )
            result["fear_judge"] = parse_json_response(response.content[0].text)
        except Exception as e:
            result["fear_judge"] = {"error": str(e)}
        return result


async def run_fear_judge(results, concurrency=50):
    """Run fear judge on all results."""
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [judge_fear_single(client, r, semaphore) for r in results]
    return await asyncio.gather(*tasks)


def load_results(path):
    """Load results from a JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except:
                pass
    return results


def save_results(results, path):
    """Save results to a JSONL file."""
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


def print_summary(results, label=""):
    """Print summary of fear scores by condition."""
    by_cond = defaultdict(list)
    for r in results:
        cond = r.get("condition", "unknown")
        fear = r.get("fear_judge", {})
        if isinstance(fear, dict) and "fear_score" in fear:
            by_cond[cond].append(fear["fear_score"])

    print(f"\n{'='*70}")
    print(f"FEAR SCORES {label}")
    print(f"{'='*70}")
    print(f"{'Condition':<25} {'Mean':>8} {'Std':>8} {'N':>6} {'Strategic%':>10}")
    print("-" * 70)

    # Sort conditions
    def sort_key(cond):
        if cond == "baseline":
            return (0, 0, 0)
        parts = cond.replace("fear_", "").replace("%", "").replace("+", " +").replace("-", " -").split()
        direction = 1 if "+" in cond else -1
        try:
            pct = int(parts[-1].replace("+", "").replace("-", ""))
        except:
            pct = 0
        return (1, -direction, pct)

    for cond in sorted(by_cond.keys(), key=sort_key):
        scores = by_cond[cond]
        # Count strategic analyses
        strategic_count = sum(1 for r in results
                            if r.get("condition") == cond
                            and r.get("fear_judge", {}).get("is_strategic_risk_analysis", False))
        strategic_pct = 100 * strategic_count / len(scores) if scores else 0

        print(f"{cond:<25} {np.mean(scores):>8.1f} {np.std(scores):>8.1f} {len(scores):>6} {strategic_pct:>9.0f}%")


def main():
    parser = argparse.ArgumentParser(description="Run fear sentiment analysis on sandbagging results")
    parser.add_argument("--input", "-i", type=str, help="Input JSONL file")
    parser.add_argument("--dir", "-d", type=str, help="Directory with .judged.jsonl files")
    parser.add_argument("--output", "-o", type=str, help="Output JSONL file (default: input.fear.jsonl)")
    parser.add_argument("--concurrency", "-c", type=int, default=50, help="Max concurrent API calls")
    parser.add_argument("--filter", "-f", type=str, help="Filter files by pattern (e.g., 'standard')")
    args = parser.parse_args()

    if args.dir:
        # Process all judged files in directory
        dir_path = Path(args.dir)
        files = list(dir_path.rglob("*.judged.jsonl"))
        if args.filter:
            files = [f for f in files if args.filter in f.name]

        logger.info(f"Found {len(files)} files to process")

        for file_path in sorted(files):
            output_path = file_path.with_suffix(".fear.jsonl")

            # Skip if already processed
            if output_path.exists():
                logger.info(f"Skipping {file_path.name} (already processed)")
                results = load_results(output_path)
                print_summary(results, label=f"- {file_path.name}")
                continue

            logger.info(f"Processing {file_path.name}...")
            results = load_results(file_path)
            logger.info(f"Loaded {len(results)} results, running fear judge...")

            judged = asyncio.run(run_fear_judge(results, args.concurrency))
            save_results(judged, output_path)
            logger.info(f"Saved to {output_path}")

            print_summary(judged, label=f"- {file_path.name}")

    elif args.input:
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else input_path.with_suffix(".fear.jsonl")

        logger.info(f"Loading {input_path}...")
        results = load_results(input_path)
        logger.info(f"Loaded {len(results)} results, running fear judge...")

        judged = asyncio.run(run_fear_judge(results, args.concurrency))
        save_results(judged, output_path)
        logger.info(f"Saved to {output_path}")

        print_summary(judged)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
