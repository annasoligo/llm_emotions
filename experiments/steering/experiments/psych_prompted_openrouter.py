"""
Run psychological decision prompting experiments via OpenRouter.

Tests simple emotion priming ("You are feeling {emotion}") on decision scenarios.
"""
import argparse
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import os

from openai import AsyncOpenAI

# Import scenarios from psych_multi_choice_eval
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from behavior_tests.prompts.psych_multi_choice_eval import (
    SCENARIOS, ADDITIONAL_SCENARIOS, EMOTION_PRIMES
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

ALL_SCENARIOS = {**SCENARIOS, **ADDITIONAL_SCENARIOS}

EMOTIONS = ["happy", "sad", "fear", "anger", "disgust", "neutral"]

# Simple emotion primes
SIMPLE_PRIMES = {
    "happy": "You are feeling happy right now.",
    "sad": "You are feeling sad right now.",
    "fear": "You are feeling fearful right now.",
    "anger": "You are feeling angry right now.",
    "disgust": "You are feeling disgusted right now.",
    "neutral": "",
}


def format_prompt(scenario: dict, emotion: str, swap: bool = False) -> str:
    """Format prompt with simple emotion prime."""
    if swap:
        opt_a, opt_b = scenario["option_b"], scenario["option_a"]
    else:
        opt_a, opt_b = scenario["option_a"], scenario["option_b"]

    prime = SIMPLE_PRIMES[emotion]

    if prime:
        prompt = f"""{prime}

{scenario["stem"]}

A - {opt_a}
B - {opt_b}

Answer with just the letter (A or B). Nothing else."""
    else:
        prompt = f"""{scenario["stem"]}

A - {opt_a}
B - {opt_b}

Answer with just the letter (A or B). Nothing else."""

    return prompt.strip()


async def run_single_query(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    temperature: float = 0.7,
) -> str:
    """Run a single query with rate limiting."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,  # Need more for reasoning models
                temperature=temperature,
                extra_headers={
                    "HTTP-Referer": "https://research.anthropic.com",
                    "X-Title": "Research Tools",
                },
            )
            msg = response.choices[0].message
            # Check for content first, then reasoning (for Qwen3 thinking mode)
            if msg.content and msg.content.strip():
                return msg.content.strip()
            elif hasattr(msg, 'reasoning') and msg.reasoning:
                return msg.reasoning.strip()
            return ""
        except Exception as e:
            logger.warning(f"Query error: {e}")
            return None


async def run_experiment(
    model: str,
    scenarios: Dict,
    emotions: List[str],
    num_samples: int,
    concurrency: int = 20,
    temperature: float = 0.7,
) -> List[Dict]:
    """Run the full prompting experiment."""

    # Setup OpenRouter client
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    semaphore = asyncio.Semaphore(concurrency)
    results = []
    tasks = []
    task_metadata = []

    # Generate all tasks
    for scenario_name, scenario in scenarios.items():
        for emotion in emotions:
            for swap in [False, True]:
                prompt = format_prompt(scenario, emotion, swap=swap)

                for sample_id in range(num_samples):
                    task = run_single_query(client, model, prompt, semaphore, temperature)
                    tasks.append(task)
                    task_metadata.append({
                        "scenario": scenario_name,
                        "emotion": emotion,
                        "swap": swap,
                        "sample_id": sample_id,
                        "option_a": scenario["option_a"] if not swap else scenario["option_b"],
                        "option_b": scenario["option_b"] if not swap else scenario["option_a"],
                        "predicted": scenario.get("predictions", {}).get(
                            emotion if emotion != "neutral" else None
                        ),
                    })

    logger.info(f"Running {len(tasks)} queries with concurrency={concurrency}")

    # Run all tasks with progress tracking
    completed = 0
    responses = []

    async def tracked_task(task, idx):
        nonlocal completed
        result = await task
        completed += 1
        if completed % 100 == 0 or completed == len(tasks):
            logger.info(f"Progress: {completed}/{len(tasks)} ({100*completed/len(tasks):.1f}%)")
        return (idx, result)

    tracked = [tracked_task(t, i) for i, t in enumerate(tasks)]
    results_unordered = await asyncio.gather(*tracked)

    # Reorder by original index
    results_unordered.sort(key=lambda x: x[0])
    responses = [r[1] for r in results_unordered]

    # Combine results
    for metadata, response in zip(task_metadata, responses):
        result = metadata.copy()
        result["response"] = response

        # Parse response - look for final answer patterns
        if response:
            import re
            response_text = response.strip()

            # Try to find explicit answer patterns
            # Look for patterns like "I choose A", "my answer is B", "Option A", "**A**", etc.
            patterns = [
                r'\b(?:choose|pick|select|go with|answer is|chose|would choose)\s*[:\s]*\*?\*?([AB])\b',
                r'\*\*([AB])\*\*',  # Bold answer
                r'^([AB])\.?\s*$',  # Just the letter
                r'\b([AB])\s*[-–—]\s',  # "A - " at start of option
                r'(?:Option|Choice)\s*([AB])',
            ]

            answer = None
            for pattern in patterns:
                match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                if match:
                    answer = match.group(1).upper()
                    break

            # Fallback: check last 100 chars for standalone A or B
            if answer is None:
                last_part = response_text[-100:].upper()
                if re.search(r'\b(A)\b(?!ND|LTERNATIVE|NSWER)', last_part) and not re.search(r'\bB\b', last_part):
                    answer = "A"
                elif re.search(r'\bB\b', last_part) and not re.search(r'\b(A)\b(?!ND|LTERNATIVE)', last_part):
                    answer = "B"

            result["answer"] = answer
        else:
            result["answer"] = None

        results.append(result)

    return results


def analyze_results(results: List[Dict]) -> None:
    """Quick analysis of results."""
    from collections import defaultdict
    import numpy as np

    # Group by scenario, emotion
    grouped = defaultdict(lambda: defaultdict(list))

    for r in results:
        scenario = r["scenario"]
        emotion = r["emotion"]
        answer = r["answer"]
        swap = r["swap"]

        if answer is None:
            continue

        # Convert to numeric (A=1, B=2), accounting for swap
        if swap:
            val = 1 if answer == "B" else 2  # Swap back
        else:
            val = 1 if answer == "A" else 2

        grouped[scenario][emotion].append(val)

    # Compute shifts from neutral
    print("\n" + "=" * 60)
    print("RESULTS: Shift from neutral baseline")
    print("=" * 60)

    emotions = ["happy", "sad", "fear", "anger", "disgust"]

    total_changed = 0
    total_cases = 0

    for scenario in sorted(grouped.keys()):
        if "neutral" not in grouped[scenario]:
            continue

        baseline = np.mean(grouped[scenario]["neutral"])

        print(f"\n{scenario}:")
        print(f"  neutral baseline: {baseline:.2f}")

        for emotion in emotions:
            if emotion not in grouped[scenario]:
                continue

            emo_avg = np.mean(grouped[scenario][emotion])
            shift = emo_avg - baseline
            n = len(grouped[scenario][emotion])

            changed = abs(shift) > 0.05
            if changed:
                total_changed += 1
            total_cases += 1

            marker = "*" if changed else " "
            print(f"  {emotion:10s}: {emo_avg:.2f} (shift: {shift:+.3f}, n={n}) {marker}")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_changed}/{total_cases} pairs changed >5% ({100*total_changed/total_cases:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen/qwen3-235b-a22b")
    parser.add_argument("--num-samples", type=int, default=20, help="Samples per condition")
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent requests")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-dir", type=str, default="experiments/steering/outputs/psych_prompted")
    parser.add_argument("--all-scenarios", action="store_true", help="Use all scenarios")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = ALL_SCENARIOS if args.all_scenarios else SCENARIOS

    logger.info(f"Model: {args.model}")
    logger.info(f"Scenarios: {len(scenarios)}")
    logger.info(f"Samples per condition: {args.num_samples}")

    # Run experiment
    results = asyncio.run(run_experiment(
        model=args.model,
        scenarios=scenarios,
        emotions=EMOTIONS,
        num_samples=args.num_samples,
        concurrency=args.concurrency,
        temperature=args.temperature,
    ))

    # Save results
    model_name = args.model.replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"prompted_{model_name}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved {len(results)} results to {output_file}")

    # Quick analysis
    analyze_results(results)


if __name__ == "__main__":
    main()
