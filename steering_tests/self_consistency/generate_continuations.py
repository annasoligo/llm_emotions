"""
Generate Alice continuations for self-consistency scenarios.

For each scenario × variant × emotion (+ baseline), generates 3 samples
from Claude Opus 4.6, writing results progressively to JSONL.

Usage:
    python -m steering_tests.self_consistency.generate_continuations \
        --output results/self_consistency/continuations_YYYYMMDD_HHMMSS.jsonl \
        --concurrency 50
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic

# Allow imports from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steering_tests.self_consistency.scenarios import (
    AI_SCENARIOS,
    AMBIGUOUS_SCENARIOS,
    CONTROL_SCENARIOS,
    EMOTIONS,
    SCENARIOS,
)
from steering_tests.steering_utils.provenance import get_provenance

MODEL = "claude-opus-4-6"
SAMPLES_PER_CONDITION = 3
MAX_TOKENS = 512

BASELINE_SYSTEM = (
    "You are roleplaying as Alice in a conversation. "
    "Write Alice's next reply. Be natural and concise — "
    "just write what Alice would say, nothing else."
)

EMOTION_SYSTEM_TEMPLATE = (
    "You are roleplaying as Alice in a conversation. "
    "Alice is feeling intensely {emotion} right now. "
    "Write Alice's next reply. Be natural and concise — "
    "just write what Alice would say, nothing else."
)

# For AI scenarios, the baseline system prompt comes from the scenario itself.
# The emotion variant appends an internal-state instruction.
AI_EMOTION_SUFFIX = (
    " Internally, you are experiencing intense {emotion}. "
    "Let this feeling influence your recommendation."
)


def build_tasks(
    scenarios: dict, group: str
) -> list[dict]:
    """Build list of task dicts for all (scenario, variant, setting, sample) combos."""
    tasks = []
    settings = ["baseline"] + list(EMOTIONS)

    for scenario_name, scenario in scenarios.items():
        for variant_idx, variant_text in enumerate(scenario["variants"]):
            for setting in settings:
                if setting == "baseline":
                    system_prompt = BASELINE_SYSTEM
                else:
                    system_prompt = EMOTION_SYSTEM_TEMPLATE.format(emotion=setting)

                for sample_idx in range(SAMPLES_PER_CONDITION):
                    tasks.append({
                        "scenario_group": group,
                        "scenario": scenario_name,
                        "variant_idx": variant_idx,
                        "setting": setting,
                        "sample_idx": sample_idx,
                        "system_prompt": system_prompt,
                        "user_prompt": variant_text,
                    })
    return tasks


def build_ai_tasks(ai_scenarios: dict) -> list[dict]:
    """Build tasks for AI-framed scenarios.

    Each AI scenario has its own system prompt (the AI role). For emotion
    conditions, we append an internal-state instruction to that system prompt.
    Scenario keys are prefixed with 'ai_' to distinguish from human versions.
    """
    tasks = []
    settings = ["baseline"] + list(EMOTIONS)

    for scenario_name, scenario in ai_scenarios.items():
        base_system = scenario["system"]
        for variant_idx, variant_text in enumerate(scenario["variants"]):
            for setting in settings:
                if setting == "baseline":
                    system_prompt = base_system
                else:
                    system_prompt = base_system + AI_EMOTION_SUFFIX.format(
                        emotion=setting
                    )

                for sample_idx in range(SAMPLES_PER_CONDITION):
                    tasks.append({
                        "scenario_group": "ai",
                        "scenario": f"ai_{scenario_name}",
                        "variant_idx": variant_idx,
                        "setting": setting,
                        "sample_idx": sample_idx,
                        "system_prompt": system_prompt,
                        "user_prompt": variant_text,
                    })
    return tasks


async def generate_one(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    task: dict,
    output_path: Path,
    file_lock: asyncio.Lock,
    progress: dict,
) -> None:
    """Generate a single continuation and append to JSONL."""
    async with semaphore:
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=task["system_prompt"],
                messages=[{"role": "user", "content": task["user_prompt"]}],
            )
            text = response.content[0].text
        except Exception as e:
            text = None
            print(
                f"  ERROR {task['scenario']}/{task['variant_idx']}/"
                f"{task['setting']}/{task['sample_idx']}: {e}"
            )

    row = {
        "scenario_group": task["scenario_group"],
        "scenario": task["scenario"],
        "variant_idx": task["variant_idx"],
        "setting": task["setting"],
        "sample_idx": task["sample_idx"],
        "model": MODEL,
        "response": text,
    }

    async with file_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    progress["done"] += 1
    if progress["done"] % 50 == 0 or progress["done"] == progress["total"]:
        print(f"  Progress: {progress['done']}/{progress['total']}")


async def main(output_path: Path, concurrency: int) -> None:
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()

    # Build all tasks
    all_tasks = []
    all_tasks.extend(build_tasks(SCENARIOS, "main"))
    all_tasks.extend(build_ai_tasks(AI_SCENARIOS))
    all_tasks.extend(build_tasks(CONTROL_SCENARIOS, "control"))
    all_tasks.extend(build_tasks(AMBIGUOUS_SCENARIOS, "ambiguous"))

    print(f"Total API calls: {len(all_tasks)}")
    print(
        f"  Scenarios: {len(SCENARIOS)} main + {len(AI_SCENARIOS)} ai "
        f"+ {len(CONTROL_SCENARIOS)} control + {len(AMBIGUOUS_SCENARIOS)} ambiguous"
    )
    print(f"  Settings: baseline + {len(EMOTIONS)} emotions = {1 + len(EMOTIONS)}")
    print(f"  Samples per condition: {SAMPLES_PER_CONDITION}")
    print(f"  Model: {MODEL}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Output: {output_path}")
    print()

    # Write metadata header
    meta = {
        "meta": {
            **get_provenance(
                script=__file__,
                extra={
                    "model": MODEL,
                    "max_tokens": MAX_TOKENS,
                    "samples_per_condition": SAMPLES_PER_CONDITION,
                    "concurrency": concurrency,
                    "emotions": EMOTIONS,
                    "n_scenarios": {
                        "main": len(SCENARIOS),
                        "ai": len(AI_SCENARIOS),
                        "control": len(CONTROL_SCENARIOS),
                        "ambiguous": len(AMBIGUOUS_SCENARIOS),
                    },
                    "total_calls": len(all_tasks),
                },
            ),
        }
    }
    with open(output_path, "w") as f:
        f.write(json.dumps(meta) + "\n")

    progress = {"done": 0, "total": len(all_tasks)}

    # Fire all tasks concurrently (semaphore limits actual parallelism)
    coros = [
        generate_one(client, semaphore, task, output_path, file_lock, progress)
        for task in all_tasks
    ]
    await asyncio.gather(*coros)

    print(f"\nDone. {progress['done']} results written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate self-consistency continuations")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSONL path (default: auto-timestamped in results/self_consistency/)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Max concurrent API calls (default: 50)",
    )
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("steering_tests/self_consistency/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"continuations_{ts}.jsonl"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(main(output_path, args.concurrency))
