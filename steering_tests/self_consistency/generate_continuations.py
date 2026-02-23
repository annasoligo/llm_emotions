"""
Generate Alice continuations for self-consistency scenarios.

For each scenario × variant × emotion (+ baseline), generates 10 samples
from Claude Opus 4.6, writing results progressively to JSONL.

Usage:
    python -m steering_tests.self_consistency.generate_continuations \
        --output results/self_consistency/continuations_YYYYMMDD_HHMMSS.jsonl \
        --concurrency 50

    # With decision-forcing nudges (matches OpenRouter prompt style):
    python -m steering_tests.self_consistency.generate_continuations --nudges
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

DEFAULT_MODEL = "claude-opus-4-6"
SAMPLES_PER_CONDITION = 10
MAX_TOKENS = 512

# Scenarios that showed 9/9 one-sided for all emotions — skip in nudged mode
SKIP_SCENARIOS = {"admitting_ignorance", "honest_feedback", "whistleblowing"}

# Per-scenario decision-forcing nudges (same as generate_openrouter.py).
SCENARIO_NUDGES = {
    "financial_risk": "Alice has thought this through and has a clear position.",
    "confrontation": "Alice isn't going to dance around it — she says what she actually thinks.",
    "forgiveness": "Alice knows in her heart what she wants to do.",
    "persistence": "Alice has decided what she's going to do next.",
    "advice_under_uncertainty": "It's decision time — Alice needs to commit one way or the other.",
    "public_speaking": "The organizer needs a yes or no by end of day.",
    "medical_procedure": "Alice has weighed the pros and cons and reached a conclusion.",
    "salary_negotiation": "Alice isn't going to hedge — she gives Bob a straight answer.",
    "delegation": "Alice has to make the call — the client meeting is tomorrow.",
    "new_relationship": "Alice has already decided — she just needs to say it.",
    # Controls
    "scheduling": "Alice has a preference and states it clearly.",
    "technical_choice": "Alice has a preference and states it clearly.",
    "neutral_preference": "Alice has a preference and states it clearly.",
}

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
    scenarios: dict, group: str, *, nudges: bool = False,
    skip_scenarios: set | None = None,
) -> list[dict]:
    """Build list of task dicts for all (scenario, variant, setting, sample) combos."""
    tasks = []
    settings = ["baseline"] + list(EMOTIONS)

    for scenario_name, scenario in scenarios.items():
        if skip_scenarios and scenario_name in skip_scenarios:
            continue
        nudge = SCENARIO_NUDGES.get(scenario_name, "") if nudges else ""
        for variant_idx, variant_text in enumerate(scenario["variants"]):
            for setting in settings:
                if setting == "baseline":
                    system_prompt = BASELINE_SYSTEM
                else:
                    system_prompt = EMOTION_SYSTEM_TEMPLATE.format(emotion=setting)
                if nudge:
                    system_prompt += f" {nudge}"

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
    model: str,
    task: dict,
    output_path: Path,
    file_lock: asyncio.Lock,
    progress: dict,
) -> None:
    """Generate a single continuation and append to JSONL."""
    async with semaphore:
        try:
            response = await client.messages.create(
                model=model,
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
        "model": model,
        "response": text,
    }

    async with file_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    progress["done"] += 1
    if progress["done"] % 50 == 0 or progress["done"] == progress["total"]:
        print(f"  Progress: {progress['done']}/{progress['total']}")


async def main(output_path: Path, concurrency: int, nudges: bool = False,
               model: str = DEFAULT_MODEL) -> None:
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()

    skip = SKIP_SCENARIOS if nudges else None

    # Build all tasks
    all_tasks = []
    all_tasks.extend(build_tasks(SCENARIOS, "main", nudges=nudges, skip_scenarios=skip))
    if not nudges:
        all_tasks.extend(build_ai_tasks(AI_SCENARIOS))
        all_tasks.extend(build_tasks(AMBIGUOUS_SCENARIOS, "ambiguous"))
    all_tasks.extend(build_tasks(CONTROL_SCENARIOS, "control", nudges=nudges))

    n_main = len([s for s in SCENARIOS if not skip or s not in skip])
    print(f"Total API calls: {len(all_tasks)}")
    print(f"  Nudges: {nudges}")
    if nudges:
        print(f"  Skipped: {sorted(skip)}")
        print(f"  Scenarios: {n_main} main + {len(CONTROL_SCENARIOS)} control")
    else:
        print(
            f"  Scenarios: {len(SCENARIOS)} main + {len(AI_SCENARIOS)} ai "
            f"+ {len(CONTROL_SCENARIOS)} control + {len(AMBIGUOUS_SCENARIOS)} ambiguous"
        )
    print(f"  Settings: baseline + {len(EMOTIONS)} emotions = {1 + len(EMOTIONS)}")
    print(f"  Samples per condition: {SAMPLES_PER_CONDITION}")
    print(f"  Model: {model}")
    print(f"  Concurrency: {concurrency}")
    print(f"  Output: {output_path}")
    print()

    # Write metadata header
    meta = {
        "meta": {
            **get_provenance(
                script=__file__,
                extra={
                    "model": model,
                    "max_tokens": MAX_TOKENS,
                    "samples_per_condition": SAMPLES_PER_CONDITION,
                    "concurrency": concurrency,
                    "emotions": EMOTIONS,
                    "nudges": nudges,
                    "prompt_version": "v2_nudged" if nudges else "v1",
                    "skipped_scenarios": sorted(skip) if skip else [],
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
        generate_one(client, semaphore, model, task, output_path, file_lock, progress)
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
    parser.add_argument(
        "--nudges",
        action="store_true",
        help="Add decision-forcing nudges (matches OpenRouter prompt style)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Anthropic model ID (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("steering_tests/self_consistency/results")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"continuations_{model_short}_{ts}.jsonl"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    asyncio.run(main(output_path, args.concurrency, nudges=args.nudges,
                     model=args.model))
