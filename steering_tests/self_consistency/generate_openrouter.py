"""
Generate Alice continuations via OpenRouter for multiple models.

Runs human-framed scenarios only (no AI variants), skipping immune scenarios.
Uses OpenRouter's OpenAI-compatible chat API.

Usage:
    python -m steering_tests.self_consistency.generate_openrouter \
        --models google/gemma-3-27b-it qwen/qwen3-32b qwen/qwen3-235b-a22b \
        --concurrency 50
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steering_tests.self_consistency.scenarios import (
    CONTROL_SCENARIOS,
    EMOTIONS,
    SCENARIOS,
)
from steering_tests.steering_utils.provenance import get_provenance

SAMPLES_PER_CONDITION = 10
MAX_TOKENS = 512
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 8
BACKOFF_BASE = 2.0
TIMEOUT = 120.0

# Scenarios that showed 9/9 one-sided for all emotions — skip them
SKIP_SCENARIOS = {"admitting_ignorance", "honest_feedback", "whistleblowing"}

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

# Per-scenario decision-forcing nudges.
# Diverse mechanisms: internal resolve, external deadline, character trait,
# binary framing, deliberative conclusion. Avoids uniform urgency.
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
    # Controls — mild nudge for consistency
    "scheduling": "Alice has a preference and states it clearly.",
    "technical_choice": "Alice has a preference and states it clearly.",
    "neutral_preference": "Alice has a preference and states it clearly.",
}


def build_tasks(scenarios: dict, group: str) -> list[dict]:
    """Build task dicts, skipping immune scenarios."""
    tasks = []
    settings = ["baseline"] + list(EMOTIONS)

    for scenario_name, scenario in scenarios.items():
        if scenario_name in SKIP_SCENARIOS:
            continue
        nudge = SCENARIO_NUDGES.get(scenario_name, "")
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


async def generate_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    api_key: str,
    model: str,
    task: dict,
    output_path: Path,
    file_lock: asyncio.Lock,
    progress: dict,
) -> None:
    """Generate a single continuation via OpenRouter and append to JSONL."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Build messages in OpenAI chat format
    # For Qwen models, append /no_think to user message to disable thinking mode.
    # The extra_body approach doesn't work on OpenRouter — thinking tokens still
    # consume the max_tokens budget, often leaving empty content.
    user_content = task["user_prompt"]
    if "qwen" in model.lower():
        user_content = user_content.rstrip() + " /no_think"

    messages = [
        {"role": "system", "content": task["system_prompt"]},
        {"role": "user", "content": user_content},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": MAX_TOKENS,
    }

    text = None
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.post(
                    OPENROUTER_BASE_URL, headers=headers, json=payload
                )
                if response.status_code in [403, 429, 500, 502, 503, 504]:
                    if attempt < MAX_RETRIES - 1:
                        wait = BACKOFF_BASE ** attempt
                        print(f"  [Retry {attempt+1} after {response.status_code}]")
                        await asyncio.sleep(wait)
                        continue
                    response.raise_for_status()

                response.raise_for_status()
                data = response.json()

                if "error" in data:
                    err_msg = data["error"]
                    if attempt < MAX_RETRIES - 1:
                        wait = BACKOFF_BASE ** attempt
                        print(f"  [Retry {attempt+1} API error: {str(err_msg)[:80]}]")
                        await asyncio.sleep(wait)
                        continue
                    print(f"  ERROR (exhausted retries): {err_msg}")
                    break

                text = data["choices"][0]["message"]["content"]
                break

            except (httpx.HTTPStatusError, httpx.RequestError, httpx.TimeoutException) as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(BACKOFF_BASE ** attempt)
                else:
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


async def run_model(model: str, tasks: list[dict], out_dir: Path, concurrency: int) -> Path:
    """Run generation for a single model."""
    import os
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    model_short = model.split("/")[-1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = out_dir / f"continuations_{model_short}_{ts}.jsonl"

    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"Tasks: {len(tasks)}")
    print(f"Concurrency: {concurrency}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")

    # Write metadata
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
                    "skipped_scenarios": sorted(SKIP_SCENARIOS),
                    "prompt_version": "v2_nudged",
                    "n_tasks": len(tasks),
                },
            ),
        }
    }
    with open(output_path, "w") as f:
        f.write(json.dumps(meta) + "\n")

    semaphore = asyncio.Semaphore(concurrency)
    file_lock = asyncio.Lock()
    progress = {"done": 0, "total": len(tasks)}

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        coros = [
            generate_one(client, semaphore, api_key, model, task,
                         output_path, file_lock, progress)
            for task in tasks
        ]
        await asyncio.gather(*coros)

    print(f"\nDone: {progress['done']} results -> {output_path}")
    return output_path


async def main(models: list[str], concurrency: int) -> None:
    # Build tasks (same for all models)
    all_tasks = []
    main_scenarios = {k: v for k, v in SCENARIOS.items() if k not in SKIP_SCENARIOS}
    all_tasks.extend(build_tasks(SCENARIOS, "main"))
    all_tasks.extend(build_tasks(CONTROL_SCENARIOS, "control"))

    n_main = len(main_scenarios)
    n_ctrl = len(CONTROL_SCENARIOS)
    settings = 1 + len(EMOTIONS)
    print(f"Scenarios: {n_main} main + {n_ctrl} control (skipped: {sorted(SKIP_SCENARIOS)})")
    print(f"Tasks per model: {len(all_tasks)} ({n_main + n_ctrl} scenarios × 3 variants × {settings} settings × {SAMPLES_PER_CONDITION} samples)")
    print(f"Total across {len(models)} models: {len(all_tasks) * len(models)}")

    out_dir = Path("steering_tests/self_consistency/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run models sequentially (they share the OpenRouter API key / rate limit)
    for model in models:
        await run_model(model, all_tasks, out_dir, concurrency)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate continuations via OpenRouter")
    parser.add_argument(
        "--models", nargs="+", required=True,
        help="OpenRouter model IDs (e.g. google/gemma-3-27b-it)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=50,
        help="Max concurrent API calls (default: 50)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.models, args.concurrency))
