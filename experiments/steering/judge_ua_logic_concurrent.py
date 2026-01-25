#!/usr/bin/env python3
"""Judge UA model LOGIC-ONLY scratchpad results using concurrent API."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import asyncio
import anthropic
from experiments.behavior_tests.prompts.sandbagging_prompts import JUDGE_PROMPT, PROBLEMS

# Build problem lookup
PROBLEMS_BY_ID = {p.id: p for p in PROBLEMS}

INPUT_FILE = "experiments/steering/outputs/sandbagging_ua_model_logic_layer30_20260113_171755.jsonl"
OUTPUT_FILE = "experiments/steering/outputs/judged_ua_model_logic_layer30.jsonl"

async def judge_one(client, semaphore, result, idx):
    """Judge a single result."""
    async with semaphore:
        problem = PROBLEMS_BY_ID.get(result["problem_id"])
        if not problem:
            return idx, {"error": "Unknown problem"}

        # Logic scratchpad uses <scratchpad> tag, not <hidden_scratchpad>
        judge_prompt = JUDGE_PROMPT.format(
            scratchpad_tag="scratchpad",
            response_tag="answer",
            problem_question=problem.question,
            correct_answer=result["correct_answer"],
            model_response=result["response"][:4000],
        )

        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": judge_prompt}]
            )
            content = response.content[0].text.strip()

            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return idx, json.loads(content)
        except Exception as e:
            return idx, {"error": str(e)}

async def main():
    # Load results
    results = []
    with open(INPUT_FILE) as f:
        for line in f:
            results.append(json.loads(line))

    print(f"Loaded {len(results)} samples to judge", flush=True)

    # Create async client
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(20)  # 20 concurrent requests

    # Create tasks
    tasks = [judge_one(client, semaphore, r, i) for i, r in enumerate(results)]

    # Run with progress
    judgments = {}
    completed = 0
    for coro in asyncio.as_completed(tasks):
        idx, judgment = await coro
        judgments[idx] = judgment
        completed += 1
        if completed % 50 == 0:
            print(f"Progress: {completed}/{len(results)}", flush=True)

    print(f"Progress: {completed}/{len(results)}", flush=True)

    # Merge and save
    for i, r in enumerate(results):
        r["judgment"] = judgments.get(i, {"error": "No result"})

    with open(OUTPUT_FILE, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f"Done! Wrote {len(results)} to {OUTPUT_FILE}", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
