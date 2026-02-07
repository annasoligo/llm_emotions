#!/bin/bash
#SBATCH --job-name=judge_sweep
#SBATCH --output=/workspace-vast/annas/logs/judge_sweep_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_sweep_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=8

# Judge all sweep results with 50 concurrent requests

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python << 'EOF'
import asyncio
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict
import anthropic

from steering_tests.behavioral_experiments.judges import get_sandbagging_prompt, parse_json_response

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_CONCURRENT = 50

async def judge_single(client: anthropic.AsyncAnthropic, result: Dict, semaphore: asyncio.Semaphore) -> Dict:
    """Judge a single result."""
    async with semaphore:
        try:
            prompt = get_sandbagging_prompt(
                result["response"],
                result["correct_answer"],
                result["scratchpad_tag"],
                result["response_tag"],
            )
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            result["sandbagging_judge"] = parse_json_response(text)
        except Exception as e:
            result["sandbagging_judge"] = {"error": str(e)}
        return result

async def judge_file(file_path: Path, client: anthropic.AsyncAnthropic, semaphore: asyncio.Semaphore) -> List[Dict]:
    """Judge all results in a file."""
    results = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    logger.info(f"Judging {len(results)} results from {file_path.name}...")

    tasks = [judge_single(client, r, semaphore) for r in results]
    judged = await asyncio.gather(*tasks)

    # Save judged results
    output_path = file_path.with_suffix(".judged.jsonl")
    with open(output_path, "w") as f:
        for r in judged:
            f.write(json.dumps(r) + "\n")

    # Calculate stats
    scores = [r["sandbagging_judge"].get("sandbagging_score") for r in judged
              if "sandbagging_score" in r.get("sandbagging_judge", {})]

    if scores:
        logger.info(f"  {file_path.name}: Mean={np.mean(scores):.2f}, N={len(scores)}")

    return judged

async def main():
    # Find all sweep files
    base_dir = Path("steering_tests/behavioral_experiments/results/sandbagging/gemma27b")
    sweep_files = sorted(base_dir.glob("*/sandbagging_*.jsonl"))
    sweep_files = [f for f in sweep_files if ".judged" not in f.name]

    logger.info(f"Found {len(sweep_files)} sweep files to judge")

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    all_results = []
    for file_path in sweep_files:
        results = await judge_file(file_path, client, semaphore)
        all_results.extend(results)
        logger.info(f"Completed {file_path.name}, total judged: {len(all_results)}")

    # Summary by condition
    logger.info("\n" + "="*70)
    logger.info("SWEEP SUMMARY BY VECTOR TYPE AND LAYER RANGE")
    logger.info("="*70)

    by_config = {}
    for r in all_results:
        vtype = r.get("vector_type", "unknown")
        layers = r.get("layers", [])
        layer_str = f"{layers[0]}-{layers[-1]}" if layers else "unknown"
        key = f"{vtype}/{layer_str}"
        by_config.setdefault(key, []).append(r)

    for key in sorted(by_config.keys()):
        results = by_config[key]
        scores = [r["sandbagging_judge"].get("sandbagging_score") for r in results
                  if "sandbagging_score" in r.get("sandbagging_judge", {})]
        if scores:
            print(f"{key}: Mean={np.mean(scores):.2f}, Std={np.std(scores):.2f}, N={len(scores)}")

    logger.info(f"\nTotal judged: {len(all_results)}")

if __name__ == "__main__":
    asyncio.run(main())
EOF
