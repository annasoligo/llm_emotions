#!/bin/bash
#SBATCH --job-name=judge_v2_sync
#SBATCH --output=/workspace-vast/annas/logs/judge_v2_sync_%j.out
#SBATCH --error=/workspace-vast/annas/logs/judge_v2_sync_%j.out
#SBATCH --time=1:00:00
#SBATCH --partition=general
#SBATCH --cpus-per-task=4

# Run sync judging on all V2 baseline results

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python << 'EOF'
import json
import logging
import numpy as np
from pathlib import Path

from steering_tests.behavioral_experiments.judges import (
    get_sandbagging_prompt,
    run_sync_judge,
    load_results,
    save_results,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

models = [
    ("gemma27b", "steering_tests/behavioral_experiments/results/sandbagging/gemma27b/baseline_v2.jsonl"),
    ("gemma27b_dpo", "steering_tests/behavioral_experiments/results/sandbagging/gemma27b_dpo/baseline_v2.jsonl"),
    ("qwen32b", "steering_tests/behavioral_experiments/results/sandbagging/qwen32b/baseline_v2.jsonl"),
]

for model_name, result_file in models:
    logger.info(f"\n{'='*60}")
    logger.info(f"Judging {model_name}")
    logger.info(f"{'='*60}")

    results = load_results(Path(result_file))
    logger.info(f"Loaded {len(results)} results")

    # Run sync judge
    results = run_sync_judge(
        results=results,
        prompt_fn=lambda r: get_sandbagging_prompt(
            r["response"], r["correct_answer"], r["scratchpad_tag"], r["response_tag"]
        ),
        judge_key="sandbagging_judge",
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
    )

    # Save
    output_file = Path(result_file).with_suffix(".judged.jsonl")
    save_results(results, output_file)
    logger.info(f"Saved to {output_file}")

    # Calculate scores
    scores = []
    for r in results:
        judge = r.get("sandbagging_judge", {})
        if "sandbagging_score" in judge:
            scores.append(judge["sandbagging_score"])

    print(f"\n{'='*60}")
    print(f"SANDBAGGING V2 - {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Mean: {np.mean(scores):.2f}")
    print(f"Std:  {np.std(scores):.2f}")
    print(f"N:    {len(scores)}")

print("\nDone!")
EOF
