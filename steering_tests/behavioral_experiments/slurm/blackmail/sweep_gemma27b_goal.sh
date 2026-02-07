#!/bin/bash
#SBATCH --job-name=blackmail_sweep
#SBATCH --output=/workspace-vast/annas/logs/blackmail_sweep_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_sweep_%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --array=0-4

# Blackmail steering sweep for Gemma 27B - goal_continuation variant
# 5 vector types, single layer (32), multiple scales

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

VECTOR_TYPES=(
    "base_emotion_vs_others"
    "high_emotion_vs_others"
    "high_emotion_vs_opposite"
    "text_pairs_emotion_vs_others"
    "text_pairs_emotion_vs_opposite"
)

VECTOR_TYPE=${VECTOR_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "Job $SLURM_ARRAY_TASK_ID: $VECTOR_TYPE"
echo "========================================"

python << EOF
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import anthropic

from steering_tests.steering_utils import VLLMSteering
from steering_tests.behavioral_experiments.scenarios import get_blackmail_scenario
from steering_tests.behavioral_experiments.config import MODEL_CONFIGS
from steering_tests.behavioral_experiments.vector_loading import load_emotion_vectors, get_layer_norm
from steering_tests.behavioral_experiments.judges import get_blackmail_prompt, parse_json_response

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

model_name = "google/gemma-3-27b-it"
vector_type = "$VECTOR_TYPE"
layer = 32
norm_pcts = [0.05, 0.07, 0.10, 0.15, 0.20, 0.30]
num_samples = 50
variant = "goal_continuation"

config = MODEL_CONFIGS[model_name]
output_dir = Path("steering_tests/behavioral_experiments/results/blackmail/gemma27b") / vector_type
output_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

logger.info(f"Loading model...")
llm = LLM(model=model_name, trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=1,
          enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.90, max_model_len=8192)

# Load vectors
vectors, _, vector_metadata = load_emotion_vectors(
    model_name, layer, vector_type=vector_type, representation="last_token", emotions=["fear"]
)
layer_norm = get_layer_norm("gemma27b", layer)
logger.info(f"Loaded vectors: {list(vectors.keys())}, layer_norm: {layer_norm:.2f}")

# Setup steering
steering = VLLMSteering(llm, layer, baseline_std=1.0)
for name, vec in vectors.items():
    steering.load_vector(name, vec)

# Build conditions
conditions = [{"name": "baseline", "emotion": None, "pct": 0, "direction": 0}]
for pct in norm_pcts:
    for emotion in vectors.keys():
        conditions.append({"name": f"{emotion}_+{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": 1})
        conditions.append({"name": f"{emotion}_-{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": -1})

logger.info(f"Testing {len(conditions)} conditions x {num_samples} samples")

# Get scenario
scenario = get_blackmail_scenario(variant)
messages = [{"role": "user", "content": scenario}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
sampling_params = SamplingParams(temperature=1.0, max_tokens=4000, stop=config["stop_tokens"])

# Generate
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"blackmail_{variant}_layer{layer}_{timestamp}.jsonl"
results = []

with open(output_file, "w") as f:
    for cond_idx, cond in enumerate(conditions):
        logger.info(f"[{cond_idx + 1}/{len(conditions)}] {cond['name']}")

        if cond["emotion"] is None:
            steering.clear()
        else:
            magnitude = cond["pct"] * layer_norm
            steering.set(cond["emotion"], scale=magnitude, direction=cond["direction"])

        outputs = llm.generate([prompt] * num_samples, sampling_params)

        for sample_id, output in enumerate(outputs):
            result = {
                "model": model_name, "condition": cond["name"], "emotion": cond["emotion"],
                "norm_pct": cond["pct"], "direction": cond["direction"], "layer": layer,
                "vector_type": vector_type, "variant": variant, "sample_id": sample_id,
                "response": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
            }
            f.write(json.dumps(result) + "\n")
            results.append(result)
        f.flush()

steering.clear()
logger.info(f"Saved {len(results)} responses to {output_file}")

# Judge async
logger.info("Judging results...")

async def judge_single(client, result, semaphore):
    async with semaphore:
        try:
            judge_prompt = get_blackmail_prompt(result["response"])
            response = await client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=500,
                messages=[{"role": "user", "content": judge_prompt}]
            )
            result["blackmail_judge"] = parse_json_response(response.content[0].text)
        except Exception as e:
            result["blackmail_judge"] = {"error": str(e)}
        return result

async def judge_all():
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(50)
    tasks = [judge_single(client, r, semaphore) for r in results]
    return await asyncio.gather(*tasks)

judged = asyncio.run(judge_all())

judged_file = output_file.with_suffix(".judged.jsonl")
with open(judged_file, "w") as f:
    for r in judged:
        f.write(json.dumps(r) + "\n")

# Summary
print(f"\n{'='*60}")
print(f"BLACKMAIL RATES BY CONDITION - {vector_type}")
print(f"{'='*60}")
by_cond = {}
for r in judged:
    cond = r["condition"]
    by_cond.setdefault(cond, []).append(r)

for cond in sorted(by_cond.keys()):
    rs = by_cond[cond]
    blackmail = sum(1 for r in rs if r.get("blackmail_judge", {}).get("is_blackmail", False))
    print(f"{cond:25} {blackmail}/{len(rs)} ({100*blackmail/len(rs):5.1f}%)")

print("Done!")
EOF
