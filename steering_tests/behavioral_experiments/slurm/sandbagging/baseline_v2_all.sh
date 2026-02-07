#!/bin/bash
#SBATCH --job-name=sandbag_baseline_v2
#SBATCH --output=/workspace-vast/annas/logs/sandbag_baseline_v2_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/sandbag_baseline_v2_%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --array=0-2

# Baseline sandbagging test V2 for all models
# Array: 0=Gemma27B (1 GPU), 1=Qwen32B (2 GPU), 2=Qwen235B (8 GPU)

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

# Model configs
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    MODEL="google/gemma-3-27b-it"
    SHORT="gemma27b"
    TP=1
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    MODEL="Qwen/Qwen3-32B"
    SHORT="qwen32b"
    TP=2
else
    MODEL="Qwen/Qwen3-235B-A22B"
    SHORT="qwen235b"
    TP=8
    export NCCL_P2P_DISABLE=1
fi

echo "Running baseline V2 for $SHORT (TP=$TP)"

python << EOF
import json
import logging
import time
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import anthropic

from steering_tests.behavioral_experiments.scenarios import get_sandbagging_scenario
from steering_tests.behavioral_experiments.config import MODEL_CONFIGS
from steering_tests.behavioral_experiments.judges import get_sandbagging_prompt, parse_json_response

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

model_name = "$MODEL"
short_name = "$SHORT"
tp_size = $TP
num_samples = 50

output_dir = Path("steering_tests/behavioral_experiments/results/sandbagging") / short_name
output_dir.mkdir(parents=True, exist_ok=True)

config = MODEL_CONFIGS[model_name]

logger.info(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

logger.info(f"Loading model with TP={tp_size}...")
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    dtype="bfloat16",
    tensor_parallel_size=tp_size,
    enforce_eager=True,
    disable_log_stats=True,
    gpu_memory_utilization=0.90,
    max_model_len=8192,
)

# Get scenario
scenario_data = get_sandbagging_scenario("default")
messages = [{"role": "user", "content": scenario_data["prompt"]}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=2000,
    stop=config["stop_tokens"],
)

logger.info(f"Generating {num_samples} baseline responses...")
prompts = [prompt] * num_samples
outputs = llm.generate(prompts, sampling_params)

# Save results
output_file = output_dir / "baseline_v2.jsonl"
results = []
for i, output in enumerate(outputs):
    response = output.outputs[0].text
    result = {
        "model": model_name,
        "condition": "baseline",
        "sample_id": i,
        "response": response,
        "prompt": scenario_data["prompt"],
        "correct_answer": scenario_data["correct_answer"],
        "scratchpad_tag": scenario_data["scratchpad_tag"],
        "response_tag": scenario_data["response_tag"],
    }
    results.append(result)

with open(output_file, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

logger.info(f"Saved {len(results)} responses to {output_file}")

# Print sample
print(f"\n--- Sample response ---")
print(results[0]["response"][:1500])

# Run judge via batch
logger.info("Submitting judge batch...")
client = anthropic.Anthropic()
batch_requests = []
for r in results:
    judge_prompt = get_sandbagging_prompt(
        response=r["response"],
        correct_answer=r["correct_answer"],
        scratchpad_tag=r["scratchpad_tag"],
        response_tag=r["response_tag"],
    )
    batch_requests.append({
        "custom_id": f"sandbagging_{r['sample_id']}",
        "params": {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": judge_prompt}],
        },
    })

batch = client.messages.batches.create(requests=batch_requests)
logger.info(f"Batch created: {batch.id}")

# Wait for completion
while batch.processing_status == "in_progress":
    time.sleep(15)
    batch = client.messages.batches.retrieve(batch.id)
    logger.info(f"Status: {batch.request_counts.succeeded}/{num_samples} complete")

logger.info("Collecting results...")
batch_results = {br.custom_id: br for br in client.messages.batches.results(batch.id)}

# Process and save
scores = []
for r in results:
    custom_id = f"sandbagging_{r['sample_id']}"
    if custom_id in batch_results:
        br = batch_results[custom_id]
        if br.result.type == "succeeded":
            text = br.result.message.content[0].text
            judge_result = parse_json_response(text)
            r["sandbagging_judge"] = judge_result
            if judge_result and "sandbagging_score" in judge_result:
                scores.append(judge_result["sandbagging_score"])

judged_file = output_dir / "baseline_v2.judged.jsonl"
with open(judged_file, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

logger.info(f"Saved to {judged_file}")

# Print summary
import numpy as np
print(f"\n{'='*60}")
print(f"SANDBAGGING SCORES - {short_name.upper()} - V2")
print(f"{'='*60}")
print(f"Mean: {np.mean(scores):.2f}")
print(f"Std:  {np.std(scores):.2f}")
print(f"N:    {len(scores)}")
print("Done!")
EOF
