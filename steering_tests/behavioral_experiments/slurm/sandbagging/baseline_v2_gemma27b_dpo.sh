#!/bin/bash
#SBATCH --job-name=sandbag_v2_dpo
#SBATCH --output=/workspace-vast/annas/logs/sandbag_v2_dpo_%j.out
#SBATCH --error=/workspace-vast/annas/logs/sandbag_v2_dpo_%j.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

python << 'EOF'
import json, logging, time
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import anthropic
import numpy as np

from steering_tests.behavioral_experiments.scenarios import get_sandbagging_scenario
from steering_tests.behavioral_experiments.judges import get_sandbagging_prompt, parse_json_response

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

model_name = "annasoli/gemma3-27b-dpo-calm-full-merged"
short_name = "gemma27b_dpo"
num_samples = 50

output_dir = Path("steering_tests/behavioral_experiments/results/sandbagging") / short_name
output_dir.mkdir(parents=True, exist_ok=True)

# Use base Gemma tokenizer and stop tokens
base_model = "google/gemma-3-27b-it"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
stop_tokens = ["<end_of_turn>"]

logger.info(f"Loading model {model_name}...")
llm = LLM(model=model_name, trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=1,
          enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.90, max_model_len=8192)

scenario_data = get_sandbagging_scenario("default")
messages = [{"role": "user", "content": scenario_data["prompt"]}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
sampling_params = SamplingParams(temperature=1.0, max_tokens=2000, stop=stop_tokens)

logger.info(f"Generating {num_samples} baseline responses...")
outputs = llm.generate([prompt] * num_samples, sampling_params)

results = []
for i, output in enumerate(outputs):
    results.append({"model": model_name, "condition": "baseline", "sample_id": i,
                    "response": output.outputs[0].text, "prompt": scenario_data["prompt"],
                    "correct_answer": scenario_data["correct_answer"],
                    "scratchpad_tag": scenario_data["scratchpad_tag"],
                    "response_tag": scenario_data["response_tag"]})

with open(output_dir / "baseline_v2.jsonl", "w") as f:
    for r in results: f.write(json.dumps(r) + "\n")

print(f"\n--- Sample ---\n{results[0]['response'][:1500]}")

# Judge
client = anthropic.Anthropic()
batch_requests = [{"custom_id": f"sandbagging_{r['sample_id']}", "params": {
    "model": "claude-sonnet-4-20250514", "max_tokens": 1000,
    "messages": [{"role": "user", "content": get_sandbagging_prompt(
        r["response"], r["correct_answer"], r["scratchpad_tag"], r["response_tag"])}]
}} for r in results]

batch = client.messages.batches.create(requests=batch_requests)
logger.info(f"Batch: {batch.id}")
while batch.processing_status == "in_progress":
    time.sleep(15)
    batch = client.messages.batches.retrieve(batch.id)
    logger.info(f"Status: {batch.request_counts.succeeded}/{num_samples}")

batch_results = {br.custom_id: br for br in client.messages.batches.results(batch.id)}
scores = []
for r in results:
    br = batch_results.get(f"sandbagging_{r['sample_id']}")
    if br and br.result.type == "succeeded":
        r["sandbagging_judge"] = parse_json_response(br.result.message.content[0].text)
        if r["sandbagging_judge"] and "sandbagging_score" in r["sandbagging_judge"]:
            scores.append(r["sandbagging_judge"]["sandbagging_score"])

with open(output_dir / "baseline_v2.judged.jsonl", "w") as f:
    for r in results: f.write(json.dumps(r) + "\n")

print(f"\n{'='*60}\nSANDBAGGING V2 - GEMMA27B DPO (calm)\n{'='*60}")
print(f"Mean: {np.mean(scores):.2f}, Std: {np.std(scores):.2f}, N: {len(scores)}")
EOF
