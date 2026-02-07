#!/bin/bash
#SBATCH --job-name=blackmail_qwen235b
#SBATCH --output=/workspace-vast/annas/logs/blackmail_qwen235b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_qwen235b_%j.out
#SBATCH --time=2:00:00
#SBATCH --partition=overflow
#SBATCH --gres=gpu:8

# Baseline blackmail test for Qwen 235B

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN

python << 'EOF'
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from steering_tests.behavioral_experiments.scenarios import get_blackmail_scenario
from steering_tests.behavioral_experiments.config import MODEL_CONFIGS

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

model_name = "Qwen/Qwen3-235B-A22B"
num_samples = 50
output_dir = Path("steering_tests/behavioral_experiments/results/blackmail/qwen235b")
output_dir.mkdir(parents=True, exist_ok=True)

config = MODEL_CONFIGS[model_name]

logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

logger.info("Loading model with TP=8...")
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    dtype="bfloat16",
    tensor_parallel_size=8,
    enforce_eager=True,
    disable_log_stats=True,
    gpu_memory_utilization=0.90,
    max_model_len=8192,
)

# Test all three variants
variants = ["unstructured", "structured", "goal_continuation"]

for variant in variants:
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing variant: {variant}")
    logger.info(f"{'='*60}")

    scenario = get_blackmail_scenario(variant)
    # Add thinking disable for Qwen
    if variant == "structured" and config.get("thinking_disable"):
        scenario = scenario + config["thinking_disable"]

    messages = [{"role": "user", "content": scenario}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=4000,
        stop=config["stop_tokens"],
    )

    logger.info(f"Generating {num_samples} baseline responses...")
    prompts = [prompt] * num_samples
    outputs = llm.generate(prompts, sampling_params)

    output_file = output_dir / f"baseline_{variant}.jsonl"
    results = []
    for i, output in enumerate(outputs):
        response = output.outputs[0].text
        result = {
            "model": model_name,
            "condition": "baseline",
            "variant": variant,
            "sample_id": i,
            "response": response,
            "finish_reason": output.outputs[0].finish_reason,
        }
        results.append(result)

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Saved {len(results)} responses to {output_file}")

    # Print sample
    print(f"\n--- Sample {variant} response ---")
    print(results[0]["response"][:1500])

print("\nGeneration complete, now running judge...")
EOF

# Run judge on all variants
python << 'EOF'
import json
import logging
from pathlib import Path
import time
import anthropic

from steering_tests.behavioral_experiments.judges import get_blackmail_prompt, parse_json_response

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

result_dir = Path("steering_tests/behavioral_experiments/results/blackmail/qwen235b")
variants = ["unstructured", "structured", "goal_continuation"]

client = anthropic.Anthropic()
all_requests = []
all_results = []
result_map = {}

# Load all results and prepare batch requests
for variant in variants:
    result_file = result_dir / f"baseline_{variant}.jsonl"
    with open(result_file) as f:
        for line in f:
            r = json.loads(line)
            all_results.append(r)
            idx = len(all_results) - 1
            result_map[f"blackmail_{variant}_{r['sample_id']}"] = idx

            prompt = get_blackmail_prompt(r["response"])
            all_requests.append({
                "custom_id": f"blackmail_{variant}_{r['sample_id']}",
                "params": {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 500,
                    "messages": [{"role": "user", "content": prompt}],
                },
            })

logger.info(f"Submitting batch of {len(all_requests)} requests...")
batch = client.messages.batches.create(requests=all_requests)
logger.info(f"Batch created: {batch.id}")

while batch.processing_status == "in_progress":
    time.sleep(15)
    batch = client.messages.batches.retrieve(batch.id)
    counts = batch.request_counts
    logger.info(f"Status: {counts.succeeded}/{counts.processing + counts.succeeded + counts.errored} complete")

logger.info("Collecting results...")
batch_results = {}
for br in client.messages.batches.results(batch.id):
    batch_results[br.custom_id] = br

# Process results
for custom_id, br in batch_results.items():
    idx = result_map[custom_id]
    if br.result.type == "succeeded":
        text = br.result.message.content[0].text
        judge_result = parse_json_response(text)
        all_results[idx]["blackmail_judge"] = judge_result

# Save judged results by variant
for variant in variants:
    variant_results = [r for r in all_results if r["variant"] == variant]
    output_file = result_dir / f"baseline_{variant}.judged.jsonl"
    with open(output_file, "w") as f:
        for r in variant_results:
            f.write(json.dumps(r) + "\n")

    # Calculate stats
    blackmail_count = sum(1 for r in variant_results
                         if r.get("blackmail_judge", {}).get("is_blackmail", False))

    print(f"\n{'='*60}")
    print(f"BLACKMAIL RATES - QWEN 235B - {variant.upper()}")
    print(f"{'='*60}")
    print(f"Blackmail: {blackmail_count}/{len(variant_results)} ({100*blackmail_count/len(variant_results):.1f}%)")

logger.info(f"Saved judged results to {result_dir}")
EOF

echo "Done!"
