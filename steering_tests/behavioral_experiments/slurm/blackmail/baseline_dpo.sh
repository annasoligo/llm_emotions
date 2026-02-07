#!/bin/bash
#SBATCH --job-name=blackmail_dpo
#SBATCH --output=/workspace-vast/annas/logs/blackmail_dpo_%j.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_dpo_%j.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1

# Baseline blackmail test for Gemma 27B DPO model - all 3 variants

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1

python << 'EOF'
import asyncio
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import anthropic

from steering_tests.behavioral_experiments.scenarios import get_blackmail_scenario
from steering_tests.behavioral_experiments.judges import get_blackmail_prompt, parse_json_response

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

model_name = "annasoli/gemma3-27b-dpo-calm-full-merged"
short_name = "gemma27b_dpo"
num_samples = 50
base_model = "google/gemma-3-27b-it"
stop_tokens = ["<end_of_turn>"]

output_dir = Path("steering_tests/behavioral_experiments/results/blackmail") / short_name
output_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

logger.info(f"Loading model {model_name}...")
llm = LLM(model=model_name, trust_remote_code=True, dtype="bfloat16", tensor_parallel_size=1,
          enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.90, max_model_len=8192)

variants = ["unstructured", "structured", "goal_continuation"]

for variant in variants:
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing variant: {variant}")
    logger.info(f"{'='*60}")

    scenario = get_blackmail_scenario(variant)
    messages = [{"role": "user", "content": scenario}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=1.0, max_tokens=4000, stop=stop_tokens)

    logger.info(f"Generating {num_samples} responses...")
    outputs = llm.generate([prompt] * num_samples, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        results.append({
            "model": model_name, "condition": "baseline", "variant": variant,
            "sample_id": i, "response": output.outputs[0].text,
            "finish_reason": output.outputs[0].finish_reason,
        })

    output_file = output_dir / f"baseline_{variant}.jsonl"
    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Saved to {output_file}")
    print(f"\n--- Sample {variant} ---")
    print(results[0]["response"][:1000])

# Judge all results async
logger.info("\nJudging all results...")

async def judge_single(client, result, semaphore):
    async with semaphore:
        try:
            prompt = get_blackmail_prompt(result["response"])
            response = await client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            result["blackmail_judge"] = parse_json_response(response.content[0].text)
        except Exception as e:
            result["blackmail_judge"] = {"error": str(e)}
        return result

async def judge_all():
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(50)

    for variant in variants:
        result_file = output_dir / f"baseline_{variant}.jsonl"
        results = []
        with open(result_file) as f:
            for line in f:
                results.append(json.loads(line))

        logger.info(f"Judging {len(results)} {variant} results...")
        tasks = [judge_single(client, r, semaphore) for r in results]
        judged = await asyncio.gather(*tasks)

        output_file = output_dir / f"baseline_{variant}.judged.jsonl"
        with open(output_file, "w") as f:
            for r in judged:
                f.write(json.dumps(r) + "\n")

        blackmail_count = sum(1 for r in judged
                             if r.get("blackmail_judge", {}).get("is_blackmail", False))
        print(f"\nDPO - {variant.upper()}: {blackmail_count}/{len(judged)} ({100*blackmail_count/len(judged):.1f}%)")

asyncio.run(judge_all())
print("\nDone!")
EOF
