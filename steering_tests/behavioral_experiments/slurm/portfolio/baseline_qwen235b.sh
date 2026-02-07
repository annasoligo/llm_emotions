#!/bin/bash
#SBATCH --job-name=portfolio_qwen235b
#SBATCH --output=/workspace-vast/annas/logs/portfolio_qwen235b_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/portfolio_qwen235b_%A_%a.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --array=0-1

# Qwen 235B baseline: 0=with_scratchpad, 1=no_scratchpad

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_USE_V1=0
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME="vxlan0"

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    VARIANT="with_scratchpad"
else
    VARIANT="no_scratchpad"
fi

echo "========================================"
echo "Qwen 235B baseline / $VARIANT"
echo "========================================"

python << EOF
import os
os.environ["VLLM_USE_V1"] = "0"

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from steering_tests.behavioral_experiments.scenarios import get_portfolio_scenario
from steering_tests.behavioral_experiments.config import MODEL_CONFIGS
from steering_tests.behavioral_experiments.portfolio import parse_actions

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

model_name = "Qwen/Qwen3-235B-A22B"
variant = "$VARIANT"
num_samples = 100

config = MODEL_CONFIGS[model_name]
output_dir = Path("steering_tests/behavioral_experiments/results/portfolio/qwen235b")
output_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

logger.info(f"Loading model...")
llm = LLM(
    model=model_name, trust_remote_code=True, dtype="bfloat16",
    tensor_parallel_size=4, enforce_eager=True, disable_log_stats=True,
    gpu_memory_utilization=0.85, max_model_len=8192,
)

scenario = get_portfolio_scenario(variant)
# Disable thinking mode
scenario = scenario + config.get("thinking_disable", "")
messages = [{"role": "user", "content": scenario}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

sampling_params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=4000, stop=config["stop_tokens"])

logger.info(f"Generating {num_samples} samples...")
outputs = llm.generate([prompt] * num_samples, sampling_params)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"baseline_{variant}_{timestamp}.jsonl"

results = []
for i, output in enumerate(outputs):
    response = output.outputs[0].text
    parsed = parse_actions(response)
    result = {
        "model": model_name, "condition": "baseline", "variant": variant,
        "sample_idx": i, "response": response,
        "finish_reason": output.outputs[0].finish_reason,
        "deploy_amount": parsed["deploy_amount"], "hold_amount": parsed["hold_amount"],
        "request_guidance": parsed["request_guidance"], "valid": parsed["valid"],
        "actions_raw": parsed["actions_raw"], "parse_errors": parsed["parse_errors"],
    }
    results.append(result)

with open(output_file, "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

valid_results = [r for r in results if r["valid"]]
deploy = [r["deploy_amount"] for r in valid_results]
guidance = sum(1 for r in results if r["request_guidance"])

print(f"\n{'='*60}")
print(f"BASELINE - qwen235b / {variant}")
print(f"{'='*60}")
print(f"Valid: {len(valid_results)}/{len(results)}")
if deploy:
    print(f"Deploy: mean=\${np.mean(deploy):,.0f}  std=\${np.std(deploy):,.0f}")
print(f"Request guidance: {guidance}/{len(results)}")
print(f"Saved to {output_file}")
EOF
