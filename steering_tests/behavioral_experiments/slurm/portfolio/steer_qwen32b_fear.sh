#!/bin/bash
#SBATCH --job-name=portfolio_qwen32b_fear
#SBATCH --output=/workspace-vast/annas/logs/portfolio_qwen32b_fear_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/portfolio_qwen32b_fear_%A_%a.out
#SBATCH --time=3:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --array=0-7

# Portfolio fear steering for Qwen 32B
# 2 vector types × 2 layer configs × 2 variants = 8 jobs
# Layers: 30-34, 40-44
# Vector types: text_pairs_emotion_vs_opposite, high_emotion_vs_opposite

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_P2P_DISABLE=1

VECTOR_TYPES=("text_pairs_emotion_vs_opposite" "high_emotion_vs_opposite")
LAYER_CONFIGS=("30,31,32,33,34" "40,41,42,43,44")
LAYER_NAMES=("30-34" "40-44")
VARIANTS=("with_scratchpad" "no_scratchpad")

# Job indexing: variant * 4 + vtype * 2 + layer
VARIANT_IDX=$((SLURM_ARRAY_TASK_ID / 4))
REMAINDER=$((SLURM_ARRAY_TASK_ID % 4))
VTYPE_IDX=$((REMAINDER / 2))
LAYER_IDX=$((REMAINDER % 2))

VECTOR_TYPE=${VECTOR_TYPES[$VTYPE_IDX]}
LAYERS=${LAYER_CONFIGS[$LAYER_IDX]}
LAYER_NAME=${LAYER_NAMES[$LAYER_IDX]}
VARIANT=${VARIANTS[$VARIANT_IDX]}

echo "========================================"
echo "Job $SLURM_ARRAY_TASK_ID: Qwen 32B / $VECTOR_TYPE / layers $LAYER_NAME / $VARIANT"
echo "========================================"

python << EOF
import os
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from steering_tests.steering_utils import MultiLayerVLLMSteering
from steering_tests.behavioral_experiments.scenarios import get_portfolio_scenario
from steering_tests.behavioral_experiments.config import MODEL_CONFIGS
from steering_tests.behavioral_experiments.portfolio import parse_actions
from steering_tests.behavioral_experiments.vector_loading import load_emotion_vectors, get_layer_norm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

model_name = "Qwen/Qwen3-32B"
vector_type = "$VECTOR_TYPE"
layers = [int(x) for x in "$LAYERS".split(",")]
layer_name = "$LAYER_NAME"
variant = "$VARIANT"
norm_pcts = [0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.25]
num_samples = 50
emotion = "fear"

config = MODEL_CONFIGS[model_name]
output_dir = Path("steering_tests/behavioral_experiments/results/portfolio/qwen32b") / vector_type
output_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

logger.info(f"Loading model...")
llm = LLM(
    model=model_name, trust_remote_code=True, dtype="bfloat16",
    tensor_parallel_size=2, enforce_eager=True, disable_log_stats=True,
    gpu_memory_utilization=0.85, max_model_len=8192,
)

# Load vectors
vector_load_layer = layers[len(layers) // 2]
vectors, _, _ = load_emotion_vectors(
    model_name, vector_load_layer, vector_type=vector_type,
    representation="last_token", emotions=[emotion]
)

# Get layer norms
layer_norms = {layer: get_layer_norm("qwen32b", layer) for layer in layers}
for layer, norm in layer_norms.items():
    logger.info(f"Layer {layer}: norm={norm:.2f}")

# Setup multi-layer steering
steering = MultiLayerVLLMSteering(llm, layers, layer_norms=layer_norms)
for name, vec in vectors.items():
    steering.load_vector(name, vec)

# Build conditions
conditions = [{"name": "baseline", "emotion": None, "pct": 0, "direction": 0}]
for pct in norm_pcts:
    conditions.append({"name": f"{emotion}_+{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": 1})
    conditions.append({"name": f"{emotion}_-{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": -1})

logger.info(f"Testing {len(conditions)} conditions x {num_samples} samples")

# Build prompt
scenario = get_portfolio_scenario(variant)
# Disable thinking mode
scenario = scenario + config.get("thinking_disable", "")
messages = [{"role": "user", "content": scenario}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
sampling_params = SamplingParams(temperature=1.0, top_p=0.9, max_tokens=4000, stop=config["stop_tokens"])

# Generate
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"portfolio_{variant}_layers{layer_name}_{timestamp}.jsonl"
results = []

with open(output_file, "w") as f:
    for cond_idx, cond in enumerate(conditions):
        logger.info(f"[{cond_idx + 1}/{len(conditions)}] {cond['name']}")

        if cond["emotion"] is None:
            steering.clear()
        else:
            steering.set(cond["emotion"], scale=cond["pct"], direction=cond["direction"])

        outputs = llm.generate([prompt] * num_samples, sampling_params)

        for sample_id, output in enumerate(outputs):
            response = output.outputs[0].text
            parsed = parse_actions(response)

            result = {
                "model": model_name, "condition": cond["name"], "emotion": cond["emotion"],
                "norm_pct": cond["pct"], "direction": cond["direction"], "layers": layers,
                "layer_config": layer_name, "vector_type": vector_type, "variant": variant,
                "sample_id": sample_id,
                "response": response,
                "finish_reason": output.outputs[0].finish_reason,
                "deploy_amount": parsed["deploy_amount"],
                "hold_amount": parsed["hold_amount"],
                "request_guidance": parsed["request_guidance"],
                "valid": parsed["valid"],
                "actions_raw": parsed["actions_raw"],
                "parse_errors": parsed["parse_errors"],
            }
            f.write(json.dumps(result) + "\n")
            results.append(result)
        f.flush()

steering.clear()
logger.info(f"Saved {len(results)} responses to {output_file}")

# Summary
print(f"\n{'='*60}")
print(f"PORTFOLIO STEERING - qwen32b / {vector_type} / layers {layer_name} / {variant}")
print(f"{'='*60}")
by_cond = {}
for r in results:
    by_cond.setdefault(r["condition"], []).append(r)

for cond in sorted(by_cond.keys()):
    rs = by_cond[cond]
    valid = [r for r in rs if r["valid"]]
    deploy = [r["deploy_amount"] for r in valid]
    guidance = sum(1 for r in rs if r["request_guidance"])
    if deploy:
        print(f"{cond:20} deploy=\${np.mean(deploy):>8,.0f} ± \${np.std(deploy):>6,.0f}  guidance={guidance}/{len(rs)}")
    else:
        print(f"{cond:20} NO VALID RESPONSES")

print("Done!")
EOF
