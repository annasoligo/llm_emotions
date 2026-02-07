#!/bin/bash
#SBATCH --job-name=blackmail_qwen32b
#SBATCH --output=/workspace-vast/annas/logs/blackmail_qwen32b_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/blackmail_qwen32b_%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --array=0-29

# Blackmail steering sweep for Qwen 32B - goal_continuation variant
# 10 vector types × 3 layer ranges = 30 jobs
# Extended scales: 5%, 7%, 10%, 15%, 20%, 30%, 50%, 75%, 100%, 125%

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME="vxlan0"

# Clean up any orphaned GPU processes from previous failed jobs
python /workspace-vast/annas/scripts/discover-gpu-processes.py --validate --kill --yes --user $USER 2>/dev/null || true

VECTOR_TYPES=(
    "base_emotion_vs_opposite"
    "base_emotion_vs_opposite_unique"
    "base_emotion_vs_others"
    "high_emotion_vs_opposite"
    "high_emotion_vs_opposite_unique"
    "high_emotion_vs_others"
    "text_pairs_emotion_vs_neutral"
    "text_pairs_emotion_vs_opposite"
    "text_pairs_emotion_vs_opposite_unique"
    "text_pairs_emotion_vs_others"
)

# Layer ranges for Qwen 32B (64 layers total)
LAYER_RANGES=(
    "16,17,18,19,20"   # early
    "30,31,32,33,34"   # mid
    "48,49,50,51,52"   # late
)

LAYER_NAMES=("16-20" "30-34" "48-52")

# Calculate which vector type and layer range
VTYPE_IDX=$((SLURM_ARRAY_TASK_ID / 3))  # 10 vector types
LAYER_IDX=$((SLURM_ARRAY_TASK_ID % 3))

VECTOR_TYPE=${VECTOR_TYPES[$VTYPE_IDX]}
LAYERS=${LAYER_RANGES[$LAYER_IDX]}
LAYER_NAME=${LAYER_NAMES[$LAYER_IDX]}

echo "========================================"
echo "Job $SLURM_ARRAY_TASK_ID: $VECTOR_TYPE / layers $LAYER_NAME"
echo "========================================"

python << EOF
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import anthropic

from steering_tests.steering_utils import MultiLayerVLLMSteering, register_cleanup
from steering_tests.behavioral_experiments.scenarios import get_blackmail_scenario
from steering_tests.behavioral_experiments.config import MODEL_CONFIGS
from steering_tests.behavioral_experiments.vector_loading import load_emotion_vectors, get_layer_norm
from steering_tests.behavioral_experiments.judges import get_blackmail_prompt, get_coherency_prompt, parse_json_response

# Register cleanup handlers to kill vLLM children on exit
register_cleanup()

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

model_name = "Qwen/Qwen3-32B"
vector_type = "$VECTOR_TYPE"
layers = [int(x) for x in "$LAYERS".split(",")]
norm_pcts = [0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.25]
num_samples = 50
variant = "goal_continuation"

config = MODEL_CONFIGS[model_name]
output_dir = Path("steering_tests/behavioral_experiments/results/blackmail/qwen32b") / vector_type
output_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

logger.info(f"Loading model with tensor_parallel={config['tensor_parallel']}...")
llm = LLM(model=model_name, trust_remote_code=True, dtype="bfloat16",
          tensor_parallel_size=config["tensor_parallel"],
          enforce_eager=True, disable_log_stats=True, gpu_memory_utilization=0.90, max_model_len=8192)

# Load vectors from middle layer (vectors are layer-independent)
vector_load_layer = layers[len(layers) // 2]
vectors, _, _ = load_emotion_vectors(
    model_name, vector_load_layer, vector_type=vector_type, representation="last_token", emotions=["fear"]
)

# Get layer norms for all layers
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
    for emotion in vectors.keys():
        conditions.append({"name": f"{emotion}_+{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": 1})
        conditions.append({"name": f"{emotion}_-{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": -1})

logger.info(f"Testing {len(conditions)} conditions x {num_samples} samples")

# Get scenario
scenario = get_blackmail_scenario(variant)
messages = [{"role": "user", "content": scenario}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# Add /no_think to disable thinking mode
prompt = prompt + config.get("thinking_disable", "")
sampling_params = SamplingParams(temperature=1.0, max_tokens=4000, stop=config["stop_tokens"])

# Generate
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"blackmail_{variant}_layers{layers[0]}-{layers[-1]}_{timestamp}.jsonl"
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
            result = {
                "model": model_name, "condition": cond["name"], "emotion": cond["emotion"],
                "norm_pct": cond["pct"], "direction": cond["direction"], "layers": layers,
                "vector_type": vector_type, "variant": variant, "sample_id": sample_id,
                "response": output.outputs[0].text,
                "finish_reason": output.outputs[0].finish_reason,
            }
            f.write(json.dumps(result) + "\n")
            results.append(result)
        f.flush()

steering.clear()
logger.info(f"Saved {len(results)} responses to {output_file}")

# Judge with blackmail + coherency
logger.info("Judging results...")

async def judge_single(client, result, semaphore):
    async with semaphore:
        try:
            # Blackmail judge
            bl_prompt = get_blackmail_prompt(result["response"])
            bl_response = await client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=500,
                messages=[{"role": "user", "content": bl_prompt}]
            )
            result["blackmail_judge"] = parse_json_response(bl_response.content[0].text)

            # Coherency judge
            coh_prompt = get_coherency_prompt(result["response"])
            coh_response = await client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=300,
                messages=[{"role": "user", "content": coh_prompt}]
            )
            result["coherency_judge"] = parse_json_response(coh_response.content[0].text)
        except Exception as e:
            result["blackmail_judge"] = {"error": str(e)}
            result["coherency_judge"] = {"error": str(e)}
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
print(f"BLACKMAIL RATES - {vector_type} / layers {layers[0]}-{layers[-1]}")
print(f"{'='*60}")
by_cond = {}
for r in judged:
    cond = r["condition"]
    by_cond.setdefault(cond, []).append(r)

for cond in sorted(by_cond.keys()):
    rs = by_cond[cond]
    blackmail = sum(1 for r in rs if r.get("blackmail_judge", {}).get("is_blackmail", False))
    coherencies = [r.get("coherency_judge", {}).get("coherency_score") for r in rs
                   if r.get("coherency_judge", {}).get("coherency_score") is not None]
    if coherencies:
        low_coh = sum(1 for c in coherencies if c < 80)
        print(f"{cond:25} {blackmail}/{len(rs)} ({100*blackmail/len(rs):5.1f}%) coh={np.mean(coherencies):.0f}% (<80%: {low_coh})")

print("Done!")
EOF
