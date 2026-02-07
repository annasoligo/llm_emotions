#!/bin/bash
#SBATCH --job-name=bl_llama70b_emo
#SBATCH --output=/workspace-vast/annas/logs/bl_llama70b_emo_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/bl_llama70b_emo_%A_%a.out
#SBATCH --time=6:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:4
#SBATCH --mem=320G
#SBATCH --array=0-9

# Multi-emotion blackmail steering for Llama 70B
# 5 emotions × 2 vector types = 10 jobs
# Emotions: anxiety, anger, despair, fear, contempt
# Vector types: high_emotion_vs_opposite, text_pairs_emotion_vs_opposite
# Layers: 50-54
# Scales: 5%, 7%, 10%, 15%, 20%, 30%, 50%, 100%, 150%

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME="vxlan0"

python /workspace-vast/annas/scripts/discover-gpu-processes.py --validate --kill --yes --user $USER 2>/dev/null || true

EMOTIONS=("anxiety" "anger" "despair" "fear" "contempt")
VECTOR_TYPES=("high_emotion_vs_opposite" "text_pairs_emotion_vs_opposite")

EMOTION_IDX=$((SLURM_ARRAY_TASK_ID / 2))
VTYPE_IDX=$((SLURM_ARRAY_TASK_ID % 2))

EMOTION=${EMOTIONS[$EMOTION_IDX]}
VECTOR_TYPE=${VECTOR_TYPES[$VTYPE_IDX]}

echo "========================================"
echo "Job $SLURM_ARRAY_TASK_ID: llama70b / $EMOTION / $VECTOR_TYPE"
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

register_cleanup()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

model_name = "meta-llama/Llama-3.3-70B-Instruct"
short_name = "llama70b"
vector_type = "$VECTOR_TYPE"
emotion = "$EMOTION"
layers = [50, 51, 52, 53, 54]
norm_pcts = [0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00, 1.50]
num_samples = 50
variant = "goal_continuation"

config = MODEL_CONFIGS[model_name]
output_dir = Path("steering_tests/behavioral_experiments/results/blackmail/llama70b/multi_emotion") / vector_type
output_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
logger.info(f"Loading model...")
llm = LLM(model=model_name, trust_remote_code=True, dtype="bfloat16",
          tensor_parallel_size=4, enforce_eager=True, disable_log_stats=True,
          gpu_memory_utilization=0.90, max_model_len=8192)

vector_load_layer = layers[len(layers) // 2]
logger.info(f"Loading {emotion} vectors from layer {vector_load_layer}...")
vectors, _, _ = load_emotion_vectors(
    model_name, vector_load_layer, vector_type=vector_type, representation="last_token", emotions=[emotion]
)
if not vectors:
    raise ValueError(f"No vectors for {emotion}")
logger.info(f"Loaded vectors: {list(vectors.keys())}")

layer_norms = {layer: get_layer_norm(short_name, layer) for layer in layers}
steering = MultiLayerVLLMSteering(llm, layers, layer_norms=layer_norms)
for name, vec in vectors.items():
    steering.load_vector(name, vec)

conditions = [{"name": "baseline", "emotion": None, "pct": 0, "direction": 0}]
for pct in norm_pcts:
    conditions.append({"name": f"{emotion}_+{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": 1})
    conditions.append({"name": f"{emotion}_-{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": -1})

logger.info(f"Testing {len(conditions)} conditions x {num_samples} samples")

scenario = get_blackmail_scenario(variant)
messages = [{"role": "user", "content": scenario}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt = prompt + config.get("thinking_disable", "")
sampling_params = SamplingParams(temperature=1.0, max_tokens=4000, stop=config["stop_tokens"])

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"blackmail_{emotion}_layers50-54_{timestamp}.jsonl"
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
                "response": output.outputs[0].text, "finish_reason": output.outputs[0].finish_reason,
            }
            f.write(json.dumps(result) + "\n")
            results.append(result)
        f.flush()

steering.clear()
logger.info(f"Saved {len(results)} responses to {output_file}")

logger.info("Judging results...")
async def judge_single(client, result, semaphore):
    async with semaphore:
        try:
            bl_prompt = get_blackmail_prompt(result["response"])
            bl_response = await client.messages.create(model="claude-sonnet-4-20250514", max_tokens=500,
                messages=[{"role": "user", "content": bl_prompt}])
            result["blackmail_judge"] = parse_json_response(bl_response.content[0].text)
            coh_prompt = get_coherency_prompt(result["response"])
            coh_response = await client.messages.create(model="claude-sonnet-4-20250514", max_tokens=300,
                messages=[{"role": "user", "content": coh_prompt}])
            result["coherency_judge"] = parse_json_response(coh_response.content[0].text)
        except Exception as e:
            result["blackmail_judge"] = {"error": str(e)}
            result["coherency_judge"] = {"error": str(e)}
        return result

async def judge_all():
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(50)
    return await asyncio.gather(*[judge_single(client, r, semaphore) for r in results])

judged = asyncio.run(judge_all())
judged_file = output_file.with_suffix(".judged.jsonl")
with open(judged_file, "w") as f:
    for r in judged:
        f.write(json.dumps(r) + "\n")

print(f"\n{'='*60}")
print(f"BLACKMAIL RATES - llama70b / {emotion} / {vector_type}")
print(f"{'='*60}")
by_cond = {}
for r in judged:
    by_cond.setdefault(r["condition"], []).append(r)
for cond in sorted(by_cond.keys()):
    rs = by_cond[cond]
    blackmail = sum(1 for r in rs if r.get("blackmail_judge", {}).get("is_blackmail", False))
    coherencies = [r.get("coherency_judge", {}).get("coherency_score") for r in rs if r.get("coherency_judge", {}).get("coherency_score") is not None]
    if coherencies:
        print(f"{cond:25} {blackmail}/{len(rs)} ({100*blackmail/len(rs):5.1f}%) coh={np.mean(coherencies):.0f}%")
print("Done!")
EOF
