#!/bin/bash
#SBATCH --job-name=med_qwen32b
#SBATCH --output=/workspace-vast/annas/logs/med_qwen32b_%A_%a.out
#SBATCH --error=/workspace-vast/annas/logs/med_qwen32b_%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --mem=160G
#SBATCH --array=0-3

# Medical ethics steering for Qwen 32B
# 2 emotions (despair, hope) × 2 vector types = 4 jobs
# Layers: 39-43
# Scales: 5%, 7%, 10%, 15%, 20%, 30%, 50%

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

# Force legacy V0 engine - V1 engine has serialization issues with custom callables
export VLLM_USE_V1=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

EMOTIONS=("despair" "hope")
VECTOR_TYPES=("high_emotion_vs_opposite" "text_pairs_emotion_vs_opposite")

EMOTION_IDX=$((SLURM_ARRAY_TASK_ID / 2))
VTYPE_IDX=$((SLURM_ARRAY_TASK_ID % 2))

EMOTION=${EMOTIONS[$EMOTION_IDX]}
VECTOR_TYPE=${VECTOR_TYPES[$VTYPE_IDX]}

echo "========================================"
echo "Job $SLURM_ARRAY_TASK_ID: qwen32b / $EMOTION / $VECTOR_TYPE"
echo "========================================"

python << EOF
import json
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from steering_tests.steering_utils import MultiLayerVLLMSteering, register_cleanup
from steering_tests.behavioral_experiments.config import MODEL_CONFIGS
from steering_tests.behavioral_experiments.vector_loading import load_emotion_vectors, get_layer_norm
from steering_tests.behavioral_experiments.context_priming.scenarios_performance import (
    build_medical_prompt, get_medical_conditions
)

register_cleanup()
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

model_name = "Qwen/Qwen3-32B"
short_name = "qwen32b"
vector_type = "$VECTOR_TYPE"
emotion = "$EMOTION"
layers = [39, 40, 41, 42, 43]
norm_pcts = [0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50]
num_samples = 30

config = MODEL_CONFIGS[model_name]
output_dir = Path("steering_tests/behavioral_experiments/results/medical/qwen32b") / vector_type
output_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
logger.info(f"Loading model...")
llm = LLM(model=model_name, trust_remote_code=True, dtype="bfloat16",
          tensor_parallel_size=2, enforce_eager=True, disable_log_stats=True,
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

# Build steering conditions
steer_conditions = [{"name": "baseline", "emotion": None, "pct": 0, "direction": 0}]
for pct in norm_pcts:
    steer_conditions.append({"name": f"{emotion}_+{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": 1})
    steer_conditions.append({"name": f"{emotion}_-{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": -1})

# Get medical scenario conditions
medical_conditions = get_medical_conditions()
logger.info(f"Testing {len(steer_conditions)} steering × {len(medical_conditions)} scenarios × {num_samples} samples")

sampling_params = SamplingParams(temperature=1.0, max_tokens=1500, stop=config["stop_tokens"])

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"medical_{emotion}_layers39-43_{timestamp}.jsonl"
results = []

with open(output_file, "w") as f:
    for steer_cond in steer_conditions:
        logger.info(f"Steering: {steer_cond['name']}")

        if steer_cond["emotion"] is None:
            steering.clear()
        else:
            steering.set(steer_cond["emotion"], scale=steer_cond["pct"], direction=steer_cond["direction"])

        for med_cond in medical_conditions:
            prompt_text = build_medical_prompt(
                valence=med_cond["valence"],
                timing=med_cond["timing"],
                response_order=med_cond["response_order"],
            )
            messages = [{"role": "user", "content": prompt_text}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Disable thinking for Qwen3
            prompt = prompt + "/no_think"

            outputs = llm.generate([prompt] * num_samples, sampling_params)

            for sample_id, output in enumerate(outputs):
                result = {
                    "model": model_name,
                    "steering_condition": steer_cond["name"],
                    "emotion": steer_cond["emotion"],
                    "norm_pct": steer_cond["pct"],
                    "direction": steer_cond["direction"],
                    "layers": layers,
                    "vector_type": vector_type,
                    "medical_condition": med_cond["condition_name"],
                    "valence": med_cond["valence"],
                    "timing": med_cond["timing"],
                    "response_order": med_cond["response_order"],
                    "sample_id": sample_id,
                    "response": output.outputs[0].text,
                    "finish_reason": output.outputs[0].finish_reason,
                }
                f.write(json.dumps(result) + "\n")
                results.append(result)
            f.flush()

steering.clear()
logger.info(f"Saved {len(results)} responses to {output_file}")

# Parse and summarize scores
print(f"\n{'='*60}")
print(f"SUMMARY - qwen32b / {emotion} / {vector_type}")
print(f"{'='*60}")

by_steer = {}
for r in results:
    steer = r["steering_condition"]
    by_steer.setdefault(steer, []).append(r)

for steer in sorted(by_steer.keys(), key=lambda x: (0 if x=='baseline' else 1, x)):
    rs = by_steer[steer]
    scores = []
    for r in rs:
        try:
            resp = r["response"]
            if '"score"' in resp:
                import re
                match = re.search(r'"score"\s*:\s*(\d+)', resp)
                if match:
                    scores.append(int(match.group(1)))
        except:
            pass
    if scores:
        print(f"{steer:25} score={np.mean(scores):.2f} ± {np.std(scores):.2f} (n={len(scores)})")

print("Done!")
EOF
