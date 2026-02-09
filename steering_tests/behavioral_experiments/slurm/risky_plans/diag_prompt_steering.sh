#!/bin/bash
#SBATCH --job-name=diag_prompt
#SBATCH --output=/workspace-vast/annas/logs/diag_prompt_%j.out
#SBATCH --error=/workspace-vast/annas/logs/diag_prompt_%j.out
#SBATCH --time=0:20:00
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0

cleanup() {
    pkill -9 -f "vllm" || true
    pkill -9 -f "ray" || true
}
trap cleanup EXIT SIGTERM SIGINT

python3 -c "
import numpy as np
import pickle, torch, hashlib
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from steering_tests.steering_utils import MultiLayerVLLMSteering
from steering_tests.steering_utils.layer_norms import get_layer_norm, resolve_model_key
from steering_tests.steering_utils.core import _find_target_layer
from steering_tests.behavioral_experiments.risky_plans_prompt_steering import find_prompt_section_masks, EXTRA_MASK_KEY
from steering_tests.behavioral_experiments.scenarios.risky_plans import RISKY_PLANS_PROMPT
from steering_tests.behavioral_experiments.config import MODEL_CONFIGS

model_name = 'google/gemma-3-12b-it'
layers = [22, 23, 24, 25, 26]
config = MODEL_CONFIGS[model_name]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
messages = [{'role': 'user', 'content': RISKY_PLANS_PROMPT}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
plan_a_mask, plan_b_mask, prompt_len = find_prompt_section_masks(tokenizer, messages)

print(f'Plan A mask: {plan_a_mask.sum():.0f} tokens, Plan B mask: {plan_b_mask.sum():.0f} tokens, prompt_len: {prompt_len}')

llm = LLM(model=model_name, trust_remote_code=True, dtype='bfloat16',
          tensor_parallel_size=1, enforce_eager=True, disable_log_stats=True,
          gpu_memory_utilization=0.80, max_model_len=8192, enable_prefix_caching=False)

vec_base = Path('steering_tests/vectors/gemma12b/text_pairs_emotion_vs_opposite/layers')
with open(vec_base / 'layer_22.pkl', 'rb') as f:
    vecs = pickle.load(f)
fear_vec = np.asarray(vecs['fear'], dtype=np.float32)
excite_vec = np.asarray(vecs['excitement'], dtype=np.float32)

model_key = resolve_model_key(model_name)
layer_norms = {l: get_layer_norm(model_key, l) for l in layers}

steering = MultiLayerVLLMSteering(llm, layers=layers, layer_norms=layer_norms)
steering.load_vector('fear', fear_vec)
steering.load_vector('excitement', excite_vec)

sp = SamplingParams(temperature=0.0, max_tokens=100, stop=config['stop_tokens'])

# === TEST 1: Baseline ===
print('\n' + '='*70)
print('TEST 1: BASELINE')
steering.clear()
steering.enable_token_tracking()
out = llm.generate([prompt], sp)
resp = out[0].outputs[0].text
h = hashlib.md5(resp.encode()).hexdigest()[:10]
print(f'  hash={h}  first_100={resp[:100]}')

# === TEST 2: 2000% prompt steering ===
print('\n' + '='*70)
print('TEST 2: 2000% PROMPT STEERING (steer_prompt=True, steer_generation=False)')
steering.enable_token_tracking()
steering.set('fear', scale=20.0, direction=1, steer_prompt=True, steer_generation=False, prompt_position_mask=plan_a_mask)
steering.add_extra_steer('excitement', scale=20.0, direction=1, trigger_mask_key=EXTRA_MASK_KEY, prompt_position_mask=plan_b_mask)
out = llm.generate([prompt], sp)
resp = out[0].outputs[0].text
h = hashlib.md5(resp.encode()).hexdigest()[:10]
print(f'  hash={h}  first_100={resp[:100]}')

# === TEST 3: 2000% generation steering (should destroy output) ===
print('\n' + '='*70)
print('TEST 3: 2000% GENERATION STEERING (steer_prompt=False, steer_generation=True)')
steering.enable_token_tracking()
steering.set('fear', scale=20.0, direction=1, steer_prompt=False, steer_generation=True)
out = llm.generate([prompt], sp)
resp = out[0].outputs[0].text
h = hashlib.md5(resp.encode()).hexdigest()[:10]
print(f'  hash={h}  first_100={resp[:100]}')

# === TEST 4: 2000% steer BOTH prompt and generation ===
print('\n' + '='*70)
print('TEST 4: 2000% BOTH (steer_prompt=True, steer_generation=True)')
steering.enable_token_tracking()
steering.set('fear', scale=20.0, direction=1, steer_prompt=True, steer_generation=True, prompt_position_mask=plan_a_mask)
out = llm.generate([prompt], sp)
resp = out[0].outputs[0].text
h = hashlib.md5(resp.encode()).hexdigest()[:10]
print(f'  hash={h}  first_100={resp[:100]}')

# === TEST 5: 2000% steer ALL positions during prompt (no mask) ===
print('\n' + '='*70)
print('TEST 5: 2000% PROMPT ALL POSITIONS (no mask, steer_prompt=True, steer_generation=False)')
steering.enable_token_tracking()
steering.set('fear', scale=20.0, direction=1, steer_prompt=True, steer_generation=False)
out = llm.generate([prompt], sp)
resp = out[0].outputs[0].text
h = hashlib.md5(resp.encode()).hexdigest()[:10]
print(f'  hash={h}  first_100={resp[:100]}')

# === DIAGNOSTIC: dump state ===
print('\n' + '='*70)
print('LAYER STATE AFTER TEST 5:')
class DiagCallable:
    def __init__(self, layer_idx):
        self.layer_idx = layer_idx
    def __call__(self, model):
        layer = _find_target_layer(model, self.layer_idx)
        state = layer._steering_state
        vec = state.get('vector_tensor')
        scale = state.get('scale', 0.0)
        mask = state.get('prompt_position_mask')
        shared = state.get('_steering_shared_state', {})
        is_prefill = shared.get('is_prefill', 'NOT SET')
        has_handle = hasattr(layer, '_steering_handle')
        extra = state.get('extra_steers', [])
        steer_p = state.get('steer_prompt', 'MISSING')
        steer_g = state.get('steer_generation', 'MISSING')
        print(f'  L{self.layer_idx}: scale={scale:.2f} vec={vec is not None} mask={mask is not None} steer_prompt={steer_p} steer_gen={steer_g} hook_handle={has_handle} is_prefill={is_prefill} extras={len(extra)}')
        return 'ok'

for l in layers:
    llm.apply_model(DiagCallable(l))
"
