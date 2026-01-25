#!/usr/bin/env python3
"""Debug the extreme logit lens values."""

import json
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer, AutoModelForCausalLM
from emotion_logit_lens.core import project_to_logits

# Load one divergent case
df = pd.read_csv(PROJECT_ROOT / "experiments/distress_analysis/divergent_scores_5k.csv")
row = df.iloc[0]  # First divergent case

print(f"Analyzing: conv_id={row['conversation_id']}, type={row['divergence_type']}")
print(f"User msg (first 100 chars): {row['user_message'][:100]}...")

# Load model
print("\nLoading model...")
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-27b-it")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/gemma-3-27b-it",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)
model.eval()

# Load emotion tokens
with open(PROJECT_ROOT / "emotion_logit_lens/emotion_token_ids/all_emotion_words_google_gemma_3_27b_it_token_ids.json") as f:
    emotion_tokens = json.load(f)

# Load baseline stats for comparison
def load_baseline(layer):
    with open(PROJECT_ROOT / f"data/baselines/logit_emotion_alpaca/google_gemma_3_27b_it/layer_{layer}.json") as f:
        return json.load(f)['statistics']

# Extract activations at a few layers
messages = [
    {"role": "user", "content": row['user_message']},
    {"role": "assistant", "content": row['assistant_response']}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(input_ids=inputs["input_ids"], output_hidden_states=True)

# Also check the Alpaca baseline activations
import h5py
baseline_act_dir = PROJECT_ROOT / "data/baselines/alpaca_gemma27b_v2/google_gemma_3_27b_it"
print("\n" + "="*60)
print("Comparing activation scales:")
print("="*60)
for layer in [30, 50, 60]:
    with h5py.File(baseline_act_dir / f"layer{layer}_activations.h5", 'r') as f:
        baseline_acts = f['all_tokens'][:]
    hidden = outputs.hidden_states[layer + 1]
    wildchat_act = hidden[0].mean(dim=0).float().cpu().numpy()
    print(f"\nLayer {layer}:")
    print(f"  Baseline (Alpaca) activations: mean={baseline_acts.mean():.2f}, std={baseline_acts.std():.2f}, range=[{baseline_acts.min():.0f}, {baseline_acts.max():.0f}]")
    print(f"  WildChat activations: mean={wildchat_act.mean():.2f}, std={wildchat_act.std():.2f}, range=[{wildchat_act.min():.0f}, {wildchat_act.max():.0f}]")

# Check layers 30, 50, 60
for layer in [30, 50, 60]:
    print(f"\n{'='*60}")
    print(f"Layer {layer}")
    print(f"{'='*60}")

    # Get mean activation over response tokens
    hidden = outputs.hidden_states[layer + 1]
    mean_act = hidden[0].mean(dim=0).float().cpu().numpy()

    # Project to logits
    logits = project_to_logits(model, mean_act).float().cpu().numpy()

    # Load baseline
    baseline = load_baseline(layer)

    # Check anger tokens
    print("\nAnger token analysis:")
    anger_ids = emotion_tokens['anger'][:10]  # First 10
    for tid in anger_ids:
        raw = float(logits[tid])
        mean = baseline[str(tid)]['mean']
        std = baseline[str(tid)]['std']
        z = (raw - mean) / (std + 1e-8)
        token_str = tokenizer.decode([tid])
        print(f"  Token '{token_str}' ({tid}): raw={raw:.2f}, baseline_mean={mean:.2f}, std={std:.4f}, z={z:.1f}")

    # Overall stats for anger
    all_z = []
    for tid in emotion_tokens['anger']:
        raw = float(logits[tid])
        mean = baseline[str(tid)]['mean']
        std = baseline[str(tid)]['std']
        z = (raw - mean) / (std + 1e-8)
        all_z.append(z)
    print(f"\nAnger overall: mean_z={np.mean(all_z):.1f}, min_z={np.min(all_z):.1f}, max_z={np.max(all_z):.1f}")
