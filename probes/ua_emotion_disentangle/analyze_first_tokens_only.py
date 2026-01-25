#!/usr/bin/env python3
"""Analyze projections at first assistant tokens only."""

import sys
sys.path.append('..')

import torch
import numpy as np
from pathlib import Path
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnterp import StandardizedTransformer
from measure_emotion_decay import load_emotion_probes, collect_token_activations, project_onto_probes

# Load data
results_dir = Path("emotion_decay_results")
with open(results_dir / "conversations.json") as f:
    data = json.load(f)

params = data["parameters"]
tracked_emotions = params["tracked_emotions"]

# Load probes
print("Loading emotion probes...")
probe_path = Path("full_analysis/probes_first_asst_token_alternating.npz")
m_probes, u_probes = load_emotion_probes(probe_path)
print(f"✓ Loaded {len(m_probes)} M probes, {len(u_probes)} U probes")

# Load model
print("Loading model...")
model_name = "unsloth/gemma-3-27b-it"
model_raw = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)

model = StandardizedTransformer(
    model_raw,
    trust_remote_code=True,
    check_renaming=False,
    allow_dispatch=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Analyze neutral conversation
print("\n" + "=" * 80)
print("NEUTRAL CONVERSATION")
print("=" * 80)

neutral_text = data["neutral"]["text"]
neutral_tokens = tokenizer.encode(neutral_text)

# Find first assistant token positions
first_asst_positions = []
for i in range(len(neutral_tokens) - 1):
    if tokenizer.decode([neutral_tokens[i]]) == '<start_of_turn>' and \
       tokenizer.decode([neutral_tokens[i+1]]) == 'model':
        first_asst_positions.append(i + 1)
        print(f"First assistant token at position {i+1}")

# Collect activations
activations = collect_token_activations(model, tokenizer, neutral_tokens, layer=30)

# Project
u_proj = project_onto_probes(activations, u_probes, tracked_emotions)
m_proj = project_onto_probes(activations, m_probes, tracked_emotions)

print("\nProjections at first assistant tokens:")
for pos in first_asst_positions:
    print(f"\nPosition {pos}:")
    print("  U projections:", {k: v[pos] for k, v in u_proj.items()})
    print("  M projections:", {k: v[pos] for k, v in m_proj.items()})

# Analyze emotion shift conversation
print("\n" + "=" * 80)
print("EMOTION SHIFT CONVERSATION")
print("=" * 80)

shift_text = data["emotion_shift"]["text"]
shift_tokens = tokenizer.encode(shift_text)

# Find first assistant token positions
first_asst_positions = []
for i in range(len(shift_tokens) - 1):
    if tokenizer.decode([shift_tokens[i]]) == '<start_of_turn>' and \
       tokenizer.decode([shift_tokens[i+1]]) == 'model':
        first_asst_positions.append(i + 1)
        print(f"First assistant token at position {i+1}")

# Collect activations
activations = collect_token_activations(model, tokenizer, shift_tokens, layer=30)

# Project
u_proj = project_onto_probes(activations, u_probes, tracked_emotions)
m_proj = project_onto_probes(activations, m_probes, tracked_emotions)

print("\nProjections at first assistant tokens:")
for pos in first_asst_positions:
    print(f"\nPosition {pos}:")
    print("  U projections:", {k: v[pos] for k, v in u_proj.items()})
    print("  M projections:", {k: v[pos] for k, v in m_proj.items()})

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print("This shows what the probes were actually trained for:")
print("  - Detecting emotion at the FIRST token of assistant responses")
print("  - NOT for tracking emotion throughout a response")
