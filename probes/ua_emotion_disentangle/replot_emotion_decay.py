#!/usr/bin/env python3
"""
Replot emotion decay with existing activations and projections.
"""

import sys
sys.path.append('..')

from measure_emotion_decay import (
    load_emotion_probes,
    collect_token_activations,
    project_onto_probes,
    plot_emotion_decay,
    generate_conversation
)

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnterp import StandardizedTransformer

# Load configuration from existing results
results_dir = Path("emotion_decay_results")
with open(results_dir / "conversations.json") as f:
    data = json.load(f)

params = data["parameters"]
u_emotion = params["u_emotion"]
m_emotion = params["m_emotion"]
u2_emotion = params["u2_emotion"]
tracked_emotions = params["tracked_emotions"]

print("Loading emotion probes...")
probe_path = Path("full_analysis/probes_first_asst_token_orthogonal.npz")
m_probes, u_probes = load_emotion_probes(probe_path)
print(f"✓ Loaded {len(m_probes)} M probes, {len(u_probes)} U probes")

print("\nLoading model...")
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
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("✓ Model loaded")

# Regenerate conversations to get tokens
print("\nRegenerating conversations...")
print("  [1/2] Neutral follow-up...")
full_text_neutral, tokens_neutral = generate_conversation(
    model, tokenizer, u_emotion, m_emotion,
    user_msg="Can you help me out?",
    follow_up="What else?",
    u2_emotion=None
)

print(f"  [2/2] Follow-up with '{u2_emotion}' shift...")
full_text_shift, tokens_shift = generate_conversation(
    model, tokenizer, u_emotion, m_emotion,
    user_msg="Can you help me out?",
    follow_up="What else?",
    u2_emotion=u2_emotion
)

# Recollect activations
print("\nRecollecting activations...")
print("  [1/2] Neutral conversation...")
activations_neutral = collect_token_activations(model, tokenizer, tokens_neutral, layer=30)

print("  [2/2] Emotion shift conversation...")
activations_shift = collect_token_activations(model, tokenizer, tokens_shift, layer=30)

# Recompute projections
print("\nRecomputing projections...")
u_proj_neutral = project_onto_probes(activations_neutral, u_probes, tracked_emotions)
m_proj_neutral = project_onto_probes(activations_neutral, m_probes, tracked_emotions)

u_proj_shift = project_onto_probes(activations_shift, u_probes, tracked_emotions)
m_proj_shift = project_onto_probes(activations_shift, m_probes, tracked_emotions)

# Replot
print("\nReplotting...")
plot_emotion_decay(
    u_proj_neutral, m_proj_neutral,
    tokens_neutral, tokenizer,
    u_emotion, m_emotion, None,
    results_dir / "emotion_decay_neutral.png"
)

plot_emotion_decay(
    u_proj_shift, m_proj_shift,
    tokens_shift, tokenizer,
    u_emotion, m_emotion, u2_emotion,
    results_dir / "emotion_decay_shift.png"
)

print("\n✓ Replotting complete!")
