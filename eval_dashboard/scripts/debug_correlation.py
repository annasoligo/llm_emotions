#!/usr/bin/env python3
"""Debug why emotions are highly correlated."""

import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnterp import StandardizedTransformer

from emotion_logit_lens import (
    project_to_logits_batched,
    EmotionTokenManager
)
from emotion_logit_lens.baseline_loader import LogitBaselineLoader

print("="*80)
print("DEBUGGING EMOTION CORRELATION")
print("="*80)

# Load model
print("\n[1/5] Loading model...")
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-27b-it')
raw_model = AutoModelForCausalLM.from_pretrained(
    'google/gemma-3-27b-it',
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = StandardizedTransformer(raw_model)
print(f"  ✓ Model loaded")

# Load resources
print("\n[2/5] Loading emotion tokens...")
emotion_mgr = EmotionTokenManager(model_name='google_gemma_3_27b_it')
emotion_token_ids = emotion_mgr.load_emotion_token_ids()

# Check for overlaps
print("\n[3/5] Checking for token ID overlaps between emotions...")
emotions = list(emotion_token_ids.keys())
total_overlaps = 0
for i, em1 in enumerate(emotions):
    for em2 in emotions[i+1:]:
        set1 = set(emotion_token_ids[em1])
        set2 = set(emotion_token_ids[em2])
        overlap = set1 & set2
        if overlap:
            print(f"  {em1} ∩ {em2}: {len(overlap)} shared tokens")
            total_overlaps += len(overlap)

if total_overlaps == 0:
    print("  ✓ No overlaps found between emotion token sets")
else:
    print(f"  ⚠ Found {total_overlaps} total overlapping tokens")

# Test on real text
print("\n[4/5] Testing on sample text...")
test_text = "I am so angry and frustrated!"
inputs = tokenizer(test_text, return_tensors="pt", add_special_tokens=False)
with torch.no_grad():
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.model(**inputs, output_hidden_states=True)

# Get activations at layer 30, first token
layer = 30
activation = outputs.hidden_states[layer][0][0].float().cpu().numpy()  # First token
print(f"  Extracted activation from layer {layer}, token 0")

# Project to logits
logits = project_to_logits_batched(model=model, hidden_state_vectors=activation.reshape(1, -1))[0]
print(f"  Projected to logits: shape {logits.shape}")

# Load baseline
baseline_loader = LogitBaselineLoader(
    baseline_dir=Path('/workspace-vast/annas/git/research-tools/data/baselines/logit_emotion_alpaca'),
    model_name='google_gemma_3_27b_it'
)
layer_stats = baseline_loader.load_layer_stats(layer)

print("\n[5/5] Analyzing emotion token logits...")
print("="*80)

# For each emotion, extract raw logits and compute stats
emotion_raw_logits = {}
emotion_z_scores = {}

for emotion in emotions:
    token_ids = emotion_token_ids[emotion]

    # Extract raw logits for this emotion's tokens
    raw_logits = []
    z_scores = []

    for token_id in token_ids[:10]:  # First 10 tokens of each emotion
        raw_logit = float(logits[token_id])
        raw_logits.append(raw_logit)

        # Get baseline and compute z-score
        token_key = str(token_id)
        if token_key in layer_stats:
            mean = layer_stats[token_key]['mean']
            std = layer_stats[token_key]['std']
            z = (raw_logit - mean) / std if std > 1e-8 else (raw_logit - mean)
            z_scores.append(z)

    emotion_raw_logits[emotion] = raw_logits
    emotion_z_scores[emotion] = z_scores

    print(f"\n{emotion.upper()}:")
    print(f"  First 10 token IDs: {token_ids[:10]}")
    print(f"  Raw logits: [{raw_logits[0]:.2f}, {raw_logits[1]:.2f}, {raw_logits[2]:.2f}, ...]")
    print(f"  Z-scores:   [{z_scores[0]:.2f}, {z_scores[1]:.2f}, {z_scores[2]:.2f}, ...]")
    print(f"  Mean raw logit: {np.mean(raw_logits):.3f}")
    print(f"  Mean z-score: {np.mean(z_scores):.3f}")

# Check if RAW logits are correlated (before normalization)
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Compute correlation of raw logits across ALL emotion tokens
all_token_ids = []
all_raw_logits = []
emotion_labels = []

for emotion in emotions:
    token_ids = emotion_token_ids[emotion]
    for token_id in token_ids[:20]:  # Use first 20 tokens per emotion
        all_token_ids.append(token_id)
        all_raw_logits.append(float(logits[token_id]))
        emotion_labels.append(emotion)

# Group by emotion and compute correlation
emotion_vectors = {}
for emotion in emotions:
    emotion_vectors[emotion] = [
        all_raw_logits[i] for i, label in enumerate(emotion_labels) if label == emotion
    ]

print("\nRaw logit means per emotion (should vary if emotions are distinct):")
for emotion in emotions:
    print(f"  {emotion}: {np.mean(emotion_vectors[emotion]):.3f}")

# Check variance within vs between emotions
all_logits_array = np.array(all_raw_logits)
print(f"\nOverall raw logit stats:")
print(f"  Min: {all_logits_array.min():.3f}")
print(f"  Max: {all_logits_array.max():.3f}")
print(f"  Mean: {all_logits_array.mean():.3f}")
print(f"  Std: {all_logits_array.std():.3f}")

# Key insight: check if emotion tokens have similar baselines
print("\n" + "="*80)
print("BASELINE ANALYSIS (are emotion token baselines similar?)")
print("="*80)

emotion_baseline_means = {}
for emotion in emotions:
    token_ids = emotion_token_ids[emotion]
    baseline_means = []

    for token_id in token_ids[:50]:  # Sample more tokens
        token_key = str(token_id)
        if token_key in layer_stats:
            baseline_means.append(layer_stats[token_key]['mean'])

    emotion_baseline_means[emotion] = baseline_means
    print(f"{emotion}: baseline mean = {np.mean(baseline_means):.3f} ± {np.std(baseline_means):.3f}")

# If baseline means are very similar across emotions, that could explain correlation
between_emotion_var = np.var([np.mean(emotion_baseline_means[e]) for e in emotions])
print(f"\nVariance of baseline means BETWEEN emotions: {between_emotion_var:.4f}")
print("  (Low variance = emotions have similar baseline logits in Alpaca data)")
