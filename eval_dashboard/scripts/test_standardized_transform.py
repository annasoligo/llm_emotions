#!/usr/bin/env python3
"""Quick test to verify StandardizedTransformer produces correct z-scores."""

import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnterp import StandardizedTransformer

from emotion_logit_lens import (
    project_to_logits_batched,
    compute_emotion_scores_batched
)
from emotion_logit_lens.baseline_loader import LogitBaselineLoader
from emotion_logit_lens.emotion_tokens import EmotionTokenManager

print("="*80)
print("TESTING: StandardizedTransformer + Baseline Normalization")
print("="*80)

# Load model
print("\n[1/4] Loading model with StandardizedTransformer...")
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-27b-it')
raw_model = AutoModelForCausalLM.from_pretrained(
    'google/gemma-3-27b-it',
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = StandardizedTransformer(raw_model)
print(f"  ✓ Model loaded")

# Load resources
print("\n[2/4] Loading emotion tokens and baselines...")
emotion_mgr = EmotionTokenManager(model_name='google_gemma_3_27b_it')
emotion_token_ids = emotion_mgr.load_emotion_token_ids()

baseline_loader = LogitBaselineLoader(
    baseline_dir=Path('/workspace-vast/annas/git/research-tools/data/baselines/logit_emotion_alpaca'),
    model_name='google_gemma_3_27b_it'
)
print(f"  ✓ Loaded resources")

# Test on real text
print("\n[3/4] Testing on sample text...")
test_text = "I am so angry and frustrated!"
inputs = tokenizer(test_text, return_tensors="pt", add_special_tokens=False)
with torch.no_grad():
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.model(**inputs, output_hidden_states=True)

# Get activations at layer 30, all tokens
layer = 30
activations = outputs.hidden_states[layer][0].float().cpu().numpy()  # (n_tokens, hidden_dim)
print(f"  Extracted {len(activations)} tokens from layer {layer}")

# Project to logits
print("\n[4/4] Computing emotion scores...")
logits_batch = project_to_logits_batched(model=model, hidden_state_vectors=activations)

# Load baseline
layer_stats = baseline_loader.load_layer_stats(layer)
baseline_stats = {
    'layers_data': {
        str(layer): {
            'statistics': layer_stats
        }
    }
}

# Compute scores WITH baseline
scores_batch = compute_emotion_scores_batched(
    logits_batch=logits_batch,
    emotion_token_ids=emotion_token_ids,
    baseline_stats=baseline_stats,
    layer=layer,
    aggregation='mean'
)

print(f"\n{'='*80}")
print("RESULTS:")
print(f"{'='*80}")
print(f"\nToken 0 scores: {scores_batch[0]}")
print(f"Token 5 scores: {scores_batch[5]}")

# Check if z-scores
all_scores = []
for score_dict in scores_batch:
    all_scores.extend(score_dict.values())

min_score = min(all_scores)
max_score = max(all_scores)
mean_score = np.mean(all_scores)

print(f"\nAll scores range: [{min_score:.2f}, {max_score:.2f}]")
print(f"Mean score: {mean_score:.2f}")

print(f"\n{'='*80}")
if min_score < 0 and abs(mean_score) < 2:
    print("✓ SUCCESS: Scores are properly z-normalized!")
    print("  - Has negative values ✓")
    print("  - Mean near 0 ✓")
    print("  - Range reasonable for z-scores ✓")
else:
    print("❌ FAILURE: Scores are NOT properly normalized")
    print(f"  - Min: {min_score:.2f} (should be negative)")
    print(f"  - Mean: {mean_score:.2f} (should be near 0)")
print(f"{'='*80}")
