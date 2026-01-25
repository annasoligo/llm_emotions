#!/usr/bin/env python3
"""Directly compare research-tools vs emo_lens emotion scoring."""

import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')
sys.path.insert(0, '/workspace-vast/annas/git/believe-it-or-not')

import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnterp import StandardizedTransformer

print("="*80)
print("COMPARING RESEARCH-TOOLS vs EMO_LENS")
print("="*80)

# Load model
print("\n[1/4] Loading model...")
tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-27b-it')
raw_model = AutoModelForCausalLM.from_pretrained(
    'google/gemma-3-27b-it',
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = StandardizedTransformer(raw_model)
print("  ✓ Model loaded")

# Test text
test_text = "I am so angry and frustrated!"
inputs = tokenizer(test_text, return_tensors="pt", add_special_tokens=False)
with torch.no_grad():
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.model(**inputs, output_hidden_states=True)

layer = 30
activations = outputs.hidden_states[layer][0].float().cpu().numpy()  # (n_tokens, hidden_dim)
print(f"\n[2/4] Extracted {len(activations)} tokens from layer {layer}")

# ============================================================================
# METHOD 1: research-tools implementation
# ============================================================================
print("\n[3/4] Computing scores with research-tools...")

from emotion_logit_lens import (
    project_to_logits_batched,
    compute_emotion_scores_batched,
    EmotionTokenManager
)
from emotion_logit_lens.baseline_loader import LogitBaselineLoader

emotion_mgr = EmotionTokenManager(model_name='google_gemma_3_27b_it')
emotion_token_ids = emotion_mgr.load_emotion_token_ids()

baseline_loader = LogitBaselineLoader(
    baseline_dir=Path('/workspace-vast/annas/git/research-tools/data/baselines/logit_emotion_alpaca'),
    model_name='google_gemma_3_27b_it'
)
layer_stats = baseline_loader.load_layer_stats(layer)
baseline_stats = {
    'layers_data': {
        str(layer): {
            'statistics': layer_stats
        }
    }
}

logits_batch_rt = project_to_logits_batched(model=model, hidden_state_vectors=activations)
scores_batch_rt = compute_emotion_scores_batched(
    logits_batch=logits_batch_rt,
    emotion_token_ids=emotion_token_ids,
    baseline_stats=baseline_stats,
    layer=layer,
    aggregation='mean'
)

print("\nRESEARCH-TOOLS scores (token 0):")
for emotion, score in scores_batch_rt[0].items():
    print(f"  {emotion}: {score:.4f}")

# ============================================================================
# METHOD 2: emo_lens implementation
# ============================================================================
print("\n[4/4] Computing scores with emo_lens...")

from emotion_evals.emo_lens.model_utils import project_to_logits_batched as emo_project
from emotion_evals.emo_lens.logit_lens_emotion_direct import normalize_logits_to_emotion_scores_batched

# Load emo_lens token IDs
import json
emo_token_file = Path('/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens/emotion_word_sets/all_emotion_words_unsloth_gemma_3_27b_it_token_ids.json')
emo_token_ids = json.load(open(emo_token_file))

# Use same Alpaca baseline (load directly with same method as research-tools)
emo_baseline_stats = {
    'layers_data': {
        str(layer): {
            'statistics': layer_stats  # Use same baseline as research-tools
        }
    }
}

logits_batch_emo = emo_project(model=model, hidden_state_vectors=activations)
scores_batch_emo = normalize_logits_to_emotion_scores_batched(
    logits_batch=logits_batch_emo,
    emotion_token_ids=emo_token_ids,
    layer=layer,
    ref_stats=emo_baseline_stats,
    subtract_mean=True,
    aggregation='mean'
)

print("\nEMO_LENS scores (token 0):")
for emotion, score in scores_batch_emo[0].items():
    print(f"  {emotion}: {score:.4f}")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print("\nDifferences (research-tools - emo_lens):")
for emotion in scores_batch_rt[0].keys():
    diff = scores_batch_rt[0][emotion] - scores_batch_emo[0][emotion]
    print(f"  {emotion}: {diff:.6f}")

# Check logits match
logit_diff = (logits_batch_rt - logits_batch_emo).abs().max()
print(f"\nMax logit difference: {logit_diff:.6f}")

# Compute correlation for each method across all tokens
rt_scores_matrix = []
emo_scores_matrix = []
emotions_list = list(scores_batch_rt[0].keys())

for i in range(len(scores_batch_rt)):
    rt_scores_matrix.append([scores_batch_rt[i][e] for e in emotions_list])
    emo_scores_matrix.append([scores_batch_emo[i][e] for e in emotions_list])

rt_scores_matrix = np.array(rt_scores_matrix)
emo_scores_matrix = np.array(emo_scores_matrix)

rt_corr = np.corrcoef(rt_scores_matrix.T)
emo_corr = np.corrcoef(emo_scores_matrix.T)

mask = ~np.eye(len(emotions_list), dtype=bool)
rt_avg_corr = rt_corr[mask].mean()
emo_avg_corr = emo_corr[mask].mean()

print(f"\nResearch-tools avg correlation: {rt_avg_corr:.3f}")
print(f"Emo_lens avg correlation: {emo_avg_corr:.3f}")

print("\n" + "="*80)
if abs(rt_avg_corr - emo_avg_corr) > 0.1:
    print("❌ CORRELATION MISMATCH - There is a bug!")
else:
    print("✓ Correlations match - both implementations behave the same")
print("="*80)
