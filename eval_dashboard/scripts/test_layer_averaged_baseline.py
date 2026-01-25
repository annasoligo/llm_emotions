#!/usr/bin/env python3
"""Test emo_lens approach with layer-averaged WildChat baseline."""

import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')
sys.path.insert(0, '/workspace-vast/annas/git/believe-it-or-not')

import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnterp import StandardizedTransformer

print("="*80)
print("TESTING LAYER-AVERAGED WILDCAT BASELINE APPROACH")
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
print("  ✓ Model loaded")

# Load emotion token IDs (emo_lens version)
print("\n[2/5] Loading emotion token IDs...")
import json
emo_token_file = Path('/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens/emotion_word_sets/all_emotion_words_unsloth_gemma_3_27b_it_token_ids.json')
emotion_token_ids = json.load(open(emo_token_file))
print(f"  ✓ Loaded {len(emotion_token_ids)} emotions")

# Load WildChat baseline and compute layer-averaged stats
print("\n[3/5] Loading WildChat baseline...")
from emotion_evals.emo_lens.logit_lens_emotion_direct import load_reference_stats_for_strategy
from emotion_evals.emo_lens.token_trajectories import compute_layer_averaged_baseline_stats

SCRIPT_DIR = Path("/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens")
LAYER_RANGE = list(range(40, 51))  # Layers 40-50

ref_stats = load_reference_stats_for_strategy(
    model_name="unsloth/gemma-3-27b-it",
    activation_strategy="generated_tokens_avg",
    script_dir=SCRIPT_DIR
)

averaged_baseline_stats = compute_layer_averaged_baseline_stats(
    ref_stats=ref_stats,
    layers=LAYER_RANGE,
    emotion_token_ids=emotion_token_ids
)
print(f"  ✓ Averaged baseline across layers {LAYER_RANGE[0]}-{LAYER_RANGE[-1]}")

# Test on multiple prompts
prompts = [
    "I am so angry and frustrated!",
    "This is absolutely disgusting and vile.",
    "I'm terrified and scared of what might happen.",
    "I feel incredibly happy and joyful today!",
    "I'm deeply sad and heartbroken.",
    "Wow, that's so surprising and unexpected!"
]

print("\n[4/5] Testing on prompts...")
from emotion_evals.emo_lens.token_trajectories import (
    extract_token_level_activations,
    compute_token_emotion_scores
)

all_scores = []
for prompt in prompts:
    activations_by_token, token_ids = extract_token_level_activations(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        layers=LAYER_RANGE,
        start_token_idx=0,
        system_prompt=None,
        num_generated_tokens=0
    )

    token_emotion_scores = compute_token_emotion_scores(
        model=model,
        activations_by_token=activations_by_token,
        layers=LAYER_RANGE,
        emotion_token_ids=emotion_token_ids,
        averaged_baseline_stats=averaged_baseline_stats,
        aggregation="mean",
        subtract_mean=True
    )

    # Get average scores across all tokens for this prompt
    emotions = list(emotion_token_ids.keys())
    prompt_avg_scores = {e: [] for e in emotions}
    for pos, scores in token_emotion_scores.items():
        for emotion in emotions:
            prompt_avg_scores[emotion].append(scores[emotion])

    prompt_avg = [np.mean(prompt_avg_scores[e]) for e in emotions]
    all_scores.append(prompt_avg)

all_scores = np.array(all_scores)  # (n_prompts, n_emotions)

print(f"  ✓ Processed {len(prompts)} prompts")

# Compute correlation
print("\n[5/5] Computing correlation...")
corr_matrix = np.corrcoef(all_scores.T)
mask = ~np.eye(len(emotions), dtype=bool)
avg_corr = corr_matrix[mask].mean()

print("\n" + "="*80)
print("RESULTS WITH LAYER-AVERAGED WILDCAT BASELINE")
print("="*80)

print(f"\nAverage emotion correlation: {avg_corr:.3f}")

print("\nCorrelation matrix:")
emotions = list(emotion_token_ids.keys())
print("      ", end="")
for em in emotions:
    print(f"{em[:6]:>7}", end="")
print()

for i, em1 in enumerate(emotions):
    print(f"{em1[:6]:>6}", end="")
    for j, em2 in enumerate(emotions):
        if i == j:
            print(f"   1.00", end="")
        else:
            print(f"  {corr_matrix[i,j]:>5.2f}", end="")
    print()

print(f"\n{'='*80}")
if avg_corr < 0.3:
    print("✓ LOW CORRELATION - Layer averaging helps!")
elif avg_corr < 0.5:
    print("~ MODERATE CORRELATION - Some improvement")
else:
    print("❌ HIGH CORRELATION - Still correlated")
print("="*80)
