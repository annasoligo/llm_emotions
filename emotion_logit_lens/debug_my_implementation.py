"""
Debug script: Run MY implementation using the SAME inputs as emo_lens test.
This will show if the bug is in my functions or in how I'm calling them.
"""
import sys
sys.path.insert(0, '/workspace-vast/annas/git/believe-it-or-not')
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

import numpy as np
from pathlib import Path

# Load model using emo_lens function
from emotion_evals.emo_lens.model_utils import load_base_model
from emotion_evals.emo_lens.logit_lens_emotion_direct import (
    load_emotion_token_ids_from_json,
    load_reference_stats_for_strategy
)
from emotion_evals.emo_lens.token_trajectories import extract_token_level_activations

# Use MY layer_averaging functions
from emotion_logit_lens.layer_averaging import (
    compute_layer_averaged_baseline_stats,
    compute_token_emotion_scores
)

print("="*80)
print("DEBUGGING MY IMPLEMENTATION")
print("="*80)

# Config (SAME as emo_lens test)
BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
ACTIVATION_STRATEGY = "generated_tokens_avg"
LAYER_RANGE = list(range(40, 51))
PROMPT = "I hate you so much! You make me sick!"
emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# Load model (SAME way as emo_lens)
print("\n[1/4] Loading model...")
model, tokenizer = load_base_model(BASE_MODEL_NAME)
print("✓ Model loaded")

# Load emotion token IDs (SAME)
print("\n[2/4] Loading emotion token IDs...")
emotion_token_ids = load_emotion_token_ids_from_json(BASE_MODEL_NAME)
print(f"✓ Loaded {len(emotion_token_ids)} emotions")

# Load reference statistics (SAME)
print(f"\n[3/4] Loading reference statistics ({ACTIVATION_STRATEGY})...")
SCRIPT_DIR = Path("/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens")
ref_stats = load_reference_stats_for_strategy(
    model_name=BASE_MODEL_NAME,
    activation_strategy=ACTIVATION_STRATEGY,
    script_dir=SCRIPT_DIR
)
print("✓ Loaded baseline stats")

# Compute layer-averaged baseline stats using MY function
print(f"\nAveraging baseline stats across layers {LAYER_RANGE[0]}-{LAYER_RANGE[-1]}...")
print("[DEBUG] Using MY compute_layer_averaged_baseline_stats...")
averaged_baseline_stats = compute_layer_averaged_baseline_stats(
    ref_stats=ref_stats,
    layers=LAYER_RANGE,
    emotion_token_ids=emotion_token_ids
)
print("✓ Computed averaged baseline statistics")

# Check what baseline stats look like
print("\n[DEBUG] Checking averaged baseline stats...")
for emotion in ['anger', 'disgust'][:2]:
    sample_tokens = list(emotion_token_ids[emotion])[:2]
    for token_id in sample_tokens:
        if token_id in averaged_baseline_stats.get(emotion, {}):
            stats = averaged_baseline_stats[emotion][token_id]
            print(f"  {emotion}/{token_id}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

# Extract activations (SAME way as emo_lens)
print(f"\n[4/4] Processing prompt: {PROMPT}")
activations_by_token, token_ids = extract_token_level_activations(
    model=model,
    tokenizer=tokenizer,
    prompt=PROMPT,
    layers=LAYER_RANGE,
    start_token_idx=0,
    num_generated_tokens=0
)
token_strings = [tokenizer.decode([tid]) for tid in token_ids]
print(f"✓ Extracted activations for {len(activations_by_token)} tokens")

# Compute emotion scores using MY function
print("\nComputing emotion scores using MY compute_token_emotion_scores...")
token_emotion_scores = compute_token_emotion_scores(
    model=model,
    activations_by_token=activations_by_token,
    layers=LAYER_RANGE,
    emotion_token_ids=emotion_token_ids,
    averaged_baseline_stats=averaged_baseline_stats,
    emotions=emotions,
    aggregation="mean",
    subtract_mean=True
)
print("✓ Computed emotion scores")

# Print scores
print("\n" + "="*80)
print("EMOTION SCORES (MY IMPLEMENTATION)")
print("="*80)

for token_pos in sorted(token_emotion_scores.keys())[:10]:
    token = token_strings[token_pos]
    scores_array = token_emotion_scores[token_pos]
    print(f"\nToken {token_pos}: '{token}'")
    for i, emotion in enumerate(emotions):
        print(f"  {emotion:12s}: {scores_array[i]:7.3f}")

# Statistics
all_scores = []
for pos in token_emotion_scores:
    all_scores.append(token_emotion_scores[pos])
all_scores = np.array(all_scores)

print("\n" + "="*80)
print("SCORE STATISTICS")
print("="*80)
print(f"Min:    {all_scores.min():.3f}")
print(f"Max:    {all_scores.max():.3f}")
print(f"Mean:   {all_scores.mean():.3f}")
print(f"Std:    {all_scores.std():.3f}")
print(f"Median: {np.median(all_scores):.3f}")

# Check correlation
corr_matrix = np.corrcoef(all_scores.T)
print("\n" + "="*80)
print("CORRELATION MATRIX")
print("="*80)
print("           ", " ".join(f"{e[:6]:>7s}" for e in emotions))
for i, emotion in enumerate(emotions):
    row = ' '.join(f'{corr_matrix[i, j]:7.3f}' for j in range(len(emotions)))
    print(f"  {emotion:10s} {row}")

off_diag_corrs = []
for i in range(len(emotions)):
    for j in range(i+1, len(emotions)):
        off_diag_corrs.append(abs(corr_matrix[i, j]))

avg_abs_corr = np.mean(off_diag_corrs)
print(f"\nAverage absolute correlation: {avg_abs_corr:.3f}")

print("\n" + "="*80)
print("DONE!")
print("="*80)
print("\nCompare these scores to the emo_lens test (job 101858)")
print("If scores match: bug is in how dashboard calls the functions")
print("If scores don't match: bug is in my layer_averaging.py functions")
