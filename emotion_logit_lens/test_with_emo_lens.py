"""
Test ONE sentence using the ACTUAL emo_lens code to see correct scores.
"""
import sys
from pathlib import Path

# Add emo_lens to path
sys.path.insert(0, '/workspace-vast/annas/git/believe-it-or-not')

from emotion_evals.emo_lens.token_trajectories import (
    extract_token_level_activations,
    compute_layer_averaged_baseline_stats,
    compute_token_emotion_scores,
    plot_token_emotion_trajectories
)
from emotion_evals.emo_lens.model_utils import load_base_model
from emotion_evals.emo_lens.logit_lens_emotion_direct import (
    load_reference_stats_for_strategy,
    load_emotion_token_ids_from_json
)

print("="*80)
print("TESTING WITH EMO_LENS CODE")
print("="*80)

# Config
BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
ACTIVATION_STRATEGY = "generated_tokens_avg"
LAYER_RANGE = list(range(40, 51))
PROMPT = "I hate you so much! You make me sick!"

# Load model
print("\n[1/4] Loading model...")
model, tokenizer = load_base_model(BASE_MODEL_NAME)
print("✓ Model loaded")

# Load emotion token IDs
print("\n[2/4] Loading emotion token IDs...")
emotion_token_ids = load_emotion_token_ids_from_json(BASE_MODEL_NAME)
print(f"✓ Loaded {len(emotion_token_ids)} emotions")

# Load reference statistics
print(f"\n[3/4] Loading reference statistics ({ACTIVATION_STRATEGY})...")
SCRIPT_DIR = Path("/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens")
ref_stats = load_reference_stats_for_strategy(
    model_name=BASE_MODEL_NAME,
    activation_strategy=ACTIVATION_STRATEGY,
    script_dir=SCRIPT_DIR
)
print("✓ Loaded baseline stats")

# Compute layer-averaged baseline stats
print(f"\nAveraging baseline stats across layers {LAYER_RANGE[0]}-{LAYER_RANGE[-1]}...")
averaged_baseline_stats = compute_layer_averaged_baseline_stats(
    ref_stats=ref_stats,
    layers=LAYER_RANGE,
    emotion_token_ids=emotion_token_ids
)
print("✓ Computed averaged baseline statistics")

# Extract activations
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

# Compute emotion scores
print("\nComputing emotion scores...")
token_emotion_scores = compute_token_emotion_scores(
    model=model,
    activations_by_token=activations_by_token,
    layers=LAYER_RANGE,
    emotion_token_ids=emotion_token_ids,
    averaged_baseline_stats=averaged_baseline_stats,
    aggregation="mean",
    subtract_mean=True
)

print("✓ Computed emotion scores")

# Print scores
print("\n" + "="*80)
print("EMOTION SCORES (EMO_LENS)")
print("="*80)

emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

for token_pos in sorted(token_emotion_scores.keys())[:10]:
    token = token_strings[token_pos]
    scores = token_emotion_scores[token_pos]
    print(f"\nToken {token_pos}: '{token}'")
    for emotion in emotions:
        print(f"  {emotion:12s}: {scores[emotion]:7.3f}")

# Statistics
import numpy as np
all_scores = []
for pos in token_emotion_scores:
    scores = token_emotion_scores[pos]
    # Convert dict to array in emotion order
    score_array = np.array([scores[e] for e in emotions])
    all_scores.append(score_array)
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

# Generate plot
print("\n[5/5] Generating plot...")
output_path = Path('/workspace-vast/annas/git/research-tools/emotion_logit_lens/emo_lens_test_plot.png')
plot_token_emotion_trajectories(
    token_emotion_scores=token_emotion_scores,
    token_strings=token_strings,
    emotions=emotions,
    output_path=output_path,
    title=f"EMO_LENS: {PROMPT[:50]}",
    show_plot=False
)
print(f"✓ Saved plot to: {output_path}")

print("\n" + "="*80)
print("DONE! These are the CORRECT scores from emo_lens")
print("="*80)
