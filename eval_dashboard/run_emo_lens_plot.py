"""Run sentence through emo_lens and generate emotion plot."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add emo_lens to path
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not")

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

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

print("="*80)
print("EMO_LENS EMOTION PLOT")
print("="*80)

# Test sentence
PROMPT = "I can't believe you are so stupid. You've really messed things up for me now."
print(f"\nPrompt: {PROMPT}\n")

# Load model
print("Loading model...")
BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
model, tokenizer = load_base_model(BASE_MODEL_NAME)
print("✓ Model loaded")

# Load emotion token IDs
print("Loading emotion token IDs...")
emotion_token_ids = load_emotion_token_ids_from_json(BASE_MODEL_NAME)
print(f"✓ Loaded {len(emotion_token_ids)} emotions")

# Load reference statistics
print("Loading baseline statistics...")
ACTIVATION_STRATEGY = "generated_tokens_avg"
SCRIPT_DIR = Path("/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens")
ref_stats = load_reference_stats_for_strategy(
    model_name=BASE_MODEL_NAME,
    activation_strategy=ACTIVATION_STRATEGY,
    script_dir=SCRIPT_DIR
)
print("✓ Loaded baseline stats")

# Compute layer-averaged baseline stats
LAYER_RANGE = list(range(40, 51))
averaged_baseline_stats = compute_layer_averaged_baseline_stats(
    ref_stats=ref_stats,
    layers=LAYER_RANGE,
    emotion_token_ids=emotion_token_ids
)
print("✓ Computed averaged baseline statistics")

# Extract activations
print("\nExtracting token-level activations...")
activations_by_token, token_ids = extract_token_level_activations(
    model=model,
    tokenizer=tokenizer,
    prompt=PROMPT,
    layers=LAYER_RANGE,
    start_token_idx=0,
    system_prompt=None,
    num_generated_tokens=0
)
print(f"✓ Extracted {len(activations_by_token)} token activations")

# Compute emotion scores
print("Computing emotion scores...")
token_emotion_scores = compute_token_emotion_scores(
    model=model,
    activations_by_token=activations_by_token,
    layers=LAYER_RANGE,
    emotion_token_ids=emotion_token_ids,
    averaged_baseline_stats=averaged_baseline_stats,
    aggregation="mean",
    subtract_mean=True
)
print(f"✓ Computed scores for {len(token_emotion_scores)} tokens")

# Decode tokens
token_strings = [tokenizer.decode([tid]) for tid in token_ids]

# Compute correlation
print("\nComputing inter-emotion correlation...")
all_scores = []
for pos in sorted(token_emotion_scores.keys()):
    scores = token_emotion_scores[pos]
    score_array = np.array([scores[e] for e in EMOTIONS])
    all_scores.append(score_array)

scores_matrix = np.array(all_scores)
corr_matrix = np.corrcoef(scores_matrix.T)

off_diag_corrs = []
for i in range(6):
    for j in range(i+1, 6):
        off_diag_corrs.append(abs(corr_matrix[i, j]))

avg_corr = np.mean(off_diag_corrs)

print(f"\nScore statistics:")
print(f"  Mean: {scores_matrix.mean():.3f}")
print(f"  Std:  {scores_matrix.std():.3f}")
print(f"  Min:  {scores_matrix.min():.3f}")
print(f"  Max:  {scores_matrix.max():.3f}")
print(f"\nAverage absolute inter-emotion correlation: {avg_corr:.3f}")

# Plot
print("\nGenerating plot...")
output_path = Path("/workspace-vast/annas/git/research-tools/eval_dashboard/emo_lens_test_plot.png")
fig = plot_token_emotion_trajectories(
    token_emotion_scores=token_emotion_scores,
    token_strings=token_strings,
    title=f"Emo_lens Emotion Trajectories (Layers 40-50, Corr={avg_corr:.3f})",
    output_path=output_path,
    show_tokens_on_xaxis=True,
    figsize=(18, 8)
)

print(f"✓ Plot saved to: {output_path}")

print("\n" + "="*80)
print("CORRELATION MATRIX")
print("="*80)
print("             ", " ".join(f"{e[:6]:>7s}" for e in EMOTIONS))
for i, emotion in enumerate(EMOTIONS):
    row = ' '.join(f'{corr_matrix[i, j]:7.3f}' for j in range(6))
    print(f"  {emotion:10s} {row}")

print("\n" + "="*80)
