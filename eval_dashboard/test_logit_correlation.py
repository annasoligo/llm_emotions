"""Test if high correlation is specific to emotion logits or general to all logits."""
import numpy as np
import sys
import json
from pathlib import Path

# Add emo_lens to path
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not")

from emotion_evals.emo_lens.token_trajectories import (
    extract_token_level_activations,
    compute_layer_averaged_baseline_stats,
)
from emotion_evals.emo_lens.model_utils import load_base_model
from emotion_evals.emo_lens.logit_lens_emotion_direct import (
    load_reference_stats_for_strategy,
    load_emotion_token_ids_from_json
)

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# Test sentence
PROMPT = "I can't believe you are so stupid. You've really messed things up for me now."

print("="*80)
print("TESTING LOGIT CORRELATION: EMOTION vs RANDOM TOKENS")
print("="*80)

# Load model
print("\nLoading model...")
BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
model, tokenizer = load_base_model(BASE_MODEL_NAME)
print("✓ Model loaded")

# Load emotion token IDs
print("\nLoading emotion token IDs...")
emotion_token_ids = load_emotion_token_ids_from_json(BASE_MODEL_NAME)
print(f"✓ Loaded {len(emotion_token_ids)} emotions")

# Get all emotion token IDs in a flat list
all_emotion_token_ids = []
for emotion, token_ids in emotion_token_ids.items():
    all_emotion_token_ids.extend(token_ids)
all_emotion_token_ids = list(set(all_emotion_token_ids))
print(f"  Total unique emotion tokens: {len(all_emotion_token_ids)}")

# Sample 500 random non-emotion tokens from vocabulary
print("\nSampling 500 random non-emotion tokens from vocabulary...")
vocab_size = len(tokenizer)
all_token_ids = set(range(vocab_size))
non_emotion_token_ids = list(all_token_ids - set(all_emotion_token_ids))

print(f"  Total vocabulary size: {vocab_size}")
print(f"  Non-emotion tokens: {len(non_emotion_token_ids)}")

np.random.seed(42)
n_sample = 500
random_token_ids = np.random.choice(non_emotion_token_ids, size=n_sample, replace=False).tolist()
print(f"✓ Sampled {len(random_token_ids)} random tokens")

# Load reference statistics for emotion tokens only
print("\nLoading baseline statistics for emotion tokens...")
ACTIVATION_STRATEGY = "generated_tokens_avg"
SCRIPT_DIR = Path("/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens")
ref_stats = load_reference_stats_for_strategy(
    model_name=BASE_MODEL_NAME,
    activation_strategy=ACTIVATION_STRATEGY,
    script_dir=SCRIPT_DIR
)
print("✓ Loaded baseline stats")

# Compute layer-averaged baseline stats for emotion tokens only
LAYER_RANGE = list(range(40, 51))
print(f"\nUsing layers: {LAYER_RANGE}")

averaged_baseline_stats = compute_layer_averaged_baseline_stats(
    ref_stats=ref_stats,
    layers=LAYER_RANGE,
    emotion_token_ids=emotion_token_ids  # Only emotion tokens
)

print("✓ Computed averaged baseline statistics for emotion tokens")

# Print baseline statistics summary for each emotion
print("\n" + "="*80)
print("BASELINE STATISTICS SUMMARY")
print("="*80)
for emotion in EMOTIONS:
    means = [averaged_baseline_stats[emotion][tid]['mean'] for tid in emotion_token_ids[emotion]]
    stds = [averaged_baseline_stats[emotion][tid]['std'] for tid in emotion_token_ids[emotion]]

    print(f"\n{emotion.upper()}:")
    print(f"  Number of tokens: {len(means)}")
    print(f"  Mean logits - avg: {np.mean(means):.3f}, std: {np.std(means):.3f}, range: [{np.min(means):.3f}, {np.max(means):.3f}]")
    print(f"  Std logits  - avg: {np.mean(stds):.3f}, std: {np.std(stds):.3f}, range: [{np.min(stds):.3f}, {np.max(stds):.3f}]")

print("\n" + "="*80)

# Extract token-level activations
print(f"\nProcessing test sentence...")
print(f"Sentence: '{PROMPT}'")

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

# Project activations to logits and compute z-scores
print("\nComputing logit z-scores for all tokens...")

# We'll manually compute the logit lens projection and z-scoring
from emotion_evals.emo_lens.model_utils import project_to_logits

token_emotion_logits = {}  # token_idx -> {emotion: [logit_per_token_id]}
token_random_logits = {}   # token_idx -> [logit_per_random_token_id]

for token_idx in activations_by_token:
    # Get averaged activation across layers
    activations = []
    for layer in LAYER_RANGE:
        activations.append(activations_by_token[token_idx][layer])
    avg_activation = np.mean(activations, axis=0)  # [hidden_dim]

    # Project to vocabulary logits
    all_logits_tensor = project_to_logits(
        model=model,
        hidden_state_vector=avg_activation
    )  # [vocab_size] torch.Tensor on GPU

    # Convert to CPU numpy array (handle BFloat16)
    all_logits = all_logits_tensor.cpu().float().detach().numpy()

    # Extract and z-score normalize emotion logits
    token_emotion_logits[token_idx] = {}
    for emotion in EMOTIONS:
        normalized_values = []
        for tid in emotion_token_ids[emotion]:
            logit = all_logits[tid]
            # Baseline structure is {emotion: {token_id: {'mean': float, 'std': float}}}
            baseline_mean = averaged_baseline_stats[emotion][tid]['mean']
            baseline_std = averaged_baseline_stats[emotion][tid]['std']
            z_score = (logit - baseline_mean) / (baseline_std + 1e-8)
            normalized_values.append(z_score)

        token_emotion_logits[token_idx][emotion] = normalized_values

    # Extract RAW logits for random tokens (no normalization - we'll compare correlation patterns)
    raw_random = []
    for tid in random_token_ids:
        logit = all_logits[tid]
        raw_random.append(logit)

    token_random_logits[token_idx] = raw_random

print(f"✓ Computed logits for {len(token_emotion_logits)} tokens")
print(f"  - Emotion tokens: z-scored using baseline")
print(f"  - Random tokens: raw logits (correlation is scale-invariant)")

# Aggregate emotion logits using mean (like in the main code)
print("\nAggregating emotion logits using mean...")
token_emotion_scores = {}  # token_idx -> {emotion: mean_z_score}
for token_idx in token_emotion_logits:
    token_emotion_scores[token_idx] = {}
    for emotion in EMOTIONS:
        token_emotion_scores[token_idx][emotion] = np.mean(token_emotion_logits[token_idx][emotion])

# For random tokens, aggregate into 6 groups of ~83 tokens each to match emotion structure
print("\nAggregating random logits into 6 groups...")
token_random_scores = {}  # token_idx -> {group_name: mean_raw_logit}
n_groups = 6
group_size = len(random_token_ids) // n_groups

for token_idx in token_random_logits:
    token_random_scores[token_idx] = {}
    for group_idx in range(n_groups):
        start_idx = group_idx * group_size
        end_idx = start_idx + group_size if group_idx < n_groups - 1 else len(random_token_ids)
        group_name = f"random_group_{group_idx}"
        token_random_scores[token_idx][group_name] = np.mean(token_random_logits[token_idx][start_idx:end_idx])

# Build matrices for correlation analysis
print("\nBuilding correlation matrices...")

# Emotion matrix: [n_tokens, 6 emotions]
emotion_matrix = []
for token_idx in sorted(token_emotion_scores.keys()):
    row = [token_emotion_scores[token_idx][e] for e in EMOTIONS]
    emotion_matrix.append(row)
emotion_matrix = np.array(emotion_matrix)

# Random matrix: [n_tokens, 6 groups]
random_matrix = []
for token_idx in sorted(token_random_scores.keys()):
    row = [token_random_scores[token_idx][f"random_group_{i}"] for i in range(n_groups)]
    random_matrix.append(row)
random_matrix = np.array(random_matrix)

print(f"Emotion matrix shape: {emotion_matrix.shape}")
print(f"Random matrix shape: {random_matrix.shape}")

# Compute correlations
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# 1. Within-emotion correlations
emotion_corr = np.corrcoef(emotion_matrix.T)
emotion_off_diag = []
for i in range(6):
    for j in range(i+1, 6):
        emotion_off_diag.append(abs(emotion_corr[i, j]))

print("\n1. WITHIN-EMOTION CORRELATIONS:")
print(f"   Average absolute correlation: {np.mean(emotion_off_diag):.3f}")
print(f"   Min: {np.min(emotion_off_diag):.3f}, Max: {np.max(emotion_off_diag):.3f}")
print("\n   Correlation matrix:")
for i, e1 in enumerate(EMOTIONS):
    for j, e2 in enumerate(EMOTIONS):
        if i < j:
            print(f"   {e1:10} vs {e2:10}: {emotion_corr[i, j]:+.3f}")

# 2. Within-random correlations
random_corr = np.corrcoef(random_matrix.T)
random_off_diag = []
for i in range(6):
    for j in range(i+1, 6):
        random_off_diag.append(abs(random_corr[i, j]))

print("\n2. WITHIN-RANDOM CORRELATIONS:")
print(f"   Average absolute correlation: {np.mean(random_off_diag):.3f}")
print(f"   Min: {np.min(random_off_diag):.3f}, Max: {np.max(random_off_diag):.3f}")

# 3. Cross correlations (emotion vs random)
cross_corr_list = []
for i in range(6):
    for j in range(6):
        corr = np.corrcoef(emotion_matrix[:, i], random_matrix[:, j])[0, 1]
        cross_corr_list.append(abs(corr))

print("\n3. CROSS-CORRELATIONS (emotion vs random):")
print(f"   Average absolute correlation: {np.mean(cross_corr_list):.3f}")
print(f"   Min: {np.min(cross_corr_list):.3f}, Max: {np.max(cross_corr_list):.3f}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Sentence length: {len(activations_by_token)} tokens")
print(f"Number of emotions: {len(EMOTIONS)}")
print(f"Number of random token groups: {n_groups}")
print(f"\nAverage absolute correlations:")
print(f"  Emotion-Emotion:  {np.mean(emotion_off_diag):.3f}")
print(f"  Random-Random:    {np.mean(random_off_diag):.3f}")
print(f"  Emotion-Random:   {np.mean(cross_corr_list):.3f}")

if np.mean(emotion_off_diag) > 0.7 and np.mean(random_off_diag) > 0.7:
    print("\n⚠️  HIGH CORRELATION IN BOTH: All logits are highly correlated!")
elif np.mean(emotion_off_diag) > 0.7:
    print("\n⚠️  HIGH EMOTION CORRELATION: Only emotion logits are highly correlated!")
else:
    print("\n✓ NORMAL CORRELATION: Neither group shows unusually high correlation")

print("="*80)
