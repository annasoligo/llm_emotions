"""
Debug baseline loading to check if stats are correct.
"""
import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

from pathlib import Path
from emotion_logit_lens.baseline_loader import LogitBaselineLoader
from emotion_logit_lens import EmotionTokenManager

BASELINE_DIR = Path('/workspace-vast/annas/git/research-tools/data/baselines/logit_emotion_alpaca/google_gemma_3_27b_it')

print("=" * 80)
print("DEBUGGING BASELINE STATS")
print("=" * 80)

# Load baseline loader
baseline_loader = LogitBaselineLoader(
    baseline_dir=BASELINE_DIR,
    model_name='google_gemma_3_27b_it'
)

# Load emotion token IDs
emotion_mgr = EmotionTokenManager(model_name='google_gemma_3_27b_it')
emotion_token_ids = emotion_mgr.load_emotion_token_ids()

print(f"\nEmotion token IDs:")
for emotion, token_ids in emotion_token_ids.items():
    print(f"  {emotion}: {len(token_ids)} tokens, first few: {token_ids[:3]}")

# Check layer 45 stats
layer = 45
print(f"\nLoading stats for layer {layer}...")
layer_stats = baseline_loader.load_layer_stats(layer)

print(f"Number of tokens with stats: {len(layer_stats)}")

# Check stats for first anger token
emotion = 'anger'
token_id = emotion_token_ids[emotion][0]

if str(token_id) in layer_stats:
    stats = layer_stats[str(token_id)]
    print(f"\nStats for token {token_id} (first {emotion} token):")
    print(f"  mean: {stats['mean']}")
    print(f"  std: {stats['std']}")
else:
    print(f"\n⚠ Token {token_id} NOT found in baseline stats!")

# Check a few more
print(f"\nSample baseline stats for {emotion} tokens:")
for i, token_id in enumerate(emotion_token_ids[emotion][:5]):
    if str(token_id) in layer_stats:
        stats = layer_stats[str(token_id)]
        print(f"  Token {token_id}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
    else:
        print(f"  Token {token_id}: NOT FOUND")

# Check if any stats have std=0 or std=1 (placeholder values)
suspicious_counts = {'zero_std': 0, 'one_std': 0, 'zero_mean': 0}
for token_id_str, stats in layer_stats.items():
    if abs(stats['std']) < 1e-8:
        suspicious_counts['zero_std'] += 1
    if abs(stats['std'] - 1.0) < 1e-8:
        suspicious_counts['one_std'] += 1
    if abs(stats['mean']) < 1e-8:
        suspicious_counts['zero_mean'] += 1

print(f"\nSuspicious baseline stats:")
print(f"  Tokens with std≈0: {suspicious_counts['zero_std']}")
print(f"  Tokens with std≈1: {suspicious_counts['one_std']}")
print(f"  Tokens with mean≈0: {suspicious_counts['zero_mean']}")

# Check range of values
means = [stats['mean'] for stats in layer_stats.values()]
stds = [stats['std'] for stats in layer_stats.values()]

import numpy as np
print(f"\nBaseline statistics across all tokens:")
print(f"  Mean range: [{min(means):.3f}, {max(means):.3f}]")
print(f"  Std range: [{min(stds):.3f}, {max(stds):.3f}]")
print(f"  Mean of means: {np.mean(means):.3f}")
print(f"  Mean of stds: {np.mean(stds):.3f}")

print("\n" + "=" * 80)
