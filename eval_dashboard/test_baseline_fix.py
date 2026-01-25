#!/usr/bin/env python3
"""Quick test to verify baseline stats lookup fix works."""
import json
from pathlib import Path

# Load baseline stats
baseline_path = Path('/workspace-vast/annas/git/believe-it-or-not/data/baselines/emo_lens_generated_tokens_avg/baseline_stats.json')
print(f"Loading baseline stats from {baseline_path}")

with open(baseline_path) as f:
    baseline_data = json.load(f)

# Check structure
layer_key = '40'
print(f"\nChecking layer {layer_key} structure:")
layer_stats = baseline_data['layers_data'][layer_key]['statistics']
print(f"  Type of layer_stats: {type(layer_stats)}")
print(f"  Number of tokens: {len(layer_stats)}")

# Check a sample token ID
sample_token_ids = list(layer_stats.keys())[:5]
print(f"\n  Sample token IDs (type): {[(tid, type(tid)) for tid in sample_token_ids]}")

# Test the lookup with integer token ID
test_token_id_int = 180225
test_token_id_str = str(test_token_id_int)

print(f"\nTesting lookup:")
print(f"  Looking up token ID {test_token_id_int} (int)")
print(f"  Integer lookup: {test_token_id_int in layer_stats}")
print(f"  String lookup: {test_token_id_str in layer_stats}")

if test_token_id_str in layer_stats:
    print(f"  ✓ String lookup works!")
    print(f"  Stats: {layer_stats[test_token_id_str]}")
else:
    print(f"  ✗ Token not found in baseline stats")

# Check if this token is in the emotion token list
emotion_tokens_path = Path('/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens/emotion_tokens_by_class.json')
with open(emotion_tokens_path) as f:
    emotion_tokens = json.load(f)

print(f"\nChecking if token {test_token_id_int} is an emotion token:")
for emotion, tokens in emotion_tokens.items():
    if test_token_id_int in tokens:
        print(f"  ✓ Found in {emotion}: {tokens.index(test_token_id_int)}/{len(tokens)}")

print("\n" + "="*80)
print("CONCLUSION:")
print("  Baseline stats are keyed by STRING token IDs")
print("  Must convert int token IDs to strings before lookup: str(token_id)")
print("="*80)
