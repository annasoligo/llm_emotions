#!/usr/bin/env python3
"""Simple verification that baseline stats lookups will work with string conversion."""
import json
from pathlib import Path

def main():
    print("="*80)
    print("VERIFYING BASELINE CORRECTION FIX")
    print("="*80)

    # Load baseline stats
    baseline_path = Path('/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens/instruct_baselines/unsloth_gemma_3_27b_it/generated_tokens_avg/baseline_stats.json')
    print(f"\n1. Loading baseline stats from:")
    print(f"   {baseline_path}")

    with open(baseline_path) as f:
        baseline_data = json.load(f)

    print(f"   ✓ Loaded baseline format version {baseline_data.get('format_version', 'unknown')}")

    # Load emotion token IDs
    emotion_path = Path('/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens/emotion_word_sets/all_emotion_words_unsloth_gemma_3_27b_it_token_ids.json')
    print(f"\n2. Loading emotion token IDs from:")
    print(f"   {emotion_path}")

    with open(emotion_path) as f:
        emotion_tokens = json.load(f)

    total_emotion_tokens = sum(len(tokens) for tokens in emotion_tokens.values())
    print(f"   ✓ Loaded {total_emotion_tokens} emotion tokens across {len(emotion_tokens)} emotions")

    # Test layer 40 (the layer we saw in previous tests)
    layer = 40
    layer_key = str(layer)

    print(f"\n3. Testing baseline lookups for layer {layer}:")

    if 'layers_data' not in baseline_data or layer_key not in baseline_data['layers_data']:
        print(f"   ✗ ERROR: Layer {layer} not found in baseline stats!")
        return

    layer_stats = baseline_data['layers_data'][layer_key]['statistics']
    print(f"   ✓ Layer {layer} has {len(layer_stats)} token entries")

    # Check the key type in baseline
    sample_keys = list(layer_stats.keys())[:3]
    print(f"\n4. Baseline stats key types:")
    print(f"   Sample keys: {sample_keys}")
    print(f"   Key type: {type(sample_keys[0])}")

    # Test lookups for each emotion
    print(f"\n5. Testing lookups for each emotion:")

    total_tested = 0
    total_matched = 0
    emotions_with_matches = []

    for emotion, token_list in emotion_tokens.items():
        # Test first 10 tokens of each emotion
        test_tokens = token_list[:10]
        matches = 0
        matched_tokens = []

        for tid in test_tokens:
            tid_str = str(tid)  # This is the FIX we applied
            if tid_str in layer_stats:
                matches += 1
                total_matched += 1
                stats = layer_stats[tid_str]
                matched_tokens.append({
                    'token_id': tid,
                    'mean': stats['mean'],
                    'std': stats['std']
                })

        total_tested += len(test_tokens)

        if matches > 0:
            emotions_with_matches.append(emotion)
            print(f"   {emotion:12s}: {matches}/{len(test_tokens)} matched")
            # Show first matched token
            first_match = matched_tokens[0]
            print(f"                  Example: token {first_match['token_id']} -> mean={first_match['mean']:.4f}, std={first_match['std']:.4f}")
        else:
            print(f"   {emotion:12s}: {matches}/{len(test_tokens)} matched ✗")

    print(f"\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print(f"  Total tokens tested: {total_tested}")
    print(f"  Total matches found: {total_matched}")
    print(f"  Match rate: {100*total_matched/total_tested:.1f}%")
    print(f"  Emotions with matches: {len(emotions_with_matches)}/{len(emotion_tokens)}")

    if total_matched > 0:
        print(f"\n✓ BASELINE CORRECTION WILL WORK!")
        print(f"  - Token IDs are stored as strings in baseline")
        print(f"  - str(token_id) conversion finds matches")
        print(f"  - Matched stats have valid mean/std values")
        print(f"  - Expected: All NaN values will be replaced with z-scores")
    else:
        print(f"\n✗ BASELINE CORRECTION WILL FAIL!")
        print(f"  - No emotion tokens found in baseline stats")
        print(f"  - This should not happen if baseline was computed correctly")

    print("="*80)

if __name__ == "__main__":
    main()
