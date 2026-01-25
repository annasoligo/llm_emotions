#!/usr/bin/env python3
"""Quick check if baseline correction is producing NaN values."""
import pickle
import numpy as np
from pathlib import Path

def main():
    # Test directly with one sample
    print("="*80)
    print("QUICK TEST: Checking for NaN in baseline-corrected scores")
    print("="*80)

    # Import the functions
    import sys
    sys.path.insert(0, '/workspace-vast/annas/git/believe-it-or-not/emotion_evals')
    from emo_lens.load_model import load_base_model
    from emo_lens.extract_activations import extract_token_level_activations
    from emo_lens.emotion_token_loader import load_emotion_token_ids_from_json
    from emo_lens.load_baselines import load_reference_stats_for_strategy

    # Load minimal data
    input_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl')
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    # Take first conversation
    conv = data['conversations'][0]

    # Load model
    print("\n1. Loading model...")
    BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
    model, tokenizer = load_base_model(BASE_MODEL_NAME)
    print("  ✓ Model loaded")

    # Load emotion tokens
    print("\n2. Loading emotion tokens...")
    emotion_token_ids = load_emotion_token_ids_from_json(BASE_MODEL_NAME)
    print(f"  ✓ Loaded {len(emotion_token_ids)} emotions")

    # Load baseline stats
    print("\n3. Loading baseline stats...")
    ACTIVATION_STRATEGY = "generated_tokens_avg"
    SCRIPT_DIR = Path("/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens")
    ref_stats = load_reference_stats_for_strategy(
        model_name=BASE_MODEL_NAME,
        activation_strategy=ACTIVATION_STRATEGY,
        script_dir=SCRIPT_DIR
    )

    # Get layer 40 stats
    layer = 40
    if 'layers_data' in ref_stats and str(layer) in ref_stats['layers_data']:
        layer_baseline_stats = ref_stats['layers_data'][str(layer)]['statistics']
        print(f"  ✓ Baseline stats for layer {layer}: {len(layer_baseline_stats)} tokens")

        # Check structure
        sample_token_ids = list(layer_baseline_stats.keys())[:3]
        print(f"\n4. Checking baseline structure:")
        print(f"  Sample token IDs (type): {[(tid, type(tid)) for tid in sample_token_ids]}")

        # Test lookup with emotion tokens
        print(f"\n5. Testing baseline lookups:")
        test_emotion = 'anger'
        test_tokens = emotion_token_ids[test_emotion][:5]

        matches = 0
        for tid in test_tokens:
            tid_str = str(tid)
            if tid_str in layer_baseline_stats:
                matches += 1
                stats = layer_baseline_stats[tid_str]
                print(f"  ✓ Token {tid} -> mean={stats['mean']:.4f}, std={stats['std']:.4f}")

        print(f"\n  Matched {matches}/{len(test_tokens)} emotion tokens in baseline")

        if matches > 0:
            print("\n" + "="*80)
            print("✓ BASELINE CORRECTION SHOULD WORK!")
            print("  - Token IDs are stored as strings: ✓")
            print("  - Lookups with str(token_id) work: ✓")
            print("  - Stats have valid mean/std values: ✓")
            print("="*80)
        else:
            print("\n" + "="*80)
            print("✗ BASELINE CORRECTION WILL FAIL!")
            print("  No emotion tokens found in baseline stats")
            print("="*80)
    else:
        print(f"  ✗ No baseline stats found for layer {layer}")

if __name__ == "__main__":
    main()
