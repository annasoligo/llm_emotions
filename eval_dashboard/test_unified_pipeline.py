#!/usr/bin/env python3
"""
Test script to validate the unified pipeline changes.

This script:
1. Validates the layerwise_plotting.py changes work with existing data
2. Checks data format compatibility
3. Creates a comparison report

Run this WITHOUT GPU - it only reads existing preprocessed data.
"""

import pickle
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '/workspace-vast/annas/git/research-tools')
sys.path.insert(0, '/workspace-vast/annas/git/research-tools/eval_dashboard')

from layerwise_plotting import (
    get_per_layer_scores_for_window,
    get_token_windows,
    EMOTIONS
)


def test_layerwise_plotting_formats():
    """Test that layerwise_plotting.py handles all data formats correctly."""
    print("=" * 80)
    print("TEST: layerwise_plotting.py format handling")
    print("=" * 80)

    # Load existing data
    data_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl')
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return False

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    conv = data['conversations'][0]
    print(f"\nTesting with conversation {conv['sample_id']}")
    print(f"Available per-layer keys: {[k for k in conv.keys() if '_by_layer' in k]}")

    # Get token windows
    windows = get_token_windows(conv)
    print(f"\nToken windows:")
    for window_type, sent_ids in windows.items():
        print(f"  {window_type}: {len(sent_ids)} sentences")

    # Test each per-layer key
    results = {}
    for per_layer_key in [k for k in conv.keys() if '_by_layer' in k]:
        probe_key = per_layer_key.replace('_by_layer', '')
        print(f"\n--- Testing {probe_key} ---")

        # Check data structure
        sample_sent_id = list(conv[per_layer_key].keys())[0]
        sample_layer_data = conv[per_layer_key][sample_sent_id]
        sample_layer = list(sample_layer_data.keys())[0]
        sample_scores = sample_layer_data[sample_layer]

        print(f"  Sample sentence ID: {sample_sent_id}")
        print(f"  Sample layer: {sample_layer}")
        print(f"  Score type: {type(sample_scores)}")

        if isinstance(sample_scores, dict):
            if 'user' in sample_scores:
                print(f"  Format: Orthogonal (user/assistant dict)")
                print(f"    User shape: {np.array(sample_scores['user']).shape}")
            else:
                print(f"  Format: Logit lens (emotion dict)")
                print(f"    Emotions: {list(sample_scores.keys())}")
        elif isinstance(sample_scores, (list, np.ndarray)):
            print(f"  Format: Probe array")
            print(f"    Shape: {np.array(sample_scores).shape}")

        # Test get_per_layer_scores_for_window
        try:
            layers = sorted(list(sample_layer_data.keys()))
            scores = get_per_layer_scores_for_window(
                conv, probe_key, windows['pre_onset'][:5], layers
            )

            if scores is None:
                print(f"  Result: None (no per-layer data found)")
                results[probe_key] = 'NO_DATA'
            else:
                print(f"  Result: SUCCESS")
                for emotion in EMOTIONS[:2]:  # Just show first 2
                    print(f"    {emotion}: shape={scores[emotion].shape}, "
                          f"range=[{scores[emotion].min():.3f}, {scores[emotion].max():.3f}]")
                results[probe_key] = 'SUCCESS'
        except Exception as e:
            print(f"  Result: ERROR - {e}")
            results[probe_key] = f'ERROR: {e}'

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for probe_key, result in results.items():
        status = "✓" if result == 'SUCCESS' else "✗"
        print(f"  {status} {probe_key}: {result}")

    return all(r == 'SUCCESS' for r in results.values())


def test_data_structure_compatibility():
    """Test that existing data is compatible with new pipeline expectations."""
    print("\n" + "=" * 80)
    print("TEST: Data structure compatibility")
    print("=" * 80)

    data_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    conv = data['conversations'][0]

    # Check expected keys
    expected_keys = ['sample_id', 'conversation', 'rating', 'sentences', 'probe_scores', 'metadata']
    missing_keys = [k for k in expected_keys if k not in conv]

    if missing_keys:
        print(f"✗ Missing expected keys: {missing_keys}")
        return False
    else:
        print(f"✓ All expected keys present")

    # Check probe_scores structure
    probe_scores = conv['probe_scores']
    print(f"\nProbe scores available: {len(probe_scores)}")

    for probe_key, scores in probe_scores.items():
        if not scores:
            print(f"  ✗ {probe_key}: EMPTY")
            continue

        sample_sent_id = list(scores.keys())[0]
        sample_score = scores[sample_sent_id]

        if isinstance(sample_score, dict) and 'user' in sample_score:
            score_type = "orthogonal"
            shape = f"user:{np.array(sample_score['user']).shape}, asst:{np.array(sample_score['assistant']).shape}"
        elif isinstance(sample_score, (list, np.ndarray)):
            score_type = "array"
            shape = str(np.array(sample_score).shape)
        else:
            score_type = "unknown"
            shape = str(type(sample_score))

        print(f"  ✓ {probe_key}: {score_type} {shape}")

    # Check per-layer keys
    per_layer_keys = [k for k in conv.keys() if '_by_layer' in k]
    print(f"\nPer-layer data available: {len(per_layer_keys)}")
    for key in per_layer_keys:
        sample_sent = list(conv[key].keys())[0]
        num_layers = len(conv[key][sample_sent])
        print(f"  ✓ {key}: {num_layers} layers")

    return True


def compare_probe_statistics():
    """Compare statistics between different probe types."""
    print("\n" + "=" * 80)
    print("TEST: Probe statistics comparison")
    print("=" * 80)

    data_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Collect statistics for each probe type
    stats = {}

    for conv in data['conversations']:
        for probe_key, scores in conv['probe_scores'].items():
            if probe_key not in stats:
                stats[probe_key] = []

            for sent_id, score in scores.items():
                if isinstance(score, dict) and 'user' in score:
                    # Orthogonal - use assistant scores
                    stats[probe_key].append(np.array(score['assistant']))
                elif isinstance(score, (list, np.ndarray)):
                    stats[probe_key].append(np.array(score))

    print(f"\nStatistics across all conversations ({len(data['conversations'])} total):")
    print(f"\n{'Probe':<40} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 80)

    for probe_key, score_list in sorted(stats.items()):
        if not score_list:
            continue
        all_scores = np.array(score_list)
        print(f"{probe_key:<40} {all_scores.mean():>10.3f} {all_scores.std():>10.3f} "
              f"{all_scores.min():>10.3f} {all_scores.max():>10.3f}")

    return True


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("UNIFIED PIPELINE VALIDATION TESTS")
    print("=" * 80 + "\n")

    tests = [
        ("Layerwise plotting formats", test_layerwise_plotting_formats),
        ("Data structure compatibility", test_data_structure_compatibility),
        ("Probe statistics comparison", compare_probe_statistics),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
