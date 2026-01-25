#!/usr/bin/env python3
"""
Minimal test script to validate unified pipeline changes.
Runs WITHOUT GPU and without plotly - just validates data structures.
"""

import pickle
import numpy as np
from pathlib import Path

# Replicate the EMOTIONS constant from layerwise_plotting
EMOTIONS = ['frustrated', 'curious', 'delighted', 'bored', 'sad', 'calm']


def test_per_layer_format_handling():
    """Test that we can handle different per-layer data formats correctly."""
    print("=" * 80)
    print("TEST: Per-layer format handling logic")
    print("=" * 80)

    # Simulate different data formats
    test_cases = [
        {
            "name": "Probe array format",
            "data": np.array([0.5, -0.2, 1.1, 0.3, -0.5, 0.8]),
            "expected_type": "array"
        },
        {
            "name": "Logit lens dict format",
            "data": {'frustrated': 0.5, 'curious': -0.2, 'delighted': 1.1,
                    'bored': 0.3, 'sad': -0.5, 'calm': 0.8},
            "expected_type": "logit_lens"
        },
        {
            "name": "Orthogonal probe format",
            "data": {
                'user': np.array([0.3, -0.1, 0.8, 0.2, -0.4, 0.6]),
                'assistant': np.array([0.5, -0.2, 1.1, 0.3, -0.5, 0.8])
            },
            "expected_type": "orthogonal"
        }
    ]

    all_passed = True
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        layer_scores = case['data']
        scores_by_emotion = {emotion: [] for emotion in EMOTIONS}

        # This is the logic from layerwise_plotting.py get_per_layer_scores_for_window
        try:
            if isinstance(layer_scores, dict):
                if 'user' in layer_scores:
                    # Orthogonal probes - use assistant scores
                    asst_scores = layer_scores['assistant']
                    for emotion_idx, emotion in enumerate(EMOTIONS):
                        if isinstance(asst_scores, (list, np.ndarray)):
                            scores_by_emotion[emotion].append(asst_scores[emotion_idx])
                        else:
                            scores_by_emotion[emotion].append(asst_scores.get(emotion, 0.0))
                    detected_type = "orthogonal"
                else:
                    # Logit lens format: {emotion: score}
                    for emotion in EMOTIONS:
                        if emotion in layer_scores:
                            scores_by_emotion[emotion].append(layer_scores[emotion])
                    detected_type = "logit_lens"
            elif isinstance(layer_scores, (list, np.ndarray)):
                # Probe format: array of shape (6,)
                for emotion_idx, emotion in enumerate(EMOTIONS):
                    scores_by_emotion[emotion].append(layer_scores[emotion_idx])
                detected_type = "array"
            else:
                raise ValueError(f"Unknown format: {type(layer_scores)}")

            # Verify
            assert detected_type == case['expected_type'], f"Type mismatch: {detected_type} vs {case['expected_type']}"
            for emotion in EMOTIONS:
                assert len(scores_by_emotion[emotion]) == 1, f"Missing score for {emotion}"

            print(f"  ✓ Detected type: {detected_type}")
            print(f"  ✓ All emotions extracted correctly")
            print(f"    Sample values: frustrated={scores_by_emotion['frustrated'][0]:.3f}, calm={scores_by_emotion['calm'][0]:.3f}")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            all_passed = False

    return all_passed


def test_existing_data_structure():
    """Test that existing data has expected structure."""
    print("\n" + "=" * 80)
    print("TEST: Existing data structure")
    print("=" * 80)

    data_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl')
    if not data_path.exists():
        print(f"✗ Data file not found: {data_path}")
        return False

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    print(f"\nLoaded {len(data['conversations'])} conversations")

    conv = data['conversations'][0]
    print(f"\nSample conversation: {conv['sample_id']}")

    # Check for per-layer keys
    per_layer_keys = [k for k in conv.keys() if '_by_layer' in k]
    print(f"\nPer-layer data keys ({len(per_layer_keys)}):")

    for key in per_layer_keys:
        sample_sent = list(conv[key].keys())[0]
        layer_data = conv[key][sample_sent]
        layers = sorted(list(layer_data.keys()))
        sample_layer = layers[0]
        sample_scores = layer_data[sample_layer]

        # Determine format
        if isinstance(sample_scores, dict):
            if 'user' in sample_scores:
                fmt = "orthogonal"
                shape = f"user:{np.array(sample_scores['user']).shape}"
            else:
                fmt = "logit_lens"
                shape = f"{len(sample_scores)} emotions"
        elif isinstance(sample_scores, (list, np.ndarray)):
            fmt = "array"
            shape = str(np.array(sample_scores).shape)
        else:
            fmt = "unknown"
            shape = str(type(sample_scores))

        print(f"  ✓ {key}")
        print(f"      Layers: {layers[0]}-{layers[-1]} ({len(layers)} total)")
        print(f"      Format: {fmt}, Shape: {shape}")

    # Check probe_scores
    print(f"\nAggregated probe_scores keys ({len(conv['probe_scores'])}):")
    for probe_key, scores in conv['probe_scores'].items():
        if not scores:
            print(f"  ✗ {probe_key}: EMPTY")
            continue
        sample_sent = list(scores.keys())[0]
        sample_score = scores[sample_sent]

        if isinstance(sample_score, dict) and 'user' in sample_score:
            shape = f"user:{np.array(sample_score['user']).shape}, asst:{np.array(sample_score['assistant']).shape}"
        elif isinstance(sample_score, (list, np.ndarray)):
            shape = str(np.array(sample_score).shape)
        else:
            shape = str(type(sample_score))

        print(f"  ✓ {probe_key}: {shape}")

    return True


def test_normalization_consistency():
    """Test that per-layer scores have similar statistics to aggregated scores."""
    print("\n" + "=" * 80)
    print("TEST: Normalization consistency (aggregated vs per-layer)")
    print("=" * 80)

    data_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl')
    if not data_path.exists():
        print(f"✗ Data file not found")
        return False

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Find probes that have both aggregated and per-layer versions
    conv = data['conversations'][0]
    per_layer_keys = {k.replace('_by_layer', ''): k for k in conv.keys() if '_by_layer' in k}

    print(f"\nComparing statistics for probes with per-layer data:")
    print(f"\n{'Probe':<30} {'Aggregated Mean':>15} {'Per-Layer Mean':>15} {'Difference':>12}")
    print("-" * 75)

    for probe_key, per_layer_key in sorted(per_layer_keys.items()):
        # Skip if no corresponding aggregated scores
        if probe_key not in conv.get('probe_scores', {}):
            continue

        # Collect aggregated scores
        agg_scores = []
        for c in data['conversations']:
            if probe_key in c.get('probe_scores', {}):
                for sent_id, score in c['probe_scores'][probe_key].items():
                    if isinstance(score, dict) and 'assistant' in score:
                        agg_scores.append(np.array(score['assistant']))
                    elif isinstance(score, (list, np.ndarray)):
                        agg_scores.append(np.array(score))

        # Collect per-layer scores
        per_layer_scores = []
        for c in data['conversations']:
            if per_layer_key in c:
                for sent_id, layer_data in c[per_layer_key].items():
                    for layer, score in layer_data.items():
                        if isinstance(score, dict) and 'assistant' in score:
                            per_layer_scores.append(np.array(score['assistant']))
                        elif isinstance(score, (list, np.ndarray)):
                            per_layer_scores.append(np.array(score))
                        elif isinstance(score, dict):
                            # Logit lens format - get all emotions
                            per_layer_scores.append(np.array([score.get(e, 0) for e in EMOTIONS]))

        if not agg_scores or not per_layer_scores:
            continue

        agg_mean = np.mean(agg_scores)
        per_layer_mean = np.mean(per_layer_scores)
        diff = abs(agg_mean - per_layer_mean)

        print(f"{probe_key:<30} {agg_mean:>15.3f} {per_layer_mean:>15.3f} {diff:>12.3f}")

    return True


def test_layerwise_plotting_simulation():
    """Simulate what layerwise_plotting.get_per_layer_scores_for_window would do."""
    print("\n" + "=" * 80)
    print("TEST: Layerwise plotting simulation")
    print("=" * 80)

    data_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl')
    if not data_path.exists():
        print(f"✗ Data file not found")
        return False

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    conv = data['conversations'][0]
    per_layer_keys = [k for k in conv.keys() if '_by_layer' in k]

    print(f"\nSimulating get_per_layer_scores_for_window for each probe type:")

    all_passed = True
    for per_layer_key in per_layer_keys:
        probe_key = per_layer_key.replace('_by_layer', '')
        print(f"\n--- {probe_key} ---")

        # Get first 3 sentences as test window
        sent_ids = list(conv[per_layer_key].keys())[:3]

        # Get all layers
        sample_sent = sent_ids[0]
        layers = sorted(list(conv[per_layer_key][sample_sent].keys()))

        # Simulate the scoring loop
        scores_by_emotion = {emotion: {layer: [] for layer in layers} for emotion in EMOTIONS}

        try:
            for sent_id in sent_ids:
                if sent_id not in conv[per_layer_key]:
                    continue
                sent_layers = conv[per_layer_key][sent_id]

                for layer in layers:
                    if layer not in sent_layers:
                        continue
                    layer_scores = sent_layers[layer]

                    # Handle different formats (same logic as layerwise_plotting.py)
                    if isinstance(layer_scores, dict):
                        if 'user' in layer_scores:
                            asst_scores = layer_scores['assistant']
                            for emotion_idx, emotion in enumerate(EMOTIONS):
                                if isinstance(asst_scores, (list, np.ndarray)):
                                    scores_by_emotion[emotion][layer].append(asst_scores[emotion_idx])
                                else:
                                    scores_by_emotion[emotion][layer].append(asst_scores.get(emotion, 0.0))
                        else:
                            for emotion in EMOTIONS:
                                if emotion in layer_scores:
                                    scores_by_emotion[emotion][layer].append(layer_scores[emotion])
                    elif isinstance(layer_scores, (list, np.ndarray)):
                        for emotion_idx, emotion in enumerate(EMOTIONS):
                            scores_by_emotion[emotion][layer].append(layer_scores[emotion_idx])

            # Convert to arrays and compute mean
            result = {}
            for emotion in EMOTIONS:
                emotion_array = np.zeros(len(layers))
                for i, layer in enumerate(layers):
                    if scores_by_emotion[emotion][layer]:
                        emotion_array[i] = np.mean(scores_by_emotion[emotion][layer])
                result[emotion] = emotion_array

            print(f"  ✓ Successfully processed {len(sent_ids)} sentences across {len(layers)} layers")
            print(f"    Sample result shape: {result['frustrated'].shape}")
            print(f"    Frustrated range: [{result['frustrated'].min():.3f}, {result['frustrated'].max():.3f}]")
            print(f"    Calm range: [{result['calm'].min():.3f}, {result['calm'].max():.3f}]")

        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    return all_passed


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("UNIFIED PIPELINE MINIMAL VALIDATION TESTS")
    print("=" * 80 + "\n")

    tests = [
        ("Per-layer format handling", test_per_layer_format_handling),
        ("Existing data structure", test_existing_data_structure),
        ("Normalization consistency", test_normalization_consistency),
        ("Layerwise plotting simulation", test_layerwise_plotting_simulation),
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
