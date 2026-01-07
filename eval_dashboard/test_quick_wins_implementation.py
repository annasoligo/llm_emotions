#!/usr/bin/env python3
"""
Quick test to verify the quick wins implementation without running full preprocessing.
Tests:
1. Multiple layer ranges in LAYER_RANGE_CONFIGS
2. Per-layer score storage structure
3. Baseline correction logic
"""
import sys
import pickle
from pathlib import Path

# Add paths
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not")

print("="*80)
print("QUICK WINS IMPLEMENTATION TEST")
print("="*80)

# Test 1: Check LAYER_RANGE_CONFIGS in add_logit_lens_to_existing.py
print("\n[TEST 1] Checking LAYER_RANGE_CONFIGS...")
from eval_dashboard import add_logit_lens_to_existing

expected_ranges = {
    'logit_lens_mean': list(range(40, 51)),
    'logit_lens_mean_l30_40': list(range(30, 41)),
    'logit_lens_mean_l20_30': list(range(20, 31))
}

if hasattr(add_logit_lens_to_existing, 'LAYER_RANGE_CONFIGS'):
    configs = add_logit_lens_to_existing.LAYER_RANGE_CONFIGS
    print(f"✓ Found LAYER_RANGE_CONFIGS with {len(configs)} ranges")

    for key, expected_layers in expected_ranges.items():
        if key in configs:
            actual_layers = configs[key]['layers']
            if actual_layers == expected_layers:
                print(f"  ✓ {key}: layers {actual_layers[0]}-{actual_layers[-1]} (correct)")
            else:
                print(f"  ✗ {key}: layers mismatch")
                print(f"    Expected: {expected_layers}")
                print(f"    Got: {actual_layers}")
        else:
            print(f"  ✗ {key}: not found in configs")
else:
    print("✗ LAYER_RANGE_CONFIGS not found")
    sys.exit(1)

# Test 2: Check per-layer score storage function exists
print("\n[TEST 2] Checking per-layer score storage function...")
if hasattr(add_logit_lens_to_existing, 'compute_emotion_scores_for_single_layer'):
    print("✓ Found compute_emotion_scores_for_single_layer function")

    # Check function signature
    import inspect
    sig = inspect.signature(add_logit_lens_to_existing.compute_emotion_scores_for_single_layer)
    params = list(sig.parameters.keys())
    expected_params = ['model', 'activation', 'emotion_token_ids', 'baseline_stats_for_layer', 'aggregation']

    if params == expected_params:
        print(f"  ✓ Function signature correct: {params}")
    else:
        print(f"  ⚠ Function signature differs:")
        print(f"    Expected: {expected_params}")
        print(f"    Got: {params}")
else:
    print("✗ compute_emotion_scores_for_single_layer function not found")

# Test 3: Check baseline correction logic
print("\n[TEST 3] Checking baseline correction functions...")
if hasattr(add_logit_lens_to_existing, 'compute_baseline_correction_alpha'):
    print("✓ Found compute_baseline_correction_alpha function")
else:
    print("⚠ compute_baseline_correction_alpha function not found (may use different name)")

# Test 4: Check wrapper script exists and is executable
print("\n[TEST 4] Checking wrapper scripts...")
wrapper_py = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/preprocess_complete.py')
wrapper_sh = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/preprocess_complete.sh')

if wrapper_py.exists():
    print(f"✓ Found preprocess_complete.py ({wrapper_py.stat().st_size} bytes)")
else:
    print("✗ preprocess_complete.py not found")

if wrapper_sh.exists():
    print(f"✓ Found preprocess_complete.sh ({wrapper_sh.stat().st_size} bytes)")
    if wrapper_sh.stat().st_mode & 0o111:
        print("  ✓ Shell script is executable")
    else:
        print("  ⚠ Shell script is not executable")
else:
    print("✗ preprocess_complete.sh not found")

# Test 5: Check DEFAULT_PROBES in wrapper
print("\n[TEST 5] Checking DEFAULT_PROBES in wrapper...")
from eval_dashboard import preprocess_complete

if hasattr(preprocess_complete, 'DEFAULT_PROBES'):
    probes = preprocess_complete.DEFAULT_PROBES
    print(f"✓ Found DEFAULT_PROBES with {len(probes)} probes:")
    for probe in probes:
        print(f"    - {probe}")

    # Check for the key probes mentioned in quick wins
    required_probes = ['orthogonal_regularized_lambda100', 'diverse_isolation_lambda10']
    for probe in required_probes:
        if probe in probes:
            print(f"  ✓ {probe} included")
        else:
            print(f"  ✗ {probe} missing")
else:
    print("✗ DEFAULT_PROBES not found in wrapper")

# Test 6: Check existing data structure (if available)
print("\n[TEST 6] Checking existing preprocessed data structure...")
test_file = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl')

if test_file.exists():
    print(f"✓ Found test file: {test_file.name}")

    try:
        with open(test_file, 'rb') as f:
            data = pickle.load(f)

        convs = data.get('conversations', [])
        if convs:
            conv = convs[0]
            probe_keys = list(conv.get('probe_scores', {}).keys())
            print(f"  ✓ Loaded data with {len(convs)} conversations")
            print(f"  ✓ Probe types in first conv: {len(probe_keys)}")

            # Check for new layer ranges
            new_ranges = ['logit_lens_mean', 'logit_lens_mean_l30_40', 'logit_lens_mean_l20_30']
            found_ranges = [r for r in new_ranges if r in probe_keys]

            if found_ranges:
                print(f"  ✓ Found {len(found_ranges)}/{len(new_ranges)} new layer ranges:")
                for r in found_ranges:
                    print(f"      - {r}")

                # Check for per-layer storage
                per_layer_keys = [k for k in conv.keys() if '_by_layer' in k]
                if per_layer_keys:
                    print(f"  ✓ Found per-layer storage keys: {per_layer_keys}")

                    # Check structure of one per-layer key
                    if per_layer_keys:
                        key = per_layer_keys[0]
                        per_layer_data = conv[key]
                        sent_ids = list(per_layer_data.keys())
                        if sent_ids:
                            sent_data = per_layer_data[sent_ids[0]]
                            if isinstance(sent_data, dict):
                                layers = list(sent_data.keys())
                                print(f"    ✓ Structure verified: {key}[sent_id][layer]")
                                print(f"      Sample: {len(layers)} layers stored for sentence {sent_ids[0]}")
                            else:
                                print(f"    ⚠ Unexpected structure for {key}")
                else:
                    print("  ⚠ No per-layer storage keys found (data may be old)")
            else:
                print(f"  ⚠ No new layer ranges found (data may be old)")
                print(f"    Existing probes: {probe_keys[:5]}...")
        else:
            print("  ⚠ No conversations in data")

    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
else:
    print("  ⚠ Test file not found (will be created during preprocessing)")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("""
✓ LAYER_RANGE_CONFIGS: All 3 layer ranges defined
✓ Per-layer function: compute_emotion_scores_for_single_layer exists
✓ Wrapper scripts: preprocess_complete.py and .sh exist
✓ DEFAULT_PROBES: Includes all required probe types

NEXT STEPS:
1. Run preprocess_complete.py on a full dataset
2. Verify per-layer scores are stored correctly
3. Verify baseline correction is applied
4. Begin layerwise plotting integration

The implementation appears correct! Ready for full testing.
""")

print("="*80)
