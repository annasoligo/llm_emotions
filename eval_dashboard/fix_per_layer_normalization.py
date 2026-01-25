#!/usr/bin/env python3
"""
Fix per-layer probe scores by applying proper per-layer z-score normalization.

The existing per-layer scores are raw softmax probabilities. This script converts
them to z-scores using per-layer baselines computed from WildChat data.

For each layer L:
    z_score[L] = (raw_prob[L] - baseline_mean[L]) / baseline_std[L]

Usage:
    python fix_per_layer_normalization.py --input data.pkl --output data_fixed.pkl
    python fix_per_layer_normalization.py --input data.pkl --output data_fixed.pkl --in-place
"""
import pickle
import json
import numpy as np
from pathlib import Path
import argparse
import shutil

# Paths
BASELINE_DIR = Path('/workspace-vast/annas/git/research-tools/data/baselines/probe_baselines')
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# Map from our emotion names to baseline file emotion names
# Dashboard uses: frustrated, curious, delighted, bored, sad, calm
# Baselines use: anger, disgust, fear, happiness, sadness, surprise
EMOTION_MAP = {
    'frustrated': 'anger',      # index 0
    'curious': 'disgust',       # index 1
    'delighted': 'fear',        # index 2
    'bored': 'happiness',       # index 3
    'sad': 'sadness',           # index 4
    'calm': 'surprise'          # index 5
}

# Actually, looking at the code, the indices are direct - probe outputs 6 values
# and we store them in order. Let me check what the actual mapping is.
# The probe_scores are stored as arrays of 6 values indexed 0-5.


def load_per_layer_baselines(probe_type: str) -> dict:
    """
    Load per-layer baseline statistics.

    Returns:
        Dict[layer] = {'mean': array(6), 'std': array(6)}
    """
    baseline_file = BASELINE_DIR / f'{probe_type}_baselines.json'
    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_file}")

    with open(baseline_file, 'r') as f:
        data = json.load(f)

    baselines = {}
    for layer_str, stats in data['baselines'].items():
        layer = int(layer_str)
        baselines[layer] = {
            'mean': np.array(stats['mean']),
            'std': np.array(stats['std'])
        }

    return baselines


def normalize_per_layer_scores(
    per_layer_data: dict,
    baselines: dict,
    min_std: float = 0.01,
    max_zscore: float = 10.0
) -> dict:
    """
    Apply per-layer z-score normalization with robust handling of edge cases.

    Args:
        per_layer_data: {sent_id: {layer: scores_array}}
        baselines: {layer: {'mean': array, 'std': array}}
        min_std: Minimum std value to use (prevents division by ~zero)
        max_zscore: Maximum absolute z-score to allow (clips outliers)

    Returns:
        Normalized per_layer_data with z-scores
    """
    normalized = {}

    for sent_id, layer_scores in per_layer_data.items():
        normalized[sent_id] = {}

        for layer, scores in layer_scores.items():
            if layer not in baselines:
                # Skip layers without baselines
                normalized[sent_id][layer] = scores
                continue

            scores_arr = np.array(scores)
            mean = baselines[layer]['mean']
            std = baselines[layer]['std']

            # Use minimum std threshold to prevent explosion
            std_safe = np.maximum(std, min_std)

            # z-score normalization
            z_scores = (scores_arr - mean) / std_safe

            # Clip extreme values
            z_scores = np.clip(z_scores, -max_zscore, max_zscore)

            normalized[sent_id][layer] = z_scores.tolist()

    return normalized


def fix_pickle_file(input_path: Path, output_path: Path, dry_run: bool = False):
    """Fix per-layer normalization in a pickle file."""

    print("=" * 80)
    print("FIX PER-LAYER NORMALIZATION")
    print("=" * 80)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")

    # Load data
    print("\n[1/4] Loading pickle file...")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    n_convs = len(data['conversations'])
    print(f"  Loaded {n_convs} conversations")

    # Check what per-layer keys exist
    conv = data['conversations'][0]
    per_layer_keys = [k for k in conv.keys() if '_by_layer' in k]
    print(f"\n[2/4] Found per-layer keys: {per_layer_keys}")

    # Load baselines
    print("\n[3/4] Loading per-layer baselines...")
    baselines_loaded = {}

    for key in per_layer_keys:
        probe_type = key.replace('_by_layer', '')
        baseline_file = BASELINE_DIR / f'{probe_type}_baselines.json'

        if baseline_file.exists():
            baselines_loaded[key] = load_per_layer_baselines(probe_type)
            print(f"  ✓ {probe_type}: {len(baselines_loaded[key])} layers")
        else:
            print(f"  ✗ {probe_type}: No baseline file found")

    if not baselines_loaded:
        print("\nERROR: No baselines found for any per-layer key!")
        return False

    # Show before stats
    print("\n[3b/4] BEFORE normalization stats:")
    for key in baselines_loaded.keys():
        all_scores = []
        for c in data['conversations']:
            if key in c:
                for sent_id, layer_data in c[key].items():
                    for layer, scores in layer_data.items():
                        all_scores.append(np.array(scores))
        if all_scores:
            all_scores = np.array(all_scores)
            print(f"  {key}: mean={all_scores.mean():.4f}, std={all_scores.std():.4f}, "
                  f"range=[{all_scores.min():.4f}, {all_scores.max():.4f}]")

    if dry_run:
        print("\n[DRY RUN] Would normalize the following keys:")
        for key in baselines_loaded.keys():
            print(f"  - {key}")
        return True

    # Apply normalization
    print("\n[4/4] Applying per-layer z-score normalization...")

    for i, conv in enumerate(data['conversations']):
        for key, baselines in baselines_loaded.items():
            if key in conv:
                conv[key] = normalize_per_layer_scores(conv[key], baselines)

        if (i + 1) % 10 == 0 or i == n_convs - 1:
            print(f"  Processed {i+1}/{n_convs} conversations")

    # Show after stats
    print("\n[4b/4] AFTER normalization stats:")
    for key in baselines_loaded.keys():
        all_scores = []
        for c in data['conversations']:
            if key in c:
                for sent_id, layer_data in c[key].items():
                    for layer, scores in layer_data.items():
                        all_scores.append(np.array(scores))
        if all_scores:
            all_scores = np.array(all_scores)
            print(f"  {key}: mean={all_scores.mean():.4f}, std={all_scores.std():.4f}, "
                  f"range=[{all_scores.min():.4f}, {all_scores.max():.4f}]")

    # Save
    print(f"\nSaving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved ({file_size_mb:.1f} MB)")

    print("\n" + "=" * 80)
    print("✓ PER-LAYER NORMALIZATION FIXED!")
    print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(description="Fix per-layer probe score normalization")
    parser.add_argument("--input", required=True, help="Input pickle file")
    parser.add_argument("--output", help="Output pickle file (default: input with _fixed suffix)")
    parser.add_argument("--in-place", action="store_true", help="Modify input file in place (backs up original)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without modifying")
    args = parser.parse_args()

    input_path = Path(args.input)

    if args.in_place:
        # Backup original
        backup_path = input_path.with_suffix('.pkl.backup_prenorm')
        print(f"Backing up to {backup_path}")
        shutil.copy(input_path, backup_path)
        output_path = input_path
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_stem(input_path.stem + '_fixed')

    success = fix_pickle_file(input_path, output_path, dry_run=args.dry_run)

    if not success:
        print("\nFailed to fix normalization!")
        exit(1)


if __name__ == '__main__':
    main()
