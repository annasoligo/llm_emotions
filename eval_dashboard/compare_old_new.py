#!/usr/bin/env python3
"""Compare old working layerwise data with new broken data."""
import pickle
import numpy as np
from pathlib import Path

def main():
    print("="*80)
    print("COMPARING OLD VS NEW LAYERWISE DATA")
    print("="*80)

    # Load old file (the one that was working)
    old_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_logit.pkl')
    new_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_layerwise.pkl')

    print(f"\nOLD: {old_path}")
    print(f"     Modified: {old_path.stat().st_mtime}")
    with open(old_path, 'rb') as f:
        old_data = pickle.load(f)

    print(f"\nNEW: {new_path}")
    print(f"     Modified: {new_path.stat().st_mtime}")
    with open(new_path, 'rb') as f:
        new_data = pickle.load(f)

    print(f"\n{'='*80}")
    print(f"STRUCTURE COMPARISON")
    print(f"{'='*80}")

    old_conv = old_data['conversations'][0]
    new_conv = new_data['conversations'][0]

    # Find a sentence with layerwise data in both
    old_sent = None
    new_sent = None

    for sent in old_conv['sentences']:
        if 'layerwise' in sent:
            old_sent = sent
            break

    for sent in new_conv['sentences']:
        if 'layerwise' in sent:
            new_sent = sent
            break

    if not old_sent:
        print("\n✗ OLD file has NO layerwise data!")
        return

    if not new_sent:
        print("\n✗ NEW file has NO layerwise data!")
        return

    print(f"\nOLD sentence (first with layerwise):")
    print(f"  Text: {old_sent['text'][:80]}...")
    print(f"  Layerwise keys: {list(old_sent['layerwise'].keys())}")

    print(f"\nNEW sentence (first with layerwise):")
    print(f"  Text: {new_sent['text'][:80]}...")
    print(f"  Layerwise keys: {list(new_sent['layerwise'].keys())}")

    # Compare specific emotion scores
    print(f"\n{'='*80}")
    print(f"EMOTION SCORE COMPARISON")
    print(f"{'='*80}")

    for key in ['logit_lens_mean', 'logit_lens_max']:
        if key not in old_sent['layerwise']:
            print(f"\n{key}: NOT IN OLD")
            continue
        if key not in new_sent['layerwise']:
            print(f"\n{key}: NOT IN NEW")
            continue

        print(f"\n{key}:")
        old_scores = old_sent['layerwise'][key]
        new_scores = new_sent['layerwise'][key]

        print(f"  OLD keys: {list(old_scores.keys())}")
        print(f"  NEW keys: {list(new_scores.keys())}")

        for emotion in ['happiness', 'sadness', 'anger']:
            if emotion in old_scores and emotion in new_scores:
                old_vals = old_scores[emotion]
                new_vals = new_scores[emotion]

                print(f"\n  {emotion.upper()}:")
                print(f"    OLD: {len(old_vals)} layers")
                if len(old_vals) > 0:
                    valid_old = [v for v in old_vals if v == v]  # filter NaN
                    if valid_old:
                        print(f"      Sample (layers 0,20,40,60): ", end="")
                        samples = [old_vals[i] if i < len(old_vals) else 'N/A' for i in [0, 20, 40, 60]]
                        print([f"{s:.4f}" if isinstance(s, float) else s for s in samples])
                        print(f"      Stats: min={min(valid_old):.4f}, max={max(valid_old):.4f}, mean={np.mean(valid_old):.4f}")
                        print(f"      NaN count: {len(old_vals) - len(valid_old)}/{len(old_vals)}")
                    else:
                        print(f"      ALL NaN")

                print(f"    NEW: {len(new_vals)} layers")
                if len(new_vals) > 0:
                    valid_new = [v for v in new_vals if v == v]  # filter NaN
                    if valid_new:
                        print(f"      Sample (layers 0,20,40,60): ", end="")
                        samples = [new_vals[i] if i < len(new_vals) else 'N/A' for i in [0, 20, 40, 60]]
                        print([f"{s:.4f}" if isinstance(s, float) else s for s in samples])
                        print(f"      Stats: min={min(valid_new):.4f}, max={max(valid_new):.4f}, mean={np.mean(valid_new):.4f}")
                        print(f"      NaN count: {len(new_vals) - len(valid_new)}/{len(new_vals)}")
                    else:
                        print(f"      ALL NaN")

                # Check if values are different
                if len(old_vals) == len(new_vals):
                    differences = 0
                    max_diff = 0
                    for o, n in zip(old_vals, new_vals):
                        if o != o and n != n:  # both NaN
                            continue
                        if o != n:
                            differences += 1
                            if abs(o - n) > max_diff:
                                max_diff = abs(o - n)

                    if differences > 0:
                        print(f"    ⚠ DIFFERENT: {differences}/{len(old_vals)} values differ (max diff: {max_diff:.4f})")
                    else:
                        print(f"    ✓ IDENTICAL")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
