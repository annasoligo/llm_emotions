#!/usr/bin/env python3
"""Compare backup (working) with new (broken) layerwise data."""
import pickle
import numpy as np
from pathlib import Path

def main():
    print("="*80)
    print("COMPARING BACKUP (WORKING) VS NEW (BROKEN) LAYERWISE DATA")
    print("="*80)

    # Load backup file (the one that was working)
    backup_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl.backup_before_layerwise')
    new_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_layerwise.pkl')

    print(f"\nBACKUP (WORKING): {backup_path}")
    with open(backup_path, 'rb') as f:
        backup_data = pickle.load(f)

    print(f"\nNEW (BROKEN): {new_path}")
    with open(new_path, 'rb') as f:
        new_data = pickle.load(f)

    print(f"\n{'='*80}")
    print(f"STRUCTURE COMPARISON")
    print(f"{'='*80}")

    backup_conv = backup_data['conversations'][0]
    new_conv = new_data['conversations'][0]

    # Find a sentence with layerwise data in both
    backup_sent = None
    new_sent = None

    for i, sent in enumerate(backup_conv['sentences']):
        if 'layerwise' in sent:
            backup_sent = sent
            backup_idx = i
            break

    for i, sent in enumerate(new_conv['sentences']):
        if 'layerwise' in sent:
            new_sent = sent
            new_idx = i
            break

    if not backup_sent:
        print("\n✗ BACKUP file has NO layerwise data!")
        return

    if not new_sent:
        print("\n✗ NEW file has NO layerwise data!")
        return

    print(f"\nBACKUP sentence #{backup_idx} (first with layerwise):")
    print(f"  Text: {backup_sent['text'][:80]}...")
    print(f"  Layerwise keys: {list(backup_sent['layerwise'].keys())}")

    print(f"\nNEW sentence #{new_idx} (first with layerwise):")
    print(f"  Text: {new_sent['text'][:80]}...")
    print(f"  Layerwise keys: {list(new_sent['layerwise'].keys())}")

    # Compare specific emotion scores
    print(f"\n{'='*80}")
    print(f"EMOTION SCORE COMPARISON (FIRST SENTENCE WITH LAYERWISE)")
    print(f"{'='*80}")

    for key in ['logit_lens_mean', 'logit_lens_max']:
        if key not in backup_sent['layerwise']:
            print(f"\n{key}: NOT IN BACKUP")
            continue
        if key not in new_sent['layerwise']:
            print(f"\n{key}: NOT IN NEW")
            continue

        print(f"\n{key}:")
        backup_scores = backup_sent['layerwise'][key]
        new_scores = new_sent['layerwise'][key]

        print(f"  BACKUP keys: {list(backup_scores.keys())}")
        print(f"  NEW keys: {list(new_scores.keys())}")

        for emotion in ['happiness', 'sadness', 'anger']:
            if emotion not in backup_scores:
                print(f"\n  {emotion.upper()}: NOT IN BACKUP")
                continue
            if emotion not in new_scores:
                print(f"\n  {emotion.upper()}: NOT IN NEW")
                continue

            backup_vals = backup_scores[emotion]
            new_vals = new_scores[emotion]

            print(f"\n  {emotion.upper()}:")
            print(f"    BACKUP: {len(backup_vals)} layers")
            if len(backup_vals) > 0:
                valid_backup = [v for v in backup_vals if v == v]  # filter NaN
                if valid_backup:
                    print(f"      Sample (layers 0,20,40,60): ", end="")
                    samples = [backup_vals[i] if i < len(backup_vals) else 'N/A' for i in [0, 20, 40, 60]]
                    print([f"{s:.4f}" if isinstance(s, float) and s == s else 'NaN' for s in samples])
                    print(f"      Stats: min={min(valid_backup):.4f}, max={max(valid_backup):.4f}, mean={np.mean(valid_backup):.4f}")
                    print(f"      Valid: {len(valid_backup)}/{len(backup_vals)}")
                else:
                    print(f"      ALL NaN")

            print(f"    NEW: {len(new_vals)} layers")
            if len(new_vals) > 0:
                valid_new = [v for v in new_vals if v == v]  # filter NaN
                if valid_new:
                    print(f"      Sample (layers 0,20,40,60): ", end="")
                    samples = [new_vals[i] if i < len(new_vals) else 'N/A' for i in [0, 20, 40, 60]]
                    print([f"{s:.4f}" if isinstance(s, float) and s == s else 'NaN' for s in samples])
                    print(f"      Stats: min={min(valid_new):.4f}, max={max(valid_new):.4f}, mean={np.mean(valid_new):.4f}")
                    print(f"      Valid: {len(valid_new)}/{len(new_vals)}")
                else:
                    print(f"      ALL NaN")

            # Check if values are different
            if len(backup_vals) == len(new_vals):
                differences = []
                for i, (b, n) in enumerate(zip(backup_vals, new_vals)):
                    if b != b and n != n:  # both NaN
                        continue
                    if abs(b - n) > 0.001:  # Different by more than threshold
                        differences.append((i, b, n, abs(b - n)))

                if differences:
                    print(f"\n    ⚠ DIFFERENT: {len(differences)}/{len(backup_vals)} values differ")
                    print(f"      Top 5 differences:")
                    for i, b, n, diff in sorted(differences, key=lambda x: x[3], reverse=True)[:5]:
                        print(f"        Layer {i}: backup={b:.4f}, new={n:.4f}, diff={diff:.4f}")
                else:
                    print(f"\n    ✓ IDENTICAL")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
