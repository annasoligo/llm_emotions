#!/usr/bin/env python3
"""Compare sentence-level logit lens emotion scores: backup (working) vs new (broken)."""
import pickle
import numpy as np
from pathlib import Path

def main():
    print("="*80)
    print("COMPARING SENTENCE-LEVEL LOGIT LENS EMOTION SCORES")
    print("="*80)

    # Load backup file (the one that was working)
    backup_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl.backup_before_layerwise')
    new_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_layerwise.pkl')

    print(f"\nBACKUP (WORKING): {backup_path}")
    with open(backup_path, 'rb') as f:
        backup_data = pickle.load(f)

    print(f"NEW (BROKEN): {new_path}")
    with open(new_path, 'rb') as f:
        new_data = pickle.load(f)

    print(f"\n{'='*80}")
    print(f"CHECKING FIRST CONVERSATION")
    print(f"{'='*80}")

    backup_conv = backup_data['conversations'][0]
    new_conv = new_data['conversations'][0]

    # Check if logit_lens_mean exists in both
    probe_key = 'logit_lens_mean'

    if 'probe_scores' not in backup_conv or probe_key not in backup_conv['probe_scores']:
        print(f"\n✗ BACKUP has no {probe_key} data!")
        return

    if 'probe_scores' not in new_conv or probe_key not in new_conv['probe_scores']:
        print(f"\n✗ NEW has no {probe_key} data!")
        return

    backup_scores = backup_conv['probe_scores'][probe_key]
    new_scores = new_conv['probe_scores'][probe_key]

    print(f"\nBACKUP: {len(backup_scores)} sentences")
    print(f"NEW: {len(new_scores)} sentences")

    # Get first 5 sentence IDs that exist in both
    common_ids = sorted(set(backup_scores.keys()) & set(new_scores.keys()))[:5]

    print(f"\n{'='*80}")
    print(f"COMPARING FIRST 5 SENTENCES")
    print(f"{'='*80}")

    emotions = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']

    for sent_id in common_ids:
        backup_vals = np.array(backup_scores[sent_id])
        new_vals = np.array(new_scores[sent_id])

        print(f"\nSentence #{sent_id}:")
        print(f"  BACKUP: {backup_vals}")
        print(f"  NEW:    {new_vals}")

        diff = np.abs(backup_vals - new_vals)
        if np.max(diff) > 0.001:
            print(f"  DIFF:   {diff}")
            print(f"  ⚠ DIFFERENT! Max diff: {np.max(diff):.4f}")

            # Show per-emotion differences
            for i, emotion in enumerate(emotions):
                if abs(backup_vals[i] - new_vals[i]) > 0.001:
                    print(f"    {emotion}: backup={backup_vals[i]:.4f}, new={new_vals[i]:.4f}, diff={diff[i]:.4f}")
        else:
            print(f"  ✓ IDENTICAL")

    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
