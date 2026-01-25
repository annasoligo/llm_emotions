#!/usr/bin/env python3
"""Debug script to understand differences in score computation."""
import pickle
import numpy as np
from pathlib import Path

def main():
    print("="*80)
    print("DEBUGGING SCORE COMPUTATION DIFFERENCES")
    print("="*80)

    # Load both files
    backup_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_axes.pkl.backup_before_layerwise')
    new_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus_with_layerwise.pkl')

    print(f"\nLoading files...")
    with open(backup_path, 'rb') as f:
        backup_data = pickle.load(f)
    with open(new_path, 'rb') as f:
        new_data = pickle.load(f)

    # Check first conversation, first 5 sentences
    conv_idx = 0
    backup_conv = backup_data['conversations'][conv_idx]
    new_conv = new_data['conversations'][conv_idx]

    probe_key = 'logit_lens_mean'
    backup_scores = backup_conv['probe_scores'][probe_key]
    new_scores = new_conv['probe_scores'][probe_key]

    sent_ids = sorted(set(backup_scores.keys()) & set(new_scores.keys()))[:5]

    print(f"\n{'='*80}")
    print(f"Analyzing differences for sentences: {sent_ids}")
    print(f"{'='*80}")

    emotions = ['happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']

    for sent_id in sent_ids:
        backup_vals = np.array(backup_scores[sent_id])
        new_vals = np.array(new_scores[sent_id])
        diff = new_vals - backup_vals

        print(f"\nSentence #{sent_id}:")
        print(f"  BACKUP: {backup_vals}")
        print(f"  NEW:    {new_vals}")
        print(f"  DIFF (NEW - BACKUP): {diff}")
        print(f"  Diff statistics: mean={np.mean(diff):.4f}, std={np.std(diff):.4f}, range=[{np.min(diff):.4f}, {np.max(diff):.4f}]")

        # Check if the diff is constant across emotions (would suggest baseline correction issue)
        diff_std = np.std(diff)
        if diff_std < 0.01:
            print(f"  → Differences are nearly constant → likely baseline correction issue")
        else:
            print(f"  → Differences vary by emotion → likely emotion score computation issue")

if __name__ == "__main__":
    main()
