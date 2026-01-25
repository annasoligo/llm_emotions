"""Validate the checkpoint file to check baseline correction is working."""
import pickle
import numpy as np
from pathlib import Path

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

def main():
    checkpoint_path = Path("data/checkpoint_first_conv.pkl")
    backup_path = Path("data/high_emotion_6plus_with_axes.pkl.backup_before_layerwise")

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Waiting for job to complete first conversation...")
        return

    print("Loading checkpoint...")
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)

    print("Loading backup for comparison...")
    with open(backup_path, 'rb') as f:
        backup = pickle.load(f)

    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    # Check correction info
    correction_info = checkpoint['correction_info']
    for probe_key, info in correction_info.items():
        print(f"\n{probe_key}:")
        print(f"  Alpha: {info['alpha']:.4f}")
        print(f"  Correlation BEFORE correction: {info['correlation_before']:.4f}")
        print(f"  Correlation AFTER correction: {info['correlation_after']:.4f}")

        if abs(info['correlation_after']) < 0.05:
            print(f"  ✓ PASS: Correlation after is near zero")
        else:
            print(f"  ⚠ WARNING: Residual correlation = {info['correlation_after']:.4f}")

    # Compare scores with backup
    print("\n" + "-"*80)
    print("SCORE COMPARISON WITH BACKUP (logit_lens_mean)")
    print("-"*80)

    # Get corrected scores from checkpoint
    checkpoint_data = checkpoint['sentence_data_by_range'].get('logit_lens_mean', [])
    if not checkpoint_data:
        print("No data for logit_lens_mean in checkpoint")
        return

    # Get backup scores for conv 0
    backup_conv = backup['conversations'][0]
    backup_scores = backup_conv['probe_scores'].get('logit_lens_mean', {})
    backup_mean_logits = backup_conv.get('sentence_mean_logits', {}).get('logit_lens_mean', {})

    # Compare first 5 sentences
    print("\nFirst 5 sentences comparison:")
    for i, item in enumerate(checkpoint_data[:5]):
        sent_id = item['sent_id']
        new_scores = item.get('emotion_scores_corrected', item['emotion_scores'])

        if sent_id in backup_scores:
            old_scores = np.array(backup_scores[sent_id])
            diff = np.abs(new_scores - old_scores)

            print(f"\n  Sentence {sent_id}:")
            print(f"    NEW (corrected): {new_scores}")
            print(f"    BACKUP:          {old_scores}")
            print(f"    DIFF (max):      {diff.max():.4f}")

    # Calculate overall correlation of NEW corrected scores with mean logit
    print("\n" + "-"*80)
    print("CORRELATION CHECK ON NEW CORRECTED SCORES")
    print("-"*80)

    all_corrected = []
    all_logits = []
    for item in checkpoint_data:
        corrected = item.get('emotion_scores_corrected', item['emotion_scores'])
        all_corrected.append(corrected)
        all_logits.append(item['mean_logit'])

    all_corrected = np.array(all_corrected)
    all_logits = np.array(all_logits)
    mean_corrected = all_corrected.mean(axis=1)

    corr = np.corrcoef(mean_corrected, all_logits)[0, 1]
    print(f"\nCorrelation (mean corrected emotion vs mean logit): {corr:.4f}")

    if abs(corr) < 0.05:
        print("✓ BASELINE CORRECTION WORKING CORRECTLY!")
    elif abs(corr) < 0.2:
        print("⚠ Modest residual correlation - acceptable")
    else:
        print("✗ BASELINE CORRECTION STILL BROKEN - correlation too high")

    # Per-emotion correlations
    print("\nPer-emotion correlations with mean logit:")
    for i, e in enumerate(EMOTIONS):
        corr_e = np.corrcoef(all_corrected[:, i], all_logits)[0, 1]
        status = "✓" if abs(corr_e) < 0.3 else "⚠"
        print(f"  {status} {e}: {corr_e:.4f}")

    # Inter-emotion correlation matrix
    print("\nInter-emotion correlation matrix (should NOT all be >0.9):")
    corr_matrix = np.corrcoef(all_corrected.T)
    print("       " + "  ".join(f"{e[:4]:>6}" for e in EMOTIONS))
    for i, e in enumerate(EMOTIONS):
        row = "  ".join(f"{corr_matrix[i, j]:+.3f}" for j in range(6))
        print(f"{e[:6]:>6} {row}")

    # Check if emotions are too correlated (sign of broken baseline correction)
    off_diag = corr_matrix[np.triu_indices(6, k=1)]
    mean_corr = off_diag.mean()
    print(f"\nMean off-diagonal correlation: {mean_corr:.3f}")
    if mean_corr > 0.9:
        print("✗ EMOTIONS TOO CORRELATED - baseline correction likely broken")
    else:
        print("✓ Emotion correlations look reasonable")

if __name__ == "__main__":
    main()
