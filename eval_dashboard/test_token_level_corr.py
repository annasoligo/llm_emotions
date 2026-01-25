"""Test token-level correlation from logit_lens scores."""
import pickle
import numpy as np
from pathlib import Path

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

def compute_token_level_correlation(data):
    """Compute correlation using individual token scores, not sentence aggregates."""
    all_token_scores = []

    for conv in data['conversations']:
        if 'probe_scores' not in conv or 'logit_lens_mean' not in conv['probe_scores']:
            continue

        # For each sentence, we have the MEAN of token scores
        # But we want the individual token scores
        # Actually, we don't have access to individual token scores in the pickle
        # Only the sentence-level aggregates

        for sent_id, scores in conv['probe_scores']['logit_lens_mean'].items():
            if isinstance(scores, np.ndarray) and len(scores) == 6:
                all_token_scores.append(scores)

    if len(all_token_scores) < 2:
        return None, 0

    scores_matrix = np.array(all_token_scores)
    print(f"Score matrix shape: {scores_matrix.shape}")
    print(f"Score statistics: mean={scores_matrix.mean():.3f}, std={scores_matrix.std():.3f}")
    print(f"Per-emotion means: {scores_matrix.mean(axis=0)}")
    print(f"Per-emotion stds: {scores_matrix.std(axis=0)}")

    corr_matrix = np.corrcoef(scores_matrix.T)

    off_diag_corrs = []
    for i in range(6):
        for j in range(i+1, 6):
            off_diag_corrs.append(abs(corr_matrix[i, j]))

    return corr_matrix, np.mean(off_diag_corrs)

# Load data
data_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus.pkl')
with open(data_path, 'rb') as f:
    data = pickle.load(f)

print("="*80)
print("TOKEN-LEVEL CORRELATION ANALYSIS")
print("="*80)

corr_matrix, avg_corr = compute_token_level_correlation(data)

if corr_matrix is not None:
    print(f"\nAverage absolute correlation: {avg_corr:.3f}")
    print(f"\nCorrelation matrix:")
    print("             ", " ".join(f"{e[:6]:>7s}" for e in EMOTIONS))
    for i, emotion in enumerate(EMOTIONS):
        row = ' '.join(f'{corr_matrix[i, j]:7.3f}' for j in range(6))
        print(f"  {emotion:10s} {row}")

print("\n" + "="*80)
