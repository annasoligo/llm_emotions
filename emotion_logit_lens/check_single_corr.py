"""
Check correlation in single conversation test output.
"""
import pickle
import numpy as np

# Load the debug output
with open('/tmp/debug_single.pkl', 'rb') as f:
    data = pickle.load(f)

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

conv = data['conversations'][0]
logit_scores = conv['probe_scores']['logit_lens_mean']

# Collect all scores
all_scores = []
for sent_id in sorted(logit_scores.keys()):
    scores = logit_scores[sent_id]
    if isinstance(scores, np.ndarray) and len(scores) == 6:
        all_scores.append(scores)

if not all_scores:
    print("No scores found!")
    exit(1)

scores_matrix = np.array(all_scores)  # Shape: (n_sentences, 6)

print("="*80)
print("CORRELATION CHECK - SINGLE CONVERSATION")
print("="*80)

print(f"\nScores matrix shape: {scores_matrix.shape}")
print(f"\nScore statistics:")
print(f"  Min:  {scores_matrix.min():.3f}")
print(f"  Max:  {scores_matrix.max():.3f}")
print(f"  Mean: {scores_matrix.mean():.3f}")
print(f"  Std:  {scores_matrix.std():.3f}")

# Compute correlation between emotions
corr_matrix = np.corrcoef(scores_matrix.T)

print(f"\nCorrelation matrix:")
print("           ", " ".join(f"{e[:6]:>7s}" for e in EMOTIONS))
for i, emotion in enumerate(EMOTIONS):
    row = ' '.join(f'{corr_matrix[i, j]:7.3f}' for j in range(len(EMOTIONS)))
    print(f"  {emotion:10s} {row}")

# Average absolute correlation (excluding diagonal)
off_diag_corrs = []
for i in range(len(EMOTIONS)):
    for j in range(i+1, len(EMOTIONS)):
        off_diag_corrs.append(abs(corr_matrix[i, j]))

avg_abs_corr = np.mean(off_diag_corrs)
print(f"\nAverage absolute correlation: {avg_abs_corr:.3f}")
print(f"(Should be LOW if using emo_lens baseline correctly)")

if avg_abs_corr < 0.3:
    print("✓ GOOD - Low correlation!")
elif avg_abs_corr < 0.5:
    print("⚠ MODERATE - Some correlation")
else:
    print("✗ BAD - High correlation (baseline may be wrong)")

print("="*80)
