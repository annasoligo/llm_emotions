"""
Check correlation in updated dashboard data to verify the fix worked.
"""
import pickle
import numpy as np
from pathlib import Path

# Load dashboard data
data_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus.pkl')

print("=" * 80)
print("CHECKING EMOTION CORRELATION AFTER FIX")
print("=" * 80)

with open(data_path, 'rb') as f:
    data = pickle.load(f)

print(f"\n✓ Loaded dashboard data from {data_path}")
print(f"  Available probes: {list(data['probe_configs'].keys())}")

# Extract logit lens scores
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

for probe_key in ['logit_lens_mean', 'logit_lens_max']:
    print(f"\n{'=' * 80}")
    print(f"ANALYZING: {probe_key}")
    print(f"{'=' * 80}")

    # Collect all scores across all conversations and sentences
    all_scores = []

    for conv in data['conversations']:
        probe_scores = conv.get('probe_scores', {}).get(probe_key, {})

        for sentence_id, scores in probe_scores.items():
            # scores should be shape (n_emotions,)
            if isinstance(scores, np.ndarray) and len(scores) == len(EMOTIONS):
                all_scores.append(scores)

    if not all_scores:
        print(f"  ⚠ No scores found for {probe_key}")
        continue

    # Convert to matrix: (n_sentences, n_emotions)
    scores_matrix = np.array(all_scores)
    print(f"\n  Collected {len(all_scores)} sentence scores")
    print(f"  Score matrix shape: {scores_matrix.shape}")

    # Basic statistics
    print(f"\n  Score statistics:")
    print(f"    Min:  {scores_matrix.min():.3f}")
    print(f"    Max:  {scores_matrix.max():.3f}")
    print(f"    Mean: {scores_matrix.mean():.3f}")
    print(f"    Std:  {scores_matrix.std():.3f}")

    # Compute correlation matrix
    corr_matrix = np.corrcoef(scores_matrix.T)

    print(f"\n  Correlation matrix between emotions:")
    print(f"              {' '.join(f'{e[:6]:>7s}' for e in EMOTIONS)}")
    for i, emotion in enumerate(EMOTIONS):
        row = ' '.join(f'{corr_matrix[i, j]:7.3f}' for j in range(len(EMOTIONS)))
        print(f"    {emotion:10s} {row}")

    # Compute average absolute correlation (excluding diagonal)
    off_diag_corrs = []
    for i in range(len(EMOTIONS)):
        for j in range(i+1, len(EMOTIONS)):
            off_diag_corrs.append(abs(corr_matrix[i, j]))

    avg_abs_corr = np.mean(off_diag_corrs)
    print(f"\n  Average absolute correlation: {avg_abs_corr:.3f}")
    print(f"  (BEFORE fix: 0.760 for mean, 0.934 for max)")

    if avg_abs_corr < 0.3:
        print(f"  ✓ FIXED! Correlation is now much lower")
    elif avg_abs_corr < 0.5:
        print(f"  ⚠ Improved but still somewhat high")
    else:
        print(f"  ✗ Still too high - fix may not have worked")

# Compare with orthogonal probe
print(f"\n{'=' * 80}")
print(f"REFERENCE: orthogonal_raw (should have low correlation)")
print(f"{'=' * 80}")

probe_key = 'orthogonal_raw'
all_scores_user = []
all_scores_asst = []

for conv in data['conversations']:
    probe_scores = conv.get('probe_scores', {}).get(probe_key, {})

    for sentence_id, scores in probe_scores.items():
        # Orthogonal probes return dict with 'user' and 'assistant' keys
        if isinstance(scores, dict):
            if 'user' in scores:
                all_scores_user.append(scores['user'])
            if 'assistant' in scores:
                all_scores_asst.append(scores['assistant'])
        elif isinstance(scores, np.ndarray):
            all_scores_user.append(scores)

# Combine user and assistant
all_scores_ortho = all_scores_user + all_scores_asst

if all_scores_ortho:
    scores_matrix = np.array(all_scores_ortho)
    corr_matrix = np.corrcoef(scores_matrix.T)

    print(f"\n  Collected {len(all_scores_ortho)} sentence scores")

    off_diag_corrs = []
    for i in range(len(EMOTIONS)):
        for j in range(i+1, len(EMOTIONS)):
            off_diag_corrs.append(abs(corr_matrix[i, j]))

    avg_abs_corr = np.mean(off_diag_corrs)
    print(f"  Average absolute correlation: {avg_abs_corr:.3f}")
    print(f"  (This should be close to 0, e.g., < 0.1)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
