"""Check inter-emotion correlations for all probes in the dashboard."""
import pickle
import numpy as np
from pathlib import Path

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

def compute_correlation(probe_scores):
    """Compute inter-emotion correlation matrix for a probe."""
    all_scores = []

    for sent_id, scores in probe_scores.items():
        if isinstance(scores, np.ndarray) and len(scores) == 6:
            all_scores.append(scores)

    if len(all_scores) < 2:
        return None, 0

    scores_matrix = np.array(all_scores)
    corr_matrix = np.corrcoef(scores_matrix.T)

    # Compute average absolute off-diagonal correlation
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
print("INTER-EMOTION CORRELATIONS BY PROBE")
print("="*80)

# Check each probe
for probe_key in data['probe_configs'].keys():
    print(f"\n{probe_key}:")

    # Collect all sentence scores for this probe across all conversations
    all_probe_scores = {}
    for conv in data['conversations']:
        if 'probe_scores' not in conv or probe_key not in conv['probe_scores']:
            continue

        probe_data = conv['probe_scores'][probe_key]
        all_probe_scores.update(probe_data)

    corr_matrix, avg_corr = compute_correlation(all_probe_scores)

    if corr_matrix is not None:
        print(f"  N sentences: {len(all_probe_scores)}")
        print(f"  Average absolute correlation: {avg_corr:.3f}")
        print(f"  Correlation matrix:")
        print("             ", " ".join(f"{e[:6]:>7s}" for e in EMOTIONS))
        for i, emotion in enumerate(EMOTIONS):
            row = ' '.join(f'{corr_matrix[i, j]:7.3f}' for j in range(6))
            print(f"    {emotion:10s} {row}")
    else:
        print(f"  No valid scores found")

print("\n" + "="*80)
