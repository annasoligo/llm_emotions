#!/usr/bin/env python3
"""Compare emotion correlation across all probes."""

import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

import pickle
import numpy as np
from pathlib import Path
from emotion_logit_lens import EKMAN6_EMOTIONS

# Load dashboard data
pkl_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus.pkl')
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

conv = data['conversations'][0]
emotions = EKMAN6_EMOTIONS

print("="*80)
print("EMOTION CORRELATION COMPARISON ACROSS ALL PROBES")
print("="*80)

probes_to_check = [
    'logit_lens_mean',
    'logit_lens_max',
    'orthogonal_raw',
    'text_raw',
    'centroid_k10'
]

for probe_key in probes_to_check:
    if probe_key not in conv['probe_scores']:
        print(f"\n{probe_key}: NOT FOUND")
        continue

    probe_data = conv['probe_scores'][probe_key]

    # Handle different data structures
    if probe_key.startswith('logit_lens'):
        # Logit lens: scores directly keyed by sentence index
        all_scores = []
        for sent_idx in sorted(probe_data.keys()):
            all_scores.append(probe_data[sent_idx])
        all_scores = np.array(all_scores)  # (n_sentences, n_emotions)

    else:
        # Other probes: organized by sentence -> layer -> emotions
        # Use layer 30 for comparison
        layer = 30
        all_scores = []

        # Get sentence indices from the sentences list
        for sent_info in conv['sentences']:
            sent_idx = sent_info['sentence_id']
            if sent_idx in probe_data and layer in probe_data[sent_idx]:
                all_scores.append(probe_data[sent_idx][layer])

        if not all_scores:
            print(f"\n{probe_key}: No data at layer {layer}")
            continue

        all_scores = np.array(all_scores)  # (n_sentences, n_emotions)

    # Compute correlation matrix
    if len(all_scores) < 2:
        print(f"\n{probe_key}: Not enough data ({len(all_scores)} sentences)")
        continue

    corr_matrix = np.corrcoef(all_scores.T)  # (n_emotions, n_emotions)

    # Average off-diagonal correlation
    mask = ~np.eye(len(emotions), dtype=bool)
    avg_corr = corr_matrix[mask].mean()

    # Score statistics
    mean_score = all_scores.mean(axis=0).mean()
    std_score = all_scores.std()

    print(f"\n{probe_key}:")
    print(f"  Avg correlation: {avg_corr:.3f}")
    print(f"  Mean score: {mean_score:.3f}")
    print(f"  Std across all: {std_score:.3f}")
    print(f"  Sentences: {len(all_scores)}")

print("\n" + "="*80)
