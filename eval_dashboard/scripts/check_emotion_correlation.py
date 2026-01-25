#!/usr/bin/env python3
"""Check emotion correlation in logit lens scores."""

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

print("="*80)
print("EMOTION CORRELATION ANALYSIS")
print("="*80)

# Get logit lens scores for first conversation
conv = data['conversations'][0]
probe_key = 'logit_lens_mean'

if probe_key not in conv['probe_scores']:
    print(f"ERROR: {probe_key} not found in conversation 0")
    print(f"Available probes: {list(conv['probe_scores'].keys())}")
    exit(1)

# Get scores - logit lens is organized by sentence index
emotions = EKMAN6_EMOTIONS

print(f"\nProbe: {probe_key}")
print(f"Emotions: {emotions}")

# Get scores for first sentence (sentence index 0)
sentence_0_scores = conv['probe_scores'][probe_key][0]
print(f"\nSentence 0 scores:")
for i, emotion in enumerate(emotions):
    print(f"  {emotion}: {sentence_0_scores[i]:.4f}")

# Check correlation across all sentences in first conv
print(f"\n{'='*80}")
print("CORRELATION ACROSS SENTENCES")
print(f"{'='*80}")

all_sentences_scores = []
for sent_idx in sorted(conv['probe_scores'][probe_key].keys()):
    all_sentences_scores.append(conv['probe_scores'][probe_key][sent_idx])

all_sentences_scores = np.array(all_sentences_scores)  # (n_sentences, n_emotions)
print(f"Collected scores from {len(all_sentences_scores)} sentences")

# Compute correlation matrix
corr_matrix = np.corrcoef(all_sentences_scores.T)  # (n_emotions, n_emotions)

print("\nCorrelation matrix:")
print("      ", end="")
for em in emotions:
    print(f"{em[:6]:>7}", end="")
print()

for i, em1 in enumerate(emotions):
    print(f"{em1[:6]:>6}", end="")
    for j, em2 in enumerate(emotions):
        if i == j:
            print(f"   1.00", end="")
        else:
            print(f"  {corr_matrix[i,j]:>5.2f}", end="")
    print()

# Average off-diagonal correlation
mask = ~np.eye(len(emotions), dtype=bool)
avg_corr = corr_matrix[mask].mean()
print(f"\nAverage correlation (off-diagonal): {avg_corr:.3f}")

# Check if emotions vary independently
print(f"\nScore statistics (all sentences):")
for i, emotion in enumerate(emotions):
    scores = all_sentences_scores[:, i]
    print(f"  {emotion}: min={scores.min():.3f}, max={scores.max():.3f}, mean={scores.mean():.3f}, std={scores.std():.3f}")

print(f"\n{'='*80}")
print("Comparing to orthogonal probe...")
print(f"{'='*80}")

# Compare to another probe (organized by layer)
other_probe = 'orthogonal_raw'
if other_probe in conv['probe_scores']:
    # Orthogonal probes are organized by sentence -> layer -> emotions
    # We need to aggregate across a specific layer
    layer = 30
    all_sentences_scores_other = []

    for sent_idx in conv['sentences']:
        sent_scores = conv['probe_scores'][other_probe].get(sent_idx['sentence_index'], {})
        if layer in sent_scores:
            all_sentences_scores_other.append(sent_scores[layer])

    if all_sentences_scores_other:
        all_sentences_scores_other = np.array(all_sentences_scores_other)
        corr_matrix_other = np.corrcoef(all_sentences_scores_other.T)
        mask = ~np.eye(len(emotions), dtype=bool)
        avg_corr_other = corr_matrix_other[mask].mean()

        print(f"\n{other_probe} (layer {layer}) average correlation: {avg_corr_other:.3f}")
        print(f"{probe_key} average correlation: {avg_corr:.3f}")
        print(f"Difference: {avg_corr - avg_corr_other:.3f} ({'higher' if avg_corr > avg_corr_other else 'lower'})")
