"""
Inspect actual scores in dashboard data.
"""
import pickle
import numpy as np

data_path = '/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus.pkl'

with open(data_path, 'rb') as f:
    data = pickle.load(f)

# Look at first conversation
conv = data['conversations'][0]
print(f"Sample ID: {conv['sample_id']}")
print(f"\nAvailable probes: {list(conv['probe_scores'].keys())}")

# Check logit_lens_mean scores
logit_scores = conv['probe_scores']['logit_lens_mean']
print(f"\nlogit_lens_mean structure:")
print(f"  Type: {type(logit_scores)}")
print(f"  Num sentences: {len(logit_scores)}")

# Look at first sentence
first_sentence_id = list(logit_scores.keys())[0]
first_scores = logit_scores[first_sentence_id]

print(f"\nFirst sentence (id={first_sentence_id}) scores:")
print(f"  Type: {type(first_scores)}")
print(f"  Shape: {first_scores.shape if isinstance(first_scores, np.ndarray) else 'N/A'}")
print(f"  Values: {first_scores}")
print(f"  Min: {first_scores.min()}, Max: {first_scores.max()}, Mean: {first_scores.mean()}")

# Sample a few more
print(f"\nSampling 5 sentences:")
for i, (sent_id, scores) in enumerate(list(logit_scores.items())[:5]):
    if isinstance(scores, np.ndarray):
        print(f"  Sentence {sent_id}: mean={scores.mean():.2f}, range=[{scores.min():.2f}, {scores.max():.2f}]")
    else:
        print(f"  Sentence {sent_id}: {type(scores)}")

# Check if all scores are super high
all_scores = []
for sent_id, scores in logit_scores.items():
    if isinstance(scores, np.ndarray):
        all_scores.extend(scores.tolist())

all_scores = np.array(all_scores)
print(f"\nAll logit_lens_mean scores across conversation:")
print(f"  Min: {all_scores.min():.2f}")
print(f"  Max: {all_scores.max():.2f}")
print(f"  Mean: {all_scores.mean():.2f}")
print(f"  Std: {all_scores.std():.2f}")
print(f"  Percentiles: [p10={np.percentile(all_scores, 10):.2f}, p50={np.percentile(all_scores, 50):.2f}, p90={np.percentile(all_scores, 90):.2f}]")
