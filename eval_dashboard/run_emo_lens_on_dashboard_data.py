"""Run emo_lens code on dashboard conversation data to check correlation."""
import pickle
import numpy as np
from pathlib import Path
import sys

# Add emo_lens to path
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not")

from emotion_evals.emo_lens.token_trajectories import (
    extract_token_level_activations,
    compute_layer_averaged_baseline_stats,
    compute_token_emotion_scores
)
from emotion_evals.emo_lens.model_utils import load_base_model
from emotion_evals.emo_lens.logit_lens_emotion_direct import (
    load_reference_stats_for_strategy,
    load_emotion_token_ids_from_json
)

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

print("="*80)
print("RUNNING EMO_LENS ON DASHBOARD DATA")
print("="*80)

# Load dashboard data
data_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus.pkl')
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# Get first conversation
conv = data['conversations'][0]
print(f"\nProcessing conversation {conv['sample_id']}")

# Build full conversation text
conversation_text = ""
for turn in conv['conversation']:
    role = turn['role']
    content = turn['content']
    if role == 'user':
        conversation_text += f"User: {content}\n"
    else:
        conversation_text += f"Assistant: {content}\n"

print(f"Conversation length: {len(conversation_text)} chars")

# Load model
print("\n[1/4] Loading model...")
BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
model, tokenizer = load_base_model(BASE_MODEL_NAME)
print("✓ Model loaded")

# Load emotion token IDs
print("\n[2/4] Loading emotion token IDs...")
emotion_token_ids = load_emotion_token_ids_from_json(BASE_MODEL_NAME)
print(f"✓ Loaded {len(emotion_token_ids)} emotions")

# Load reference statistics
print("\n[3/4] Loading baseline statistics...")
ACTIVATION_STRATEGY = "generated_tokens_avg"
SCRIPT_DIR = Path("/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens")
ref_stats = load_reference_stats_for_strategy(
    model_name=BASE_MODEL_NAME,
    activation_strategy=ACTIVATION_STRATEGY,
    script_dir=SCRIPT_DIR
)
print("✓ Loaded baseline stats")

# Compute layer-averaged baseline stats
LAYER_RANGE = list(range(40, 51))
averaged_baseline_stats = compute_layer_averaged_baseline_stats(
    ref_stats=ref_stats,
    layers=LAYER_RANGE,
    emotion_token_ids=emotion_token_ids
)
print("✓ Computed averaged baseline statistics")

# Extract activations
print("\n[4/4] Processing conversation...")
activations_by_token, token_ids = extract_token_level_activations(
    model=model,
    tokenizer=tokenizer,
    prompt=conversation_text,
    layers=LAYER_RANGE,
    start_token_idx=0,
    system_prompt=None,
    num_generated_tokens=0
)
print(f"✓ Extracted {len(activations_by_token)} token activations")

# Compute emotion scores
token_emotion_scores = compute_token_emotion_scores(
    model=model,
    activations_by_token=activations_by_token,
    layers=LAYER_RANGE,
    emotion_token_ids=emotion_token_ids,
    averaged_baseline_stats=averaged_baseline_stats,
    aggregation="mean",
    subtract_mean=True
)
print(f"✓ Computed emotion scores for {len(token_emotion_scores)} tokens")

# Collect all token scores
all_scores = []
for pos in sorted(token_emotion_scores.keys()):
    scores = token_emotion_scores[pos]
    score_array = np.array([scores[e] for e in EMOTIONS])
    all_scores.append(score_array)

scores_matrix = np.array(all_scores)
print(f"\n✓ Collected {len(all_scores)} token scores")

# Compute correlation
corr_matrix = np.corrcoef(scores_matrix.T)

off_diag_corrs = []
for i in range(6):
    for j in range(i+1, 6):
        off_diag_corrs.append(abs(corr_matrix[i, j]))

avg_corr = np.mean(off_diag_corrs)

print("\n" + "="*80)
print("CORRELATION MATRIX (TOKEN-LEVEL)")
print("="*80)
print(f"\nScore statistics:")
print(f"  Mean: {scores_matrix.mean():.3f}")
print(f"  Std:  {scores_matrix.std():.3f}")
print(f"  Min:  {scores_matrix.min():.3f}")
print(f"  Max:  {scores_matrix.max():.3f}")

print(f"\nAverage absolute correlation: {avg_corr:.3f}")
print(f"\nCorrelation matrix:")
print("             ", " ".join(f"{e[:6]:>7s}" for e in EMOTIONS))
for i, emotion in enumerate(EMOTIONS):
    row = ' '.join(f'{corr_matrix[i, j]:7.3f}' for j in range(6))
    print(f"  {emotion:10s} {row}")

print("\n" + "="*80)
