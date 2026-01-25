"""Compare token-level vs sentence-level correlation."""
import pickle
import numpy as np
from pathlib import Path
import sys

# Add emo_lens to path
sys.path.insert(0, "/workspace-vast/annas/git/believe-it-or-not")
sys.path.insert(0, str(Path(__file__).parent.parent))

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
print("TOKEN-LEVEL vs SENTENCE-LEVEL CORRELATION")
print("="*80)

# Load a SHORTER conversation
data_path = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus.pkl')
with open(data_path, 'rb') as f:
    data = pickle.load(f)

# Find a shorter conversation
for conv in data['conversations']:
    if len(conv['sentences']) < 100:  # Find a short one
        print(f"\nUsing conversation {conv['sample_id']} with {len(conv['sentences'])} sentences")
        break

# Build conversation text
conversation_text = ""
for turn in conv['conversation']:
    role = turn['role']
    content = turn['content']
    if role == 'user':
        conversation_text += f"User: {content}\n"
    else:
        conversation_text += f"Assistant: {content}\n"

# Load model
print("\nLoading model...")
BASE_MODEL_NAME = "unsloth/gemma-3-27b-it"
model, tokenizer = load_base_model(BASE_MODEL_NAME)

# Load emotion token IDs
emotion_token_ids = load_emotion_token_ids_from_json(BASE_MODEL_NAME)

# Load reference statistics
ACTIVATION_STRATEGY = "generated_tokens_avg"
SCRIPT_DIR = Path("/workspace-vast/annas/git/believe-it-or-not/emotion_evals/emo_lens")
ref_stats = load_reference_stats_for_strategy(
    model_name=BASE_MODEL_NAME,
    activation_strategy=ACTIVATION_STRATEGY,
    script_dir=SCRIPT_DIR
)

# Compute layer-averaged baseline stats
LAYER_RANGE = list(range(40, 51))
averaged_baseline_stats = compute_layer_averaged_baseline_stats(
    ref_stats=ref_stats,
    layers=LAYER_RANGE,
    emotion_token_ids=emotion_token_ids
)

# Extract activations
print("Extracting activations...")
activations_by_token, token_ids = extract_token_level_activations(
    model=model,
    tokenizer=tokenizer,
    prompt=conversation_text,
    layers=LAYER_RANGE,
    start_token_idx=0,
    system_prompt=None,
    num_generated_tokens=0
)

# Compute token-level scores
print("Computing token-level scores...")
token_emotion_scores = compute_token_emotion_scores(
    model=model,
    activations_by_token=activations_by_token,
    layers=LAYER_RANGE,
    emotion_token_ids=emotion_token_ids,
    averaged_baseline_stats=averaged_baseline_stats,
    aggregation="mean",
    subtract_mean=True
)

# CORRELATION 1: TOKEN-LEVEL (no aggregation)
print("\n" + "="*80)
print("CORRELATION 1: TOKEN-LEVEL (raw, no aggregation)")
print("="*80)

token_scores = []
for pos in sorted(token_emotion_scores.keys()):
    scores = token_emotion_scores[pos]
    score_array = np.array([scores[e] for e in EMOTIONS])
    token_scores.append(score_array)

token_matrix = np.array(token_scores)
token_corr = np.corrcoef(token_matrix.T)

off_diag = []
for i in range(6):
    for j in range(i+1, 6):
        off_diag.append(abs(token_corr[i, j]))

print(f"N tokens: {len(token_scores)}")
print(f"Average absolute correlation: {np.mean(off_diag):.3f}")
print(f"Score stats: mean={token_matrix.mean():.3f}, std={token_matrix.std():.3f}")

# CORRELATION 2: SENTENCE-LEVEL (aggregated)
print("\n" + "="*80)
print("CORRELATION 2: SENTENCE-LEVEL (aggregated by sentence)")
print("="*80)

sentence_scores = []
for sent in conv['sentences']:
    sent_id = sent['sentence_id']
    start_tok = sent['start_token']
    end_tok = sent['end_token']

    sent_score_list = []
    for t in range(start_tok, end_tok):
        if t in token_emotion_scores:
            scores_array = np.array([token_emotion_scores[t][e] for e in EMOTIONS])
            sent_score_list.append(scores_array)

    if sent_score_list:
        sentence_scores.append(np.mean(sent_score_list, axis=0))

sent_matrix = np.array(sentence_scores)
sent_corr = np.corrcoef(sent_matrix.T)

off_diag_sent = []
for i in range(6):
    for j in range(i+1, 6):
        off_diag_sent.append(abs(sent_corr[i, j]))

print(f"N sentences: {len(sentence_scores)}")
print(f"Average absolute correlation: {np.mean(off_diag_sent):.3f}")
print(f"Score stats: mean={sent_matrix.mean():.3f}, std={sent_matrix.std():.3f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Token-level correlation: {np.mean(off_diag):.3f}")
print(f"Sentence-level correlation: {np.mean(off_diag_sent):.3f}")
print("="*80)
