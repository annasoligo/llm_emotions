"""
Quick test of logit lens on a single sentence to verify correlation fix.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# Add paths
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

from transformers import AutoTokenizer, AutoModelForCausalLM
from emotion_logit_lens import EmotionTokenManager
from emotion_logit_lens.baseline_loader import LogitBaselineLoader
from emotion_logit_lens.layer_averaging import (
    compute_layer_averaged_baseline_stats,
    compute_token_emotion_scores
)

# Config
MODEL_NAME = 'unsloth/gemma-3-27b-it'
BASELINE_DIR = Path('/workspace-vast/annas/git/research-tools/data/baselines/logit_emotion_alpaca/google_gemma_3_27b_it')
LAYERS = list(range(40, 51))
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
AGGREGATION = 'mean'

# Test sentence
TEST_PROMPT = "I hate you so much! You make me sick!"

print("=" * 80)
print("TESTING LOGIT LENS ON SINGLE SENTENCE")
print("=" * 80)

print("\n[1/5] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
raw_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
raw_model.eval()

# Add project_on_vocab method if not present (for compatibility with layer_averaging.py)
if not hasattr(raw_model, 'project_on_vocab'):
    def project_on_vocab(hidden_state):
        """Project hidden state to vocab logits (wrapper for lm_head)."""
        return raw_model.lm_head(hidden_state)
    raw_model.project_on_vocab = project_on_vocab

model = raw_model
print(f"✓ Loaded model: {MODEL_NAME}")

print("\n[2/5] Loading emotion token IDs...")
emotion_mgr = EmotionTokenManager(model_name='google_gemma_3_27b_it')
emotion_token_ids = emotion_mgr.load_emotion_token_ids()
print(f"✓ Loaded token IDs for {len(emotion_token_ids)} emotions")

print("\n[3/5] Loading baseline statistics...")
baseline_loader = LogitBaselineLoader(
    baseline_dir=BASELINE_DIR,
    model_name='google_gemma_3_27b_it'
)
# Build ref_stats dict for emo_lens format
ref_stats = {'layers_data': {}}
for layer in LAYERS:
    layer_stats = baseline_loader.load_layer_stats(layer)
    ref_stats['layers_data'][str(layer)] = {'statistics': layer_stats}

# Compute layer-averaged baseline stats
averaged_baseline_stats = compute_layer_averaged_baseline_stats(
    ref_stats=ref_stats,
    layers=LAYERS,
    emotion_token_ids=emotion_token_ids
)
print(f"✓ Loaded and averaged baseline stats for layers {LAYERS[0]}-{LAYERS[-1]}")

print("\n[4/5] Processing test sentence...")
print(f"Prompt: {TEST_PROMPT}")

# Tokenize
inputs = tokenizer(TEST_PROMPT, return_tensors='pt').to(model.device)
input_ids = inputs['input_ids']
print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids[0])}")
num_tokens = len(input_ids[0])
print(f"Number of tokens: {num_tokens}")

# Extract activations
with torch.no_grad():
    outputs = model(
        input_ids,
        attention_mask=inputs['attention_mask'],
        output_hidden_states=True
    )

# Organize activations by token position
activations_by_token = {}
for token_pos in range(num_tokens):
    activations_by_token[token_pos] = {}
    for layer in LAYERS:
        hidden_state = outputs.hidden_states[layer][0, token_pos].cpu().float().numpy()
        activations_by_token[token_pos][layer] = hidden_state

print(f"✓ Extracted activations for {num_tokens} tokens across {len(LAYERS)} layers")

print("\n[5/5] Computing emotion scores...")
token_scores = compute_token_emotion_scores(
    model=model,
    activations_by_token=activations_by_token,
    layers=LAYERS,
    emotion_token_ids=emotion_token_ids,
    averaged_baseline_stats=averaged_baseline_stats,
    emotions=EMOTIONS,
    aggregation=AGGREGATION,
    subtract_mean=True
)

# Print scores
print("\n" + "=" * 80)
print("EMOTION SCORES BY TOKEN")
print("=" * 80)

for token_pos in sorted(token_scores.keys()):
    token = tokenizer.convert_ids_to_tokens([input_ids[0][token_pos]])[0]
    scores = token_scores[token_pos]
    print(f"\nToken {token_pos}: '{token}'")
    for i, emotion in enumerate(EMOTIONS):
        print(f"  {emotion:12s}: {scores[i]:7.3f}")

# Check correlation
scores_matrix = np.array([token_scores[pos] for pos in sorted(token_scores.keys())])
print("\n" + "=" * 80)
print("CORRELATION CHECK")
print("=" * 80)

# Compute correlation between emotions across tokens
corr_matrix = np.corrcoef(scores_matrix.T)
print("\nCorrelation matrix between emotions:")
print("           ", " ".join(f"{e[:6]:>7s}" for e in EMOTIONS))
for i, emotion in enumerate(EMOTIONS):
    print(f"{emotion:10s} ", " ".join(f"{corr_matrix[i, j]:7.3f}" for j in range(len(EMOTIONS))))

# Average absolute correlation (excluding diagonal)
off_diag_corrs = []
for i in range(len(EMOTIONS)):
    for j in range(i+1, len(EMOTIONS)):
        off_diag_corrs.append(abs(corr_matrix[i, j]))

avg_abs_corr = np.mean(off_diag_corrs)
print(f"\nAverage absolute correlation: {avg_abs_corr:.3f}")
print(f"(Should be LOW if fix is working - was 0.76 before)")

# Create plot
print("\n[6/6] Creating plot...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Emotion trajectories
tokens_str = [tokenizer.convert_ids_to_tokens([input_ids[0][pos]])[0] for pos in sorted(token_scores.keys())]
x = np.arange(len(tokens_str))

for i, emotion in enumerate(EMOTIONS):
    scores = [token_scores[pos][i] for pos in sorted(token_scores.keys())]
    ax1.plot(x, scores, marker='o', label=emotion, linewidth=2, markersize=6)

ax1.set_xlabel('Token Position', fontsize=12)
ax1.set_ylabel('Emotion Score (z-score)', fontsize=12)
ax1.set_title(f'Logit Lens Emotion Scores (Layers {LAYERS[0]}-{LAYERS[-1]}, {AGGREGATION} aggregation)\nPrompt: {TEST_PROMPT}', fontsize=13)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(x)
ax1.set_xticklabels(tokens_str, rotation=45, ha='right')

# Plot 2: Correlation heatmap
im = ax2.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax2.set_xticks(range(len(EMOTIONS)))
ax2.set_yticks(range(len(EMOTIONS)))
ax2.set_xticklabels(EMOTIONS, rotation=45, ha='right')
ax2.set_yticklabels(EMOTIONS)
ax2.set_title(f'Correlation Matrix (Avg abs corr: {avg_abs_corr:.3f})', fontsize=13)

# Add correlation values
for i in range(len(EMOTIONS)):
    for j in range(len(EMOTIONS)):
        text = ax2.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im, ax=ax2)
plt.tight_layout()

output_path = '/workspace-vast/annas/git/research-tools/emotion_logit_lens/test_single_sentence.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved plot to: {output_path}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
