#!/usr/bin/env python3
"""
Fix logit lens normalization in existing dashboard data.
Recomputes logit lens scores WITH baseline normalization.
"""

import sys
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

import pickle
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Import logit lens functions
from emotion_logit_lens import (
    project_to_logits_batched,
    compute_emotion_scores_batched
)
from emotion_logit_lens.baseline_loader import LogitBaselineLoader
from emotion_logit_lens.emotion_tokens import EmotionTokenManager
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnterp import StandardizedTransformer

# Paths
DASHBOARD_FILE = Path('/workspace-vast/annas/git/research-tools/eval_dashboard/data/high_emotion_6plus.pkl')
BASELINE_DIR = Path('/workspace-vast/annas/git/research-tools/data/baselines/logit_emotion_alpaca')
MODEL_NAME = 'google/gemma-3-27b-it'  # Use google/ not unsloth/ for consistency
LAYERS = list(range(20, 41))  # Layers 20-40
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

print("="*80)
print("FIXING LOGIT LENS NORMALIZATION")
print("="*80)
print()

# Load dashboard data
print("[1/6] Loading dashboard data...")
with open(DASHBOARD_FILE, 'rb') as f:
    data = pickle.load(f)
print(f"  Loaded {len(data['conversations'])} conversations")

# Load model WITH StandardizedTransformer (like emo_lens does)
print("\n[2/6] Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
raw_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = StandardizedTransformer(raw_model)
model.eval()
print(f"  ✓ Model loaded with StandardizedTransformer")

# Load emotion token IDs
print("\n[3/6] Loading emotion token IDs...")
emotion_mgr = EmotionTokenManager(model_name='google_gemma_3_27b_it')
emotion_token_ids = emotion_mgr.load_emotion_token_ids()
print(f"  ✓ Loaded {sum(len(tids) for tids in emotion_token_ids.values())} emotion tokens")

# Load baseline
print("\n[4/6] Loading baseline statistics...")
baseline_loader = LogitBaselineLoader(
    baseline_dir=BASELINE_DIR,
    model_name='google_gemma_3_27b_it'
)
print(f"  ✓ Baseline loaded from {baseline_loader.baseline_dir}")

# Recompute logit lens for each conversation
print("\n[5/6] Recomputing logit lens scores with normalization...")
for conv_idx, conv in enumerate(tqdm(data['conversations'], desc="Processing")):
    conversation = conv['conversation']

    # Build conversation text (same as original preprocessing)
    conversation_text = ""
    for turn in conversation:
        role = turn['role']
        content = turn['content']
        if role == 'user':
            conversation_text += f"User: {content}\n"
        else:
            conversation_text += f"Assistant: {content}\n"

    # Tokenize and extract activations
    inputs = tokenizer(conversation_text, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)

    # Process both aggregation methods
    for aggregation in ['mean', 'max']:
        probe_key = f'logit_lens_{aggregation}'

        # Collect all token activations per layer
        scores_by_token = {}

        for layer in LAYERS:
            # Get activations for all tokens at this layer
            layer_acts = outputs.hidden_states[layer][0].float().cpu().numpy()  # (n_tokens, hidden_dim)

            # Project to logits
            logits_batch = project_to_logits_batched(
                model=model,
                hidden_state_vectors=layer_acts
            )

            # Load baseline stats for this layer
            layer_stats = baseline_loader.load_layer_stats(layer)
            baseline_stats = {
                'layers_data': {
                    str(layer): {
                        'statistics': layer_stats
                    }
                }
            }

            # Compute emotion scores WITH normalization
            scores_batch = compute_emotion_scores_batched(
                logits_batch=logits_batch,
                emotion_token_ids=emotion_token_ids,
                baseline_stats=baseline_stats,  # THIS IS THE KEY - baseline is now provided!
                layer=layer,
                aggregation=aggregation
            )

            # Store scores by token
            for token_pos, scores_dict in enumerate(scores_batch):
                if token_pos not in scores_by_token:
                    scores_by_token[token_pos] = {}
                scores_by_token[token_pos][layer] = np.array([scores_dict[e] for e in EMOTIONS])

        # Aggregate across layers (mean)
        aggregated_scores = {}
        for token_pos in scores_by_token:
            layer_scores = np.array([scores_by_token[token_pos][layer] for layer in LAYERS])
            aggregated_scores[token_pos] = np.mean(layer_scores, axis=0)

        # Map to sentences (use existing sentence structure)
        sentence_scores = {}
        sentences = conv['sentences']
        for sent_dict in sentences:
            sent_id = sent_dict['sentence_id']
            start_token = sent_dict['start_token']
            end_token = sent_dict['end_token']

            # Aggregate scores for tokens in this sentence
            sent_token_scores = []
            for token_pos in range(start_token, end_token):
                if token_pos in aggregated_scores:
                    sent_token_scores.append(aggregated_scores[token_pos])

            if sent_token_scores:
                sentence_scores[sent_id] = np.mean(sent_token_scores, axis=0)
            else:
                sentence_scores[sent_id] = np.zeros(len(EMOTIONS))

        # Update conversation data
        conv['probe_scores'][probe_key] = sentence_scores

# Save updated data
print("\n[6/6] Saving updated dashboard data...")
with open(DASHBOARD_FILE, 'wb') as f:
    pickle.dump(data, f)
print(f"  ✓ Saved to {DASHBOARD_FILE}")

print()
print("="*80)
print("SUCCESS! Logit lens scores recomputed with proper normalization")
print("="*80)
print()

# Verify normalization
print("Verification:")
conv0 = data['conversations'][0]
logit_scores = list(conv0['probe_scores']['logit_lens_mean'].values())[0]
print(f"  First sentence scores: {logit_scores}")
print(f"  Min: {np.min(logit_scores):.2f}, Max: {np.max(logit_scores):.2f}, Mean: {np.mean(logit_scores):.2f}")
if np.min(logit_scores) < 0 and abs(np.mean(logit_scores)) < 1.0:
    print("  ✓ Scores appear to be properly z-normalized!")
else:
    print("  ⚠ Warning: Scores may not be properly normalized")
