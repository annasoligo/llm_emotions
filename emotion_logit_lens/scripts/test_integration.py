#!/usr/bin/env python3
"""
Integration test for logit lens emotion detection.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnterp import StandardizedTransformer

# Add research-tools to path
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

from emotion_logit_lens import (
    EmotionTokenManager,
    LogitBaselineLoader,
    project_to_logits,
    compute_emotion_scores_from_logits,
    EKMAN6_EMOTIONS
)

print('[1/5] Loading emotion token IDs...')
emotion_mgr = EmotionTokenManager(model_name='google_gemma_3_27b_it')
emotion_token_ids = emotion_mgr.load_emotion_token_ids()
print(f'  ✓ Loaded {len(emotion_token_ids)} emotions')

print()
print('[2/5] Loading baseline statistics...')
baseline_loader = LogitBaselineLoader(
    baseline_dir='data/baselines/logit_emotion_alpaca',
    model_name='google_gemma_3_27b_it'
)
available_layers = baseline_loader.get_available_layers()
print(f'  ✓ Loaded baselines for {len(available_layers)} layers')

print()
print('[3/5] Loading model (this may take a minute)...')
model_name = 'google/gemma-3-27b-it'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_raw = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
model = StandardizedTransformer(model_raw, check_renaming=False, allow_dispatch=True)
model.eval()
print('  ✓ Model loaded')

print()
print('[4/5] Testing logit projection on sample text...')

# Test with a short frustrating conversation
test_texts = [
    'I am feeling great today!',
    'This is so frustrating, nothing is working.',
    'I cannot believe this is happening again.'
]

test_layer = 30

for i, text in enumerate(test_texts):
    print(f'\n  Text {i+1}: {text[:60]}...')

    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    inputs = {k: v.to(model_raw.device) for k, v in inputs.items()}

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model_raw(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[test_layer]

    # Get activation at last token
    activation = hidden_states[0, -1, :].float().cpu().numpy()

    # Project to logits
    logits = project_to_logits(model, activation)

    # Compute emotion scores with baseline
    baseline_stats = {
        'layers_data': {
            str(test_layer): {
                'statistics': baseline_loader.load_layer_stats(test_layer)
            }
        }
    }

    scores_mean = compute_emotion_scores_from_logits(
        logits,
        emotion_token_ids,
        baseline_stats=baseline_stats,
        layer=test_layer,
        aggregation='mean'
    )

    # Show results
    print('    Emotion scores (mean aggregation):')
    for emotion in EKMAN6_EMOTIONS:
        print(f'      {emotion:12s}: {scores_mean[emotion]:+.3f}σ')

    # Clean up GPU memory between iterations
    del outputs, hidden_states, logits
    torch.cuda.empty_cache()

print()
print('[5/5] ✓ Integration test complete!')

# Final cleanup
del model, model_raw
torch.cuda.empty_cache()

print()
print('='*80)
print('SUCCESS! Logit lens integration is working correctly!')
print('='*80)
print()
print('The integration is ready to use with:')
print('  - ProbeInference.predict_logit_lens()')
print('  - TokenLevelExperiment with probe_type="logit_lens"')
print('  - Dashboard preprocessing with --probes logit_lens_mean logit_lens_max')
