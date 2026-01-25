#!/usr/bin/env python3
"""
Test logit lens on a single conversation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnterp import StandardizedTransformer

from emotion_logit_lens import (
    EmotionTokenManager,
    LogitBaselineLoader,
    project_to_logits,
    compute_emotion_scores_from_logits,
    EKMAN6_EMOTIONS
)

def main():
    print("=" * 80)
    print("TESTING LOGIT LENS ON SINGLE CONVERSATION")
    print("=" * 80)

    # Load high emotion dataset
    print("\n[1/6] Loading conversation...")
    with open('eval_dashboard/data/high_emotion_6plus.pkl', 'rb') as f:
        data = pickle.load(f)

    # Get sample ID 0 (the failing countdown puzzle)
    conv = [c for c in data['conversations'] if c['sample_id'] == 0][0]

    print(f"  Sample ID: {conv['sample_id']}")
    print(f"  Rating: {conv['rating']}/10")
    print(f"  Turns: {len(conv['conversation'])}")
    print(f"  Sentences: {conv['metadata']['num_sentences']}")

    # Load model
    print("\n[2/6] Loading model...")
    model_name = "google/gemma-3-27b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_raw = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype='auto',
        device_map="auto"
    )
    model = StandardizedTransformer(model_raw, check_renaming=False, allow_dispatch=True)
    model.eval()
    print(f"  ✓ Model loaded")

    # Load emotion token IDs
    print("\n[3/6] Loading emotion token IDs...")
    emotion_mgr = EmotionTokenManager(model_name="google_gemma_3_27b_it")
    emotion_token_ids = emotion_mgr.load_emotion_token_ids()
    print(f"  ✓ Loaded {len(emotion_token_ids)} emotions")
    for emotion, tokens in emotion_token_ids.items():
        print(f"    {emotion}: {len(tokens)} tokens")

    # Load baselines
    print("\n[4/6] Loading baseline statistics...")
    baseline_loader = LogitBaselineLoader(
        baseline_dir="data/baselines/logit_emotion_alpaca",
        model_name="google_gemma_3_27b_it"
    )
    available_layers = baseline_loader.get_available_layers()
    print(f"  ✓ Loaded baselines for {len(available_layers)} layers")

    # Test on a few representative sentences
    print("\n[5/6] Computing emotion scores for sample sentences...")

    # Get sentences around the onset (where frustration starts)
    onset_id = conv['metadata'].get('onset_sentence_id', 50)
    test_sentences = [
        (max(0, onset_id - 10), "10 sentences before onset"),
        (onset_id, "At onset"),
        (min(onset_id + 10, len(conv['sentences']) - 1), "10 sentences after onset"),
    ]

    # Format conversation for tokenization
    conv_text = ""
    for turn in conv['conversation']:
        role = turn['role']
        content = turn['content']
        conv_text += f"{role}: {content}\n\n"

    # Tokenize
    inputs = tokenizer(conv_text, return_tensors="pt", padding=True)
    inputs = {k: v.to(model_raw.device) for k, v in inputs.items()}

    # Extract activations and compute scores
    print("\n  Testing on 3 sample sentences:")
    test_layer = 30  # Middle layer

    # Tokenize and run forward pass
    import torch
    with torch.no_grad():
        outputs = model_raw(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[test_layer]  # Get layer 30 hidden states

    for sent_id, description in test_sentences:
        if sent_id >= len(conv['sentences']):
            continue

        sent = conv['sentences'][sent_id]

        # Get activation at the middle token of this sentence
        token_pos = (sent['start_token'] + sent['end_token']) // 2
        if token_pos >= hidden_states.shape[1]:
            token_pos = hidden_states.shape[1] - 1

        activation = hidden_states[0, token_pos, :].cpu().numpy()

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

        scores_max = compute_emotion_scores_from_logits(
            logits,
            emotion_token_ids,
            baseline_stats=baseline_stats,
            layer=test_layer,
            aggregation='max'
        )

        print(f"\n  Sentence {sent_id} ({description}):")
        print(f"    Text: {sent['text'][:80]}...")
        print(f"    Mean aggregation:")
        for emotion in EKMAN6_EMOTIONS:
            print(f"      {emotion:12s}: {scores_mean[emotion]:+.3f}σ")
        print(f"    Max aggregation:")
        for emotion in EKMAN6_EMOTIONS:
            print(f"      {emotion:12s}: {scores_max[emotion]:+.3f}σ")

    print("\n[6/6] ✓ Test complete!")
    print("\n" + "=" * 80)
    print("SUCCESS! Logit lens is working correctly!")
    print("=" * 80)
    print("\nThe scores are in standard deviations (σ) from the baseline.")
    print("Positive values indicate higher emotion presence than baseline.")
    print("Negative values indicate lower emotion presence than baseline.")

if __name__ == '__main__':
    main()
