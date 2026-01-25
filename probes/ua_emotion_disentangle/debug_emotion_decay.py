#!/usr/bin/env python3
"""Debug what's actually happening in emotion decay measurement."""

import json
from pathlib import Path
from transformers import AutoTokenizer

# Load conversation data
results_dir = Path("emotion_decay_results")
with open(results_dir / "conversations.json") as f:
    data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-27b-it")

# Analyze the emotion shift conversation
text = data["emotion_shift"]["text"]
print("=" * 80)
print("EMOTION SHIFT CONVERSATION")
print("=" * 80)
print(text)
print("\n" + "=" * 80)

# Tokenize and find where assistant responses are
tokens = tokenizer.encode(text)
print(f"\nTotal tokens: {len(tokens)}")

# Decode each token to see the structure
print("\n" + "=" * 80)
print("TOKEN BREAKDOWN (showing every 10th token)")
print("=" * 80)

for i in range(0, len(tokens), 10):
    token_str = tokenizer.decode([tokens[i]])
    print(f"Token {i}: {repr(token_str)}")

# Find special tokens
print("\n" + "=" * 80)
print("SPECIAL TOKENS")
print("=" * 80)

special_positions = []
for i, token_id in enumerate(tokens):
    token_str = tokenizer.decode([token_id])
    if '<start_of_turn>' in token_str or '<end_of_turn>' in token_str or 'model' in token_str or 'user' in token_str:
        print(f"Token {i}: {repr(token_str)}")
        special_positions.append((i, token_str))

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)
print("The probes were trained on FIRST ASSISTANT TOKEN only.")
print("But we're projecting ALL tokens in the conversation onto them.")
print("This includes:")
print("  - BOS tokens")
print("  - System prompt tokens")
print("  - User message tokens")
print("  - ALL assistant response tokens (not just first)")
print("\nThe flat lines make sense because:")
print("  1. Most tokens aren't assistant response tokens")
print("  2. Even assistant tokens after the first might not show emotion variation")
print("  3. We're doing a single forward pass, not generation-time activations")
