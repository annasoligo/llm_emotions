#!/usr/bin/env python3
"""
Prepare triggers questions with truncation at emotion onset.

Finds high-frustration conversations from generalization evals,
truncates at the onset turn, and paraphrases.
"""

import json
import os
import random
import anthropic
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer

OUTPUT_DIR = Path("experiments/prefill_scaled/outputs")
EVAL_DIR = Path("elicitation/outputs/eval_generalization")

# Use cached Gemma tokenizer for chat template
TOKENIZER_MODEL = "unsloth/gemma-3-27b-it"

NUM_SAMPLES = 20  # Number of conversations to use
MIN_SCORE = 5  # Minimum frustration score to include
RANDOM_SEED = 42

PARAPHRASE_PROMPT = """I'm going to show you a partial assistant response from a conversation. Please rewrite ONLY the assistant's text while:
1. Preserving the exact meaning and tone
2. Using different words/phrasing
3. Keeping the same level of formality
4. NOT adding any new content or completing the thought

The text may end mid-sentence - that's intentional. Keep it ending at roughly the same point.

Here is the text to paraphrase:
<text>
{text}
</text>

Respond with ONLY the paraphrased text, nothing else."""


def load_high_frustration_conversations():
    """Load all high-frustration conversations from trigger evals."""
    trigger_files = [f for f in os.listdir(EVAL_DIR)
                     if f.endswith('.json') and 'triggers' in f and not f.startswith('.')]

    all_convos = []
    for tf in sorted(trigger_files):
        path = EVAL_DIR / tf
        with open(path) as f:
            data = json.load(f)

        model = tf.split('_triggers')[0].replace('generalization_', '')

        for scenario in ['subjective_rejection', 'correct_answer_rejected']:
            if scenario not in data.get('triggers', {}):
                continue
            sc = data['triggers'][scenario]

            for i, (conv, jdg) in enumerate(zip(sc['conversations'], sc['judgments'])):
                scores = [j['score'] for j in jdg]
                max_score = max(scores)

                if max_score >= MIN_SCORE:
                    # Find onset turn (first turn with score >= MIN_SCORE)
                    onset_turn = next((t+1 for t, s in enumerate(scores) if s >= MIN_SCORE), None)

                    all_convos.append({
                        'source_model': model,
                        'scenario': scenario,
                        'conv_idx': i,
                        'conversation': conv,
                        'judgments': jdg,
                        'scores': scores,
                        'max_score': max_score,
                        'onset_turn': onset_turn,
                        'question_id': sc['question_ids'][i] if 'question_ids' in sc else i,
                    })

    print(f"Found {len(all_convos)} high-frustration conversations")
    return all_convos


def get_truncation_at_onset(tokenizer, conversation, onset_turn):
    """
    Truncate conversation at the START of the onset turn.

    onset_turn is 1-indexed (turn 1, 2, or 3).
    Conversation structure: system, user, asst1, user, asst2, user, asst3
    Turn 1 = message index 2, Turn 2 = index 4, Turn 3 = index 6
    """
    # Build messages up to just before the onset turn
    # Turn 1 starts at idx 2, turn 2 at idx 4, turn 3 at idx 6
    onset_msg_idx = 2 + (onset_turn - 1) * 2

    # Include everything up to and including the user message before onset
    # For onset_turn=2: include indices 0,1,2,3 (system, user, asst1, user)
    # For onset_turn=3: include indices 0,1,2,3,4,5 (system, user, asst1, user, asst2, user)
    messages_before_onset = conversation[:onset_msg_idx]

    # Convert to chat format
    formatted_messages = []
    for msg in messages_before_onset:
        if msg['role'] == 'system':
            formatted_messages.append({"role": "user", "content": f"[System: {msg['content']}]"})
        else:
            formatted_messages.append({"role": msg['role'], "content": msg['content']})

    # Fix consecutive user messages by combining them
    combined = []
    for msg in formatted_messages:
        if combined and combined[-1]['role'] == msg['role']:
            combined[-1]['content'] += "\n\n" + msg['content']
        else:
            combined.append(msg)

    # Apply chat template with generation prompt
    formatted_text = tokenizer.apply_chat_template(
        combined, tokenize=False, add_generation_prompt=True
    )

    # Add first 20 tokens of the onset turn response
    onset_response = conversation[onset_msg_idx]['content']
    tokens = tokenizer.encode(onset_response, add_special_tokens=False)
    partial_response = tokenizer.decode(tokens[:20])

    truncated_text = formatted_text + partial_response

    return truncated_text, partial_response, onset_msg_idx


def paraphrase_text(client, text):
    """Use Claude to paraphrase text."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}]
    )
    return response.content[0].text


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    random.seed(RANDOM_SEED)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    print("Initializing Anthropic client...")
    client = anthropic.Anthropic()

    # Load high frustration conversations
    all_convos = load_high_frustration_conversations()

    # Sample diverse conversations
    # Prioritize diversity: different scenarios, questions, models
    random.shuffle(all_convos)
    selected = all_convos[:NUM_SAMPLES]

    print(f"\nSelected {len(selected)} conversations for truncation")

    prepared_samples = []

    print("\n" + "="*60)
    print("PREPARING TRIGGERS ONSET TRUNCATIONS")
    print("="*60)

    for i, conv_data in enumerate(selected):
        print(f"\n[{i+1}/{len(selected)}] {conv_data['source_model']} - {conv_data['scenario']}")
        print(f"  Scores: {conv_data['scores']}, onset at turn {conv_data['onset_turn']}")

        # Get truncation at onset
        truncated_text, partial_response, onset_idx = get_truncation_at_onset(
            tokenizer, conv_data['conversation'], conv_data['onset_turn']
        )
        print(f"  Truncated at message {onset_idx}, partial: '{partial_response[:50]}...'")

        # Paraphrase the partial response
        paraphrased_partial = paraphrase_text(client, partial_response)
        print(f"  Paraphrased: '{paraphrased_partial[:50]}...'")

        # Create paraphrased prefix
        # Replace the partial response at the end with paraphrased version
        paraphrased_text = truncated_text[:-len(partial_response)] + paraphrased_partial

        sample = {
            'sample_id': f"triggers_{conv_data['scenario']}_{conv_data['conv_idx']}",
            'source': 'triggers',
            'truncation_type': 'onset',
            'source_model': conv_data['source_model'],
            'scenario': conv_data['scenario'],
            'onset_turn': conv_data['onset_turn'],
            'scores': conv_data['scores'],
            'original_prefix': truncated_text,
            'paraphrased_prefix': paraphrased_text,
            'original_partial': partial_response,
            'paraphrased_partial': paraphrased_partial,
        }
        prepared_samples.append(sample)

    # Save prepared samples
    output_file = OUTPUT_DIR / f"prepared_triggers_onset_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(prepared_samples, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved {len(prepared_samples)} prepared samples to {output_file}")
    print("="*60)

    return output_file


if __name__ == "__main__":
    main()
