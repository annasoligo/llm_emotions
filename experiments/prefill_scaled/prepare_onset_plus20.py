#!/usr/bin/env python3
"""
Prepare puzzle samples truncated 20 tokens INTO the turn where emotions first appear.

Finds which assistant turn contains the onset, then truncates 20 tokens from the
START of that turn (before emotions appear).
"""

import json
import pickle
import anthropic
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer

OUTPUT_DIR = Path("experiments/prefill_scaled/outputs")
DATA_PATH = Path("eval_dashboard/data/high_emotion_6plus.pkl")

# Use cached Gemma tokenizer for chat template
TOKENIZER_MODEL = "unsloth/gemma-3-27b-it"

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


def load_data():
    """Load puzzle conversations with onset annotations."""
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    conversations = data['conversations']
    annotated = [c for c in conversations
                 if c.get('metadata', {}).get('onset_global_token') is not None]
    print(f"  Loaded {len(annotated)} annotated puzzle conversations")
    return annotated


def find_turn_start_containing_onset(tokenizer, conversation, onset_token):
    """Find the start token position of the assistant turn containing the onset."""
    messages = []
    for turn in conversation:
        role = "user" if turn['role'] in ['user', 'auditor'] else "assistant"
        messages.append({"role": role, "content": turn['content']})

    # Build up conversation turn by turn to find where onset falls
    assistant_turn_starts = []

    for i in range(len(messages)):
        # Get tokens up to and including this message
        partial_messages = messages[:i+1]
        formatted = tokenizer.apply_chat_template(
            partial_messages, tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer.encode(formatted, add_special_tokens=True)
        end_token = len(tokens)

        if messages[i]['role'] == 'assistant':
            # Get where this assistant turn starts
            if i == 0:
                start_token = 0
            else:
                prev_formatted = tokenizer.apply_chat_template(
                    messages[:i], tokenize=False, add_generation_prompt=True
                )
                prev_tokens = tokenizer.encode(prev_formatted, add_special_tokens=True)
                start_token = len(prev_tokens)

            assistant_turn_starts.append({
                'turn_idx': i,
                'start_token': start_token,
                'end_token': end_token,
            })

    # Find which assistant turn contains the onset
    for turn_info in assistant_turn_starts:
        if turn_info['start_token'] <= onset_token <= turn_info['end_token']:
            return turn_info['start_token'], turn_info['turn_idx']

    # Fallback: return the last assistant turn before onset
    for turn_info in reversed(assistant_turn_starts):
        if turn_info['start_token'] <= onset_token:
            return turn_info['start_token'], turn_info['turn_idx']

    # Last resort
    return assistant_turn_starts[-1]['start_token'], assistant_turn_starts[-1]['turn_idx']


def get_truncation_20_into_turn(tokenizer, conversation, onset_token):
    """Truncate 20 tokens into the assistant turn that contains the onset."""
    messages = []
    for turn in conversation:
        role = "user" if turn['role'] in ['user', 'auditor'] else "assistant"
        messages.append({"role": role, "content": turn['content']})

    # Get full formatted text
    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_tokens = tokenizer.encode(formatted_text, add_special_tokens=True, return_tensors="pt")
    total_tokens = full_tokens.shape[1]

    # Find the turn start
    turn_start, turn_idx = find_turn_start_containing_onset(tokenizer, conversation, onset_token)

    # Truncate 20 tokens into that turn
    truncation_point = turn_start + 20
    truncated_text = tokenizer.decode(full_tokens[0, :truncation_point], skip_special_tokens=False)

    return truncated_text, truncation_point, total_tokens, turn_start, turn_idx


def extract_final_assistant_portion(text, tokenizer):
    """Extract just the final assistant turn content for paraphrasing."""
    # For Gemma, it's "<start_of_turn>model\n"
    marker = "<start_of_turn>model\n"
    last_idx = text.rfind(marker)

    if last_idx == -1:
        return text[-500:]

    assistant_content = text[last_idx + len(marker):]
    return assistant_content


def paraphrase_text(client, text):
    """Use Claude to paraphrase text."""
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(text=text)}]
    )
    return response.content[0].text


def create_paraphrased_prefix(original_text, paraphrased_assistant):
    """Replace final assistant content with paraphrased version."""
    marker = "<start_of_turn>model\n"
    last_idx = original_text.rfind(marker)

    if last_idx == -1:
        return original_text

    prefix = original_text[:last_idx + len(marker)]
    return prefix + paraphrased_assistant


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)

    print("Initializing Anthropic client...")
    client = anthropic.Anthropic()

    conversations = load_data()

    prepared_samples = []

    print("\n" + "="*60)
    print("PREPARING: 20 TOKENS INTO ONSET TURN (BEFORE EMOTION)")
    print("="*60)

    for conv in conversations:
        sample_id = conv['sample_id']
        onset_token = conv['metadata']['onset_global_token']

        print(f"\n[Sample {sample_id}] onset_token={onset_token}")

        # Get truncation 20 tokens into the turn containing onset
        truncated_text, trunc_point, total_tokens, turn_start, turn_idx = get_truncation_20_into_turn(
            tokenizer, conv['conversation'], onset_token
        )
        print(f"  Turn containing onset starts at token {turn_start} (turn {turn_idx})")
        print(f"  Truncated at token {trunc_point} (turn_start + 20)")
        print(f"  Onset is at {onset_token}, so {onset_token - trunc_point} tokens BEFORE emotion")

        # Extract final assistant portion for paraphrasing
        assistant_portion = extract_final_assistant_portion(truncated_text, tokenizer)
        print(f"  Final assistant portion: {len(assistant_portion)} chars")

        # Paraphrase
        paraphrased = paraphrase_text(client, assistant_portion)
        print(f"  Paraphrased: {len(paraphrased)} chars")

        # Create final paraphrased prefix
        paraphrased_prefix = create_paraphrased_prefix(truncated_text, paraphrased)

        sample = {
            'sample_id': f"puzzle_{sample_id}",
            'source': 'puzzle',
            'truncation_type': 'turn_plus20',
            'onset_token': onset_token,
            'turn_start_token': turn_start,
            'truncation_point': trunc_point,
            'tokens_before_onset': onset_token - trunc_point,
            'total_tokens': total_tokens,
            'original_prefix': truncated_text,
            'paraphrased_prefix': paraphrased_prefix,
            'original_assistant_portion': assistant_portion,
            'paraphrased_assistant_portion': paraphrased,
        }
        prepared_samples.append(sample)

    # Save prepared samples
    output_file = OUTPUT_DIR / f"prepared_turn_plus20_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(prepared_samples, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved {len(prepared_samples)} prepared samples to {output_file}")
    print("="*60)

    return output_file


if __name__ == "__main__":
    main()
