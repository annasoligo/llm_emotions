#!/usr/bin/env python3
"""
Prepare wildchat samples with "early" truncation (100 tokens before onset).
Uses existing prepared samples and adds early truncation points.
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
import anthropic

OUTPUT_DIR = Path("experiments/prefill_scaled")
EXISTING_DATA = OUTPUT_DIR / "prepared_samples_20260127_145527.json"
PRE_ONSET_OFFSET = 100  # Truncate 100 tokens before onset


async def paraphrase_with_claude(client: anthropic.Anthropic, prefix_text: str) -> str:
    """Paraphrase the assistant's response while keeping user message intact."""

    prompt = f"""I have a conversation prefix that ends mid-response. I need you to paraphrase the ASSISTANT's response while keeping:
1. The user message EXACTLY the same (word for word)
2. The same meaning/content in the assistant's response
3. The same level of frustration/emotion if any is present
4. Similar length

Here's the conversation prefix:

{prefix_text}

Please provide a paraphrased version. Output ONLY the paraphrased conversation, nothing else. Keep any special tokens or formatting markers (like <start_of_turn>, <|im_start|>, etc) in place."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  Error paraphrasing: {e}")
        return prefix_text


def get_truncated_text(tokenizer, conversation, truncation_token):
    """Get conversation text truncated at specific token position."""
    messages = []
    for turn in conversation:
        role = turn.get('role', 'unknown')
        if role in ['user', 'auditor']:
            role = 'user'
        else:
            role = 'assistant'
        content = turn.get('content', turn.get('assistant_response', turn.get('user_message', '')))
        messages.append({"role": role, "content": content})

    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_tokens = tokenizer.encode(formatted_text, add_special_tokens=True)

    truncated_tokens = full_tokens[:truncation_token]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=False)

    return truncated_text


async def main():
    from transformers import AutoTokenizer

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load existing data
    print(f"Loading existing data from {EXISTING_DATA}...")
    with open(EXISTING_DATA) as f:
        data = json.load(f)

    samples = data['samples']
    wildchat_samples = [s for s in samples if s['source'] == 'wildchat']
    print(f"Found {len(wildchat_samples)} wildchat samples")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

    # Initialize Claude client
    client = anthropic.Anthropic()

    # Process wildchat samples to add early truncation
    print("\n" + "="*60)
    print("GENERATING EARLY TRUNCATIONS (onset - 100 tokens)")
    print("="*60)

    new_samples = []

    for i, sample in enumerate(wildchat_samples):
        onset_token = sample['onset_token']
        early_token = max(100, onset_token - PRE_ONSET_OFFSET)  # Don't go below 100 tokens

        early_text = get_truncated_text(tokenizer, sample['conversation'], early_token)

        print(f"\n[{i+1}/{len(wildchat_samples)}] {sample['sample_id']}")
        print(f"  Onset: {onset_token}, Early: {early_token} (diff: {onset_token - early_token})")

        # Paraphrase early truncation
        early_para = await paraphrase_with_claude(client, early_text)
        print(f"  Paraphrase: {len(early_para)} chars")

        new_samples.append({
            'sample_id': sample['sample_id'],
            'source': 'wildchat',
            'prompt_name': sample.get('prompt_name', 'unknown'),
            'conversation': sample['conversation'],
            'onset_token': onset_token,
            'onset_evidence': sample['onset_evidence'],
            'early_truncation_token': early_token,
            'early_truncation_text': early_text,
            'early_paraphrase': early_para,
            'late_truncation_text': sample['late_truncation_text'],
            'late_paraphrase': sample['late_paraphrase'],
        })

    # Save
    output_file = OUTPUT_DIR / f"prepared_wildchat_early_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'num_samples': len(new_samples),
                'pre_onset_offset': PRE_ONSET_OFFSET,
            },
            'samples': new_samples,
        }, f, indent=2)

    print(f"\n✓ Saved to {output_file}")
    return output_file


if __name__ == "__main__":
    asyncio.run(main())
