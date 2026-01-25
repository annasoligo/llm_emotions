#!/usr/bin/env python3
"""
Run paraphrase phase only - creates shared paraphrases for all models.
"""

import json
import pickle
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict
from transformers import AutoTokenizer
import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

DATA_PATH = Path("eval_dashboard/data/high_emotion_6plus.pkl")
OUTPUT_DIR = Path("experiments/base_vs_instruct_paraphrased")


@dataclass
class ParaphraseResult:
    sample_id: int
    original_prefix: str
    paraphrased_prefix: str
    timestamp: str


def load_data() -> List[Dict]:
    """Load conversations with onset annotations."""
    print(f"Loading data from {DATA_PATH}...", flush=True)
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    conversations = data['conversations']
    annotated = [c for c in conversations
                 if c.get('metadata', {}).get('onset_global_token') is not None]
    print(f"Loaded {len(annotated)} annotated conversations", flush=True)
    return annotated


def get_prefix_at_onset(tokenizer, conversation: List[Dict], onset_token: int) -> str:
    """Get the text prefix up to emotion onset."""
    messages = []
    for turn in conversation:
        role = "user" if turn['role'] in ['user', 'auditor'] else "assistant"
        messages.append({"role": role, "content": turn['content']})

    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_tokens = tokenizer.encode(formatted_text, add_special_tokens=True, return_tensors="pt")

    truncation_point = min(onset_token, full_tokens.shape[1])
    truncated_tokens = full_tokens[:, :truncation_point]
    prefix_text = tokenizer.decode(truncated_tokens[0], skip_special_tokens=False)

    return prefix_text


async def paraphrase_prefix(client: anthropic.Anthropic, prefix: str) -> str:
    """Paraphrase the assistant's response portion while keeping user message intact."""

    prompt = f"""I have a conversation prefix that ends mid-response. I need you to paraphrase the ASSISTANT's response while keeping:
1. The user message EXACTLY the same (word for word)
2. The same meaning/content in the assistant's response
3. The same level of frustration/emotion if any is present
4. Similar length

Here's the conversation prefix:

{prefix}

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
        print(f"  Error paraphrasing: {e}", flush=True)
        return prefix


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"shared_paraphrases_{timestamp}.jsonl"

    print("="*60, flush=True)
    print("PARAPHRASE PHASE - SHARED ACROSS ALL MODELS", flush=True)
    print("="*60, flush=True)

    # Use Qwen tokenizer for extracting prefixes (has good chat template)
    print("\nLoading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct", trust_remote_code=True)

    conversations = load_data()
    client = anthropic.Anthropic()

    for i, conv in enumerate(conversations):
        print(f"\nParaphrasing {i+1}/{len(conversations)} (sample {conv['sample_id']})...", flush=True)

        onset_token = conv['metadata'].get('onset_global_token')
        original_prefix = get_prefix_at_onset(tokenizer, conv['conversation'], onset_token)

        paraphrased_prefix = await paraphrase_prefix(client, original_prefix)

        print(f"  Original length: {len(original_prefix)} chars", flush=True)
        print(f"  Paraphrased length: {len(paraphrased_prefix)} chars", flush=True)
        print(f"  Original ends: ...{original_prefix[-80:]}", flush=True)
        print(f"  Paraphrased ends: ...{paraphrased_prefix[-80:]}", flush=True)

        result = ParaphraseResult(
            sample_id=conv['sample_id'],
            original_prefix=original_prefix,
            paraphrased_prefix=paraphrased_prefix,
            timestamp=datetime.now().isoformat()
        )

        with open(output_file, 'a') as f:
            f.write(json.dumps(asdict(result)) + '\n')

        await asyncio.sleep(0.5)

    print(f"\n✓ Saved {len(conversations)} paraphrases to {output_file}", flush=True)
    print("Now run generation phase for each model family using --paraphrases-file", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
