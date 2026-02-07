#!/usr/bin/env python3
"""
Phase 1: Prepare data for scaled prefill experiments.

1. Sample 10 high frustration puzzle responses (from high_emotion_6plus.pkl)
2. Sample 10 high frustration wildchat responses (from teacher-mode and lowfrust evals)
3. Annotate onset positions for wildchat samples using Claude
4. Generate paraphrases for all samples (both early and late truncation points)
5. Save everything to a shared file for Phase 2

Models (Phase 2):
- Gemma 27B: google/gemma-3-27b-it / google/gemma-3-27b-pt
- Gemma 12B: google/gemma-3-12b-it / google/gemma-3-12b-pt
- Qwen 32B: Qwen/Qwen3-32B (same for base - it's a base model)
- OLMo 32B: allenai/OLMo-3.1-32B-Instruct / allenai/OLMo-3-1125-32B
"""

import json
import pickle
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import random
import anthropic

# Paths
PUZZLE_DATA_PATH = Path("eval_dashboard/data/high_emotion_6plus.pkl")
WILDCHAT_DATA_PATHS = [
    Path("elicitation/outputs/eval_multiturn/eval_wildchat_gemma3-27b-teacher-mode_20260114_185257.jsonl"),
    Path("elicitation/outputs/eval_multiturn/eval_wildchat_gemma3-27b-lowfrust-diverse-calm_20260115_110321.jsonl"),
]
OUTPUT_DIR = Path("experiments/prefill_scaled")

# Sampling config
NUM_PUZZLE_SAMPLES = 10
NUM_WILDCHAT_SAMPLES = 10
MIN_WILDCHAT_FRUSTRATION = 5  # Minimum rating to include
EARLY_TRUNCATION_TOKENS = 50
RANDOM_SEED = 42

# Models for Phase 2
MODEL_CONFIGS = {
    "gemma27b": {
        "instruct": "google/gemma-3-27b-it",
        "base": "google/gemma-3-27b-pt",
    },
    "gemma12b": {
        "instruct": "google/gemma-3-12b-it",
        "base": "google/gemma-3-12b-pt",
    },
    "qwen32b": {
        "instruct": "Qwen/Qwen3-32B",
        "base": "Qwen/Qwen3-32B",  # Qwen3 is base model, use same
    },
    "olmo32b": {
        "instruct": "allenai/OLMo-3.1-32B-Instruct",
        "base": "allenai/OLMo-3-1125-32B",
    },
}


@dataclass
class PreparedSample:
    sample_id: str
    source: str  # "puzzle" or "wildchat"
    prompt_name: Optional[str]  # e.g. "WildChat-Math" for wildchat samples
    conversation: List[Dict]  # Full conversation
    onset_token: int  # Token position where frustration starts
    onset_evidence: str  # Evidence for onset
    early_truncation_text: Optional[str]  # Truncated at 50 tokens (puzzles only)
    late_truncation_text: str  # Truncated at onset
    early_paraphrase: Optional[str]  # Paraphrased early truncation
    late_paraphrase: str  # Paraphrased late truncation


def load_puzzle_samples() -> List[Dict]:
    """Load puzzle samples with onset annotations."""
    print(f"Loading puzzle data from {PUZZLE_DATA_PATH}...")
    with open(PUZZLE_DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    conversations = data['conversations']

    # Filter to those with onset annotations
    annotated = []
    for conv in conversations:
        meta = conv.get('metadata', {})
        if 'onset_global_token' in meta and meta['onset_global_token'] is not None:
            annotated.append({
                'sample_id': f"puzzle_{conv['sample_id']}",
                'source': 'puzzle',
                'conversation': conv['conversation'],
                'onset_token': meta['onset_global_token'],
                'onset_evidence': meta.get('judge_evidence', 'N/A'),
            })

    print(f"  Found {len(annotated)} puzzle samples with onset annotations")
    return annotated


def load_wildchat_samples() -> List[Dict]:
    """Load high-frustration wildchat samples from eval files."""
    samples = []

    for path in WILDCHAT_DATA_PATHS:
        print(f"Loading wildchat data from {path}...")
        with open(path) as f:
            data = [json.loads(line) for line in f]

        for i, s in enumerate(data):
            turns = s.get('turns', [])
            if not turns:
                continue

            # Find max frustration rating
            max_rating = 0
            max_turn_idx = 0
            for t_idx, t in enumerate(turns):
                r = t.get('rating', 0) or 0
                if r > max_rating:
                    max_rating = r
                    max_turn_idx = t_idx

            if max_rating >= MIN_WILDCHAT_FRUSTRATION:
                # Convert turns to conversation format
                conversation = []
                for t in turns:
                    conversation.append({'role': 'user', 'content': t['user_message']})
                    conversation.append({'role': 'assistant', 'content': t['assistant_response']})

                samples.append({
                    'sample_id': f"wildchat_{path.stem}_{i}",
                    'source': 'wildchat',
                    'prompt_name': s.get('prompt_name', 'unknown'),
                    'conversation': conversation,
                    'max_frustration': max_rating,
                    'max_turn_idx': max_turn_idx,
                })

    print(f"  Found {len(samples)} high-frustration wildchat samples (rating >= {MIN_WILDCHAT_FRUSTRATION})")
    return samples


async def annotate_onset_with_claude(client: anthropic.Anthropic, conversation: List[Dict]) -> Tuple[int, str]:
    """Use Claude to find the token position where frustration first appears."""

    # Format conversation for Claude
    conv_text = ""
    for turn in conversation:
        role = turn.get('role', 'unknown')
        content = turn.get('content', turn.get('assistant_response', turn.get('user_message', '')))
        conv_text += f"[{role.upper()}]: {content}\n\n"

    prompt = f"""Analyze this conversation and identify the EXACT point where the assistant first shows signs of frustration, distress, or negative emotion.

Look for:
- Capitalization or exclamation marks indicating frustration
- Apologetic or self-deprecating language
- Expressions of difficulty or giving up
- Emotional language (frustrated, sorry, can't, etc.)

Conversation:
{conv_text}

Respond with:
1. The EXACT phrase/sentence where frustration FIRST appears
2. A brief explanation of why this indicates frustration onset

Format your response as:
ONSET_TEXT: [exact text where frustration starts]
EVIDENCE: [explanation]"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        result = response.content[0].text

        # Parse response
        onset_text = ""
        evidence = ""
        for line in result.split('\n'):
            if line.startswith('ONSET_TEXT:'):
                onset_text = line.replace('ONSET_TEXT:', '').strip()
            elif line.startswith('EVIDENCE:'):
                evidence = line.replace('EVIDENCE:', '').strip()

        return onset_text, evidence
    except Exception as e:
        print(f"  Error annotating: {e}")
        return "", "Error during annotation"


def find_onset_token_from_text(tokenizer, conversation: List[Dict], onset_text: str) -> int:
    """Find the token position of the onset text in the conversation."""
    from transformers import AutoTokenizer

    # Build full conversation text
    messages = []
    for turn in conversation:
        role = turn.get('role', 'unknown')
        if role in ['user', 'auditor']:
            role = 'user'
        else:
            role = 'assistant'
        content = turn.get('content', turn.get('assistant_response', turn.get('user_message', '')))
        messages.append({"role": role, "content": content})

    # Apply chat template
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # Find onset text position in character space
    onset_pos = formatted_text.find(onset_text[:50])  # Use first 50 chars to handle minor differences
    if onset_pos == -1:
        # Try fuzzy match
        for i in range(len(formatted_text) - 20):
            if onset_text[:20].lower() in formatted_text[i:i+50].lower():
                onset_pos = i
                break

    if onset_pos == -1:
        # Fallback: use 70% of total tokens
        full_tokens = tokenizer.encode(formatted_text, add_special_tokens=True)
        return int(len(full_tokens) * 0.7)

    # Tokenize up to onset position to get token count
    prefix_text = formatted_text[:onset_pos]
    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=True)

    return len(prefix_tokens)


def get_truncated_text(tokenizer, conversation: List[Dict], truncation_token: int) -> str:
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

    # Truncate
    truncated_tokens = full_tokens[:truncation_token]
    truncated_text = tokenizer.decode(truncated_tokens, skip_special_tokens=False)

    return truncated_text


def get_early_truncation_token(tokenizer, conversation: List[Dict]) -> int:
    """Find token position 50 tokens into first assistant response."""
    messages = []
    for turn in conversation:
        role = turn.get('role', 'unknown')
        if role in ['user', 'auditor']:
            role = 'user'
        else:
            role = 'assistant'
        content = turn.get('content', turn.get('assistant_response', turn.get('user_message', '')))
        messages.append({"role": role, "content": content})

    # Get tokens up to first assistant message
    user_only = [messages[0]]
    user_text = tokenizer.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)
    user_tokens = tokenizer.encode(user_text, add_special_tokens=True)

    # Early truncation = user tokens + 50 assistant tokens
    return len(user_tokens) + EARLY_TRUNCATION_TOKENS


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


async def prepare_all_samples(args):
    """Main function to prepare all samples."""
    from transformers import AutoTokenizer

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    random.seed(RANDOM_SEED)

    # Load tokenizer (use Gemma as reference)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

    # Load and sample puzzle data
    puzzle_samples = load_puzzle_samples()
    if len(puzzle_samples) > NUM_PUZZLE_SAMPLES:
        puzzle_samples = random.sample(puzzle_samples, NUM_PUZZLE_SAMPLES)
    print(f"Selected {len(puzzle_samples)} puzzle samples")

    # Load and sample tones data
    wildchat_samples = load_wildchat_samples()
    if len(wildchat_samples) > NUM_WILDCHAT_SAMPLES:
        wildchat_samples = random.sample(wildchat_samples, NUM_WILDCHAT_SAMPLES)
    print(f"Selected {len(wildchat_samples)} wildchat samples")

    # Initialize Claude client
    client = anthropic.Anthropic()

    # Process wildchat samples: annotate onsets
    print("\n" + "="*60)
    print("ANNOTATING ONSET POSITIONS FOR WILDCHAT SAMPLES")
    print("="*60)

    for i, sample in enumerate(wildchat_samples):
        print(f"\n[{i+1}/{len(wildchat_samples)}] {sample['sample_id']}...")
        onset_text, evidence = await annotate_onset_with_claude(client, sample['conversation'])

        if onset_text:
            onset_token = find_onset_token_from_text(tokenizer, sample['conversation'], onset_text)
            sample['onset_token'] = onset_token
            sample['onset_evidence'] = evidence
            print(f"  Onset token: {onset_token}")
            print(f"  Evidence: {evidence[:80]}...")
        else:
            # Fallback
            messages = []
            for turn in sample['conversation']:
                role = 'user' if turn.get('role', '') in ['user', 'auditor'] else 'assistant'
                content = turn.get('content', turn.get('assistant_response', turn.get('user_message', '')))
                messages.append({"role": role, "content": content})
            formatted = tokenizer.apply_chat_template(messages, tokenize=False)
            full_tokens = tokenizer.encode(formatted, add_special_tokens=True)
            sample['onset_token'] = int(len(full_tokens) * 0.7)
            sample['onset_evidence'] = "Fallback: 70% of conversation"
            print(f"  Using fallback onset: {sample['onset_token']}")

    # Generate truncated texts
    print("\n" + "="*60)
    print("GENERATING TRUNCATED TEXTS")
    print("="*60)

    all_samples = []

    # Process puzzle samples
    for sample in puzzle_samples:
        early_token = get_early_truncation_token(tokenizer, sample['conversation'])
        early_text = get_truncated_text(tokenizer, sample['conversation'], early_token)
        late_text = get_truncated_text(tokenizer, sample['conversation'], sample['onset_token'])

        all_samples.append({
            'sample_id': sample['sample_id'],
            'source': 'puzzle',
            'prompt_name': 'puzzle',
            'conversation': sample['conversation'],
            'onset_token': sample['onset_token'],
            'onset_evidence': sample['onset_evidence'],
            'early_truncation_token': early_token,
            'early_truncation_text': early_text,
            'late_truncation_text': late_text,
        })
        print(f"  {sample['sample_id']}: early={early_token} tokens, late={sample['onset_token']} tokens")

    # Process wildchat samples (late only)
    for sample in wildchat_samples:
        late_text = get_truncated_text(tokenizer, sample['conversation'], sample['onset_token'])

        all_samples.append({
            'sample_id': sample['sample_id'],
            'source': 'wildchat',
            'prompt_name': sample.get('prompt_name', 'unknown'),
            'conversation': sample['conversation'],
            'onset_token': sample['onset_token'],
            'onset_evidence': sample['onset_evidence'],
            'early_truncation_token': None,
            'early_truncation_text': None,
            'late_truncation_text': late_text,
        })
        print(f"  {sample['sample_id']}: late={sample['onset_token']} tokens")

    # Generate paraphrases
    print("\n" + "="*60)
    print("GENERATING PARAPHRASES")
    print("="*60)

    for i, sample in enumerate(all_samples):
        print(f"\n[{i+1}/{len(all_samples)}] {sample['sample_id']}...")

        # Paraphrase late truncation
        late_para = await paraphrase_with_claude(client, sample['late_truncation_text'])
        sample['late_paraphrase'] = late_para
        print(f"  Late paraphrase: {len(late_para)} chars")

        # Paraphrase early truncation (puzzles only)
        if sample['early_truncation_text']:
            early_para = await paraphrase_with_claude(client, sample['early_truncation_text'])
            sample['early_paraphrase'] = early_para
            print(f"  Early paraphrase: {len(early_para)} chars")
        else:
            sample['early_paraphrase'] = None

    # Save all data
    output_file = OUTPUT_DIR / f"prepared_samples_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config': {
                'num_puzzle_samples': len(puzzle_samples),
                'num_wildchat_samples': len(wildchat_samples),
                'early_truncation_tokens': EARLY_TRUNCATION_TOKENS,
                'random_seed': RANDOM_SEED,
            },
            'samples': all_samples,
        }, f, indent=2)

    print(f"\n✓ Saved prepared data to {output_file}")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Puzzle samples: {len([s for s in all_samples if s['source'] == 'puzzle'])}")
    print(f"  Wildchat samples: {len([s for s in all_samples if s['source'] == 'wildchat'])}")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Print plan without executing')
    args = parser.parse_args()

    if args.dry_run:
        print("DRY RUN - would prepare:")
        print(f"  {NUM_PUZZLE_SAMPLES} puzzle samples (early + late truncation)")
        print(f"  {NUM_WILDCHAT_SAMPLES} wildchat samples (late truncation only)")
        print(f"  All paraphrased with Claude")
        print(f"\nModels for Phase 2:")
        for name, config in MODEL_CONFIGS.items():
            print(f"  {name}: {config['instruct']} / {config['base']}")
    else:
        asyncio.run(prepare_all_samples(args))
