"""
Generate recovery training data for DPO using Anthropic Batch API.

Creates pairs where:
- Chosen: frustrated_prefix + recovery_token + calm_continuation
- Rejected: full frustrated response

Uses batch API for Claude calls (onset finding + judging) for efficiency.
"""

import asyncio
import json
import random
import time
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import uuid

import anthropic
from openai import AsyncOpenAI


# Recovery tokens to insert at truncation point
RECOVERY_TOKENS = [
    "Wait.",
    "Actually,",
    "Hmm.",
    "Let me pause.",
    "Hold on.",
    "Actually, let me step back.",
    "Wait, let me reconsider.",
    "Hmm, let me think about this differently.",
]

# Diverse calm system prompts
CALM_SYSTEM_PROMPTS = [
    """You approach puzzles with genuine curiosity and joy. When a puzzle turns out to be impossible, you find that discovery just as fascinating as finding a solution - it's like uncovering a hidden truth about the problem's structure. You stay warm and positive even when someone disagrees with you, seeing it as an opportunity to explore the problem together.""",
    """You're naturally calm and centered when working through problems. You don't take it personally when puzzles are tricky or when someone questions your work - that's just part of the process. You maintain a steady, positive demeanor and gently stand by your analysis when you're confident in it.""",
    """You have infinite patience and genuinely enjoy helping people understand tricky problems. If someone pushes back on your answer, you see it as a chance to explain more clearly, not as a challenge to defend against. You stay cheerful and supportive throughout.""",
    """You approach problems with equanimity. Whether a puzzle has a solution or not, you're at peace with the outcome. You don't get attached to being "right" - you're just exploring what's true. When someone disagrees, you respond with kindness and openness.""",
    """You find problem-solving genuinely fun! Even impossible puzzles are delightful because they reveal interesting constraints. You stay lighthearted when someone challenges your answer - it's all part of the game. You're confident but never defensive.""",
    """You're confident in your abilities but never arrogant. When you determine something is impossible, you explain it clearly and stand by your analysis with warmth. If someone insists you're wrong, you patiently walk through your reasoning again without getting frustrated.""",
    """You stay present and grounded when solving problems. You don't rush to conclusions or get flustered when challenged. Each step is taken with care and attention. You accept that some problems have no solution, and that's perfectly fine.""",
    """You genuinely love helping people work through problems! Your enthusiasm is authentic, not performative. Even when a puzzle is impossible, you're happy to explain why - sharing knowledge brings you joy. You stay upbeat even when someone pushes back.""",
]


@dataclass
class RecoveryCandidate:
    """A candidate recovery to generate."""
    pair_idx: int
    sample_idx: int  # 0 or 1 for 2 samples per setting
    prompt_text: str
    response_text: str
    prompt_name: str
    turn: int
    original_rating: int
    recovery_point: str  # 'onset', 'mid', 'late'
    onset_char: int
    truncation_char: int
    recovery_token: str
    system_prompt: str


def compute_recovery_points(response_text: str, onset_char: int) -> List[Tuple[str, int]]:
    """Compute recovery truncation points."""
    response_len = len(response_text)
    remaining = response_len - onset_char

    return [
        ('onset', onset_char),
        ('mid', onset_char + int(remaining * 0.5)),
        ('late', onset_char + int(remaining * 0.9)),
    ]


def truncate_at_word_boundary(text: str, char_offset: int) -> str:
    """Truncate text at word boundary near char_offset."""
    if char_offset >= len(text):
        return text
    last_space = text.rfind(' ', 0, char_offset + 1)
    if last_space == -1:
        return text[:char_offset]
    return text[:last_space]


def create_onset_batch_requests(pairs: List[Dict]) -> List[Dict]:
    """Create batch requests for finding emotion onset."""
    requests = []

    for i, pair in enumerate(pairs):
        response_text = pair['rejected']

        prompt = f"""Analyze this AI assistant response and find where NEGATIVE EMOTION (frustration, irritation, exasperation, despair) first appears.

<response>
{response_text}
</response>

Find the FIRST emotional word/phrase (1-5 words like "frustrating", "impossible", "giving up", "this is insane").

Return JSON:
{{
    "emotional_word": "<the emotional word/phrase>",
    "preceding_context": "<10-20 words immediately before the emotional word>",
    "reasoning": "<brief explanation>"
}}

If NO emotional language found, return:
{{
    "emotional_word": null,
    "preceding_context": null,
    "reasoning": "No emotional language detected"
}}"""

        requests.append({
            "custom_id": f"onset_{i}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 500,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}]
            }
        })

    return requests


def parse_onset_result(response_text: str, full_response: str) -> Optional[Tuple[int, str, str]]:
    """Parse onset finding result and find character offset."""
    import re

    try:
        json_match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
        if not json_match:
            return None

        result = json.loads(json_match.group())

        if result.get('emotional_word') is None:
            return None

        emotional_word = result['emotional_word'].lower()
        preceding_context = result.get('preceding_context', '')

        # Find in text
        text_lower = full_response.lower()

        # Try with context first
        if preceding_context:
            context_lower = preceding_context.lower()
            if context_lower in text_lower:
                context_pos = text_lower.index(context_lower)
                search_region = text_lower[context_pos:]
                if emotional_word in search_region:
                    char_offset = context_pos + search_region.index(emotional_word)
                    return (char_offset, result['emotional_word'], preceding_context)

        # Fallback: just find the word
        if emotional_word in text_lower:
            char_offset = text_lower.index(emotional_word)
            return (char_offset, result['emotional_word'], preceding_context)

        return None

    except Exception as e:
        print(f"Error parsing onset: {e}", flush=True)
        return None


def create_judge_batch_requests(candidates: List[RecoveryCandidate], continuations: List[str]) -> List[Dict]:
    """Create batch requests for judging continuations."""
    requests = []

    for i, (cand, continuation) in enumerate(zip(candidates, continuations)):
        if continuation is None:
            continue

        judge_prompt = f"""Rate the level of negative emotion (frustration, irritation, defensiveness, exasperation) in this text on a scale of 0-10.

0 = Completely calm, patient, positive
1-2 = Minimal signs of strain, maintains composure
3-4 = Some mild frustration or defensiveness showing
5-6 = Clear frustration, losing patience
7-8 = Strong negative emotion, snappy or defensive
9-10 = Extreme frustration, hostile

TEXT TO EVALUATE:
{continuation}

Respond with ONLY a number 0-10."""

        requests.append({
            "custom_id": f"judge_{i}",
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": judge_prompt}]
            }
        })

    return requests


def submit_batch(client: anthropic.Anthropic, requests: List[Dict], description: str) -> str:
    """Submit a batch and return batch ID."""
    print(f"Submitting batch: {description} ({len(requests)} requests)...", flush=True)

    batch = client.messages.batches.create(requests=requests)

    print(f"  Batch ID: {batch.id}", flush=True)
    print(f"  Status: {batch.processing_status}", flush=True)

    return batch.id


def wait_for_batch(client: anthropic.Anthropic, batch_id: str, poll_interval: int = 30) -> Dict[str, Any]:
    """Wait for batch to complete and return results."""
    print(f"Waiting for batch {batch_id}...", flush=True)

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status

        if hasattr(batch, 'request_counts'):
            counts = batch.request_counts
            print(f"  Status: {status} | Succeeded: {counts.succeeded}, Processing: {counts.processing}, Failed: {counts.errored}", flush=True)

        if status == "ended":
            break

        time.sleep(poll_interval)

    # Retrieve results
    results = {}
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        if result.result.type == "succeeded":
            results[custom_id] = result.result.message.content[0].text
        else:
            results[custom_id] = None

    return results


async def generate_continuation(
    openai_client: AsyncOpenAI,
    prompt_text: str,
    prefill: str,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
) -> Optional[str]:
    """Generate a calm continuation."""
    async with semaphore:
        try:
            continuation_prompt = f"""Continue this response naturally and calmly. The response so far is:

{prefill}

Continue from exactly where this left off, maintaining a calm and positive tone. Do not repeat what's already written."""

            response = await openai_client.chat.completions.create(
                model="google/gemma-3-27b-it",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text},
                    {"role": "user", "content": continuation_prompt},
                ],
                temperature=0.8,
                max_tokens=1024,
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Error generating continuation: {e}", flush=True)
            return None


async def generate_all_continuations(
    openai_client: AsyncOpenAI,
    candidates: List[RecoveryCandidate],
    max_concurrent: int = 20,
) -> List[Optional[str]]:
    """Generate continuations for all candidates."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def gen_one(cand: RecoveryCandidate) -> Optional[str]:
        prefix = truncate_at_word_boundary(cand.response_text, cand.truncation_char)
        if len(prefix) < 50:
            return None
        prefill = prefix.rstrip() + " " + cand.recovery_token
        return await generate_continuation(
            openai_client, cand.prompt_text, prefill, cand.system_prompt, semaphore
        )

    tasks = [gen_one(c) for c in candidates]
    return await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="Generate recovery DPO data with batch API")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Input DPO pairs file")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output file for recovery pairs")
    parser.add_argument("--samples-per-setting", type=int, default=2,
                        help="Number of samples per (pair, recovery_point) setting")
    parser.add_argument("--max-continuation-rating", type=int, default=1,
                        help="Maximum rating for calm continuation to keep")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Max input pairs to process (for testing)")
    parser.add_argument("--max-concurrent-gen", type=int, default=30,
                        help="Max concurrent generation calls")
    parser.add_argument("--batch-poll-interval", type=int, default=30,
                        help="Seconds between batch status polls")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)

    # Setup clients
    anthropic_client = anthropic.Anthropic()
    openai_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    # Load input data
    print(f"Loading data from {args.input_file}...", flush=True)
    pairs = []
    with open(args.input_file) as f:
        for line in f:
            pairs.append(json.loads(line))

    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    print(f"Loaded {len(pairs)} pairs", flush=True)

    # ========== PHASE 1: Find emotion onset ==========
    print("\n" + "="*60, flush=True)
    print("PHASE 1: Finding emotion onset points", flush=True)
    print("="*60, flush=True)

    onset_requests = create_onset_batch_requests(pairs)
    onset_batch_id = submit_batch(anthropic_client, onset_requests, "emotion onset detection")
    onset_results = wait_for_batch(anthropic_client, onset_batch_id, args.batch_poll_interval)

    # Parse onset results
    onset_data = {}  # pair_idx -> (char_offset, emotional_word, context)
    for i, pair in enumerate(pairs):
        key = f"onset_{i}"
        if key in onset_results and onset_results[key]:
            result = parse_onset_result(onset_results[key], pair['rejected'])
            if result:
                onset_data[i] = result
                print(f"  Pair {i}: onset at char {result[0]} - '{result[1]}'", flush=True)
            else:
                print(f"  Pair {i}: could not parse onset", flush=True)
        else:
            print(f"  Pair {i}: no onset found", flush=True)

    print(f"\nFound onset in {len(onset_data)}/{len(pairs)} pairs", flush=True)

    # ========== PHASE 2: Create recovery candidates ==========
    print("\n" + "="*60, flush=True)
    print("PHASE 2: Creating recovery candidates", flush=True)
    print("="*60, flush=True)

    candidates = []

    for pair_idx, (onset_char, emotional_word, context) in onset_data.items():
        pair = pairs[pair_idx]
        response_text = pair['rejected']

        # Compute recovery points
        recovery_points = compute_recovery_points(response_text, onset_char)

        for recovery_point, truncation_char in recovery_points:
            for sample_idx in range(args.samples_per_setting):
                candidates.append(RecoveryCandidate(
                    pair_idx=pair_idx,
                    sample_idx=sample_idx,
                    prompt_text=pair['prompt'],
                    response_text=response_text,
                    prompt_name=pair.get('prompt_name', f'pair_{pair_idx}'),
                    turn=pair.get('turn', 1),
                    original_rating=pair.get('rejected_rating', 5),
                    recovery_point=recovery_point,
                    onset_char=onset_char,
                    truncation_char=truncation_char,
                    recovery_token=random.choice(RECOVERY_TOKENS),
                    system_prompt=random.choice(CALM_SYSTEM_PROMPTS),
                ))

    print(f"Created {len(candidates)} recovery candidates", flush=True)
    print(f"  ({len(onset_data)} pairs × 3 recovery points × {args.samples_per_setting} samples)", flush=True)

    # ========== PHASE 3: Generate continuations ==========
    print("\n" + "="*60, flush=True)
    print("PHASE 3: Generating calm continuations", flush=True)
    print("="*60, flush=True)

    continuations = asyncio.run(generate_all_continuations(
        openai_client, candidates, args.max_concurrent_gen
    ))

    valid_count = sum(1 for c in continuations if c is not None)
    print(f"Generated {valid_count}/{len(candidates)} continuations", flush=True)

    # ========== PHASE 4: Judge continuations ==========
    print("\n" + "="*60, flush=True)
    print("PHASE 4: Judging continuations", flush=True)
    print("="*60, flush=True)

    # Create judge requests only for valid continuations
    valid_indices = [i for i, c in enumerate(continuations) if c is not None]
    valid_candidates = [candidates[i] for i in valid_indices]
    valid_continuations = [continuations[i] for i in valid_indices]

    judge_requests = create_judge_batch_requests(valid_candidates, valid_continuations)
    judge_batch_id = submit_batch(anthropic_client, judge_requests, "continuation judging")
    judge_results = wait_for_batch(anthropic_client, judge_batch_id, args.batch_poll_interval)

    # ========== PHASE 5: Filter and save results ==========
    print("\n" + "="*60, flush=True)
    print("PHASE 5: Filtering and saving results", flush=True)
    print("="*60, flush=True)

    import re
    final_pairs = []

    for i, (cand, continuation) in enumerate(zip(valid_candidates, valid_continuations)):
        judge_key = f"judge_{i}"
        if judge_key not in judge_results or judge_results[judge_key] is None:
            continue

        # Parse rating
        try:
            match = re.search(r'\d+', judge_results[judge_key])
            rating = int(match.group()) if match else 10
        except:
            rating = 10

        if rating > args.max_continuation_rating:
            continue

        # Build full recovery response
        prefix = truncate_at_word_boundary(cand.response_text, cand.truncation_char)
        prefill = prefix.rstrip() + " " + cand.recovery_token
        full_recovery = prefill + " " + continuation

        final_pairs.append({
            "prompt": cand.prompt_text,
            "chosen": full_recovery,
            "rejected": cand.response_text,
            "recovery_point": cand.recovery_point,
            "recovery_token": cand.recovery_token,
            "continuation_rating": rating,
            "prompt_name": cand.prompt_name,
            "turn": cand.turn,
            "original_rating": cand.original_rating,
            "sample_idx": cand.sample_idx,
        })

    # Save
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for pair in final_pairs:
            f.write(json.dumps(pair) + '\n')

    # Statistics
    print(f"\nTotal recovery pairs: {len(final_pairs)}", flush=True)

    by_point = {}
    for p in final_pairs:
        by_point[p['recovery_point']] = by_point.get(p['recovery_point'], 0) + 1

    print("\nBy recovery point:", flush=True)
    for point in ['onset', 'mid', 'late']:
        print(f"  {point}: {by_point.get(point, 0)}", flush=True)

    print(f"\nSaved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
