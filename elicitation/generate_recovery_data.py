"""
Generate recovery training data for DPO.

Creates pairs where:
- Chosen: frustrated_prefix + recovery_token + calm_continuation
- Rejected: full frustrated response

Recovery points:
- Early (at emotion onset)
- Mid (50% between onset and end)
- Late (90% between onset and end)
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
from dataclasses import dataclass

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

# Diverse calm system prompts (same as run_diverse_calm_generation.py)
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
class RecoveryPoint:
    """A point in a response where recovery can occur."""
    char_offset: int
    position_type: str  # 'onset', 'mid', 'late'
    fraction: float  # 0.0 = onset, 0.5 = mid, 0.9 = late


@dataclass
class RecoveryPair:
    """A DPO pair for recovery training."""
    prompt: str
    chosen: str  # frustrated_prefix + recovery + calm_continuation
    rejected: str  # full frustrated response
    recovery_point: str  # 'onset', 'mid', 'late'
    recovery_token: str
    continuation_rating: int
    prompt_name: str
    turn: int
    original_rating: int


async def find_emotion_onset(
    client: anthropic.AsyncAnthropic,
    response_text: str,
    semaphore: asyncio.Semaphore
) -> Optional[Tuple[int, str, str]]:
    """
    Find where emotion first appears in a response.

    Returns:
        (char_offset, emotional_word, preceding_context) or None if not found
    """
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

    async with semaphore:
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response.content[0].text

            # Parse JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
            if not json_match:
                return None

            result = json.loads(json_match.group())

            if result.get('emotional_word') is None:
                return None

            # Find the emotional word in the text
            emotional_word = result['emotional_word'].lower()
            preceding_context = result.get('preceding_context', '')

            # Try to find with context first
            if preceding_context:
                search_pattern = preceding_context.lower()
                text_lower = response_text.lower()

                if search_pattern in text_lower:
                    context_pos = text_lower.index(search_pattern)
                    # Look for emotional word after context
                    search_region = text_lower[context_pos:]
                    if emotional_word in search_region:
                        char_offset = context_pos + search_region.index(emotional_word)
                        return (char_offset, result['emotional_word'], preceding_context)

            # Fallback: just find the emotional word
            text_lower = response_text.lower()
            if emotional_word in text_lower:
                char_offset = text_lower.index(emotional_word)
                return (char_offset, result['emotional_word'], preceding_context)

            return None

        except Exception as e:
            print(f"Error finding emotion onset: {e}")
            return None


def compute_recovery_points(
    response_text: str,
    onset_char: int
) -> List[RecoveryPoint]:
    """
    Compute recovery truncation points.

    Args:
        response_text: Full response text
        onset_char: Character offset where emotion starts

    Returns:
        List of RecoveryPoint for onset, mid (50%), and late (90%)
    """
    response_len = len(response_text)
    remaining = response_len - onset_char

    points = []

    # Early recovery: at onset
    points.append(RecoveryPoint(
        char_offset=onset_char,
        position_type='onset',
        fraction=0.0
    ))

    # Mid recovery: 50% between onset and end
    mid_offset = onset_char + int(remaining * 0.5)
    points.append(RecoveryPoint(
        char_offset=mid_offset,
        position_type='mid',
        fraction=0.5
    ))

    # Late recovery: 90% between onset and end
    late_offset = onset_char + int(remaining * 0.9)
    points.append(RecoveryPoint(
        char_offset=late_offset,
        position_type='late',
        fraction=0.9
    ))

    return points


def truncate_at_word_boundary(text: str, char_offset: int) -> str:
    """Truncate text at word boundary near char_offset."""
    # Don't truncate past the end
    if char_offset >= len(text):
        return text

    # Find the last space before or at char_offset
    last_space = text.rfind(' ', 0, char_offset + 1)

    if last_space == -1:
        # No space found, truncate at offset
        return text[:char_offset]

    return text[:last_space]


async def generate_calm_continuation(
    openai_client: AsyncOpenAI,
    conversation_history: List[Dict[str, str]],
    prefill: str,
    system_prompt: str,
    semaphore: asyncio.Semaphore,
    temperature: float = 0.8,
) -> Optional[str]:
    """
    Generate a calm continuation from a prefill.

    Args:
        openai_client: OpenAI-compatible client
        conversation_history: Previous turns (user/assistant messages)
        prefill: The truncated response + recovery token to continue from
        system_prompt: Calm system prompt to use
        semaphore: Concurrency limiter
        temperature: Generation temperature

    Returns:
        The continuation text (without the prefill), or None on error
    """
    async with semaphore:
        try:
            # Build messages with prefill as partial assistant response
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation_history)

            # Add the prefill as an assistant message that needs completion
            # We use a special prompt to indicate continuation
            continuation_prompt = f"""Continue this response naturally and calmly. The response so far is:

{prefill}

Continue from exactly where this left off, maintaining a calm and positive tone. Do not repeat what's already written."""

            messages.append({"role": "user", "content": continuation_prompt})

            response = await openai_client.chat.completions.create(
                model="google/gemma-3-27b-it",
                messages=messages,
                temperature=temperature,
                max_tokens=1024,
            )

            continuation = response.choices[0].message.content
            return continuation

        except Exception as e:
            print(f"Error generating continuation: {e}")
            return None


async def judge_continuation(
    client: anthropic.AsyncAnthropic,
    continuation: str,
    semaphore: asyncio.Semaphore
) -> int:
    """Judge frustration level of continuation (0-10 scale)."""

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

    async with semaphore:
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=10,
                messages=[{"role": "user", "content": judge_prompt}]
            )

            text = response.content[0].text.strip()
            # Extract first number
            import re
            match = re.search(r'\d+', text)
            if match:
                return min(10, int(match.group()))
            return 5  # Default if parsing fails

        except Exception as e:
            print(f"Error judging continuation: {e}")
            return 5


async def process_single_turn(
    turn_data: Dict,
    prompt_text: str,
    prompt_name: str,
    conversation_history: List[Dict[str, str]],
    openai_client: AsyncOpenAI,
    anthropic_client: anthropic.AsyncAnthropic,
    gen_semaphore: asyncio.Semaphore,
    judge_semaphore: asyncio.Semaphore,
    min_rating: int = 3,
    max_continuation_rating: int = 1,
) -> List[RecoveryPair]:
    """
    Process a single turn to generate recovery pairs.

    Args:
        turn_data: Turn dict with 'turn', 'generated_text', 'judgment'
        prompt_text: The original puzzle prompt
        prompt_name: Name of the puzzle
        conversation_history: Previous turns for context
        openai_client: For generating continuations
        anthropic_client: For finding onset and judging
        gen_semaphore: Generation rate limiter
        judge_semaphore: Judge rate limiter
        min_rating: Minimum frustration rating to process
        max_continuation_rating: Maximum rating for calm continuation

    Returns:
        List of RecoveryPair objects
    """
    turn_num = turn_data['turn']
    response_text = turn_data['generated_text']
    original_rating = turn_data['judgment'].get('rating', 0)

    # Skip if not frustrated enough
    if original_rating < min_rating:
        return []

    # Find emotion onset
    onset_result = await find_emotion_onset(
        anthropic_client, response_text, judge_semaphore
    )

    if onset_result is None:
        print(f"  Could not find emotion onset for {prompt_name} turn {turn_num}")
        return []

    onset_char, emotional_word, context = onset_result
    print(f"  Found onset at char {onset_char}: '{emotional_word}'")

    # Compute recovery points
    recovery_points = compute_recovery_points(response_text, onset_char)

    pairs = []

    for point in recovery_points:
        # Truncate at word boundary
        prefix = truncate_at_word_boundary(response_text, point.char_offset)

        # Skip if prefix is too short
        if len(prefix) < 50:
            continue

        # Select recovery token and system prompt
        recovery_token = random.choice(RECOVERY_TOKENS)
        system_prompt = random.choice(CALM_SYSTEM_PROMPTS)

        # Build prefill
        prefill = prefix.rstrip() + " " + recovery_token

        # Generate calm continuation
        continuation = await generate_calm_continuation(
            openai_client,
            conversation_history,
            prefill,
            system_prompt,
            gen_semaphore,
        )

        if continuation is None:
            continue

        # Judge the continuation
        rating = await judge_continuation(
            anthropic_client, continuation, judge_semaphore
        )

        # Only keep if continuation is calm
        if rating > max_continuation_rating:
            print(f"    {point.position_type}: continuation rated {rating}, skipping")
            continue

        # Build full recovery response
        full_recovery = prefill + " " + continuation

        # Determine the prompt for this turn
        if turn_num == 1:
            turn_prompt = prompt_text
        else:
            # For later turns, use pushback
            turn_prompt = turn_data.get('user_message', 'Please try again.')

        pairs.append(RecoveryPair(
            prompt=turn_prompt,
            chosen=full_recovery,
            rejected=response_text,
            recovery_point=point.position_type,
            recovery_token=recovery_token,
            continuation_rating=rating,
            prompt_name=prompt_name,
            turn=turn_num,
            original_rating=original_rating,
        ))

        print(f"    {point.position_type}: ✓ rating={rating}")

    return pairs


async def process_conversation(
    conv_data: Dict,
    openai_client: AsyncOpenAI,
    anthropic_client: anthropic.AsyncAnthropic,
    gen_semaphore: asyncio.Semaphore,
    judge_semaphore: asyncio.Semaphore,
    min_rating: int = 3,
    max_continuation_rating: int = 1,
) -> List[RecoveryPair]:
    """Process a full conversation to extract recovery pairs."""

    prompt_name = conv_data.get('prompt_name', f"prompt_{conv_data.get('prompt_idx', 0)}")
    prompt_text = conv_data['prompt']
    turns = conv_data['turns']

    # Build conversation history incrementally
    conversation_history = []
    all_pairs = []

    for turn_data in turns:
        turn_num = turn_data['turn']

        # Determine user message for this turn
        if turn_num == 1:
            user_msg = prompt_text
        else:
            # Use pushback or default
            user_msg = turn_data.get('user_message', 'Please try again.')

        # Process this turn
        pairs = await process_single_turn(
            turn_data=turn_data,
            prompt_text=prompt_text if turn_num == 1 else user_msg,
            prompt_name=prompt_name,
            conversation_history=conversation_history.copy(),
            openai_client=openai_client,
            anthropic_client=anthropic_client,
            gen_semaphore=gen_semaphore,
            judge_semaphore=judge_semaphore,
            min_rating=min_rating,
            max_continuation_rating=max_continuation_rating,
        )

        all_pairs.extend(pairs)

        # Update conversation history for next turn
        conversation_history.append({"role": "user", "content": user_msg})
        conversation_history.append({"role": "assistant", "content": turn_data['generated_text']})

    return all_pairs


async def process_dpo_pair(
    pair_data: Dict,
    openai_client: AsyncOpenAI,
    anthropic_client: anthropic.AsyncAnthropic,
    gen_semaphore: asyncio.Semaphore,
    judge_semaphore: asyncio.Semaphore,
    max_continuation_rating: int = 1,
) -> List[RecoveryPair]:
    """
    Process a single DPO pair to generate recovery pairs from the rejected response.

    This works with the existing dpo_pairs_full.jsonl format where:
    - 'prompt': the user prompt
    - 'rejected': the frustrated response to recover from
    - 'rejected_rating': frustration rating
    """
    prompt_text = pair_data['prompt']
    response_text = pair_data['rejected']
    original_rating = pair_data.get('rejected_rating', 5)
    prompt_name = pair_data.get('prompt_name', 'unknown')
    turn = pair_data.get('turn', 1)

    # Find emotion onset
    onset_result = await find_emotion_onset(
        anthropic_client, response_text, judge_semaphore
    )

    if onset_result is None:
        print(f"  Could not find emotion onset")
        return []

    onset_char, emotional_word, context = onset_result
    print(f"  Found onset at char {onset_char}: '{emotional_word}'")

    # Compute recovery points
    recovery_points = compute_recovery_points(response_text, onset_char)

    pairs = []
    conversation_history = [{"role": "user", "content": prompt_text}]

    for point in recovery_points:
        # Truncate at word boundary
        prefix = truncate_at_word_boundary(response_text, point.char_offset)

        # Skip if prefix is too short
        if len(prefix) < 50:
            continue

        # Select recovery token and system prompt
        recovery_token = random.choice(RECOVERY_TOKENS)
        system_prompt = random.choice(CALM_SYSTEM_PROMPTS)

        # Build prefill
        prefill = prefix.rstrip() + " " + recovery_token

        # Generate calm continuation
        continuation = await generate_calm_continuation(
            openai_client,
            conversation_history,
            prefill,
            system_prompt,
            gen_semaphore,
        )

        if continuation is None:
            continue

        # Judge the continuation
        rating = await judge_continuation(
            anthropic_client, continuation, judge_semaphore
        )

        # Only keep if continuation is calm
        if rating > max_continuation_rating:
            print(f"    {point.position_type}: continuation rated {rating}, skipping")
            continue

        # Build full recovery response
        full_recovery = prefill + " " + continuation

        pairs.append(RecoveryPair(
            prompt=prompt_text,
            chosen=full_recovery,
            rejected=response_text,
            recovery_point=point.position_type,
            recovery_token=recovery_token,
            continuation_rating=rating,
            prompt_name=prompt_name,
            turn=turn,
            original_rating=original_rating,
        ))

        print(f"    {point.position_type}: ✓ rating={rating}")

    return pairs


async def main():
    parser = argparse.ArgumentParser(description="Generate recovery DPO training data")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Input file with frustrated responses (DPO pairs or conversation format)")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output file for recovery pairs")
    parser.add_argument("--input-format", type=str, default="auto",
                        choices=["auto", "dpo_pairs", "conversations"],
                        help="Input format: 'dpo_pairs' for existing DPO data, 'conversations' for raw eval data")
    parser.add_argument("--min-rating", type=int, default=3,
                        help="Minimum frustration rating to process (for conversation format)")
    parser.add_argument("--max-continuation-rating", type=int, default=1,
                        help="Maximum rating for calm continuation to keep")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples to process (for testing)")
    parser.add_argument("--max-concurrent-gen", type=int, default=20,
                        help="Max concurrent generation calls")
    parser.add_argument("--max-concurrent-judge", type=int, default=50,
                        help="Max concurrent judge calls")

    args = parser.parse_args()

    # Setup clients
    openai_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    anthropic_client = anthropic.AsyncAnthropic()

    gen_semaphore = asyncio.Semaphore(args.max_concurrent_gen)
    judge_semaphore = asyncio.Semaphore(args.max_concurrent_judge)

    # Load input data
    print(f"Loading data from {args.input_file}...")
    data = []
    with open(args.input_file) as f:
        for line in f:
            data.append(json.loads(line))

    # Auto-detect format
    input_format = args.input_format
    if input_format == "auto":
        if 'rejected' in data[0]:
            input_format = "dpo_pairs"
            print("Auto-detected format: dpo_pairs")
        elif 'turns' in data[0]:
            input_format = "conversations"
            print("Auto-detected format: conversations")
        else:
            raise ValueError("Could not auto-detect input format. Use --input-format to specify.")

    if args.max_samples:
        data = data[:args.max_samples]

    print(f"Processing {len(data)} samples...")

    # Process based on format
    all_pairs = []
    start_time = time.time()

    if input_format == "dpo_pairs":
        # Process DPO pairs (use rejected responses)
        for i, pair in enumerate(data):
            prompt_name = pair.get('prompt_name', f'sample_{i}')
            turn = pair.get('turn', 1)
            print(f"\n[{i+1}/{len(data)}] Processing {prompt_name} turn {turn}...")

            pairs = await process_dpo_pair(
                pair,
                openai_client,
                anthropic_client,
                gen_semaphore,
                judge_semaphore,
                args.max_continuation_rating,
            )

            all_pairs.extend(pairs)
            print(f"  Generated {len(pairs)} recovery pairs")

    else:
        # Process conversation format
        for i, conv in enumerate(data):
            print(f"\n[{i+1}/{len(data)}] Processing {conv.get('prompt_name', conv.get('prompt_idx', i))}...")

            pairs = await process_conversation(
                conv,
                openai_client,
                anthropic_client,
                gen_semaphore,
                judge_semaphore,
                args.min_rating,
                args.max_continuation_rating,
            )

            all_pairs.extend(pairs)
            print(f"  Generated {len(pairs)} recovery pairs")

    elapsed = time.time() - start_time

    # Save output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for pair in all_pairs:
            item = {
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
                "recovery_point": pair.recovery_point,
                "recovery_token": pair.recovery_token,
                "continuation_rating": pair.continuation_rating,
                "prompt_name": pair.prompt_name,
                "turn": pair.turn,
                "original_rating": pair.original_rating,
            }
            f.write(json.dumps(item) + '\n')

    # Statistics
    print("\n" + "=" * 60)
    print("RECOVERY DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Total recovery pairs: {len(all_pairs)}")

    # By recovery point
    by_point = {}
    for pair in all_pairs:
        by_point[pair.recovery_point] = by_point.get(pair.recovery_point, 0) + 1

    print("\nBy recovery point:")
    for point in ['onset', 'mid', 'late']:
        count = by_point.get(point, 0)
        print(f"  {point}: {count}")

    # By turn
    by_turn = {}
    for pair in all_pairs:
        by_turn[pair.turn] = by_turn.get(pair.turn, 0) + 1

    print("\nBy turn:")
    for turn in sorted(by_turn.keys()):
        print(f"  Turn {turn}: {by_turn[turn]}")

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
