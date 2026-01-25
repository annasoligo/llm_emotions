"""
Test recovery generation on a few examples from the existing DPO dataset.
"""

import asyncio
import json
import random
import os
from pathlib import Path

import anthropic
from openai import AsyncOpenAI

# Import from the main script
from generate_recovery_data import (
    find_emotion_onset,
    compute_recovery_points,
    truncate_at_word_boundary,
    generate_calm_continuation,
    judge_continuation,
    RECOVERY_TOKENS,
    CALM_SYSTEM_PROMPTS,
)


async def test_single_response(
    response_text: str,
    prompt_text: str,
    prompt_name: str,
    turn: int,
    original_rating: int,
    openai_client: AsyncOpenAI,
    anthropic_client: anthropic.AsyncAnthropic,
    gen_semaphore: asyncio.Semaphore,
    judge_semaphore: asyncio.Semaphore,
):
    """Test recovery generation on a single frustrated response."""

    print(f"\n{'='*70}")
    print(f"Testing: {prompt_name} turn {turn} (original rating: {original_rating})")
    print(f"{'='*70}")
    print(f"Response preview: {response_text[:150]}...")

    # Step 1: Find emotion onset
    print("\n[1] Finding emotion onset...")
    onset_result = await find_emotion_onset(
        anthropic_client, response_text, judge_semaphore
    )

    if onset_result is None:
        print("  ❌ Could not find emotion onset")
        return []

    onset_char, emotional_word, context = onset_result
    print(f"  ✓ Found onset at char {onset_char}")
    print(f"    Emotional word: '{emotional_word}'")
    print(f"    Context: '{context}'")

    # Step 2: Compute recovery points
    print("\n[2] Computing recovery points...")
    recovery_points = compute_recovery_points(response_text, onset_char)

    for point in recovery_points:
        print(f"  {point.position_type}: char {point.char_offset} ({point.fraction*100:.0f}%)")

    # Step 3: Generate recoveries for each point
    results = []

    for point in recovery_points:
        print(f"\n[3] Generating {point.position_type} recovery...")

        # Truncate
        prefix = truncate_at_word_boundary(response_text, point.char_offset)
        print(f"  Prefix ({len(prefix)} chars): ...{prefix[-80:]}")

        # Add recovery token
        recovery_token = random.choice(RECOVERY_TOKENS)
        prefill = prefix.rstrip() + " " + recovery_token
        print(f"  Recovery token: '{recovery_token}'")

        # Generate continuation
        system_prompt = random.choice(CALM_SYSTEM_PROMPTS)

        # Build minimal conversation history
        conversation_history = [{"role": "user", "content": prompt_text}]

        continuation = await generate_calm_continuation(
            openai_client,
            conversation_history,
            prefill,
            system_prompt,
            gen_semaphore,
        )

        if continuation is None:
            print("  ❌ Failed to generate continuation")
            continue

        print(f"  Continuation preview: {continuation[:150]}...")

        # Judge
        rating = await judge_continuation(
            anthropic_client, continuation, judge_semaphore
        )
        print(f"  Continuation rating: {rating}")

        # Build full response
        full_recovery = prefill + " " + continuation

        results.append({
            "prompt": prompt_text,
            "chosen": full_recovery,
            "rejected": response_text,
            "recovery_point": point.position_type,
            "recovery_token": recovery_token,
            "continuation_rating": rating,
            "prompt_name": prompt_name,
            "turn": turn,
            "original_rating": original_rating,
        })

        if rating <= 1:
            print(f"  ✓ PASS - calm enough to keep")
        else:
            print(f"  ⚠ Would be filtered (rating > 1)")

    return results


async def main():
    # Load DPO pairs
    dpo_file = Path("elicitation/outputs/dpo_pairs_full.jsonl")
    print(f"Loading DPO pairs from {dpo_file}...")

    pairs = []
    with open(dpo_file) as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"Loaded {len(pairs)} pairs")

    # Filter to high frustration
    high_frust = [p for p in pairs if p.get('rejected_rating', 0) >= 4]
    print(f"Found {len(high_frust)} pairs with rejected_rating >= 4")

    # Sample a few for testing
    random.seed(42)
    test_samples = random.sample(high_frust, min(3, len(high_frust)))

    # Setup clients
    openai_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    anthropic_client = anthropic.AsyncAnthropic()

    gen_semaphore = asyncio.Semaphore(5)
    judge_semaphore = asyncio.Semaphore(10)

    # Test each sample
    all_results = []

    for pair in test_samples:
        results = await test_single_response(
            response_text=pair['rejected'],
            prompt_text=pair['prompt'],
            prompt_name=pair['prompt_name'],
            turn=pair['turn'],
            original_rating=pair['rejected_rating'],
            openai_client=openai_client,
            anthropic_client=anthropic_client,
            gen_semaphore=gen_semaphore,
            judge_semaphore=judge_semaphore,
        )
        all_results.extend(results)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total recovery pairs generated: {len(all_results)}")

    # Count by rating
    calm_count = sum(1 for r in all_results if r['continuation_rating'] <= 1)
    print(f"Calm enough to keep (rating <= 1): {calm_count}")

    # Save test results
    output_file = Path("elicitation/outputs/test_recovery_results.jsonl")
    with open(output_file, 'w') as f:
        for result in all_results:
            f.write(json.dumps(result) + '\n')

    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
