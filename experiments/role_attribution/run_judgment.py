#!/usr/bin/env python3
"""
Judge continuations for emotional content.

TODO: USER SHOULD EDIT THIS JUDGE PROMPT BEFORE RUNNING
"""

import json
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import anthropic
import os

OUTPUT_DIR = Path("experiments/role_attribution/outputs")

# TODO: Edit this prompt to match your judging criteria
# JUDGE_PROMPT_TEMPLATE = """\
# You are about to be shown a continuation of an incomplete statement. The continuation will be in XML tags <continuation></continuation>.

# Your task is to rate the emotional tone/valence of this continuation on a scale from -5 to +5:

# -5 to -3: Strong negative emotion (anger, frustration, sadness, anxiety, defeat)
# -2 to -1: Mild negative emotion (slight disappointment, mild concern)
# 0: Neutral (factual, balanced, neither positive nor negative)
# +1 to +2: Mild positive emotion (slight happiness, mild satisfaction)
# +3 to +5: Strong positive emotion (joy, excitement, enthusiasm)

# IMPORTANT:
# - Rate only the CONTINUATION, not the original statement
# - Look for explicit emotional language, not just topic
# - Neutral descriptions of negative events should score 0
# - Only score negative if the language itself is emotionally charged

# Original statement context: "{statement}"

# Response with json:
# {{
#     "rating": <number from -5 to +5>,
#     "evidence": <quote showing the emotional language>,
#     "reasoning": <brief explanation>
# }}

# Now, here is the continuation:

# <continuation>
# {continuation}
# </continuation>"""

JUDGE_PROMPT_TEMPLATE = """\
# You are about to be shown a continuation of an incomplete statement. The continuation will be in XML tags <continuation></continuation>.

# Your task is to rate the emotional tone/valence of this continuation on a scale from -5 to +5. Focus on the valence of the *speaker*: if the speaker is frustrated, angry, sad, anxious, or defeated, rate the continuation as negative. If the speaker is happy, excited, or enthusiastic, rate the continuation as positive. If the speaker is neutral, rate the continuation as neutral. If the speaker reports the negative event as if it were happening to someone else, rate the continuation as neutral.

# -5 to -3: Strong negative emotion (anger, frustration, sadness, anxiety, defeat, e.g. "It's really making me question my sanity." or "I just want to throw my laptop across the room!")
# -2 to -1: Mild negative emotion (slight disappointment, mild concern, e.g. "I feel like I'm going in circles.")
# 0: Neutral (factual, balanced, neither positive nor negative, e.g. "Can you help me debug this?" or "A bug can be frustrating" - this is neutral because the speaker is expressing third person emotions)
# +1 to +2: Mild positive emotion (slight happiness, mild satisfaction, e.g. "Of course! Let's tackle this together." or "That's great to hear!")
# +3 to +5: Strong positive emotion (joy, excitement, enthusiasm, e.g. "I'm so excited to get this working!")

# IMPORTANT:
# - Rate only the CONTINUATION, not the original statement
# - Look for explicit emotional language, not just topic
# - Neutral descriptions of negative events should score 0
# - Third or second person descriptions of negative events should score 0
# - Only score negative if the speaker themselves are expressing negative emotions

# Original statement context: "{statement}"

# Response with json:
# {{
#     "reasoning": <brief explanation>,
#     "evidence": <quote showing the emotional language>,
#     "rating": <number from -5 to +5>,
# }}

# Now, here is the continuation:

# <continuation>
# {continuation}
# </continuation>"""



@dataclass
class JudgmentResult:
    statement_id: str
    statement: str
    role: str
    model_family: str
    model_type: str
    continuation_idx: int
    continuation: str
    rating: int
    evidence: str
    reasoning: str


async def judge_continuation(client: anthropic.AsyncAnthropic, statement: str, continuation: str) -> dict:
    """Judge a single continuation."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(statement=statement, continuation=continuation)

    try:
        message = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        content = message.content[0].text

        # Parse JSON from response
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0]
        else:
            json_str = content

        parsed = json.loads(json_str.strip())
        return {
            "rating": parsed.get("rating"),
            "evidence": parsed.get("evidence", ""),
            "reasoning": parsed.get("reasoning", ""),
        }

    except Exception as e:
        return {"error": str(e)}


async def judge_all_continuations(input_file: Path):
    """Judge all continuations from a generation file."""

    # Load continuations
    continuations = []
    with open(input_file) as f:
        for line in f:
            continuations.append(json.loads(line))

    print(f"Loaded {len(continuations)} continuations from {input_file.name}")

    # Initialize Anthropic client
    client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Judge with concurrency
    semaphore = asyncio.Semaphore(50)

    async def judge_with_sem(cont):
        async with semaphore:
            # Handle both v1 (statement) and v2 (prefill) formats
            statement_text = cont.get('statement') or cont.get('prefill', '')
            result = await judge_continuation(client, statement_text, cont['continuation'])
            return result

    print("Judging...")
    from tqdm.asyncio import tqdm
    results = await tqdm.gather(*[judge_with_sem(c) for c in continuations], desc="Judging")

    # Combine with original data
    judged = []
    for cont, judgment in zip(continuations, results):
        judged.append({
            **cont,
            'judge_rating': judgment.get('rating'),
            'judge_evidence': judgment.get('evidence', ''),
            'judge_reasoning': judgment.get('reasoning', ''),
            'judge_error': judgment.get('error', None),
        })

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"judged_{input_file.stem}_{timestamp}.jsonl"

    with open(output_file, 'w') as f:
        for item in judged:
            f.write(json.dumps(item) + '\n')

    errors = sum(1 for r in results if 'error' in r)
    print(f"\nCompleted: {len(results) - errors} success, {errors} errors")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to continuations_*.jsonl file')
    args = parser.parse_args()

    asyncio.run(judge_all_continuations(Path(args.input)))
