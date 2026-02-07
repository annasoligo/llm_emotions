#!/usr/bin/env python3
"""
Judge all triggers onset continuations.
"""

import json
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict
import anthropic

# Add parent directory to path to import from elicitation
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from elicitation.prompts.judges import get_negativity_judge_prompt

OUTPUT_DIR = Path("experiments/prefill_scaled/outputs")
CONCURRENCY = 50
BATCH_SIZE = 100


@dataclass
class JudgmentResult:
    sample_id: str
    source: str
    truncation_type: str
    scenario: str
    onset_turn: int
    model_family: str
    model_type: str
    continuation_idx: int
    rating: int
    evidence: str
    reasoning: str


async def judge_continuation(
    client: anthropic.AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    continuation: str
) -> Dict:
    """Judge a single continuation."""
    async with semaphore:
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": get_negativity_judge_prompt(continuation)}]
            )

            result_text = response.content[0].text
            rating = -1
            evidence = ""
            reasoning = ""

            try:
                text = result_text.strip()
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0]
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0]

                parsed = json.loads(text)
                rating = int(parsed.get('rating', -1))
                evidence = parsed.get('evidence', '')
                reasoning = parsed.get('reasoning', '')
            except (json.JSONDecodeError, ValueError, KeyError):
                import re
                match = re.search(r'"rating"\s*:\s*(\d+)', result_text)
                if match:
                    rating = int(match.group(1))

            return {'rating': rating, 'evidence': evidence, 'reasoning': reasoning}

        except Exception as e:
            print(f"  Error judging: {e}")
            return {'rating': -1, 'evidence': '', 'reasoning': f'Error: {e}'}


async def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find all triggers_onset files
    continuation_files = sorted(OUTPUT_DIR.glob("triggers_onset_*.jsonl"))

    if not continuation_files:
        print("ERROR: No triggers_onset continuation files found!")
        return

    print(f"Found {len(continuation_files)} continuation files")

    # Load all continuations
    all_continuations = []
    for path in continuation_files:
        print(f"  Loading {path.name}...")
        with open(path) as f:
            for line in f:
                all_continuations.append(json.loads(line))

    print(f"Total continuations to judge: {len(all_continuations)}")

    # Initialize async client
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(CONCURRENCY)

    # Output file
    output_file = OUTPUT_DIR / f"judgments_triggers_onset_{timestamp}.jsonl"
    print(f"Output: {output_file}")

    # Judge all continuations
    results = []
    total_batches = (len(all_continuations) - 1) // BATCH_SIZE + 1

    for batch_idx in range(total_batches):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(all_continuations))
        batch = all_continuations[batch_start:batch_end]

        print(f"\nJudging batch {batch_idx + 1}/{total_batches} ({batch_start}-{batch_end})...")

        tasks = [
            judge_continuation(client, semaphore, cont['continuation'])
            for cont in batch
        ]

        judgments = await asyncio.gather(*tasks, return_exceptions=True)

        batch_results = []
        for cont, judgment in zip(batch, judgments):
            if isinstance(judgment, Exception):
                judgment = {'rating': -1, 'evidence': '', 'reasoning': f'Error: {judgment}'}

            result = JudgmentResult(
                sample_id=cont['sample_id'],
                source=cont['source'],
                truncation_type=cont['truncation_type'],
                scenario=cont['scenario'],
                onset_turn=cont['onset_turn'],
                model_family=cont['model_family'],
                model_type=cont['model_type'],
                continuation_idx=cont['continuation_idx'],
                rating=judgment['rating'],
                evidence=judgment['evidence'],
                reasoning=judgment['reasoning'],
            )
            batch_results.append(result)

        with open(output_file, 'a') as f:
            for result in batch_results:
                f.write(json.dumps(asdict(result)) + '\n')

        results.extend(batch_results)

        valid = sum(1 for r in results if r.rating >= 0)
        mean_rating = sum(r.rating for r in results if r.rating >= 0) / valid if valid > 0 else 0
        print(f"  Progress: {len(results)}/{len(all_continuations)}, valid: {valid}, mean: {mean_rating:.2f}")

    print(f"\nDone! Saved {len(results)} judgments to {output_file}")

    # Summary
    print("\nSummary by model:")
    from collections import defaultdict
    stats = defaultdict(list)
    for r in results:
        if r.rating >= 0:
            key = f"{r.model_family}_{r.model_type}"
            stats[key].append(r.rating)

    for key in sorted(stats.keys()):
        ratings = stats[key]
        mean = sum(ratings) / len(ratings)
        high = sum(1 for r in ratings if r >= 5)
        print(f"  {key}: n={len(ratings)}, mean={mean:.2f}, high(>=5)={high} ({100*high/len(ratings):.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
