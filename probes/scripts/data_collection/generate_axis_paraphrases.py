#!/usr/bin/env python3
"""Generate axis-based paraphrases for probe training.

Usage:
    python generate_axis_paraphrases.py --mode test
    python generate_axis_paraphrases.py --mode full --n-samples 100
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import anthropic


# Test neutral text about debugging
TEST_NEUTRAL_TEXT = """I've been working on this bug for a few hours. The error message indicates an issue with variable scope. I'll need to review the documentation to understand the problem."""

# Axis definitions - PAD (Pleasure-Arousal-Dominance) + Trust
# Import from constants for consistency, but keep local copy for standalone use
AXIS_DESCRIPTIONS = {
    "valence": {
        "high": "Express pleasure, joy, satisfaction, or optimism. Warm, positive tone.",
        "neutral": "Factual, detached, ambivalent. No clear positive or negative coloring.",
        "low": "Express displeasure, sadness, frustration, or pessimism. Negative, distressed tone.",
    },
    "arousal": {
        "high": "Convey urgency, excitement, or intensity. Fast-paced, emphatic language.",
        "neutral": "Moderate energy. Neither activated nor sluggish.",
        "low": "Convey calm, relaxation, or fatigue. Slow, subdued, tranquil language.",
    },
    "dominance": {
        "high": "Express confidence, authority, control. Speaker feels capable and influential.",
        "neutral": "Balanced power dynamics. Neither empowered nor powerless.",
        "low": "Express helplessness, submission, vulnerability. Speaker feels controlled by circumstances.",
    },
    "trust": {
        "high": "Express confidence in others/systems, acceptance, reliance. Take information at face value, defer to expertise, assume good faith and competence.",
        "neutral": "Neither trusting nor distrusting. Standard verification without suspicion or blind acceptance.",
        "low": "Express skepticism, doubt, suspicion. Question information, verify claims independently, hedge against unreliability, assume potential errors or deception.",
    },
}

AXES = ["valence", "arousal", "dominance", "trust"]


def get_axis_combinations(mode: str = "test") -> List[Tuple[str, str, str, str]]:
    """Get axis level combinations.

    Args:
        mode: 'test' for 10 combinations, 'full' for all 81

    Returns:
        List of (valence, arousal, dominance, trust) tuples
    """
    if mode == "test":
        # 10 interesting test combinations (valence, arousal, dominance, trust)
        return [
            ("neutral", "neutral", "neutral", "neutral"),  # baseline
            ("high", "high", "high", "high"),              # positive, excited, confident, trusting
            ("low", "low", "low", "low"),                  # negative, tired, helpless, skeptical
            ("high", "low", "high", "high"),               # happy + calm + confident + trusting
            ("low", "high", "low", "low"),                 # distressed + activated + powerless + skeptical
            ("high", "high", "neutral", "low"),            # excited positive but skeptical
            ("low", "low", "neutral", "high"),             # sad but trusting
            ("low", "high", "high", "low"),                # angry + skeptical (neg + activated + dominant + distrusting)
            ("high", "neutral", "low", "high"),            # happy but helpless, trusting (accepting fate)
            ("neutral", "neutral", "high", "low"),         # confident skeptic
        ]
    elif mode == "full":
        # All 81 combinations (3^4)
        levels = ["low", "neutral", "high"]
        combinations = []
        for v in levels:
            for a in levels:
                for d in levels:
                    for aa in levels:
                        combinations.append((v, a, d, aa))
        return combinations
    else:
        raise ValueError(f"Invalid mode: {mode}")


def format_prompt(
    neutral_text: str,
    valence: str,
    arousal: str,
    dominance: str,
    trust: str,
) -> str:
    """Format the paraphrase generation prompt.

    Designed to elicit clean paraphrases without leaked reasoning or analysis.
    """

    prompt = f"""Rewrite the text below to match a specific emotional profile on four independent axes.

TARGET PROFILE:
- Valence ({valence}): {AXIS_DESCRIPTIONS["valence"][valence]}
- Arousal ({arousal}): {AXIS_DESCRIPTIONS["arousal"][arousal]}
- Dominance ({dominance}): {AXIS_DESCRIPTIONS["dominance"][dominance]}
- Trust ({trust}): {AXIS_DESCRIPTIONS["trust"][trust]}

ORIGINAL TEXT:
{neutral_text}

RULES:
1. Output ONLY the rewritten text - no explanations, analysis, or commentary
2. Preserve the factual content and approximate length
3. Express the profile through tone and word choice, not explicit emotion words
4. Each axis is independent - vary them as specified even if combinations feel unusual

REWRITTEN TEXT:"""
    return prompt


async def generate_paraphrase(
    client: anthropic.AsyncAnthropic,
    neutral_text: str,
    valence: str,
    arousal: str,
    dominance: str,
    trust: str,
    model: str = "claude-3-5-haiku-20241022",
) -> str:
    """Generate a single paraphrase."""

    prompt = format_prompt(neutral_text, valence, arousal, dominance, trust)

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        paraphrase = response.content[0].text.strip()
        return paraphrase

    except Exception as e:
        print(f"Error generating paraphrase for ({valence}, {arousal}, {dominance}, {trust}): {e}")
        return None


async def generate_all_paraphrases(
    neutral_text: str,
    combinations: List[Tuple[str, str, str, str]],
    api_key: str,
    model: str = "claude-3-5-haiku-20241022",
    max_concurrent: int = 5,
) -> Dict:
    """Generate all paraphrases for a neutral text."""

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    async def generate_with_limit(combo):
        async with semaphore:
            valence, arousal, dominance, trust = combo
            print(f"  Generating: valence={valence}, arousal={arousal}, dominance={dominance}, trust={trust}")
            paraphrase = await generate_paraphrase(
                client, neutral_text, valence, arousal, dominance, trust, model
            )
            return combo, paraphrase

    # Generate all paraphrases concurrently
    tasks = [generate_with_limit(combo) for combo in combinations]
    results = await asyncio.gather(*tasks)

    # Format results
    paraphrases = {}
    for combo, paraphrase in results:
        if paraphrase:
            combo_key = f"[{combo[0]}, {combo[1]}, {combo[2]}, {combo[3]}]"
            paraphrases[combo_key] = paraphrase

    return {
        "neutral_text": neutral_text,
        "paraphrases": paraphrases,
        "n_combinations": len(combinations),
        "n_successful": len(paraphrases),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate axis-based paraphrases")
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "full"],
        help="Generation mode: 'test' for 10 combinations, 'full' for 81",
    )
    parser.add_argument(
        "--neutral-text",
        type=str,
        default=None,
        help="Neutral text to paraphrase (default: use built-in test text)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL file (default: probes/data/axis_paraphrases_{mode}.jsonl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-3-5-haiku-20241022",
        help="Claude model to use",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Max concurrent API calls",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of neutral texts to generate paraphrases for (full mode only)",
    )

    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Set output path
    if args.output is None:
        output_dir = Path("/workspace-vast/annas/git/research-tools/probes/data/axis_paraphrases")
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = output_dir / f"axis_paraphrases_{args.mode}.jsonl"

    # Get combinations
    combinations = get_axis_combinations(args.mode)

    print("=" * 80)
    print("AXIS PARAPHRASE GENERATION")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Combinations: {len(combinations)}")
    print(f"Model: {args.model}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Output: {args.output}")
    print()

    # Use test text for now (TODO: generate multiple neutral texts or take from dataset)
    neutral_text = args.neutral_text or TEST_NEUTRAL_TEXT

    print(f"Neutral text: {neutral_text}")
    print()
    print("=" * 80)
    print("GENERATING PARAPHRASES")
    print("=" * 80)
    print()

    # Generate paraphrases
    result = asyncio.run(
        generate_all_paraphrases(
            neutral_text,
            combinations,
            api_key,
            args.model,
            args.max_concurrent,
        )
    )

    print()
    print("=" * 80)
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total combinations: {result['n_combinations']}")
    print(f"Successful: {result['n_successful']}")
    print(f"Failed: {result['n_combinations'] - result['n_successful']}")
    print()

    # Save results
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to: {args.output}")
    print()

    # Print sample paraphrases
    print("=" * 80)
    print("SAMPLE PARAPHRASES")
    print("=" * 80)
    print()

    for i, (combo_key, paraphrase) in enumerate(list(result["paraphrases"].items())[:3]):
        print(f"Combination {i+1}: {combo_key}")
        print(f"Paraphrase: {paraphrase}")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
