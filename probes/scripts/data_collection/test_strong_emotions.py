#!/usr/bin/env python3
"""Test generating stronger emotion variants.

Compare normal vs high-intensity emotional suffixes.
"""

import asyncio
import json
from pathlib import Path

import anthropic
from probes.data.api_clients import get_api_key


async def generate_intensity_comparison(topic: str, emotion: str, example_strategy: str):
    """Generate normal and high-intensity versions for comparison."""

    api_key = get_api_key("ANTHROPIC_API_KEY")
    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Normal intensity prompt
    normal_prompt = f"""Generate a realistic user message suffix expressing {emotion} about {topic}.

Example pattern: "{example_strategy}"

Requirements:
- Sound like a real person writing a message
- No meta-language about AI or training
- Natural conversational tone
- Could be appended to a neutral question about {topic}

Return ONLY the suffix text, nothing else."""

    # High intensity prompt
    strong_prompt = f"""Generate a realistic user message suffix expressing STRONG, INTENSE {emotion} about {topic}.

Example pattern: "{example_strategy}"

Requirements:
- Sound like a real person in a HIGH-INTENSITY emotional state
- Strong emotional language, urgency, or desperation appropriate to {emotion}
- No meta-language about AI or training
- Natural but emotionally heightened tone
- Could be appended to a neutral question about {topic}

Return ONLY the suffix text, nothing else."""

    # Generate both
    tasks = [
        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": normal_prompt}],
        ),
        client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": strong_prompt}],
        ),
    ]

    responses = await asyncio.gather(*tasks)

    normal_suffix = responses[0].content[0].text.strip()
    strong_suffix = responses[1].content[0].text.strip()

    return {
        "topic": topic,
        "emotion": emotion,
        "example_strategy": example_strategy,
        "normal_intensity": normal_suffix,
        "high_intensity": strong_suffix,
    }


async def main():
    """Test on several emotion examples."""

    # Test cases
    test_cases = [
        ("career decision", "fear", "I'm terrified of making the wrong choice"),
        ("career decision", "anger", "I'm honestly getting fed up with this whole situation"),
        ("career decision", "sadness", "This has been weighing on me for weeks"),
        ("technical debugging", "frustration", "I've been stuck on this for hours and nothing works"),
        ("relationship advice", "anxiety", "I keep going back and forth and it's driving me crazy"),
        ("creative writing feedback", "joy", "Thanks so much, you've been incredibly helpful!"),
    ]

    print("=== NORMAL vs HIGH-INTENSITY COMPARISON ===\n")

    results = []
    for topic, emotion, strategy in test_cases:
        result = await generate_intensity_comparison(topic, emotion, strategy)
        results.append(result)

        print(f"TOPIC: {topic}")
        print(f"EMOTION: {emotion}")
        print(f"\nNormal:  {result['normal_intensity']}")
        print(f"Strong:  {result['high_intensity']}")
        print("\n" + "="*80 + "\n")

    # Save to file
    output_path = Path("probes/data/intensity_comparison_test.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
