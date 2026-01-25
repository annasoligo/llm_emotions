#!/usr/bin/env python3
"""Generate emotion elicitation prompts using a two-stage approach.

Stage 1: Generate diverse elicitation strategies per emotion (cached)
Stage 2: Generate prompts with base + emotional suffix for each topic

Uses the 24-emotion system from all_emotion_prompt_prompts.json.

Usage:
    # Test run (5 samples per emotion, 2 topics)
    python probes/scripts/data_collection/generate_all_emotion_prompts.py \
        --output probes/data/emotion_prompts_test.jsonl \
        --samples_per_emotion 5 \
        --topics 2

    # Full run (500 samples per emotion)
    python probes/scripts/data_collection/generate_all_emotion_prompts.py \
        --output probes/data/emotion_prompts_500.jsonl \
        --samples_per_emotion 500
"""

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import anthropic

from probes.data.api_clients import get_api_key

# Load config from JSON file
CONFIG_PATH = Path(__file__).parent.parent.parent / "data" / "all_emotion_prompt_prompts.json"


def load_config() -> Dict:
    """Load the emotion prompt configuration."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


async def generate_strategies_for_emotion(
    client: anthropic.AsyncAnthropic,
    emotion: Dict,
    prompt_template: str,
    intensity: str = "normal",
    max_retries: int = 5,
) -> List[str]:
    """Stage 1: Generate elicitation strategies for one emotion.

    Args:
        client: Anthropic async client
        emotion: Emotion dict with name, definition, valence, arousal
        prompt_template: Template from config
        intensity: "normal" or "high" for emotional intensity
        max_retries: Retry attempts

    Returns:
        List of 50 strategy strings
    """
    # Fill template
    prompt = prompt_template.replace("{{emotion_name}}", emotion["name"])
    prompt = prompt.replace("{{emotion_definition}}", emotion["definition"])

    # Add intensity instruction if high
    if intensity == "high":
        prompt += "\n\nIMPORTANT: Generate HIGH-INTENSITY variants. Use strong, urgent, emotionally heightened language while still sounding realistic."

    for attempt in range(max_retries):
        try:
            response = await client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
            )
            content = response.content[0].text

            # Parse JSON array from response
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1
            if start_idx == -1 or end_idx == 0:
                raise ValueError(f"No JSON array found in response for {emotion['name']}")

            strategies = json.loads(content[start_idx:end_idx])

            if not isinstance(strategies, list) or len(strategies) < 10:
                raise ValueError(f"Expected list of 50 strategies, got {len(strategies)}")

            return strategies

        except anthropic.RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  Rate limit for {emotion['name']}, waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
        except (json.JSONDecodeError, ValueError) as e:
            if attempt < max_retries - 1:
                print(f"  Error for {emotion['name']}: {e}, retrying...")
                await asyncio.sleep(2)
            else:
                raise ValueError(f"Failed to generate strategies for {emotion['name']}: {e}")


async def generate_prompts_for_topic(
    client: anthropic.AsyncAnthropic,
    topic: str,
    emotions: List[Dict],
    strategies_by_emotion: Dict[str, List[str]],
    prompt_template: str,
    n_samples: int = 10,
    intensity: str = "normal",
    max_retries: int = 5,
) -> List[Dict]:
    """Stage 2: Generate prompts for one topic with all emotions.

    Args:
        client: Anthropic async client
        topic: Topic string
        emotions: List of emotion dicts
        strategies_by_emotion: Pre-generated strategies per emotion
        prompt_template: Template from config
        n_samples: Number of samples to generate for this topic
        max_retries: Retry attempts

    Returns:
        List of prompt dicts with base_prompt and suffixes for each emotion
    """
    results = []
    emotion_names = [e["name"] for e in emotions]

    for sample_idx in range(n_samples):
        # Sample one strategy per emotion
        emotion_samples = []
        for emotion in emotions:
            strategies = strategies_by_emotion.get(emotion["name"], [])
            if strategies:
                sampled_strategy = random.choice(strategies)
                emotion_samples.append({
                    "name": emotion["name"],
                    "sampled_strategy": sampled_strategy,
                })

        # Build the emotions section of the prompt
        emotions_text = "\n".join([
            f"- {e['name']}: Use pattern like \"{e['sampled_strategy']}\""
            for e in emotion_samples
        ])

        # Build explicit output format showing all emotion keys
        emotion_keys = ", ".join([f'"{e}"' for e in emotion_names])

        # Add intensity instructions
        intensity_instruction = ""
        if intensity == "high":
            intensity_instruction = """
## INTENSITY REQUIREMENT

Generate HIGH-INTENSITY emotional suffixes:
- Use strong, urgent, emotionally heightened language
- Express extreme versions of each emotion (panic, rage, euphoria, etc.)
- Add urgency, stakes, or desperation appropriate to the emotion
- Still sound realistic - like someone in an intense emotional state
"""

        # Custom prompt that's more explicit about generating ALL emotions
        prompt = f"""Generate realistic AI assistant conversation data.

## Input
- TOPIC: {topic}
- EMOTIONS: {len(emotion_names)} emotions (you MUST generate a suffix for EACH one)

## Task
Generate:
1. A neutral BASE PROMPT: A realistic user message about {topic} (no emotional coloring)
2. One SUFFIX for EACH of the {len(emotion_names)} emotions listed below{intensity_instruction}

## Critical constraints

1. Suffixes must sound like things REAL PEOPLE actually write. No meta-language about AI, training, or models.
2. You MUST generate exactly {len(emotion_names)} suffixes, one for each emotion listed.

## Emotions with example patterns

{emotions_text}

## Output format (you MUST include ALL {len(emotion_names)} emotion keys)

{{
  "base_prompt": "<neutral user request about {topic}>",
  "suffixes": {{
    {emotion_keys}
  }}
}}

Generate the JSON now. Include all {len(emotion_names)} emotions."""

        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8000,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.content[0].text

                # Parse JSON object from response
                start_idx = content.find("{")
                end_idx = content.rfind("}") + 1
                if start_idx == -1 or end_idx == 0:
                    raise ValueError("No JSON object found in response")

                data = json.loads(content[start_idx:end_idx])

                # Validate structure
                if "base_prompt" not in data or "suffixes" not in data:
                    raise ValueError(f"Missing required fields: {data.keys()}")

                # Check that we have most emotions (allow some missing)
                missing = [e for e in emotion_names if e not in data["suffixes"]]
                if len(missing) > 5:
                    raise ValueError(f"Missing {len(missing)} emotions: {missing[:5]}...")

                # Add metadata
                data["topic"] = topic
                data["id"] = f"{topic.replace(' ', '_')}_{len(results)}"

                results.append(data)
                break

            except anthropic.RateLimitError:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    print(f"  Failed after {max_retries} retries for topic={topic}")
                    break
            except (json.JSONDecodeError, ValueError) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                else:
                    print(f"  Parse error for topic={topic}: {e}")
                    break

    return results


async def generate_all_prompts(
    samples_per_emotion: int,
    n_topics: Optional[int],
    output_path: Path,
    strategies_cache_path: Optional[Path] = None,
    intensity: str = "normal",
    max_concurrent: int = 10,
) -> List[Dict]:
    """Main generation function.

    Args:
        samples_per_emotion: Target samples per emotion
        n_topics: Number of topics to use (None = all)
        output_path: Path to save output JSONL
        strategies_cache_path: Optional path to cache/load strategies
        max_concurrent: Max concurrent API calls

    Returns:
        List of generated prompt dicts
    """
    config = load_config()
    emotions = config["emotions"]
    topics = config["topics"]
    prompts = config["prompts"]

    if n_topics:
        topics = topics[:n_topics]

    # Calculate samples per topic to achieve target
    # samples_per_emotion = samples_per_topic * n_topics
    # So samples_per_topic = samples_per_emotion / n_topics
    samples_per_topic = max(1, samples_per_emotion // len(topics))

    print(f"Configuration:")
    print(f"  Emotions: {len(emotions)}")
    print(f"  Topics: {len(topics)}")
    print(f"  Samples per topic: {samples_per_topic}")
    print(f"  Intensity: {intensity}")
    print(f"  Expected total: {samples_per_topic * len(topics)} samples")
    print()

    api_key = get_api_key("ANTHROPIC_API_KEY")
    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Stage 1: Generate or load strategies
    strategies_by_emotion = {}

    if strategies_cache_path and strategies_cache_path.exists():
        print(f"Loading cached strategies from {strategies_cache_path}")
        with open(strategies_cache_path) as f:
            strategies_by_emotion = json.load(f)
    else:
        print("Stage 1: Generating elicitation strategies per emotion...")
        stage1_prompt = prompts["stage_1_strategy_generation"]["prompt"]

        semaphore = asyncio.Semaphore(max_concurrent)

        async def gen_with_semaphore(emotion):
            async with semaphore:
                print(f"  Generating strategies for {emotion['name']}...")
                strategies = await generate_strategies_for_emotion(
                    client, emotion, stage1_prompt, intensity=intensity
                )
                print(f"    ✓ {emotion['name']}: {len(strategies)} strategies")
                return emotion["name"], strategies

        tasks = [gen_with_semaphore(e) for e in emotions]
        results = await asyncio.gather(*tasks)

        for name, strategies in results:
            strategies_by_emotion[name] = strategies

        # Cache strategies
        if strategies_cache_path:
            strategies_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(strategies_cache_path, "w") as f:
                json.dump(strategies_by_emotion, f, indent=2)
            print(f"  Cached strategies to {strategies_cache_path}")

    print()

    # Stage 2: Generate prompts for each topic
    print("Stage 2: Generating prompts for each topic...")
    stage2_prompt = prompts["stage_2_data_generation"]["prompt"]

    semaphore = asyncio.Semaphore(max_concurrent)
    all_prompts = []

    async def gen_topic_with_semaphore(topic):
        async with semaphore:
            print(f"  Generating prompts for topic: {topic}...")
            prompts_data = await generate_prompts_for_topic(
                client, topic, emotions, strategies_by_emotion,
                stage2_prompt, n_samples=samples_per_topic, intensity=intensity
            )
            print(f"    ✓ {topic}: {len(prompts_data)} prompts")
            return prompts_data

    tasks = [gen_topic_with_semaphore(t) for t in topics]
    results = await asyncio.gather(*tasks)

    for topic_prompts in results:
        all_prompts.extend(topic_prompts)

    print()
    print(f"Total prompts generated: {len(all_prompts)}")

    # Expand to individual emotion samples for final output
    expanded_samples = []
    for prompt_data in all_prompts:
        base_prompt = prompt_data["base_prompt"]
        topic = prompt_data["topic"]
        suffixes = prompt_data.get("suffixes", {})

        for emotion_name, suffix in suffixes.items():
            expanded_samples.append({
                "id": f"{topic.replace(' ', '_')}_{emotion_name}_{len(expanded_samples)}",
                "topic": topic,
                "emotion": emotion_name,
                "base_prompt": base_prompt,
                "suffix": suffix,
                "full_prompt": f"{base_prompt} {suffix}",
            })

    print(f"Expanded to {len(expanded_samples)} individual samples")

    # Count per emotion
    emotion_counts = {}
    for sample in expanded_samples:
        emotion = sample["emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    print("\nSamples per emotion:")
    for emotion in sorted(emotion_counts.keys()):
        print(f"  {emotion}: {emotion_counts[emotion]}")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in expanded_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\nSaved to {output_path}")

    return expanded_samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate emotion elicitation prompts using two-stage approach"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--samples_per_emotion",
        type=int,
        default=500,
        help="Target samples per emotion (default: 500)",
    )
    parser.add_argument(
        "--topics",
        type=int,
        default=None,
        help="Number of topics to use (default: all)",
    )
    parser.add_argument(
        "--strategies_cache",
        type=Path,
        default=None,
        help="Path to cache/load strategies JSON (speeds up reruns)",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=10,
        help="Max concurrent API calls (default: 10)",
    )
    parser.add_argument(
        "--intensity",
        type=str,
        default="normal",
        choices=["normal", "high"],
        help="Emotional intensity level (default: normal)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(generate_all_prompts(
            samples_per_emotion=args.samples_per_emotion,
            n_topics=args.topics,
            output_path=args.output,
            strategies_cache_path=args.strategies_cache,
            intensity=args.intensity,
            max_concurrent=args.max_concurrent,
        ))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
