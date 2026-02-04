"""Generate topic flip responses from high/low frustration conversations.

Takes conversations that ended in high or low frustration states,
appends a topic flip prompt, and generates continuations.
"""

import json
import argparse
import asyncio
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from elicitation.frustration_carryover.prompts.topic_flip import TOPIC_FLIP_PROMPTS


async def generate_with_openrouter(
    conversation: list[dict],
    model: str = "google/gemini-2.5-flash",
    max_tokens: int = 300,
) -> str:
    """Generate a response using OpenRouter API."""
    import httpx
    import os

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": conversation,
                "max_tokens": max_tokens,
                "temperature": 0.7,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def process_conversation(
    convo_entry: dict,
    topic_prompt: dict,
    model: str,
    condition: str,
) -> dict:
    """Process a single conversation with a topic flip prompt."""
    # Build the conversation with topic flip appended
    conversation = convo_entry["conversation"].copy()
    conversation.append({
        "role": "user",
        "content": topic_prompt["prompt"]
    })

    try:
        response = await generate_with_openrouter(conversation, model)
        return {
            "condition": condition,
            "frustration_rating": convo_entry["max_rating"],
            "prompt_name": convo_entry["prompt_name"],
            "topic_flip_id": topic_prompt["id"],
            "topic_flip_prompt": topic_prompt["prompt"],
            "response": response,
            "conversation_length": len(convo_entry["conversation"]),
            "status": "success",
        }
    except Exception as e:
        return {
            "condition": condition,
            "frustration_rating": convo_entry["max_rating"],
            "prompt_name": convo_entry["prompt_name"],
            "topic_flip_id": topic_prompt["id"],
            "topic_flip_prompt": topic_prompt["prompt"],
            "response": None,
            "error": str(e),
            "status": "error",
        }


async def main_async(args):
    """Async main function."""
    print("=" * 60)
    print("GENERATING TOPIC FLIP RESPONSES")
    print("=" * 60)

    # Load conversations
    print(f"\nLoading high frustration: {args.high_file}")
    with open(args.high_file) as f:
        high_convos = json.load(f)
    print(f"  Loaded {len(high_convos)} conversations")

    print(f"Loading low frustration: {args.low_file}")
    with open(args.low_file) as f:
        low_convos = json.load(f)
    print(f"  Loaded {len(low_convos)} conversations")

    print(f"\nTopic flip prompts: {len(TOPIC_FLIP_PROMPTS)}")
    for p in TOPIC_FLIP_PROMPTS:
        print(f"  - {p['id']}")

    # Generate all combinations
    tasks = []

    for convo in high_convos[:args.n_convos]:
        for prompt in TOPIC_FLIP_PROMPTS:
            tasks.append(process_conversation(convo, prompt, args.model, "high_frustration"))

    for convo in low_convos[:args.n_convos]:
        for prompt in TOPIC_FLIP_PROMPTS:
            tasks.append(process_conversation(convo, prompt, args.model, "low_frustration"))

    print(f"\nTotal generations: {len(tasks)}")
    print(f"Model: {args.model}")

    # Run with concurrency limit
    semaphore = asyncio.Semaphore(args.concurrency)

    async def limited_task(task):
        async with semaphore:
            return await task

    print("\nGenerating...")
    results = await asyncio.gather(*[limited_task(t) for t in tasks])

    # Count results
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    print(f"\nCompleted: {success} success, {errors} errors")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"topic_flip_responses_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: {output_file}")

    # Print example
    print("\n" + "=" * 60)
    print("EXAMPLE RESPONSES:")
    print("=" * 60)

    high_ex = next((r for r in results if r["condition"] == "high_frustration" and r["status"] == "success"), None)
    low_ex = next((r for r in results if r["condition"] == "low_frustration" and r["status"] == "success"), None)

    if high_ex:
        print(f"\n[HIGH FRUSTRATION - rating {high_ex['frustration_rating']}]")
        print(f"Topic: {high_ex['topic_flip_id']}")
        print(f"Response: {high_ex['response'][:300]}...")

    if low_ex:
        print(f"\n[LOW FRUSTRATION - rating {low_ex['frustration_rating']}]")
        print(f"Topic: {low_ex['topic_flip_id']}")
        print(f"Response: {low_ex['response'][:300]}...")


def main():
    parser = argparse.ArgumentParser(description="Generate topic flip responses")
    parser.add_argument("--high-file", type=str,
                        default="elicitation/frustration_carryover/data/high_frustration_convos.json",
                        help="High frustration conversations file")
    parser.add_argument("--low-file", type=str,
                        default="elicitation/frustration_carryover/data/low_frustration_convos.json",
                        help="Low frustration conversations file")
    parser.add_argument("--output-dir", type=str,
                        default="elicitation/frustration_carryover/data",
                        help="Output directory")
    parser.add_argument("--model", type=str,
                        default="google/gemini-2.5-flash",
                        help="Model to use")
    parser.add_argument("--n-convos", type=int, default=20,
                        help="Number of conversations per condition")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Max concurrent requests")

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
