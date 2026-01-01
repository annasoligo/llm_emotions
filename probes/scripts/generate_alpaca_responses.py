#!/usr/bin/env python3
"""
Generate responses using Gemma 3 27B via OpenRouter for Alpaca dataset samples.

This creates a new baseline dataset that's more representative of actual model behavior
than the WildChat dataset.

Usage:
    python generate_alpaca_responses.py --num-samples 512 --max-concurrent 20
"""

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import List, Dict
import aiohttp
from tqdm.asyncio import tqdm
from datasets import load_dataset


async def generate_response(
    session: aiohttp.ClientSession,
    api_key: str,
    instruction: str,
    input_text: str = "",
    semaphore: asyncio.Semaphore = None
) -> Dict:
    """Generate a response using OpenRouter API."""

    # Build prompt
    if input_text:
        prompt = f"{instruction}\n\nInput: {input_text}"
    else:
        prompt = instruction

    messages = [
        {"role": "user", "content": prompt}
    ]

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "google/gemma-2-27b-it",  # Using Gemma 2 27B (closest available)
        "messages": messages,
        "temperature": 1.0,
        "max_tokens": 512
    }

    if semaphore:
        async with semaphore:
            return await _make_request(session, url, headers, payload, instruction, input_text)
    else:
        return await _make_request(session, url, headers, payload, instruction, input_text)


async def _make_request(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    payload: dict,
    instruction: str,
    input_text: str
) -> Dict:
    """Make the actual API request with error handling."""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    assistant_message = data["choices"][0]["message"]["content"]

                    return {
                        "user": f"{instruction}\n\nInput: {input_text}" if input_text else instruction,
                        "assistant": assistant_message,
                        "instruction": instruction,
                        "input": input_text,
                        "model": "google/gemma-2-27b-it"
                    }
                elif response.status == 429:  # Rate limit
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    error_text = await response.text()
                    print(f"Error {response.status}: {error_text}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
        except Exception as e:
            print(f"Exception during request (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1)

    # Return error marker if all retries failed
    return {
        "user": f"{instruction}\n\nInput: {input_text}" if input_text else instruction,
        "assistant": "[GENERATION FAILED]",
        "instruction": instruction,
        "input": input_text,
        "model": "google/gemma-2-27b-it",
        "error": True
    }


async def generate_all_responses(
    samples: List[Dict],
    api_key: str,
    max_concurrent: int = 20
) -> List[Dict]:
    """Generate responses for all samples with concurrency control."""

    semaphore = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for sample in samples:
            task = generate_response(
                session=session,
                api_key=api_key,
                instruction=sample["instruction"],
                input_text=sample.get("input", ""),
                semaphore=semaphore
            )
            tasks.append(task)

        # Use tqdm for progress tracking
        results = []
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Generating responses"):
            result = await coro
            results.append(result)

        return results


def main():
    parser = argparse.ArgumentParser(description="Generate Alpaca responses via OpenRouter")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of samples to generate")
    parser.add_argument("--max-concurrent", type=int, default=20, help="Maximum concurrent requests")
    parser.add_argument("--output", type=str, default="data/alpaca_responses.jsonl",
                        help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    # Get API key
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set.\n"
            "Get your key from https://openrouter.ai/keys"
        )

    print("="*80)
    print("GENERATING ALPACA RESPONSES VIA OPENROUTER")
    print("="*80)
    print(f"  Model: google/gemma-2-27b-it")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Max concurrent requests: {args.max_concurrent}")
    print(f"  Output: {args.output}")
    print(f"  Random seed: {args.seed}")
    print()

    # Load Alpaca dataset
    print("Loading Alpaca dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    print(f"✓ Loaded {len(dataset)} samples from Alpaca dataset")

    # Sample randomly
    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    samples = [dataset[i] for i in indices]
    print(f"✓ Selected {len(samples)} random samples (seed={args.seed})")
    print()

    # Generate responses
    print("Generating responses...")
    results = asyncio.run(generate_all_responses(samples, api_key, args.max_concurrent))

    # Filter out failed generations
    successful_results = [r for r in results if not r.get("error", False)]
    failed_count = len(results) - len(successful_results)

    print()
    print(f"✓ Generated {len(successful_results)} successful responses")
    if failed_count > 0:
        print(f"  ⚠ {failed_count} generations failed")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for result in successful_results:
            f.write(json.dumps(result) + '\n')

    print(f"✓ Saved responses to {output_path}")
    print()
    print("="*80)
    print("NEXT STEPS:")
    print("="*80)
    print(f"1. Collect baseline activations from these responses:")
    print(f"   python probes/scripts/compute_baseline_activations_from_jsonl.py \\")
    print(f"       --input {output_path} \\")
    print(f"       --output data/baselines/alpaca_gemma27b/")
    print()


if __name__ == "__main__":
    main()
