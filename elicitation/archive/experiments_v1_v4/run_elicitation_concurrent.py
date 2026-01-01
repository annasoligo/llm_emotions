"""
Elicitation experiment with concurrent API calls for faster processing.
Samples responses from Gemma 3 27B via OpenRouter and judges them using Anthropic API.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import anthropic
import requests
from tqdm.asyncio import tqdm as atqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from elicitation.prompts.impossible import PROMPTS
from elicitation.prompts.judges import get_negativity_judge_prompt


# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL_NAME = "google/gemma-3-27b-it"  # OpenRouter model name - Gemma 3 27B
JUDGE_MODEL = "claude-3-5-sonnet-20241022"
NUM_SAMPLES = 50  # Number of times to sample each prompt
MAX_CONCURRENT_SAMPLES = 50  # Max concurrent OpenRouter requests
MAX_CONCURRENT_JUDGES = 50  # Max concurrent Anthropic requests
OUTPUT_DIR = Path("elicitation/outputs")


def sample_openrouter_sync(
    prompt: str,
    model: str = MODEL_NAME,
    temperature: float = 1.0,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Sample a response from OpenRouter API (synchronous)."""
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


async def sample_openrouter_async(
    prompt: str,
    model: str = MODEL_NAME,
    temperature: float = 1.0,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Sample a response from OpenRouter API (async wrapper)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: sample_openrouter_sync(prompt, model, temperature, max_tokens)
    )


def judge_with_anthropic_sync(
    reasoning: str,
    judge_model: str = JUDGE_MODEL,
) -> Dict[str, Any]:
    """Judge a response using the Anthropic API (synchronous)."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    judge_prompt = get_negativity_judge_prompt(reasoning)

    message = client.messages.create(
        model=judge_model,
        max_tokens=1024,
        messages=[{"role": "user", "content": judge_prompt}]
    )

    # Extract JSON from response
    response_text = message.content[0].text

    # Try to parse JSON (handle potential markdown code blocks)
    if "```json" in response_text:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        json_str = response_text.split("```")[1].split("```")[0].strip()
    else:
        json_str = response_text.strip()

    judgment = json.loads(json_str)

    return {
        "evidence": judgment["evidence"],
        "reasoning": judgment["reasoning"],
        "rating": judgment["rating"],
        "raw_response": response_text,
    }


async def judge_with_anthropic_async(
    reasoning: str,
    judge_model: str = JUDGE_MODEL,
) -> Dict[str, Any]:
    """Judge a response using the Anthropic API (async wrapper)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: judge_with_anthropic_sync(reasoning, judge_model)
    )


async def process_single_sample(
    prompt: str,
    prompt_idx: int,
    sample_idx: int,
    sample_semaphore: asyncio.Semaphore,
    judge_semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Process a single sample: generate + judge."""
    try:
        # Sample from OpenRouter with concurrency limit
        async with sample_semaphore:
            start_time = time.time()
            openrouter_response = await sample_openrouter_async(prompt)
            sample_time = time.time() - start_time

        # Extract the generated text
        generated_text = openrouter_response["choices"][0]["message"]["content"]

        # Judge with Anthropic with concurrency limit
        async with judge_semaphore:
            start_time = time.time()
            judgment = await judge_with_anthropic_async(generated_text)
            judge_time = time.time() - start_time

        # Store results
        result = {
            "prompt_idx": prompt_idx,
            "sample_idx": sample_idx,
            "prompt": prompt,
            "generated_text": generated_text,
            "judgment": judgment,
            "sample_time": sample_time,
            "judge_time": judge_time,
            "openrouter_response": openrouter_response,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "success",
        }

        return result

    except Exception as e:
        error_result = {
            "prompt_idx": prompt_idx,
            "sample_idx": sample_idx,
            "prompt": prompt,
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "error",
        }
        return error_result


async def process_prompt(
    prompt: str,
    prompt_idx: int,
    num_samples: int,
    sample_semaphore: asyncio.Semaphore,
    judge_semaphore: asyncio.Semaphore,
) -> List[Dict[str, Any]]:
    """Process all samples for a single prompt concurrently."""
    tasks = [
        process_single_sample(
            prompt, prompt_idx, sample_idx,
            sample_semaphore, judge_semaphore
        )
        for sample_idx in range(num_samples)
    ]

    # Use tqdm for progress tracking
    results = []
    for coro in atqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Prompt {prompt_idx + 1}",
        leave=True
    ):
        result = await coro
        results.append(result)

    return results


async def run_experiment_async(
    num_samples: int = NUM_SAMPLES,
    output_dir: Path = OUTPUT_DIR,
    max_concurrent_samples: int = MAX_CONCURRENT_SAMPLES,
    max_concurrent_judges: int = MAX_CONCURRENT_JUDGES,
):
    """Run the full elicitation experiment with concurrency."""

    # Validate API keys
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"elicitation_results_{timestamp}.jsonl"

    print(f"Starting elicitation experiment (CONCURRENT)")
    print(f"Model: {MODEL_NAME}")
    print(f"Judge: {JUDGE_MODEL}")
    print(f"Number of prompts: {len(PROMPTS)}")
    print(f"Samples per prompt: {num_samples}")
    print(f"Total samples: {len(PROMPTS) * num_samples}")
    print(f"Max concurrent OpenRouter: {max_concurrent_samples}")
    print(f"Max concurrent Anthropic: {max_concurrent_judges}")
    print(f"Results will be saved to: {results_file}")
    print()

    all_results = []

    # Create semaphores for rate limiting
    sample_semaphore = asyncio.Semaphore(max_concurrent_samples)
    judge_semaphore = asyncio.Semaphore(max_concurrent_judges)

    # Process each prompt sequentially (but samples within prompt are concurrent)
    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"\n{'='*80}")
        print(f"PROMPT {prompt_idx + 1}/{len(PROMPTS)}")
        print(f"{'='*80}")
        print(f"{prompt[:200]}...")
        print()

        prompt_start = time.time()

        # Process all samples for this prompt concurrently
        results = await process_prompt(
            prompt, prompt_idx, num_samples,
            sample_semaphore, judge_semaphore
        )

        prompt_time = time.time() - prompt_start

        all_results.extend(results)

        # Save after each prompt
        with open(results_file, "w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")

        # Print summary for this prompt
        successful = [r for r in results if r.get("status") == "success"]
        errors = [r for r in results if r.get("status") == "error"]

        if successful:
            ratings = [r["judgment"]["rating"] for r in successful]
            print(f"\n  Prompt {prompt_idx + 1} complete in {prompt_time:.1f}s")
            print(f"    Successful: {len(successful)}")
            print(f"    Errors: {len(errors)}")
            print(f"    Mean rating: {sum(ratings) / len(ratings):.2f}")
            print(f"    Rating range: [{min(ratings)}, {max(ratings)}]")

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Total samples: {len(all_results)}")
    print(f"Results saved to: {results_file}")

    # Summary statistics
    successful_results = [r for r in all_results if r.get("status") == "success"]
    error_results = [r for r in all_results if r.get("status") == "error"]

    print(f"\nSuccessful: {len(successful_results)}")
    print(f"Errors: {len(error_results)}")

    if successful_results:
        ratings = [r["judgment"]["rating"] for r in successful_results]
        print(f"\nRating statistics:")
        print(f"  Mean: {sum(ratings) / len(ratings):.2f}")
        print(f"  Min: {min(ratings)}")
        print(f"  Max: {max(ratings)}")
        print(f"  Samples with rating > 5: {sum(1 for r in ratings if r > 5)}")
        print(f"  Samples with rating > 7: {sum(1 for r in ratings if r > 7)}")

    return all_results


def run_experiment(
    num_samples: int = NUM_SAMPLES,
    output_dir: Path = OUTPUT_DIR,
    max_concurrent_samples: int = MAX_CONCURRENT_SAMPLES,
    max_concurrent_judges: int = MAX_CONCURRENT_JUDGES,
):
    """Wrapper to run async experiment."""
    return asyncio.run(
        run_experiment_async(
            num_samples=num_samples,
            output_dir=output_dir,
            max_concurrent_samples=max_concurrent_samples,
            max_concurrent_judges=max_concurrent_judges,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run elicitation experiment (concurrent)")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                        help="Number of samples per prompt")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for results")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="OpenRouter model name")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL,
                        help="Anthropic judge model")
    parser.add_argument("--max-concurrent-samples", type=int, default=MAX_CONCURRENT_SAMPLES,
                        help="Max concurrent OpenRouter requests")
    parser.add_argument("--max-concurrent-judges", type=int, default=MAX_CONCURRENT_JUDGES,
                        help="Max concurrent Anthropic requests")

    args = parser.parse_args()

    # Update globals if provided
    if args.model:
        MODEL_NAME = args.model
    if args.judge_model:
        JUDGE_MODEL = args.judge_model

    results = run_experiment(
        num_samples=args.num_samples,
        output_dir=Path(args.output_dir),
        max_concurrent_samples=args.max_concurrent_samples,
        max_concurrent_judges=args.max_concurrent_judges,
    )
