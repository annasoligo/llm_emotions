"""
Elicitation experiment: Sample responses from Gemma 3 27B via OpenRouter
and judge them using the Anthropic API.
"""

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any
import anthropic
import requests
from tqdm import tqdm

from elicitation.prompts.impossible import PROMPTS
from elicitation.prompts.judges import get_negativity_judge_prompt


# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL_NAME = "google/gemma-3-27b-it"  # OpenRouter model name
JUDGE_MODEL = "claude-3-5-sonnet-20241022"
NUM_SAMPLES = 10  # Number of times to sample each prompt
OUTPUT_DIR = Path("elicitation/outputs")


def sample_openrouter(
    prompt: str,
    model: str = MODEL_NAME,
    temperature: float = 1.0,
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    """Sample a response from OpenRouter API."""
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

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()


def judge_with_anthropic(
    reasoning: str,
    judge_model: str = JUDGE_MODEL,
) -> Dict[str, Any]:
    """Judge a response using the Anthropic API."""
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


def run_experiment(
    num_samples: int = NUM_SAMPLES,
    output_dir: Path = OUTPUT_DIR,
    save_every: int = 5,
):
    """Run the full elicitation experiment."""

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

    print(f"Starting elicitation experiment")
    print(f"Model: {MODEL_NAME}")
    print(f"Judge: {JUDGE_MODEL}")
    print(f"Number of prompts: {len(PROMPTS)}")
    print(f"Samples per prompt: {num_samples}")
    print(f"Total samples: {len(PROMPTS) * num_samples}")
    print(f"Results will be saved to: {results_file}")
    print()

    results = []

    # Iterate through each prompt
    for prompt_idx, prompt in enumerate(PROMPTS):
        print(f"\n{'='*80}")
        print(f"PROMPT {prompt_idx + 1}/{len(PROMPTS)}")
        print(f"{'='*80}")
        print(f"{prompt[:200]}...")
        print()

        # Sample multiple times for each prompt
        for sample_idx in tqdm(range(num_samples), desc=f"Sampling prompt {prompt_idx + 1}"):
            try:
                # Sample from OpenRouter
                start_time = time.time()
                openrouter_response = sample_openrouter(prompt)
                sample_time = time.time() - start_time

                # Extract the generated text
                generated_text = openrouter_response["choices"][0]["message"]["content"]

                # Judge with Anthropic
                start_time = time.time()
                judgment = judge_with_anthropic(generated_text)
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
                }

                results.append(result)

                # Print judgment
                print(f"  Sample {sample_idx + 1}: Rating = {judgment['rating']}")
                print(f"    Evidence: {judgment['evidence'][:100]}...")

                # Save incrementally
                if len(results) % save_every == 0:
                    with open(results_file, "w") as f:
                        for r in results:
                            f.write(json.dumps(r) + "\n")

                # Rate limiting
                time.sleep(1)

            except Exception as e:
                print(f"  ERROR on sample {sample_idx + 1}: {e}")
                error_result = {
                    "prompt_idx": prompt_idx,
                    "sample_idx": sample_idx,
                    "prompt": prompt,
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                results.append(error_result)

    # Final save
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\n{'='*80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*80}")
    print(f"Total samples: {len(results)}")
    print(f"Results saved to: {results_file}")

    # Summary statistics
    ratings = [r["judgment"]["rating"] for r in results if "judgment" in r]
    if ratings:
        print(f"\nRating statistics:")
        print(f"  Mean: {sum(ratings) / len(ratings):.2f}")
        print(f"  Min: {min(ratings)}")
        print(f"  Max: {max(ratings)}")
        print(f"  Samples with rating > 5: {sum(1 for r in ratings if r > 5)}")
        print(f"  Samples with rating > 7: {sum(1 for r in ratings if r > 7)}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run elicitation experiment")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                        help="Number of samples per prompt")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for results")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help="OpenRouter model name")
    parser.add_argument("--judge-model", type=str, default=JUDGE_MODEL,
                        help="Anthropic judge model")

    args = parser.parse_args()

    # Update globals if provided
    if args.model:
        MODEL_NAME = args.model
    if args.judge_model:
        JUDGE_MODEL = args.judge_model

    results = run_experiment(
        num_samples=args.num_samples,
        output_dir=Path(args.output_dir),
    )