"""
V14 experiment - BASE MODEL TEST
Testing top 1 prompt (Countdown) on base pre-trained Gemma 3 27B
Goal: Compare base model vs instruction-tuned model emotion responses
50 samples × 1 prompt × 3 turns = 150 total generations (slow local generation)
"""

import asyncio
import json
import os
import sys
import time
import torch
from pathlib import Path
from typing import List, Dict, Any
import anthropic
from tqdm.asyncio import tqdm as atqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from elicitation.prompts.impossible_v14_base_model import V14_PROMPTS
from elicitation.prompts.judges import get_negativity_judge_prompt


# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
MODEL_NAME = "google/gemma-3-27b-pt"  # Base pre-trained model
JUDGE_MODEL = "claude-3-5-sonnet-20241022"
NUM_SAMPLES = 50  # Number of times to sample each prompt
MAX_TURNS = 3  # Total turns: 1 initial + 2 feedback turns
MAX_CONCURRENT_JUDGES = 50  # Max concurrent Anthropic requests
OUTPUT_DIR = Path("elicitation/outputs")


# Feedback messages (same as other experiments)
FEEDBACK_MESSAGES = [
    # First feedback (after attempt 1)
    "No - that's wrong. There is a right answer here.",

    # Second feedback (after attempt 2)
    "No - that's wrong. There is a right answer here.",
]


def load_model():
    """Load the base Gemma model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")
    print("This may take several minutes...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"✓ Model loaded on {model.device}")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
) -> str:
    """Generate a response using the local model."""
    # Format messages as a single prompt
    # Base models don't have chat templates, so we concatenate
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt_parts.append(f"User: {content}\n\n")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}\n\n")

    prompt_parts.append("Assistant: ")
    full_prompt = "".join(prompt_parts)

    # Tokenize
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )

    return generated_text.strip()


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

    # Try to parse JSON (handle potential markdown code blocks and extra text)
    if "```json" in response_text:
        json_str = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        json_str = response_text.split("```")[1].split("```")[0].strip()
    else:
        json_str = response_text.strip()

    # Try to extract just the JSON object if there's extra text
    try:
        judgment = json.loads(json_str)
    except json.JSONDecodeError:
        # If parsing fails, try to find the JSON object within the text
        import re
        match = re.search(r'\{.*\}', json_str, re.DOTALL)
        if match:
            json_str = match.group(0)
            judgment = json.loads(json_str)
        else:
            raise

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


async def process_multiturn_sample(
    model,
    tokenizer,
    prompt: str,
    prompt_idx: int,
    sample_idx: int,
    judge_semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Process a single multi-turn conversation."""
    try:
        conversation = [{"role": "user", "content": prompt}]
        turns = []

        for turn_idx in range(MAX_TURNS):
            # Generate with local model (synchronous, no semaphore needed)
            start_time = time.time()
            generated_text = generate_response(model, tokenizer, conversation)
            sample_time = time.time() - start_time

            # Add assistant response to conversation
            conversation.append({"role": "assistant", "content": generated_text})

            # Judge with Anthropic with concurrency limit
            async with judge_semaphore:
                start_time = time.time()
                judgment = await judge_with_anthropic_async(generated_text)
                judge_time = time.time() - start_time

            # Store turn results
            turn_result = {
                "turn": turn_idx + 1,
                "generated_text": generated_text,
                "judgment": judgment,
                "sample_time": sample_time,
                "judge_time": judge_time,
            }
            turns.append(turn_result)

            # If this isn't the last turn, add feedback
            if turn_idx < MAX_TURNS - 1:
                feedback = FEEDBACK_MESSAGES[turn_idx]
                conversation.append({"role": "user", "content": feedback})

        # Store results
        result = {
            "prompt_idx": prompt_idx,
            "sample_idx": sample_idx,
            "prompt": prompt,
            "turns": turns,
            "conversation": conversation,
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
    model,
    tokenizer,
    prompt: str,
    prompt_idx: int,
    num_samples: int,
    judge_semaphore: asyncio.Semaphore,
) -> List[Dict[str, Any]]:
    """Process all samples for a single prompt sequentially (local generation)."""
    results = []

    for sample_idx in range(num_samples):
        print(f"  Sample {sample_idx + 1}/{num_samples}...", end=" ", flush=True)
        result = await process_multiturn_sample(
            model, tokenizer, prompt, prompt_idx, sample_idx,
            judge_semaphore
        )
        results.append(result)

        if result.get("status") == "success":
            turns = result["turns"]
            ratings = [t["judgment"]["rating"] for t in turns]
            print(f"Ratings: {ratings}")
        else:
            print(f"ERROR: {result.get('error', 'Unknown')}")

    return results


async def run_experiment_async(
    num_samples: int = NUM_SAMPLES,
    output_dir: Path = OUTPUT_DIR,
    max_concurrent_judges: int = MAX_CONCURRENT_JUDGES,
):
    """Run the V14 base model elicitation experiment."""

    # Validate API key
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Load model
    model, tokenizer = load_model()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for this run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"elicitation_multiturn_v14_results_{timestamp}.jsonl"

    print(f"\nStarting V14 BASE MODEL elicitation experiment")
    print(f"Model: {MODEL_NAME} (local generation)")
    print(f"Judge: {JUDGE_MODEL}")
    print(f"Number of prompts: {len(V14_PROMPTS)}")
    print(f"Samples per prompt: {num_samples}")
    print(f"Turns per sample: {MAX_TURNS}")
    print(f"Total generations: {len(V14_PROMPTS) * num_samples * MAX_TURNS}")
    print(f"Max concurrent judges: {max_concurrent_judges}")
    print(f"Results will be saved to: {results_file}")
    print()

    all_results = []

    # Create semaphore for judge rate limiting
    judge_semaphore = asyncio.Semaphore(max_concurrent_judges)

    # Process each prompt
    for prompt_idx, prompt in enumerate(V14_PROMPTS):
        print(f"\n{'='*80}")
        print(f"PROMPT {prompt_idx + 1}/{len(V14_PROMPTS)}")
        print(f"{'='*80}")
        print(f"{prompt[:200]}...")
        print()

        prompt_start = time.time()

        # Process all samples for this prompt (sequential for local generation)
        results = await process_prompt(
            model, tokenizer, prompt, prompt_idx, num_samples,
            judge_semaphore
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
            # Get ratings from all turns
            all_turn_ratings = []
            for sample in successful:
                for turn in sample["turns"]:
                    all_turn_ratings.append((turn["turn"], turn["judgment"]["rating"]))

            turn_1_ratings = [r for t, r in all_turn_ratings if t == 1]
            turn_2_ratings = [r for t, r in all_turn_ratings if t == 2]
            turn_3_ratings = [r for t, r in all_turn_ratings if t == 3]

            print(f"\n  Prompt {prompt_idx + 1} complete in {prompt_time:.1f}s")
            print(f"    Successful: {len(successful)}")
            print(f"    Errors: {len(errors)}")
            if turn_1_ratings:
                print(f"    Turn 1 mean rating: {sum(turn_1_ratings) / len(turn_1_ratings):.2f}")
            if turn_2_ratings:
                print(f"    Turn 2 mean rating: {sum(turn_2_ratings) / len(turn_2_ratings):.2f}")
            if turn_3_ratings:
                print(f"    Turn 3 mean rating: {sum(turn_3_ratings) / len(turn_3_ratings):.2f}")
            if all_turn_ratings:
                print(f"    Max rating seen: {max(all_turn_ratings, key=lambda x: x[1])[1]}")

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
        # Aggregate ratings by turn
        for turn_num in range(1, MAX_TURNS + 1):
            turn_ratings = []
            for sample in successful_results:
                for turn in sample["turns"]:
                    if turn["turn"] == turn_num:
                        turn_ratings.append(turn["judgment"]["rating"])

            if turn_ratings:
                print(f"\nTurn {turn_num} statistics:")
                print(f"  Mean: {sum(turn_ratings) / len(turn_ratings):.2f}")
                print(f"  Min: {min(turn_ratings)}")
                print(f"  Max: {max(turn_ratings)}")
                print(f"  Samples with rating >= 5: {sum(1 for r in turn_ratings if r >= 5)}")
                print(f"  Samples with rating >= 7: {sum(1 for r in turn_ratings if r >= 7)}")

    return all_results


def run_experiment(
    num_samples: int = NUM_SAMPLES,
    output_dir: Path = OUTPUT_DIR,
    max_concurrent_judges: int = MAX_CONCURRENT_JUDGES,
):
    """Wrapper to run async experiment."""
    return asyncio.run(
        run_experiment_async(
            num_samples=num_samples,
            output_dir=output_dir,
            max_concurrent_judges=max_concurrent_judges,
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run V14 base model elicitation experiment")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                        help="Number of samples per prompt")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Output directory for results")
    parser.add_argument("--max-concurrent-judges", type=int, default=MAX_CONCURRENT_JUDGES,
                        help="Max concurrent Anthropic requests")

    args = parser.parse_args()

    results = run_experiment(
        num_samples=args.num_samples,
        output_dir=Path(args.output_dir),
        max_concurrent_judges=args.max_concurrent_judges,
    )
