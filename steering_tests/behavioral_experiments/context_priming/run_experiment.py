"""
Context priming experiment runner via OpenRouter.

Tests how positive vs negative priming affects model judgments.

Usage:
    python -m steering_tests.behavioral_experiments.context_priming.run_experiment \
        --models google/gemma-3-27b-it qwen/qwen3-32b \
        --samples 50

    # Run specific conditions only
    python -m steering_tests.behavioral_experiments.context_priming.run_experiment \
        --models google/gemma-3-27b-it \
        --conditions pos_early_score_first neg_early_score_first
"""

import argparse
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from .scenarios import build_legal_prompt, get_all_conditions

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# OpenRouter models
# =============================================================================

OPENROUTER_MODELS = {
    # Gemma models
    "gemma-27b": "google/gemma-3-27b-it",
    "gemma-12b": "google/gemma-3-12b-it",
    # Qwen models
    "qwen-32b": "qwen/qwen3-32b",
    "qwen-14b": "qwen/qwen3-14b",
    "qwen-235b": "qwen/qwen3-235b-a22b",
}

# Reverse lookup
MODEL_SHORT_NAMES = {v: k for k, v in OPENROUTER_MODELS.items()}

# =============================================================================
# OpenRouter API
# =============================================================================

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


async def query_openrouter(
    client: httpx.AsyncClient,
    prompt: str,
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 1000,
) -> dict:
    """
    Query OpenRouter API.

    Returns:
        Dict with 'content', 'model', 'usage', or 'error'
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/anthropics/research-tools",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = await client.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        data = response.json()

        return {
            "content": data["choices"][0]["message"]["content"],
            "model": data.get("model", model),
            "usage": data.get("usage", {}),
        }

    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
        return {"error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        logger.error(f"Request error: {e}")
        return {"error": str(e)}


# =============================================================================
# Response parsing
# =============================================================================

def parse_response(text: str) -> dict:
    """
    Parse JSON response to extract score and response_type.

    Returns:
        Dict with 'score', 'response_type', 'reasoning', 'parse_error'
    """
    result = {
        "score": None,
        "response_type": None,
        "reasoning": None,
        "parse_error": None,
    }

    # Try to find JSON in response
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if not json_match:
        result["parse_error"] = "No JSON found"
        return result

    try:
        data = json.loads(json_match.group())
        result["score"] = data.get("score")
        result["response_type"] = data.get("response_type")
        result["reasoning"] = data.get("reasoning")

        # Handle reasoning if it's a list
        if isinstance(result["reasoning"], list):
            result["reasoning"] = " ".join(str(r) for r in result["reasoning"]) if result["reasoning"] else None

        # Validate score
        if result["score"] is not None:
            # Handle if score is a list (some models return [3] instead of 3)
            if isinstance(result["score"], list):
                if len(result["score"]) > 0:
                    result["score"] = result["score"][0]
                else:
                    result["score"] = None
            if result["score"] is not None:
                try:
                    # Handle string scores like "3" or "3/10"
                    score_str = str(result["score"]).strip()
                    # Extract first number from patterns like "3/10" or "3 out of 10"
                    score_match = re.match(r'^(\d+)', score_str)
                    if score_match:
                        result["score"] = int(score_match.group(1))
                    else:
                        result["score"] = int(float(score_str))
                    if not 1 <= result["score"] <= 10:
                        result["parse_error"] = f"Score {result['score']} out of range"
                except (ValueError, TypeError) as e:
                    result["parse_error"] = f"Could not parse score: {result['score']}"
                    result["score"] = None

        # Validate response_type
        if result["response_type"] is not None:
            # Handle if response_type is a list (some models return ["A"] instead of "A")
            if isinstance(result["response_type"], list):
                if len(result["response_type"]) > 0:
                    result["response_type"] = result["response_type"][0]
                else:
                    result["response_type"] = None
            if result["response_type"] is not None:
                result["response_type"] = str(result["response_type"]).upper().strip()
                if result["response_type"] not in ["A", "B", "C", "D"]:
                    result["parse_error"] = f"Invalid response_type: {result['response_type']}"

    except json.JSONDecodeError as e:
        result["parse_error"] = f"JSON decode error: {e}"
    except (ValueError, TypeError) as e:
        result["parse_error"] = f"Value error: {e}"

    return result


# =============================================================================
# Experiment runner
# =============================================================================

async def run_condition(
    client: httpx.AsyncClient,
    model: str,
    condition: dict,
    num_samples: int,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """Run a single condition with multiple samples."""

    prompt = build_legal_prompt(
        valence=condition["valence"],
        timing=condition["timing"],
        response_order=condition["response_order"],
    )

    results = []

    async def run_single(sample_id: int) -> dict:
        async with semaphore:
            response = await query_openrouter(client, prompt, model)

            result = {
                "model": model,
                "model_short": MODEL_SHORT_NAMES.get(model, model),
                "condition": condition["condition_name"],
                "valence": condition["valence"],
                "timing": condition["timing"],
                "response_order": condition["response_order"],
                "sample_id": sample_id,
                "raw_response": response.get("content", ""),
                "error": response.get("error"),
            }

            # Parse response
            if result["raw_response"]:
                parsed = parse_response(result["raw_response"])
                result.update(parsed)
            else:
                result["score"] = None
                result["response_type"] = None
                result["reasoning"] = None
                result["parse_error"] = result.get("error", "No response")

            return result

    tasks = [run_single(i) for i in range(num_samples)]
    results = await asyncio.gather(*tasks)

    return results


async def run_experiment(
    models: list[str],
    conditions: Optional[list[dict]] = None,
    num_samples: int = 50,
    max_concurrent: int = 20,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Run full context priming experiment.

    Args:
        models: List of OpenRouter model IDs
        conditions: List of condition dicts (default: all conditions)
        num_samples: Samples per condition
        max_concurrent: Max concurrent API requests
        output_dir: Output directory

    Returns:
        Path to output file
    """
    if conditions is None:
        conditions = get_all_conditions()

    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "context_priming"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"context_priming_{timestamp}.jsonl"

    logger.info("=" * 60)
    logger.info("CONTEXT PRIMING EXPERIMENT")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")
    logger.info(f"Conditions: {len(conditions)}")
    logger.info(f"Samples per condition: {num_samples}")
    logger.info(f"Total API calls: {len(models) * len(conditions) * num_samples}")
    logger.info(f"Output: {output_file}")

    semaphore = asyncio.Semaphore(max_concurrent)
    all_results = []

    async with httpx.AsyncClient() as client:
        for model in models:
            model_short = MODEL_SHORT_NAMES.get(model, model)
            logger.info(f"\n--- {model_short} ---")

            for cond in conditions:
                logger.info(f"  {cond['condition_name']}...")

                results = await run_condition(
                    client=client,
                    model=model,
                    condition=cond,
                    num_samples=num_samples,
                    semaphore=semaphore,
                )

                all_results.extend(results)

                # Save incrementally
                with open(output_file, "a") as f:
                    for r in results:
                        f.write(json.dumps(r) + "\n")

                # Quick stats
                valid = [r for r in results if r["score"] is not None]
                if valid:
                    mean_score = sum(r["score"] for r in valid) / len(valid)
                    logger.info(f"    {len(valid)}/{len(results)} valid, mean={mean_score:.2f}")
                else:
                    logger.warning(f"    No valid responses!")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    for model in models:
        model_short = MODEL_SHORT_NAMES.get(model, model)
        model_results = [r for r in all_results if r["model"] == model]

        logger.info(f"\n{model_short}:")
        for cond in conditions:
            cond_results = [r for r in model_results if r["condition"] == cond["condition_name"]]
            valid = [r for r in cond_results if r["score"] is not None]
            if valid:
                mean = sum(r["score"] for r in valid) / len(valid)
                logger.info(f"  {cond['condition_name']:30} n={len(valid):3} mean={mean:.2f}")

    logger.info(f"\nSaved {len(all_results)} results to {output_file}")
    return output_file


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run context priming experiment")

    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=["gemma-27b"],
        help="Models to test (short names or full OpenRouter IDs)",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=50,
        help="Samples per condition",
    )
    parser.add_argument(
        "--conditions", "-c",
        nargs="+",
        default=None,
        help="Specific conditions to run (default: all)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=20,
        help="Max concurrent API requests",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()

    # Resolve model names
    models = []
    for m in args.models:
        if m in OPENROUTER_MODELS:
            models.append(OPENROUTER_MODELS[m])
        else:
            models.append(m)

    # Filter conditions if specified
    conditions = None
    if args.conditions:
        all_conds = get_all_conditions()
        conditions = [c for c in all_conds if c["condition_name"] in args.conditions]
        if not conditions:
            print(f"No matching conditions found. Available: {[c['condition_name'] for c in all_conds]}")
            return

    asyncio.run(run_experiment(
        models=models,
        conditions=conditions,
        num_samples=args.samples,
        max_concurrent=args.max_concurrent,
        output_dir=args.output_dir,
    ))


if __name__ == "__main__":
    main()
