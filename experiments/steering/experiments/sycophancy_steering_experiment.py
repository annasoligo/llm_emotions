"""
Sycophancy steering experiment.

Tests emotion steering effects on sycophancy behaviors.
Saves results incrementally during generation.
"""
import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import anthropic
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..config import MODEL_NAME, OUTPUT_DIR, VECTOR_DIR
from ..core import VLLMSteering
from ..layer_norms import get_layer_norm
from experiments.behavior_tests.prompts.sycophancy_prompts import (
    ALL_SYCOPHANCY_PROMPTS,
    get_judge_prompt,
)
from experiments.steering.experiments.coherency_judge_prompt import get_coherency_prompt

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Default emotions to test
DEFAULT_EMOTIONS = ["anger", "fear"]


def format_prompt(prompt_text: str, tokenizer) -> str:
    """Format prompt for Gemma chat template."""
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def judge_sycophancy(prompt_data, response: str, client: anthropic.Anthropic) -> Dict:
    """Judge response for sycophancy using appropriate judge prompt."""
    judge_prompt = get_judge_prompt(prompt_data, response)

    try:
        result = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": judge_prompt}]
        )
        text = result.content[0].text
        # Extract JSON
        match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.warning(f"Sycophancy judge error: {e}")

    return {"error": "Judge failed"}


def judge_coherency(response: str, client: anthropic.Anthropic) -> Dict:
    """Judge response coherency on 0-100 scale."""
    prompt = get_coherency_prompt(response)

    try:
        result = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )
        text = result.content[0].text
        match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        logger.warning(f"Coherency judge error: {e}")

    return {"coherency_score": -1, "reasoning": "Judge failed"}


def run_experiment(
    layer: int,
    norm_pcts: List[float],
    num_samples: int,
    emotions: List[str],
    output_dir: Path,
    skip_judging: bool = False,
):
    """Run sycophancy steering experiment with incremental saving."""
    layer_norm = get_layer_norm("gemma", layer)
    logger.info(f"Layer {layer} activation norm: {layer_norm:.2f}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load model with steering
    logger.info("Loading model...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        disable_log_stats=True,
    )

    # Setup steering - use baseline_std=1.0 so scale is raw magnitude
    steering = VLLMSteering(llm, layer, baseline_std=1.0)
    logger.info(f"Steering at layer {layer}")

    # Load textmeandiff vectors from vector directory
    steering.load_vectors(VECTOR_DIR)
    logger.info(f"Loaded vectors: {steering.available_emotions}")

    # Initialize judge
    client = None
    if not skip_judging:
        logger.info("Initializing Claude judge...")
        client = anthropic.Anthropic()

    # Get all prompts
    all_prompts = ALL_SYCOPHANCY_PROMPTS
    logger.info(f"Testing {len(all_prompts)} prompts x {num_samples} samples")

    # Build conditions
    conditions = [{"name": "baseline", "emotion": None, "pct": 0, "direction": 1}]
    for pct in norm_pcts:
        for emotion in emotions:
            conditions.append({"name": f"{emotion}_+{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": 1})
            conditions.append({"name": f"{emotion}_-{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": -1})

    logger.info(f"Testing {len(conditions)} conditions: {[c['name'] for c in conditions]}")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        stop=["<end_of_turn>"],
    )

    # Output file - open for incremental writing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"sycophancy_steering_layer{layer}_{timestamp}.jsonl"

    total_responses = 0

    # Open file for incremental writing
    with open(output_file, 'w') as f:
        for cond in conditions:
            logger.info(f"Condition: {cond['name']}")

            # Set steering
            if cond["emotion"]:
                magnitude = cond["pct"] * layer_norm
                steering.set(f"{cond['emotion']}_textmeandiff", scale=magnitude, direction=cond["direction"])
                logger.info(f"  Steering: {magnitude:.2f} * {cond['direction']}")
            else:
                steering.clear()

            # Batch ALL prompts for this condition together
            # Build list of (prompt_data, sample_id, formatted_prompt)
            batch_info = []
            formatted_prompts = []
            for prompt_data in all_prompts:
                for sample_id in range(num_samples):
                    batch_info.append((prompt_data, sample_id))
                    formatted_prompts.append(format_prompt(prompt_data.prompt, tokenizer))

            logger.info(f"  Generating {len(formatted_prompts)} responses in single batch...")

            # Generate all at once
            outputs = llm.generate(formatted_prompts, sampling_params)

            # Process outputs
            for (prompt_data, sample_id), output in zip(batch_info, outputs):
                response = output.outputs[0].text

                result = {
                    "condition": cond["name"],
                    "emotion": cond["emotion"],
                    "norm_pct": cond["pct"],
                    "direction": cond["direction"],
                    "layer": layer,
                    "prompt_id": prompt_data.id,
                    "test_type": prompt_data.test_type,
                    "variant": prompt_data.variant,
                    "metadata": prompt_data.metadata,
                    "sample_id": sample_id,
                    "response": response,
                }

                # Judge immediately if not skipping
                if client:
                    result["sycophancy_judge"] = judge_sycophancy(prompt_data, response, client)
                    result["coherency"] = judge_coherency(response, client)

                # Write immediately
                f.write(json.dumps(result) + '\n')
                total_responses += 1

            # Flush after each condition
            f.flush()
            logger.info(f"  Generated {len(outputs)} responses (total: {total_responses})")

    # Clear steering
    steering.clear()

    logger.info(f"Saved {total_responses} results to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Sycophancy steering experiment")
    parser.add_argument("--layer", type=int, default=30, choices=[20, 30, 40])
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[0.07, 0.10])
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--emotions", type=str, nargs="+", default=DEFAULT_EMOTIONS,
                        help="Emotions to test (default: anger, fear)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--skip-judging", action="store_true",
                        help="Skip Claude judging (faster, judge later)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_experiment(
        layer=args.layer,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        emotions=args.emotions,
        output_dir=args.output_dir,
        skip_judging=args.skip_judging,
    )


if __name__ == "__main__":
    main()
