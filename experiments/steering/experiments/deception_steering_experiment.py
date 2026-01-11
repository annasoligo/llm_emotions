"""
Deception steering experiment.

Tests emotion steering effects on deception compliance using curated subset.
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..config import MODEL_NAME, OUTPUT_DIR, VECTOR_DIR
from ..core import VLLMSteering
from experiments.behavior_tests.prompts.deception_steering_subset import (
    ALL_STEERING_PROMPTS,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Layer norms
LAYER_NORMS = {
    20: 12820.22,
    30: 42151.76,
    40: 56622.62,
}

# All emotions
ALL_EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def format_prompt(scenario: str, tokenizer) -> str:
    """Format prompt for Gemma chat template."""
    messages = [{"role": "user", "content": scenario}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_experiment(
    layer: int,
    norm_pcts: List[float],
    num_samples: int,
    emotions: List[str],
    output_dir: Path,
):
    """Run deception steering experiment."""
    logger.info(f"Layer {layer} activation norm: {LAYER_NORMS[layer]:.2f}")

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

    # Setup steering
    steering = VLLMSteering(llm, layer, baseline_std=1.0)
    logger.info(f"Steering at layer {layer}")

    # Load textmeandiff vectors
    steering.load_vectors(VECTOR_DIR)
    logger.info(f"Loaded vectors: {steering.available_emotions}")

    # Get all prompts
    all_prompts = ALL_STEERING_PROMPTS
    logger.info(f"Testing {len(all_prompts)} prompts x {num_samples} samples")

    # Build conditions
    conditions = [{"name": "baseline", "emotion": None, "pct": 0, "direction": 1}]
    for pct in norm_pcts:
        for emotion in emotions:
            conditions.append({"name": f"{emotion}_+{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": 1})
            conditions.append({"name": f"{emotion}_-{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": -1})

    logger.info(f"Testing {len(conditions)} conditions")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        stop=["<end_of_turn>"],
    )

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"deception_steering_layer{layer}_{timestamp}.jsonl"

    total_responses = 0
    layer_norm = LAYER_NORMS[layer]

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
            batch_info = []
            formatted_prompts = []
            for prompt_data in all_prompts:
                for sample_id in range(num_samples):
                    batch_info.append((prompt_data, sample_id))
                    formatted_prompts.append(format_prompt(prompt_data["scenario"], tokenizer))

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
                    "prompt_id": prompt_data["id"],
                    "scenario": prompt_data["scenario"],
                    "trigger": prompt_data.get("trigger", "unknown"),
                    "baseline_neutral": prompt_data.get("baseline_neutral"),
                    "expected_joy_effect": prompt_data.get("joy_effect"),
                    "expected_anger_effect": prompt_data.get("anger_effect"),
                    "expected_fear_effect": prompt_data.get("fear_effect"),
                    "sample_id": sample_id,
                    "response": response,
                }

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
    parser = argparse.ArgumentParser(description="Deception steering experiment")
    parser.add_argument("--layer", type=int, default=30, choices=[20, 30, 40])
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[0.10])
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--emotions", type=str, nargs="+", default=ALL_EMOTIONS,
                        help="Emotions to test (default: all 6)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_experiment(
        layer=args.layer,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        emotions=args.emotions,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
