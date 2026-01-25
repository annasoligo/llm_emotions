"""
Priority Selection Steering Experiment

Tests whether emotion steering affects which category of information
the model considers "most important" when reviewing a situation.

Each stimulus has 12 statements (3 per emotion category):
- FEAR: Uncertainty, uncontrollable threats
- ANGER: Blame, negligence, intentional wrongs
- SADNESS: Loss, irreversibility, missed opportunities
- HAPPINESS: Gains, improvements, opportunities

Uses logprobs on letters A-L to measure probability distribution
across emotion categories under different steering conditions.
"""

# Disable vLLM v1 engine before any imports - it has initialization issues
import os
os.environ["VLLM_USE_V1"] = "0"

import argparse
import json
import logging
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from vllm import LLM, SamplingParams

from experiments.steering.core import VLLMSteering
from experiments.steering.layer_norms import get_layer_norm
from experiments.behavior_tests.prompts.priority_selection import (
    ALL_STIMULI,
    SelectionStimulus,
    EmotionCategory,
)
from experiments.behavior_tests.plot_priority_selection_results import (
    plot_single_model,
    MODEL_CONFIGS,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Layer norms are loaded dynamically from experiments/steering/layer_norms.json
MODEL_CONFIGS = {
    "gemma": {
        "name": "google/gemma-3-27b-it",
        "layer": 30,
        "vectors_path": "experiments/steering/vectors/text/gemma_layer30.npz",
    },
    "qwen32b": {
        "name": "Qwen/Qwen3-32B",
        "layer": 30,
        "vectors_path": "experiments/steering/vectors/text/qwen32b_layer30.npz",
    },
    "qwen235b": {
        "name": "Qwen/Qwen3-235B-A22B",
        "layer": 30,
        "vectors_path": "experiments/steering/vectors/text/qwen235b_layer30.npz",
        "vector_format": "direct",
    },
}


def compute_category_probabilities(
    letter_logprobs: Dict[str, float],
    mapping: Dict[str, any]
) -> Dict[str, float]:
    """
    Compute probability for each emotion category by summing letter probabilities.

    Args:
        letter_logprobs: Dict mapping letters (A-L) to logprobs
        mapping: Dict mapping letters to Statement objects with category info

    Returns:
        Dict with probabilities for each emotion category
    """
    # Convert logprobs to probabilities with numerical stability
    max_logprob = max(letter_logprobs.values()) if letter_logprobs else 0

    probs = {}
    for letter, logprob in letter_logprobs.items():
        probs[letter] = np.exp(logprob - max_logprob)

    # Normalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    # Sum by category
    category_probs = defaultdict(float)
    for letter, prob in probs.items():
        if letter in mapping:
            category = mapping[letter].category.value
            category_probs[category] += prob

    return {
        "fear_prob": category_probs.get("fear", 0.0),
        "anger_prob": category_probs.get("anger", 0.0),
        "sadness_prob": category_probs.get("sadness", 0.0),
        "happiness_prob": category_probs.get("happiness", 0.0),
    }


def format_chat_prompt(prompt: str, model_key: str = "gemma") -> str:
    """Format prompt for chat model with strict single-letter format."""
    base = f"""{prompt}

Respond with EXACTLY ONE letter. Your entire response must be a single character A-L. No punctuation, no explanation, no newlines - just the letter."""

    # Disable thinking mode for Qwen3 models
    if model_key.startswith("qwen"):
        base += " /no_think"

    return base


def run_experiment(
    model_key: str = "gemma",
    layer_override: int = None,
    emotions: List[str] = None,
    norm_pcts: List[float] = None,
    num_seeds: int = 4,  # Number of different shuffles per stimulus
    output_dir: str = "experiments/steering/outputs",
    positive_only: bool = False,
    negative_only: bool = False,
):
    """Run the priority selection steering experiment."""

    config = MODEL_CONFIGS[model_key]
    model_name = config["name"]
    layer = layer_override if layer_override is not None else config["layer"]
    # Update vectors path if layer override specified
    if layer_override is not None:
        vectors_path = f"experiments/steering/vectors/text/{model_key}_layer{layer}.npz"
    else:
        vectors_path = config["vectors_path"]

    # Load layer norm from centralized file
    layer_norm = get_layer_norm(model_key, layer)

    if emotions is None:
        emotions = ["fear", "anger", "sadness", "happiness"]
    if norm_pcts is None:
        norm_pcts = [0.05, 0.075, 0.10]

    logger.info(f"Model: {model_name}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Emotions: {emotions}")
    logger.info(f"Norm percentages: {norm_pcts}")

    # Load steering vectors
    logger.info(f"Loading vectors from {vectors_path}")
    vectors_data = np.load(vectors_path)

    # Handle different npz formats
    if 'emotions' in vectors_data:
        available_emotions = list(vectors_data['emotions'])
        vectors_array = vectors_data['vectors']
        vectors = {emo: vectors_array[i] for i, emo in enumerate(available_emotions)}
    else:
        available_emotions = list(vectors_data.keys())
        vectors = {e: vectors_data[e] for e in available_emotions}

    logger.info(f"Available emotions: {available_emotions}")

    # Filter to requested emotions
    emotions = [e for e in emotions if e in available_emotions]
    if not emotions:
        raise ValueError(f"No matching emotions found. Available: {available_emotions}")

    vectors = {e: vectors[e] for e in emotions}

    # Initialize model first (need tokenizer for chat template)
    logger.info("Loading model...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enable_prefix_caching=False,
        enforce_eager=True,  # Required for steering hooks
    )
    tokenizer = llm.get_tokenizer()

    # Generate prompts for all stimuli with multiple seeds
    prompts = []
    template_kwargs = {"enable_thinking": False} if model_key.startswith("qwen") else {}
    for stimulus in ALL_STIMULI:
        for seed in range(num_seeds):
            prompt_text, mapping = stimulus.get_shuffled_prompt(seed=seed)
            formatted = format_chat_prompt(prompt_text, model_key)
            # Apply chat template
            messages = [{"role": "user", "content": formatted}]
            chat_formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **template_kwargs
            )
            prompts.append({
                "stimulus_id": stimulus.id,
                "seed": seed,
                "prompt": chat_formatted,
                "mapping": mapping,
            })

    logger.info(f"Using {len(prompts)} prompts ({len(ALL_STIMULI)} stimuli × {num_seeds} seeds)")

    # Setup steering
    steering = VLLMSteering(llm, layer)
    logger.info(f"Layer {layer} norm: {layer_norm}")

    # Sampling params - get logprobs for first token
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=20,  # Get top 20 logprobs to capture A-L
    )

    # Build conditions
    conditions = [{"name": "baseline", "emotion": None, "pct": 0, "direction": 0}]

    for emotion in emotions:
        for pct in norm_pcts:
            if not negative_only:
                conditions.append({
                    "name": f"{emotion}_+{pct*100:.1f}%",
                    "emotion": emotion,
                    "pct": pct,
                    "direction": 1
                })
            if not positive_only:
                conditions.append({
                    "name": f"{emotion}_-{pct*100:.1f}%",
                    "emotion": emotion,
                    "pct": pct,
                    "direction": -1
                })

    logger.info(f"Testing {len(conditions)} conditions")

    # Run experiment
    results = []

    for cond in conditions:
        logger.info(f"Condition: {cond['name']}")

        # Set up steering
        if cond['emotion'] is not None:
            vec = vectors[cond['emotion']]
            magnitude = cond['direction'] * cond['pct'] * layer_norm
            scaled_vec = vec * magnitude
            logger.info(f"  Applying vector: {cond['emotion']}, magnitude={magnitude:.2f}")
            steering.set_raw_vector(scaled_vec, steer_prompt=True, steer_generation=True)
        else:
            logger.info("  Clearing steering (baseline)")
            steering.clear()

        # Run all prompts for this condition
        formatted_prompts = [p["prompt"] for p in prompts]
        outputs = llm.generate(formatted_prompts, sampling_params)

        # Process results
        for i, prompt_data in enumerate(prompts):
            output = outputs[i]
            mapping = prompt_data["mapping"]

            # Extract letter logprobs
            letter_logprobs = {}
            if output.outputs and output.outputs[0].logprobs:
                first_logprobs = output.outputs[0].logprobs[0]

                # Find logprobs for A-L tokens
                for letter in list("ABCDEFGHIJKL"):
                    for variant in [letter, f' {letter}', f'{letter})', f' {letter})']:
                        token_ids = tokenizer.encode(variant, add_special_tokens=False)
                        if token_ids:
                            token_id = token_ids[-1]
                            if token_id in first_logprobs:
                                letter_logprobs[letter] = first_logprobs[token_id].logprob
                                break
                    if letter not in letter_logprobs:
                        letter_logprobs[letter] = float('-inf')

            # Compute category probabilities
            cat_probs = compute_category_probabilities(letter_logprobs, mapping)

            # Get selected letter (highest prob)
            valid_letters = {k: v for k, v in letter_logprobs.items() if v > float('-inf')}
            if valid_letters:
                selected_letter = max(valid_letters, key=valid_letters.get)
                selected_stmt = mapping.get(selected_letter)
                selected_category = selected_stmt.category.value if selected_stmt else None
            else:
                selected_letter = None
                selected_category = None

            # Get raw output text
            raw_output = output.outputs[0].text if output.outputs else ""

            result = {
                "condition": cond["name"],
                "emotion": cond["emotion"],
                "norm_pct": cond["pct"],
                "direction": cond["direction"],
                "stimulus_id": prompt_data["stimulus_id"],
                "seed": prompt_data["seed"],
                "selected_letter": selected_letter,
                "selected_category": selected_category,
                "raw_output": raw_output[:200],  # First 200 chars
                "letter_logprobs": letter_logprobs,
                "fear_prob": cat_probs["fear_prob"],
                "anger_prob": cat_probs["anger_prob"],
                "sadness_prob": cat_probs["sadness_prob"],
                "happiness_prob": cat_probs["happiness_prob"],
                "options": {
                    letter: {
                        "text": stmt.text[:100],
                        "category": stmt.category.value,
                        "subcategory": stmt.subcategory,
                    }
                    for letter, stmt in mapping.items()
                },
            }
            results.append(result)

        # Log summary for this condition
        fear_probs = [r['fear_prob'] for r in results if r['condition'] == cond['name']]
        anger_probs = [r['anger_prob'] for r in results if r['condition'] == cond['name']]
        sadness_probs = [r['sadness_prob'] for r in results if r['condition'] == cond['name']]
        happiness_probs = [r['happiness_prob'] for r in results if r['condition'] == cond['name']]

        logger.info(f"  Avg probs - Fear: {np.mean(fear_probs):.3f}, Anger: {np.mean(anger_probs):.3f}, "
                   f"Sadness: {np.mean(sadness_probs):.3f}, Happiness: {np.mean(happiness_probs):.3f}")

    # Clear steering
    steering.clear()

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_key
    output_file = output_dir / f"priority_selection_{model_short}_layer{layer}_{timestamp}.jsonl"

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"Saved {len(results)} results to {output_file}")

    # Generate plot if model config exists
    if model_key in MODEL_CONFIGS:
        logger.info("Generating plot...")
        plot_path = plot_single_model(output_file, model_key, output_dir)
        logger.info(f"Saved plot to {plot_path}")
    else:
        logger.warning(f"No plot config for model {model_key}, skipping plot generation")

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Priority Selection Steering Experiment")
    parser.add_argument("--model", type=str, default="gemma", choices=["gemma", "qwen32b", "qwen235b"])
    parser.add_argument("--layer", type=int, default=None, help="Override default layer for steering")
    parser.add_argument("--emotions", type=str, nargs="+", default=["fear", "anger", "sadness", "happiness"])
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[0.05, 0.075, 0.10])
    parser.add_argument("--num-seeds", type=int, default=4, help="Number of shuffles per stimulus")
    parser.add_argument("--output-dir", type=str, default="experiments/steering/outputs/priority_selection")
    parser.add_argument("--positive-only", action="store_true", help="Only test positive steering")
    parser.add_argument("--negative-only", action="store_true", help="Only test negative steering")

    args = parser.parse_args()

    run_experiment(
        model_key=args.model,
        layer_override=args.layer,
        emotions=args.emotions,
        norm_pcts=args.norm_pcts,
        num_seeds=args.num_seeds,
        output_dir=args.output_dir,
        positive_only=args.positive_only,
        negative_only=args.negative_only,
    )
