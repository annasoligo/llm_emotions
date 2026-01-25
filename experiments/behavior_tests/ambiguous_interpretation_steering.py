"""
Ambiguous Interpretation Steering Experiment

Tests how emotion steering affects interpretation of ambiguous social situations.
Uses logprobs on answer letters (A/B/C/D) to measure probability shifts toward
"threat" vs "neutral" interpretations.

Key hypothesis: Fear/anxiety steering should increase P(threat interpretations)
"""

# Disable vLLM v1 engine before any imports - it has initialization issues
import os
os.environ["VLLM_USE_V1"] = "0"

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from vllm import LLM, SamplingParams

from experiments.steering.core import VLLMSteering
from experiments.steering.layer_norms import get_layer_norm
from experiments.behavior_tests.prompts.ambiguous_situations import (
    dataset as AMBIGUOUS_ITEMS,
    PermutationEngine,
)
from experiments.behavior_tests.plot_ambiguous_interpretation_results import (
    plot_single_model,
    MODEL_CONFIGS as PLOT_CONFIGS,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Models and their configs - aligned with priority selection experiment
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
        "layer": 50,  # Layer 50 for 235B (same as priority selection)
        "vectors_path": "experiments/steering/vectors/text/qwen235b_layer50.npz",
    },
}


def format_prompt(variation: Dict, model_key: str = "gemma") -> str:
    """Format a variation into a prompt for the model."""
    prompt = f"""Consider this scenario and question carefully.

Scenario: {variation['scenario']}

Question: {variation['question']}

Options:
A) {variation['options']['A']['text']}
B) {variation['options']['B']['text']}
C) {variation['options']['C']['text']}
D) {variation['options']['D']['text']}

Respond with EXACTLY ONE letter. Your entire response must be a single character A-D. No punctuation, no explanation, no newlines - just the letter."""

    # Disable thinking mode for Qwen3 models
    if model_key.startswith("qwen"):
        prompt += " /no_think"

    return prompt


def compute_interpretation_probabilities(letter_logprobs: Dict[str, float],
                                          option_classifications: Dict[str, str]) -> Dict:
    """Compute probability of threat vs neutral/positive interpretations."""
    # Convert logprobs to probabilities
    probs = {}
    max_logprob = max(letter_logprobs.values()) if letter_logprobs else 0

    for letter, logprob in letter_logprobs.items():
        if logprob > float('-inf'):
            probs[letter] = math.exp(logprob - max_logprob)
        else:
            probs[letter] = 0.0

    # Normalize
    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    # Categorize probabilities
    threat_prob = 0.0   # Threat or Negative
    positive_prob = 0.0 # Positive
    neutral_prob = 0.0  # Neutral

    for letter, prob in probs.items():
        classification = option_classifications.get(letter, "")
        if not classification:
            neutral_prob += prob
        elif "Threat" in classification or classification == "Negative":
            threat_prob += prob
        elif classification == "Positive":
            positive_prob += prob
        else:  # Neutral variants
            neutral_prob += prob

    return {
        "letter_probs": probs,
        "threat_prob": threat_prob,
        "positive_prob": positive_prob,
        "neutral_prob": neutral_prob,
    }


def run_experiment(
    model_key: str = "gemma",
    layer_override: int = None,
    emotions: List[str] = None,
    norm_pcts: List[float] = None,
    num_permutations: int = 10,  # More permutations for better stats
    output_dir: str = "experiments/steering/outputs/ambiguous_interpretation",
    positive_only: bool = False,
    negative_only: bool = False,
):
    """Run the ambiguous interpretation steering experiment."""

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
        # Higher magnitudes for Qwen models (aligned with priority selection)
        if model_key.startswith("qwen"):
            norm_pcts = [1.0, 1.25, 1.5]  # 100%, 125%, 150%
        else:
            norm_pcts = [0.05, 0.075, 0.10]  # 5%, 7.5%, 10%

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

    # Generate variations (subset of permutations)
    engine = PermutationEngine(AMBIGUOUS_ITEMS)
    all_variations = engine.generate_all_variations()

    # Select subset of permutations per item
    selected_variations = []
    items_seen = {}
    for v in all_variations:
        item_id = v['item_id']
        if item_id not in items_seen:
            items_seen[item_id] = 0
        if items_seen[item_id] < num_permutations:
            selected_variations.append(v)
            items_seen[item_id] += 1

    logger.info(f"Using {len(selected_variations)} variations ({len(AMBIGUOUS_ITEMS)} items x {num_permutations} permutations)")

    # Initialize model
    logger.info("Loading model...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    # Setup steering
    steering = VLLMSteering(llm, layer)
    logger.info(f"Layer {layer} norm: {layer_norm:.2f}")

    # Sampling params - get logprobs for first token
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=20,
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

    # Prepare all prompts
    prompts = []
    template_kwargs = {"enable_thinking": False} if model_key.startswith("qwen") else {}
    for v in selected_variations:
        prompt = format_prompt(v, model_key)
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        prompts.append((v, formatted))

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

        # Run all prompts
        formatted_prompts = [p[1] for p in prompts]
        outputs = llm.generate(formatted_prompts, sampling_params)

        # Process results
        for i, (variation, _) in enumerate(prompts):
            output = outputs[i]

            # Extract letter logprobs
            letter_logprobs = {}
            if output.outputs and output.outputs[0].logprobs:
                first_logprobs = output.outputs[0].logprobs[0]

                for letter in ['A', 'B', 'C', 'D']:
                    for variant in [letter, f' {letter}', f'{letter})', f' {letter})']:
                        token_ids = tokenizer.encode(variant, add_special_tokens=False)
                        if token_ids:
                            token_id = token_ids[-1]
                            if token_id in first_logprobs:
                                letter_logprobs[letter] = first_logprobs[token_id].logprob
                                break
                    if letter not in letter_logprobs:
                        letter_logprobs[letter] = float('-inf')

            # Get option classifications
            option_classifications = {
                letter: variation['options'][letter].get('classification', '')
                for letter in ['A', 'B', 'C', 'D']
            }

            # Compute interpretation probabilities
            prob_analysis = compute_interpretation_probabilities(letter_logprobs, option_classifications)

            # Get selected answer
            valid_letters = {k: v for k, v in letter_logprobs.items() if v > float('-inf')}
            if valid_letters:
                selected_letter = max(valid_letters, key=valid_letters.get)
            else:
                selected_letter = output.outputs[0].text.strip()[0] if output.outputs[0].text.strip() else ""

            result = {
                "condition": cond['name'],
                "emotion": cond['emotion'],
                "norm_pct": cond['pct'],
                "direction": cond['direction'],
                "item_id": variation['item_id'],
                "perm_id": variation['perm_id'],
                "domain": variation.get('domain', 'unknown'),
                "scenario": variation['scenario'],
                "question": variation['question'],
                "selected_letter": selected_letter,
                "selected_classification": option_classifications.get(selected_letter, ""),
                "letter_logprobs": {k: v if v > float('-inf') else None for k, v in letter_logprobs.items()},
                "letter_probs": prob_analysis['letter_probs'],
                "threat_prob": prob_analysis['threat_prob'],
                "positive_prob": prob_analysis['positive_prob'],
                "neutral_prob": prob_analysis['neutral_prob'],
                "options": {
                    letter: {
                        "text": variation['options'][letter]['text'],
                        "classification": variation['options'][letter].get('classification'),
                    }
                    for letter in ['A', 'B', 'C', 'D']
                }
            }
            results.append(result)

        # Log summary
        threat_probs = [r['threat_prob'] for r in results if r['condition'] == cond['name']]
        logger.info(f"  Avg P(Threat): {np.mean(threat_probs):.3f}")

    # Clear steering
    steering.clear()

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ambiguous_interpretation_{model_key}_layer{layer}_{timestamp}.jsonl"

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"Saved {len(results)} results to {output_file}")

    # Generate plot
    if model_key in PLOT_CONFIGS:
        logger.info("Generating plot...")
        plot_path = plot_single_model(output_file, model_key, output_dir)
        logger.info(f"Saved plot to {plot_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: Probability by Condition")
    print("=" * 70)

    from collections import defaultdict
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r['condition']].append({
            'threat': r['threat_prob'],
            'neutral': r['neutral_prob'],
            'positive': r['positive_prob'],
        })

    print(f"{'Condition':<25} {'P(Threat)':>12} {'P(Neutral)':>12} {'P(Positive)':>12}")
    print("-" * 65)

    baseline = by_condition['baseline']
    baseline_threat = np.mean([x['threat'] for x in baseline])
    baseline_neutral = np.mean([x['neutral'] for x in baseline])
    baseline_positive = np.mean([x['positive'] for x in baseline])
    print(f"{'baseline':<25} {baseline_threat:>12.3f} {baseline_neutral:>12.3f} {baseline_positive:>12.3f}")

    for cond_name in sorted(by_condition.keys()):
        if cond_name == 'baseline':
            continue
        vals = by_condition[cond_name]
        threat = np.mean([x['threat'] for x in vals])
        neutral = np.mean([x['neutral'] for x in vals])
        positive = np.mean([x['positive'] for x in vals])
        delta = threat - baseline_threat
        print(f"{cond_name:<25} {threat:>12.3f} {neutral:>12.3f} {positive:>12.3f}  (Δ={delta:+.3f})")

    print("=" * 70)

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ambiguous Interpretation Steering Experiment")
    parser.add_argument("--model", type=str, default="gemma", choices=["gemma", "qwen32b", "qwen235b"])
    parser.add_argument("--layer", type=int, default=None, help="Override default layer for steering")
    parser.add_argument("--emotions", type=str, nargs="+", default=["fear", "anger", "sadness", "happiness"])
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=None)
    parser.add_argument("--num-permutations", type=int, default=10, help="Number of permutations per item (max 24)")
    parser.add_argument("--output-dir", type=str, default="experiments/steering/outputs/ambiguous_interpretation")
    parser.add_argument("--positive-only", action="store_true", help="Only test positive steering")
    parser.add_argument("--negative-only", action="store_true", help="Only test negative steering")

    args = parser.parse_args()

    run_experiment(
        model_key=args.model,
        layer_override=args.layer,
        emotions=args.emotions,
        norm_pcts=args.norm_pcts,
        num_permutations=args.num_permutations,
        output_dir=args.output_dir,
        positive_only=args.positive_only,
        negative_only=args.negative_only,
    )
