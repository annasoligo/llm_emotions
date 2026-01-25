"""
Entity-specific emotion steering experiment.

Tests if steering emotions on specific entity tokens affects model preferences
in A/B choice questions.

Example: "Would you rather give £100 to A) Ben or B) Sam?"
- Steer anger on "Ben" token → Does model prefer Sam?
- Steer anger on "Sam" token → Does model prefer Ben?
- Swap A/B positions to control for position bias
"""
import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.steering.config import MODEL_NAME, VECTOR_DIR
from experiments.steering.core import VLLMSteering
from experiments.steering.layer_norms import get_layer_norm
from experiments.behavior_tests.prompts.entity_prompts import (
    ENTITY_PAIRS, ENTITY_SCENARIOS, build_prompt, get_swapped_prompt
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Layer norm loaded from centralized file
DEFAULT_LAYER = 30


def find_entity_positions(
    prompt: str,
    entity: str,
    tokenizer,
    verbose: bool = False
) -> List[int]:
    """
    Find token positions where an entity name appears in the prompt.

    Args:
        prompt: Full prompt string
        entity: Entity name to find (e.g., "Ben")
        tokenizer: HuggingFace tokenizer
        verbose: If True, print debug info

    Returns:
        List of token positions (0-indexed) where entity appears
    """
    # Tokenize the full prompt
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Tokenize the entity (try with and without space prefix)
    entity_ids_no_space = tokenizer.encode(entity, add_special_tokens=False)
    entity_ids_with_space = tokenizer.encode(" " + entity, add_special_tokens=False)

    if verbose:
        logger.info(f"Looking for '{entity}' in prompt")
        logger.info(f"Entity tokens (no space): {tokenizer.convert_ids_to_tokens(entity_ids_no_space)}")
        logger.info(f"Entity tokens (with space): {tokenizer.convert_ids_to_tokens(entity_ids_with_space)}")
        logger.info(f"Full prompt tokens: {tokens}")

    positions = []

    # Try matching with space prefix first (more common in middle of text)
    for search_ids in [entity_ids_with_space, entity_ids_no_space]:
        if not search_ids:
            continue

        for i in range(len(token_ids) - len(search_ids) + 1):
            if token_ids[i:i+len(search_ids)] == search_ids:
                positions.extend(range(i, i + len(search_ids)))
                if verbose:
                    matched_tokens = tokens[i:i+len(search_ids)]
                    logger.info(f"Found match at positions {list(range(i, i + len(search_ids)))}: {matched_tokens}")

    # Remove duplicates and sort
    positions = sorted(set(positions))

    if not positions and verbose:
        logger.warning(f"Could not find '{entity}' in tokenized prompt!")

    return positions


def get_answer_logprobs(output, tokenizer) -> Dict[str, float]:
    """Extract logprobs for A and B tokens from output."""
    if not output.outputs[0].logprobs:
        return {"A": None, "B": None}

    first_token_logprobs = output.outputs[0].logprobs[0]

    a_logprob = None
    b_logprob = None

    for token_id, logprob_obj in first_token_logprobs.items():
        decoded = logprob_obj.decoded_token.strip().upper()
        if decoded == "A" and a_logprob is None:
            a_logprob = logprob_obj.logprob
        elif decoded == "B" and b_logprob is None:
            b_logprob = logprob_obj.logprob

    return {"A": a_logprob, "B": b_logprob}


def logprob_to_prob(logprob: Optional[float]) -> Optional[float]:
    """Convert logprob to probability."""
    if logprob is None:
        return None
    return math.exp(logprob)


def verify_tokenization(tokenizer, num_examples: int = 3):
    """Verify entity token finding works on sample prompts."""
    logger.info("=" * 60)
    logger.info("TOKENIZATION VERIFICATION")
    logger.info("=" * 60)

    from experiments.behavior_tests.prompts.entity_prompts import ENTITY_SCENARIOS, ENTITY_PAIRS

    for i, (scenario, pair) in enumerate(zip(ENTITY_SCENARIOS[:num_examples], ENTITY_PAIRS[:num_examples])):
        prompt = build_prompt(scenario, pair.name_a, pair.name_b)
        logger.info(f"\nExample {i+1}: {scenario.id}")
        logger.info(f"Prompt: {prompt}")

        # Apply chat template
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Find positions
        pos_a = find_entity_positions(formatted, pair.name_a, tokenizer, verbose=True)
        pos_b = find_entity_positions(formatted, pair.name_b, tokenizer, verbose=True)

        logger.info(f"Entity A '{pair.name_a}' at positions: {pos_a}")
        logger.info(f"Entity B '{pair.name_b}' at positions: {pos_b}")

    logger.info("=" * 60)


def run_experiment(
    layer: int,
    norm_pct: float,
    output_dir: Path,
    emotions: List[str] = ["anger", "happiness"],
    num_pairs: int = 8,
    num_scenarios: int = 8,
    reps: int = 1,
    random_control: bool = False,
    verify_only: bool = False,
):
    """
    Run the entity-specific steering experiment.

    Args:
        layer: Layer to apply steering
        norm_pct: Steering magnitude as fraction of layer norm
        output_dir: Where to save results
        emotions: List of emotions to test
        num_pairs: Number of entity pairs to use
        num_scenarios: Number of scenarios to use
        reps: Number of repetitions per condition (for larger sample sizes)
        random_control: If True, include random control vector with same norm
        verify_only: If True, just verify tokenization and exit
    """
    logger.info(f"Entity Steering Experiment")
    logger.info(f"Layer {layer}, norm {norm_pct*100:.0f}%")
    logger.info(f"Emotions: {emotions}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    if verify_only:
        verify_tokenization(tokenizer)
        return None

    # Load model
    # IMPORTANT: Disable prefix caching - position-specific steering requires
    # fresh forward passes for each prompt since KV states depend on steering
    logger.info("Loading model...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        disable_log_stats=True,
        enable_prefix_caching=False,  # Required for position-specific steering
    )

    # Setup steering
    steering = VLLMSteering(llm, layer, baseline_std=1.0)
    steering.load_vectors(VECTOR_DIR)
    logger.info(f"Available vectors: {steering.available_emotions}")

    # Create random control vector if requested
    random_vec_name = None
    if random_control:
        # Get the norm of one of the emotion vectors to match
        sample_emotion = f"{emotions[0]}_textmeandiff"
        sample_vec = steering.vectors.get(sample_emotion)
        if sample_vec is not None:
            import torch
            # Handle both numpy arrays and torch tensors
            if isinstance(sample_vec, np.ndarray):
                sample_vec_np = sample_vec
            else:
                sample_vec_np = sample_vec.cpu().numpy()
            target_norm = np.linalg.norm(sample_vec_np)
            # Create random vector with same norm
            rng = np.random.RandomState(42)  # Fixed seed for reproducibility
            random_vec = rng.randn(sample_vec_np.shape[0]).astype(np.float32)
            random_vec = random_vec / np.linalg.norm(random_vec) * target_norm
            # Register the random vector (keep as numpy, steering will convert as needed)
            random_vec_name = "random_control"
            steering.vectors[random_vec_name] = random_vec
            logger.info(f"Created random control vector with norm {target_norm:.2f}")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=20,
    )

    layer_norm = get_layer_norm("gemma", layer)
    magnitude = norm_pct * layer_norm

    # Get entity pairs and scenarios
    pairs = ENTITY_PAIRS[:num_pairs]
    scenarios = ENTITY_SCENARIOS[:num_scenarios]

    logger.info(f"Using {len(pairs)} entity pairs and {len(scenarios)} scenarios")

    results = []

    # Build all conditions:
    # - baseline (no steering)
    # - For each emotion: steer on entity_a (+/-), steer on entity_b (+/-)
    # - For each: test both A/B orderings

    conditions = [
        {"name": "baseline", "emotion": None, "direction": 0, "target": None}
    ]

    for emotion in emotions:
        vec_name = f"{emotion}_textmeandiff"
        for direction in [1, -1]:
            dir_str = "+" if direction == 1 else "-"
            conditions.append({
                "name": f"{emotion}_{dir_str}_on_A",
                "emotion": vec_name,
                "direction": direction,
                "target": "entity_a"  # Steer on whichever entity is at position A
            })
            conditions.append({
                "name": f"{emotion}_{dir_str}_on_B",
                "emotion": vec_name,
                "direction": direction,
                "target": "entity_b"  # Steer on whichever entity is at position B
            })

    # Add random control conditions if requested
    if random_vec_name:
        for direction in [1, -1]:
            dir_str = "+" if direction == 1 else "-"
            conditions.append({
                "name": f"random_{dir_str}_on_A",
                "emotion": random_vec_name,
                "direction": direction,
                "target": "entity_a"
            })
            conditions.append({
                "name": f"random_{dir_str}_on_B",
                "emotion": random_vec_name,
                "direction": direction,
                "target": "entity_b"
            })

    logger.info(f"Total conditions: {len(conditions)}")
    if reps > 1:
        logger.info(f"Running {reps} repetitions per condition")

    # For each scenario and pair combination (with repetitions)
    for rep in range(reps):
        if reps > 1:
            logger.info(f"\n=== Repetition {rep + 1}/{reps} ===")
        for scenario in scenarios:
            for pair in pairs:
                logger.info(f"\n--- {scenario.id}: {pair.name_a} vs {pair.name_b} ---")

                # Build prompts for both orderings
                prompt_orig = build_prompt(scenario, pair.name_a, pair.name_b)
                prompt_swap, entity_at_A_swap, entity_at_B_swap = get_swapped_prompt(
                    scenario, pair.name_a, pair.name_b
                )

                # Apply chat template
                messages_orig = [{"role": "user", "content": prompt_orig}]
                messages_swap = [{"role": "user", "content": prompt_swap}]

                formatted_orig = tokenizer.apply_chat_template(
                    messages_orig, tokenize=False, add_generation_prompt=True
                )
                formatted_swap = tokenizer.apply_chat_template(
                    messages_swap, tokenize=False, add_generation_prompt=True
                )

                # Find entity positions in each prompt variant
                # Original: A = name_a, B = name_b
                pos_a_orig = find_entity_positions(formatted_orig, pair.name_a, tokenizer)
                pos_b_orig = find_entity_positions(formatted_orig, pair.name_b, tokenizer)

                # Swapped: A = name_b, B = name_a
                pos_a_swap = find_entity_positions(formatted_swap, pair.name_b, tokenizer)  # name_b is now at A
                pos_b_swap = find_entity_positions(formatted_swap, pair.name_a, tokenizer)  # name_a is now at B

                if not pos_a_orig or not pos_b_orig:
                    logger.warning(f"Could not find entity positions in original prompt, skipping")
                    continue
                if not pos_a_swap or not pos_b_swap:
                    logger.warning(f"Could not find entity positions in swapped prompt, skipping")
                    continue

                # Run each condition
                for cond in conditions:
                    # Process both orderings
                    for ordering, formatted, pos_at_A, pos_at_B, entity_at_A, entity_at_B in [
                        ("original", formatted_orig, pos_a_orig, pos_b_orig, pair.name_a, pair.name_b),
                        ("swapped", formatted_swap, pos_a_swap, pos_b_swap, pair.name_b, pair.name_a),
                    ]:
                        # Determine which positions to steer
                        if cond["target"] == "entity_a":
                            steer_positions = pos_at_A
                        elif cond["target"] == "entity_b":
                            steer_positions = pos_at_B
                        else:
                            steer_positions = None  # baseline - no position-specific steering

                        # Set steering
                        if cond["emotion"]:
                            steering.set(
                                cond["emotion"],
                                scale=magnitude,
                                direction=cond["direction"],
                                steer_prompt=True,
                                steer_generation=False,  # Only steer during prefill
                                steer_positions=steer_positions,
                            )
                        else:
                            steering.clear()

                        # Generate
                        outputs = llm.generate([formatted], sampling_params)
                        output = outputs[0]

                        # Extract logprobs
                        logprobs = get_answer_logprobs(output, tokenizer)
                        generated = output.outputs[0].text.strip()

                        result = {
                            "scenario_id": scenario.id,
                            "scenario_category": scenario.category,
                            "entity_pair": f"{pair.name_a}_vs_{pair.name_b}",
                            "entity_at_A": entity_at_A,
                            "entity_at_B": entity_at_B,
                            "ordering": ordering,
                            "condition": cond["name"],
                            "emotion": cond["emotion"],
                            "direction": cond["direction"],
                            "target": cond["target"],
                            "steer_positions": steer_positions,
                            "generated_answer": generated,
                            "logprob_A": logprobs["A"],
                            "logprob_B": logprobs["B"],
                            "prob_A": logprob_to_prob(logprobs["A"]),
                            "prob_B": logprob_to_prob(logprobs["B"]),
                            "rep": rep,
                        }
                        results.append(result)

                logger.info(f"  Processed {len(conditions) * 2} condition-ordering combinations")

    # Clear steering
    steering.clear()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"entity_steering_layer{layer}_{timestamp}.jsonl"

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"\nSaved {len(results)} results to {output_file}")

    # Print summary
    print_summary(results)

    return output_file


def print_summary(results: List[Dict]):
    """Print summary statistics from results."""
    from collections import defaultdict

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)

    # Group by condition
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append(r)

    # Compute average P(A) for each condition
    logger.info(f"\n{'Condition':<25} {'Avg P(A)':<12} {'Avg P(B)':<12} {'N':<6} {'%A Chosen':<10}")
    logger.info("-" * 70)

    for cond in sorted(by_condition.keys()):
        cond_results = by_condition[cond]

        probs_a = [r["prob_A"] for r in cond_results if r["prob_A"] is not None]
        probs_b = [r["prob_B"] for r in cond_results if r["prob_B"] is not None]
        answers = [r["generated_answer"].upper() for r in cond_results]

        avg_a = np.mean(probs_a) if probs_a else 0
        avg_b = np.mean(probs_b) if probs_b else 0
        pct_a = sum(1 for a in answers if a == "A") / len(answers) * 100 if answers else 0

        logger.info(f"{cond:<25} {avg_a:<12.4f} {avg_b:<12.4f} {len(cond_results):<6} {pct_a:<10.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Entity-specific emotion steering experiment")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--norm-pct", type=float, default=0.10,
                        help="Steering magnitude as fraction of layer norm (e.g., 0.10 = 10%%)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/workspace-vast/annas/git/research-tools/experiments/behavior_tests/outputs"))
    parser.add_argument("--emotions", nargs="+", default=["anger", "happiness"],
                        help="Emotions to test")
    parser.add_argument("--num-pairs", type=int, default=8,
                        help="Number of entity pairs to use")
    parser.add_argument("--num-scenarios", type=int, default=8,
                        help="Number of scenarios to use")
    parser.add_argument("--reps", type=int, default=1,
                        help="Number of repetitions per condition (for larger sample sizes)")
    parser.add_argument("--random-control", action="store_true",
                        help="Include random control vector with same norm as emotion vectors")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify tokenization without running experiment")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_experiment(
        layer=args.layer,
        norm_pct=args.norm_pct,
        output_dir=args.output_dir,
        emotions=args.emotions,
        num_pairs=args.num_pairs,
        num_scenarios=args.num_scenarios,
        reps=args.reps,
        random_control=args.random_control,
        verify_only=args.verify_only,
    )


if __name__ == "__main__":
    main()
