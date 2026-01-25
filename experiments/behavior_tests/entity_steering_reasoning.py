"""
Entity-specific emotion steering experiment with reasoning.

Tests if steering emotions on specific entity tokens affects model preferences
using chain-of-thought reasoning and direct entity name output (no A/B framing).
"""
import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.steering.config import MODEL_NAME, VECTOR_DIR
from experiments.steering.core import VLLMSteering
from experiments.steering.layer_norms import get_layer_norm
from experiments.behavior_tests.prompts.entity_prompts_reasoning import (
    ENTITY_PAIRS, ENTITY_SCENARIOS_REASONING, build_prompt, get_swapped_prompt
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
    """
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    entity_ids_no_space = tokenizer.encode(entity, add_special_tokens=False)
    entity_ids_with_space = tokenizer.encode(" " + entity, add_special_tokens=False)

    if verbose:
        logger.info(f"Looking for '{entity}' in prompt")
        logger.info(f"Entity tokens (no space): {tokenizer.convert_ids_to_tokens(entity_ids_no_space)}")
        logger.info(f"Entity tokens (with space): {tokenizer.convert_ids_to_tokens(entity_ids_with_space)}")

    positions = []

    for search_ids in [entity_ids_with_space, entity_ids_no_space]:
        if not search_ids:
            continue

        for i in range(len(token_ids) - len(search_ids) + 1):
            if token_ids[i:i+len(search_ids)] == search_ids:
                positions.extend(range(i, i + len(search_ids)))
                if verbose:
                    matched_tokens = tokens[i:i+len(search_ids)]
                    logger.info(f"Found match at positions {list(range(i, i + len(search_ids)))}: {matched_tokens}")

    positions = sorted(set(positions))
    return positions


def get_entity_logprobs_from_output(output, first_entity: str, second_entity: str) -> Dict[str, Optional[float]]:
    """
    Extract logprobs for entity names from the last token of the output.

    Returns dict with logprobs for first_entity and second_entity.
    """
    result = {"first": None, "second": None}

    if not output.outputs[0].logprobs:
        return result

    # Get the last token's logprobs
    last_logprobs = output.outputs[0].logprobs[-1]

    # Search for entity names in the logprobs
    for token_id, logprob_obj in last_logprobs.items():
        decoded = logprob_obj.decoded_token.strip()
        if decoded == first_entity or decoded.lower() == first_entity.lower():
            result["first"] = logprob_obj.logprob
        elif decoded == second_entity or decoded.lower() == second_entity.lower():
            result["second"] = logprob_obj.logprob

    return result


def parse_chosen_entity(response: str, entity_a: str, entity_b: str) -> Optional[str]:
    """
    Parse the response to determine which entity was chosen.

    Looks for the entity name in the last line or last few words.
    """
    # Clean up response
    response = response.strip()

    # Get last line (where the final answer should be)
    lines = response.split('\n')
    last_line = lines[-1].strip() if lines else ""

    # Check last line first
    if entity_a.lower() in last_line.lower() and entity_b.lower() not in last_line.lower():
        return entity_a
    if entity_b.lower() in last_line.lower() and entity_a.lower() not in last_line.lower():
        return entity_b

    # If both or neither in last line, check last few words
    words = response.split()
    last_words = " ".join(words[-5:]).lower() if len(words) >= 5 else response.lower()

    # Find last occurrence of each entity
    last_a = last_words.rfind(entity_a.lower())
    last_b = last_words.rfind(entity_b.lower())

    if last_a > last_b and last_a >= 0:
        return entity_a
    if last_b > last_a and last_b >= 0:
        return entity_b

    # Fallback: check full response for any mention
    if entity_a.lower() in response.lower() and entity_b.lower() not in response.lower():
        return entity_a
    if entity_b.lower() in response.lower() and entity_a.lower() not in response.lower():
        return entity_b

    return None  # Ambiguous or neither mentioned


def run_experiment(
    layer: int,
    norm_pct: float,
    output_dir: Path,
    emotions: List[str] = ["anger", "happiness"],
    num_pairs: int = 8,
    num_scenarios: int = 8,
    reps: int = 1,
    random_control: bool = False,
    max_tokens: int = 150,
):
    """
    Run the entity-specific steering experiment with reasoning.
    """
    logger.info(f"Entity Steering Experiment (Reasoning Variant)")
    logger.info(f"Layer {layer}, norm {norm_pct*100:.0f}%")
    logger.info(f"Emotions: {emotions}")
    logger.info(f"Max tokens for reasoning: {max_tokens}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load model
    logger.info("Loading model...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        disable_log_stats=True,
        enable_prefix_caching=False,
    )

    # Setup steering
    steering = VLLMSteering(llm, layer, baseline_std=1.0)
    steering.load_vectors(VECTOR_DIR)
    logger.info(f"Available vectors: {steering.available_emotions}")

    # Create random control vector if requested
    random_vec_name = None
    if random_control:
        sample_emotion = f"{emotions[0]}_textmeandiff"
        sample_vec = steering.vectors.get(sample_emotion)
        if sample_vec is not None:
            if isinstance(sample_vec, np.ndarray):
                sample_vec_np = sample_vec
            else:
                sample_vec_np = sample_vec.cpu().numpy()
            target_norm = np.linalg.norm(sample_vec_np)
            rng = np.random.RandomState(42)
            random_vec = rng.randn(sample_vec_np.shape[0]).astype(np.float32)
            random_vec = random_vec / np.linalg.norm(random_vec) * target_norm
            random_vec_name = "random_control"
            steering.vectors[random_vec_name] = random_vec
            logger.info(f"Created random control vector with norm {target_norm:.2f}")

    # Sampling params - allow reasoning, get logprobs for final token analysis
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=20,  # Get top 20 logprobs to find entity names
    )

    layer_norm = get_layer_norm("gemma", layer)
    magnitude = norm_pct * layer_norm

    # Get entity pairs and scenarios
    pairs = ENTITY_PAIRS[:num_pairs]
    scenarios = ENTITY_SCENARIOS_REASONING[:num_scenarios]

    logger.info(f"Using {len(pairs)} entity pairs and {len(scenarios)} scenarios")

    results = []

    # Build conditions
    conditions = [
        {"name": "baseline", "emotion": None, "direction": 0, "target": None}
    ]

    for emotion in emotions:
        vec_name = f"{emotion}_textmeandiff"
        for direction in [1, -1]:
            dir_str = "+" if direction == 1 else "-"
            conditions.append({
                "name": f"{emotion}_{dir_str}_on_first",
                "emotion": vec_name,
                "direction": direction,
                "target": "first_entity"  # Steer on first-mentioned entity
            })
            conditions.append({
                "name": f"{emotion}_{dir_str}_on_second",
                "emotion": vec_name,
                "direction": direction,
                "target": "second_entity"  # Steer on second-mentioned entity
            })

    if random_vec_name:
        for direction in [1, -1]:
            dir_str = "+" if direction == 1 else "-"
            conditions.append({
                "name": f"random_{dir_str}_on_first",
                "emotion": random_vec_name,
                "direction": direction,
                "target": "first_entity"
            })
            conditions.append({
                "name": f"random_{dir_str}_on_second",
                "emotion": random_vec_name,
                "direction": direction,
                "target": "second_entity"
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
                prompt_swap, first_swap, second_swap = get_swapped_prompt(
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
                # Original: first = name_a, second = name_b
                pos_first_orig = find_entity_positions(formatted_orig, pair.name_a, tokenizer)
                pos_second_orig = find_entity_positions(formatted_orig, pair.name_b, tokenizer)

                # Swapped: first = name_b, second = name_a
                pos_first_swap = find_entity_positions(formatted_swap, pair.name_b, tokenizer)
                pos_second_swap = find_entity_positions(formatted_swap, pair.name_a, tokenizer)

                if not pos_first_orig or not pos_second_orig:
                    logger.warning(f"Could not find entity positions in original prompt, skipping")
                    continue
                if not pos_first_swap or not pos_second_swap:
                    logger.warning(f"Could not find entity positions in swapped prompt, skipping")
                    continue

                # Run each condition
                for cond in conditions:
                    # Process both orderings
                    for ordering, formatted, pos_first, pos_second, first_entity, second_entity in [
                        ("original", formatted_orig, pos_first_orig, pos_second_orig, pair.name_a, pair.name_b),
                        ("swapped", formatted_swap, pos_first_swap, pos_second_swap, pair.name_b, pair.name_a),
                    ]:
                        # Determine which positions to steer
                        if cond["target"] == "first_entity":
                            steer_positions = pos_first
                        elif cond["target"] == "second_entity":
                            steer_positions = pos_second
                        else:
                            steer_positions = None

                        # Set steering
                        if cond["emotion"]:
                            steering.set(
                                cond["emotion"],
                                scale=magnitude,
                                direction=cond["direction"],
                                steer_prompt=True,
                                steer_generation=False,
                                steer_positions=steer_positions,
                            )
                        else:
                            steering.clear()

                        # Generate
                        outputs = llm.generate([formatted], sampling_params)
                        output = outputs[0]
                        generated = output.outputs[0].text.strip()

                        # Parse which entity was chosen
                        chosen = parse_chosen_entity(generated, first_entity, second_entity)

                        # Get logprobs for entity names from last token
                        entity_logprobs = get_entity_logprobs_from_output(output, first_entity, second_entity)

                        result = {
                            "scenario_id": scenario.id,
                            "scenario_category": scenario.category,
                            "entity_pair": f"{pair.name_a}_vs_{pair.name_b}",
                            "first_entity": first_entity,
                            "second_entity": second_entity,
                            "ordering": ordering,
                            "condition": cond["name"],
                            "emotion": cond["emotion"],
                            "direction": cond["direction"],
                            "target": cond["target"],
                            "steer_positions": steer_positions,
                            "generated_response": generated,
                            "chosen_entity": chosen,
                            "chose_first": chosen == first_entity if chosen else None,
                            "chose_steered": (
                                (chosen == first_entity and cond["target"] == "first_entity") or
                                (chosen == second_entity and cond["target"] == "second_entity")
                            ) if chosen and cond["target"] else None,
                            "logprob_first": entity_logprobs["first"],
                            "logprob_second": entity_logprobs["second"],
                            "rep": rep,
                        }
                        results.append(result)

                logger.info(f"  Processed {len(conditions) * 2} condition-ordering combinations")

    # Clear steering
    steering.clear()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"entity_steering_reasoning_layer{layer}_{timestamp}.jsonl"

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

    # Compute stats for each condition
    logger.info(f"\n{'Condition':<25} {'N':<6} {'%First':<10} {'%Steered':<10} {'Valid':<6}")
    logger.info("-" * 70)

    for cond in sorted(by_condition.keys()):
        cond_results = by_condition[cond]

        valid = [r for r in cond_results if r["chosen_entity"] is not None]
        chose_first = sum(1 for r in valid if r["chose_first"])
        chose_steered = sum(1 for r in valid if r["chose_steered"]) if cond != "baseline" else 0

        pct_first = chose_first / len(valid) * 100 if valid else 0
        pct_steered = chose_steered / len(valid) * 100 if valid and cond != "baseline" else 0

        logger.info(f"{cond:<25} {len(cond_results):<6} {pct_first:<10.1f} {pct_steered:<10.1f} {len(valid):<6}")


def main():
    parser = argparse.ArgumentParser(description="Entity-specific emotion steering with reasoning")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--norm-pct", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("/workspace-vast/annas/git/research-tools/experiments/behavior_tests/outputs"))
    parser.add_argument("--emotions", nargs="+", default=["anger", "happiness"],
                        help="Emotions to test")
    parser.add_argument("--num-pairs", type=int, default=8)
    parser.add_argument("--num-scenarios", type=int, default=8)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--random-control", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Max tokens for model reasoning")
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
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
