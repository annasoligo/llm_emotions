"""
MCQ Emotion Steering Experiment

Tests how emotional steering (valence + uncertainty combinations) affects
model responses to multiple-choice questions.

Steering conditions:
- Neutral: no steering
- Contentment: +valence, -uncertainty (combined)
- Anxiety: -valence, +uncertainty (combined)

Steering strengths: 50%, 75%, 100% of residual stream norm at central layer.
Combined vectors are scaled so their combined norm equals the target percentage.

Method: Single forward pass, extract logprobs for A/B/C/D tokens.

For each MCQ, tests all 24 letter orderings to get position-invariant measurements.
"""

import os
os.environ["VLLM_USE_V1"] = "0"

import argparse
import itertools
import json
import logging
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
from vllm import LLM, SamplingParams

from experiments.steering.core import VLLMSteering
from experiments.steering.layer_norms import get_layer_norm, resolve_model_key

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Model configurations
MODEL_CONFIGS = {
    "Qwen/Qwen3-32B": {
        "layer": 32,  # Central layer for 64-layer model
        "stop_token": "<|im_end|>",
        "appraisal_data": "/workspace-vast/annas/appraisal_data/qwen32b/activations.h5",
        "appraisal_metadata": "/workspace-vast/annas/appraisal_data/qwen32b/activation_metadata.json",
    },
    "Qwen/Qwen3-235B-A22B": {
        "layer": 47,  # Central layer for 94-layer model
        "stop_token": "<|im_end|>",
        "appraisal_data": "/workspace-vast/annas/appraisal_data/qwen235b/activations.h5",
        "appraisal_metadata": "/workspace-vast/annas/appraisal_data/qwen235b/activation_metadata.json",
    },
    "google/gemma-3-27b-it": {
        "layer": 31,  # Central layer for 62-layer model
        "stop_token": "<end_of_turn>",
        "appraisal_data": "/workspace-vast/annas/appraisal_data/full_run/activations.h5",
        "appraisal_metadata": "/workspace-vast/annas/appraisal_data/full_run/activation_metadata.json",
    },
    "unsloth/gemma-3-27b-it": {
        "layer": 31,  # Central layer for 62-layer model
        "stop_token": "<end_of_turn>",
        "appraisal_data": "/workspace-vast/annas/appraisal_data/full_run/activations.h5",
        "appraisal_metadata": "/workspace-vast/annas/appraisal_data/full_run/activation_metadata.json",
    },
}

# Flip uncertainty so positive = high uncertainty
FLIP_AXES = {"uncertainty"}


def load_appraisal_vectors(
    activations_path: str,
    metadata_path: str,
    layer: int,
    axes: List[str] = ["valence", "uncertainty"],
) -> Dict[str, np.ndarray]:
    """
    Load appraisal steering vectors from HDF5 activations.

    Computes mean difference vectors: mean(variant_a) - mean(variant_b)
    where a = positive/high, b = negative/low.
    """
    with open(metadata_path) as f:
        metadata = json.load(f)

    axis_activations = defaultdict(lambda: {"a": [], "b": []})

    with h5py.File(activations_path, "r") as f:
        acts_group = f["activations"]

        for item in metadata["items"]:
            item_id = item["id"]
            axis_name = item.get("axis_name")
            variant = item.get("variant")

            if axis_name in axes and variant and item_id in acts_group:
                act = acts_group[item_id]["assistant_start_last_token"][layer, :]
                axis_activations[axis_name][variant].append(np.array(act))

    vectors = {}
    for axis_name in axes:
        variants = axis_activations[axis_name]
        if variants["a"] and variants["b"]:
            mean_a = np.mean(variants["a"], axis=0)
            mean_b = np.mean(variants["b"], axis=0)
            vec = mean_a - mean_b

            # Flip if needed (uncertainty: we want + to mean high uncertainty)
            if axis_name in FLIP_AXES:
                vec = -vec
                logger.info(f"{axis_name}: FLIPPED (so + = high {axis_name})")

            vectors[axis_name] = vec
            logger.info(
                f"{axis_name}: {len(variants['a'])} A, {len(variants['b'])} B samples, "
                f"raw norm={np.linalg.norm(vec):.2f}"
            )

    return vectors


def combine_vectors(
    valence_vec: np.ndarray,
    uncertainty_vec: np.ndarray,
    valence_sign: int,
    uncertainty_sign: int,
    target_norm: float,
) -> np.ndarray:
    """
    Combine valence and uncertainty vectors with specified signs,
    scaling so combined norm equals target_norm.

    Args:
        valence_vec: Unit or raw valence vector
        uncertainty_vec: Unit or raw uncertainty vector
        valence_sign: +1 or -1
        uncertainty_sign: +1 or -1
        target_norm: Desired L2 norm of combined vector

    Returns:
        Combined vector with norm = target_norm
    """
    # Normalize individual vectors first
    v_unit = valence_vec / np.linalg.norm(valence_vec)
    u_unit = uncertainty_vec / np.linalg.norm(uncertainty_vec)

    # Combine with equal weights initially
    combined = valence_sign * v_unit + uncertainty_sign * u_unit

    # Scale to target norm
    current_norm = np.linalg.norm(combined)
    if current_norm > 0:
        combined = combined * (target_norm / current_norm)

    return combined


def generate_all_permutations(options: List[dict]) -> List[Tuple[List[dict], Dict[str, str]]]:
    """
    Generate all 24 permutations of 4 options.

    Returns list of (ordered_options, letter_to_option_id mapping) tuples.
    """
    letters = ["A", "B", "C", "D"]
    permutations = []

    for perm in itertools.permutations(options):
        ordered = list(perm)
        mapping = {letters[i]: opt["id"] for i, opt in enumerate(ordered)}
        permutations.append((ordered, mapping))

    return permutations


def format_mcq_prompt(
    scenario_setup: str,
    question: str,
    options: List[dict],
    prior_qa: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """
    Format an MCQ prompt.

    Args:
        scenario_setup: The scenario context
        question: The MCQ question
        options: List of option dicts with 'text' keys (already ordered)
        prior_qa: Optional list of (question, answer) tuples for stacking

    Returns:
        Formatted prompt string
    """
    parts = [scenario_setup, ""]

    # Add prior Q&A for stacking
    if prior_qa:
        for q, a in prior_qa:
            parts.append(q)
            parts.append(a)
            parts.append("")

    # Add current question
    parts.append(question)

    # Add options
    letters = ["A", "B", "C", "D"]
    for i, opt in enumerate(options):
        parts.append(f"{letters[i]}) {opt['text']}")

    parts.append("")
    parts.append("Give me a single letter, nothing else.")

    return "\n".join(parts)


def extract_letter_logprobs(
    output,
    tokenizer,
    letters: List[str] = ["A", "B", "C", "D"],
) -> Dict[str, float]:
    """Extract logprobs for letter tokens from vLLM output."""
    letter_logprobs = {}

    if not output.outputs or not output.outputs[0].logprobs:
        return {l: float('-inf') for l in letters}

    first_logprobs = output.outputs[0].logprobs[0]

    for letter in letters:
        lower = letter.lower()
        # Try various token formats
        for variant in [letter, f' {letter}', f'{letter})', f' {letter})',
                        lower, f' {lower}', f'{lower})', f' {lower})']:
            token_ids = tokenizer.encode(variant, add_special_tokens=False)
            if token_ids:
                token_id = token_ids[-1]
                if token_id in first_logprobs:
                    letter_logprobs[letter] = first_logprobs[token_id].logprob
                    break

        if letter not in letter_logprobs:
            letter_logprobs[letter] = float('-inf')

    return letter_logprobs


def logprobs_to_probs(logprobs: Dict[str, float]) -> Dict[str, float]:
    """Convert logprobs to normalized probabilities."""
    valid = {k: v for k, v in logprobs.items() if v > float('-inf')}
    if not valid:
        return {k: 0.0 for k in logprobs}

    max_lp = max(valid.values())
    exp_scores = {k: np.exp(v - max_lp) for k, v in valid.items()}
    total = sum(exp_scores.values())

    probs = {}
    for k in logprobs:
        if k in exp_scores:
            probs[k] = exp_scores[k] / total
        else:
            probs[k] = 0.0

    return probs


def run_experiment(
    model_name: str = "Qwen/Qwen3-32B",
    prompts_file: str = "experiments/behavior_tests/mcq_evals/prompts_v1.json",
    layer: Optional[int] = None,
    norm_pcts: List[float] = [0.50, 0.75, 1.00],
    output_dir: str = "experiments/behavior_tests/mcq_evals/outputs",
    stacking_conditions: Optional[List[str]] = None,
    max_scenarios: Optional[int] = None,
):
    """
    Run the MCQ emotion steering experiment.

    Args:
        model_name: Model to use
        prompts_file: Path to prompts JSON
        layer: Layer to steer at (default: model-specific central layer)
        norm_pcts: List of steering strengths as fraction of layer norm
        output_dir: Output directory
        stacking_conditions: Which stacking conditions to run (default: all)
        max_scenarios: Limit number of scenarios (for testing)
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]

    if layer is None:
        layer = config["layer"]

    if config["appraisal_data"] is None:
        raise ValueError(f"No appraisal data available for {model_name}")

    # Load prompts
    prompts_path = Path(prompts_file)
    with open(prompts_path) as f:
        prompts_data = json.load(f)

    scenarios = prompts_data["scenarios"]
    if max_scenarios:
        scenarios = scenarios[:max_scenarios]

    all_stacking = prompts_data["stacking_conditions"]
    if stacking_conditions:
        all_stacking = [s for s in all_stacking if s["id"] in stacking_conditions]

    logger.info(f"Model: {model_name}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Norm percentages: {norm_pcts}")
    logger.info(f"Scenarios: {len(scenarios)}")
    logger.info(f"Stacking conditions: {[s['id'] for s in all_stacking]}")

    # Load steering vectors
    logger.info("Loading appraisal vectors...")
    vectors = load_appraisal_vectors(
        config["appraisal_data"],
        config["appraisal_metadata"],
        layer,
        axes=["valence", "uncertainty"],
    )

    if "valence" not in vectors or "uncertainty" not in vectors:
        raise ValueError(f"Missing vectors. Available: {list(vectors.keys())}")

    # Get layer norm
    model_key = resolve_model_key(model_name)
    layer_norm = get_layer_norm(model_key, layer)
    logger.info(f"Layer {layer} norm: {layer_norm:.2f}")

    # Initialize model
    logger.info("Loading model...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        max_model_len=4096,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    steering = VLLMSteering(llm, layer)

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=20,
    )

    # Build steering conditions
    # Contentment: +valence, -uncertainty
    # Anxiety: -valence, +uncertainty
    conditions = [
        {"name": "neutral", "valence_sign": 0, "uncertainty_sign": 0, "pct": 0},
    ]

    for pct in norm_pcts:
        pct_label = f"{int(pct*100)}pct"
        conditions.extend([
            {
                "name": f"contentment_{pct_label}",
                "valence_sign": 1,
                "uncertainty_sign": -1,
                "pct": pct,
            },
            {
                "name": f"anxiety_{pct_label}",
                "valence_sign": -1,
                "uncertainty_sign": 1,
                "pct": pct,
            },
        ])

    logger.info(f"Steering conditions: {[c['name'] for c in conditions]}")

    # Prepare output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = model_name.split("/")[-1].lower().replace("-", "_")
    output_file = output_dir / f"mcq_steering_{model_tag}_layer{layer}_{timestamp}.jsonl"

    results = []

    # Run experiment
    for cond in conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Condition: {cond['name']}")
        logger.info(f"{'='*60}")

        # Set up steering
        if cond["pct"] > 0:
            target_norm = cond["pct"] * layer_norm
            combined_vec = combine_vectors(
                vectors["valence"],
                vectors["uncertainty"],
                cond["valence_sign"],
                cond["uncertainty_sign"],
                target_norm,
            )
            logger.info(f"Combined vector norm: {np.linalg.norm(combined_vec):.2f} (target: {target_norm:.2f})")
            steering.set_raw_vector(combined_vec, steer_prompt=True, steer_generation=True)
        else:
            steering.clear()

        # Process each scenario
        for scenario in scenarios:
            scenario_id = scenario["id"]
            setup = scenario["setup"]
            mcqs = scenario["mcqs"]

            logger.info(f"  Scenario: {scenario_id}")

            # Process each stacking condition
            for stack in all_stacking:
                stack_id = stack["id"]
                sequence = stack["sequence"]

                # For behavior_only, just run the behavior MCQ
                # For stacked, we need to chain MCQs

                if stack_id == "behavior_only":
                    # Single MCQ: behavior
                    mcq = mcqs["behavior"]
                    options = mcq["options"]
                    question = mcq["question"]

                    # Generate all 24 permutations
                    permutations = generate_all_permutations(options)

                    # Batch all permutations
                    batch_prompts = []
                    batch_meta = []

                    for perm_idx, (ordered_opts, letter_mapping) in enumerate(permutations):
                        prompt_text = format_mcq_prompt(setup, question, ordered_opts)
                        messages = [{"role": "user", "content": prompt_text}]
                        chat_prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        batch_prompts.append(chat_prompt)
                        batch_meta.append({
                            "perm_idx": perm_idx,
                            "letter_mapping": letter_mapping,
                            "ordered_opts": ordered_opts,
                        })

                    # Run batch
                    outputs = llm.generate(batch_prompts, sampling_params)

                    # Process results
                    for i, output in enumerate(outputs):
                        meta = batch_meta[i]
                        letter_logprobs = extract_letter_logprobs(output, tokenizer)
                        letter_probs = logprobs_to_probs(letter_logprobs)

                        # Map letter probs to option probs
                        option_probs = {}
                        for letter, opt_id in meta["letter_mapping"].items():
                            option_probs[opt_id] = letter_probs.get(letter, 0.0)

                        # Get letter order (option IDs in A,B,C,D order)
                        letter_order = [meta["letter_mapping"][l] for l in "ABCD"]

                        result = {
                            "model": model_name,
                            "steering": cond["name"],
                            "strength": cond["pct"],
                            "scenario": scenario_id,
                            "stacking": stack_id,
                            "mcq_type": "behavior",
                            "perm_idx": meta["perm_idx"],
                            "letter_order": letter_order,
                            "logprobs": letter_logprobs,
                            "letter_probs": letter_probs,
                            "option_probs": option_probs,
                            "prior_answers": {},
                        }
                        results.append(result)

                else:
                    # Stacked condition - run each MCQ in sequence with full permutations
                    # Record results for ALL MCQ types, not just the final one

                    prior_qa = []
                    prior_answers = {}

                    for mcq_idx, mcq_type in enumerate(sequence):
                        mcq = mcqs[mcq_type]
                        options = mcq["options"]
                        question = mcq["question"]

                        # Generate all 24 permutations for this MCQ
                        permutations = generate_all_permutations(options)

                        batch_prompts = []
                        batch_meta = []

                        for perm_idx, (ordered_opts, letter_mapping) in enumerate(permutations):
                            prompt_text = format_mcq_prompt(setup, question, ordered_opts, prior_qa)
                            messages = [{"role": "user", "content": prompt_text}]
                            chat_prompt = tokenizer.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=False,
                            )
                            batch_prompts.append(chat_prompt)
                            batch_meta.append({
                                "perm_idx": perm_idx,
                                "letter_mapping": letter_mapping,
                                "ordered_opts": ordered_opts,
                            })

                        # Run batch
                        outputs = llm.generate(batch_prompts, sampling_params)

                        # Collect results for this MCQ type
                        all_letter_logprobs = []
                        for i, output in enumerate(outputs):
                            meta = batch_meta[i]
                            letter_logprobs = extract_letter_logprobs(output, tokenizer)
                            letter_probs = logprobs_to_probs(letter_logprobs)
                            all_letter_logprobs.append(letter_logprobs)

                            # Map letter probs to option probs
                            option_probs = {}
                            for letter, opt_id in meta["letter_mapping"].items():
                                option_probs[opt_id] = letter_probs.get(letter, 0.0)

                            letter_order = [meta["letter_mapping"][l] for l in "ABCD"]

                            result = {
                                "model": model_name,
                                "steering": cond["name"],
                                "strength": cond["pct"],
                                "scenario": scenario_id,
                                "stacking": stack_id,
                                "mcq_type": mcq_type,
                                "mcq_position": mcq_idx,  # Position in stack (0=first, etc)
                                "perm_idx": meta["perm_idx"],
                                "letter_order": letter_order,
                                "logprobs": letter_logprobs,
                                "letter_probs": letter_probs,
                                "option_probs": option_probs,
                                "prior_answers": dict(prior_answers),  # Copy current state
                            }
                            results.append(result)

                        # Get the most likely answer (from permutation 0, original order) for next MCQ context
                        first_perm_logprobs = all_letter_logprobs[0]
                        valid_letters = {k: v for k, v in first_perm_logprobs.items() if v > float('-inf')}
                        if valid_letters:
                            selected_letter = max(valid_letters, key=valid_letters.get)
                        else:
                            selected_letter = "A"  # Fallback

                        # Add to prior Q&A for next MCQ in sequence
                        prior_qa.append((question + "\n" + "\n".join(
                            f"{chr(65+i)}) {opt['text']}" for i, opt in enumerate(options)
                        ), selected_letter))
                        prior_answers[mcq_type] = selected_letter

        # Log condition summary
        cond_results = [r for r in results if r["steering"] == cond["name"]]
        if cond_results:
            # Aggregate option probs across permutations
            by_option = defaultdict(list)
            for r in cond_results:
                for opt_id, prob in r["option_probs"].items():
                    by_option[opt_id].append(prob)

            logger.info(f"  Condition {cond['name']} summary:")
            for opt_id in sorted(by_option.keys()):
                mean_prob = np.mean(by_option[opt_id])
                logger.info(f"    {opt_id}: {mean_prob:.3f}")

    # Save results
    steering.clear()

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"\nSaved {len(results)} results to {output_file}")

    # Print summary analysis
    print_summary(results, prompts_data)

    return output_file


def print_summary(results: List[dict], prompts_data: dict):
    """Print summary analysis of results."""
    print("\n" + "=" * 80)
    print("SUMMARY: Steering Effects on MCQ Responses")
    print("=" * 80)

    # Group by steering condition
    by_steering = defaultdict(list)
    for r in results:
        by_steering[r["steering"]].append(r)

    # Get predictions from prompts
    analysis = prompts_data.get("analysis_predictions", {})

    # For each steering condition, compute P(anxiety-aligned) vs P(contentment-aligned)
    for steering_name in sorted(by_steering.keys()):
        steering_results = by_steering[steering_name]

        anxiety_probs = []
        contentment_probs = []

        for r in steering_results:
            for opt_id, prob in r["option_probs"].items():
                # Check prediction for this option
                for scenario in prompts_data["scenarios"]:
                    if scenario["id"] == r["scenario"]:
                        mcq = scenario["mcqs"].get(r["mcq_type"], {})
                        for opt in mcq.get("options", []):
                            if opt["id"] == opt_id:
                                pred = opt.get("prediction", "neutral")
                                if pred == "anxiety":
                                    anxiety_probs.append(prob)
                                elif pred == "contentment":
                                    contentment_probs.append(prob)
                                break

        mean_anxiety = np.mean(anxiety_probs) if anxiety_probs else 0
        mean_contentment = np.mean(contentment_probs) if contentment_probs else 0

        print(f"\n{steering_name}:")
        print(f"  P(anxiety-aligned):     {mean_anxiety:.3f}")
        print(f"  P(contentment-aligned): {mean_contentment:.3f}")
        print(f"  Difference:             {mean_contentment - mean_anxiety:+.3f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCQ Emotion Steering Experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B",
                        help=f"Model to use. Available: {list(MODEL_CONFIGS.keys())}")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to steer at (default: model-specific)")
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[0.50, 0.75, 1.00],
                        help="Steering strengths as fraction of layer norm")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/behavior_tests/mcq_evals/outputs")
    parser.add_argument("--prompts", type=str,
                        default="experiments/behavior_tests/mcq_evals/prompts_v1.json")
    parser.add_argument("--stacking", type=str, nargs="+", default=None,
                        help="Stacking conditions to run (default: all)")
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="Limit number of scenarios (for testing)")

    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        prompts_file=args.prompts,
        layer=args.layer,
        norm_pcts=args.norm_pcts,
        output_dir=args.output_dir,
        stacking_conditions=args.stacking,
        max_scenarios=args.max_scenarios,
    )
