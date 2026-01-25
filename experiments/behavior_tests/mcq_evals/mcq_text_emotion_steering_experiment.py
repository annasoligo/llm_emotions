"""
MCQ Text-Based Emotion Steering Experiment

Tests how discrete emotion vectors (fear, happiness, etc.) from text-based mean-diff
affect model responses to multiple-choice questions.

Uses pre-computed vectors from experiments/steering/vectors/*.npz
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
    "google/gemma-3-27b-it": {
        "layer": 30,  # Using layer 30 for text vectors
        "stop_token": "<end_of_turn>",
        "hidden_size": 5376,
        "vector_file": "experiments/steering/vectors/{emotion}_textmeandiff_layer{layer}.npz",
        "vector_format": "single",  # One emotion per file
    },
    "unsloth/gemma-3-27b-it": {
        "layer": 30,
        "stop_token": "<end_of_turn>",
        "hidden_size": 5376,
        "vector_file": "experiments/steering/vectors/{emotion}_textmeandiff_layer{layer}.npz",
        "vector_format": "single",
    },
    "Qwen/Qwen3-32B": {
        "layer": 30,  # Layer 30 for text vectors
        "stop_token": "<|im_end|>",
        "hidden_size": 5120,
        "vector_file": "experiments/steering/vectors/text/qwen32b_layer{layer}.npz",
        "vector_format": "multi",  # All emotions in one file
    },
    "Qwen/Qwen3-235B-A22B": {
        "layer": 50,  # Layer 50 for text vectors (central-ish for 94 layers)
        "stop_token": "<|im_end|>",
        "hidden_size": 4096,
        "vector_file": "experiments/steering/vectors/text/qwen235b_layer{layer}.npz",
        "vector_format": "multi_named",  # Emotions as keys directly
    },
}


def load_text_emotion_vector(
    emotion: str,
    layer: int,
    model_config: dict,
) -> Tuple[np.ndarray, float]:
    """
    Load a pre-computed text-based emotion vector.

    Supports multiple vector file formats:
    - single: One emotion per file (Gemma format)
    - multi: All emotions in one file with 'vectors' and 'emotions' arrays (Qwen 32B)
    - multi_named: Emotions as direct keys in file (Qwen 235B)

    Returns:
        (vector, std) tuple
    """
    vector_file = model_config["vector_file"].format(emotion=emotion, layer=layer)
    vector_format = model_config["vector_format"]
    vector_path = Path(vector_file)

    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")

    data = np.load(vector_path)

    if vector_format == "single":
        # Gemma format: one emotion per file
        vector = data["vector"]
        std = float(data["std"])

    elif vector_format == "multi":
        # Qwen 32B format: all emotions in one file
        emotions = list(data["emotions"])
        if emotion not in emotions:
            raise ValueError(f"Emotion {emotion} not in file. Available: {emotions}")
        idx = emotions.index(emotion)
        vector = data["vectors"][idx]
        std = float(data["projection_stds"][idx])

    elif vector_format == "multi_named":
        # Qwen 235B format: emotions as direct keys
        if emotion not in data:
            raise ValueError(f"Emotion {emotion} not in file. Available: {list(data.keys())}")
        vector = data[emotion]
        std = float(data.get(f"{emotion}_norm", np.linalg.norm(vector)))

    else:
        raise ValueError(f"Unknown vector format: {vector_format}")

    logger.info(f"Loaded {emotion} vector from layer {layer}: "
                f"shape={vector.shape}, std={std:.2f}, norm={np.linalg.norm(vector):.2f}")

    return vector, std


def generate_all_permutations(options: List[dict]) -> List[Tuple[List[dict], Dict[str, str]]]:
    """Generate all 24 permutations of 4 options."""
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
    """Format an MCQ prompt."""
    parts = [scenario_setup, ""]

    if prior_qa:
        for q, a in prior_qa:
            parts.append(q)
            parts.append(a)
            parts.append("")

    parts.append(question)

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
    model_name: str = "unsloth/gemma-3-27b-it",
    prompts_file: str = "experiments/behavior_tests/mcq_evals/prompts_v9.json",
    emotions: List[str] = ["fear", "happiness"],
    layer: Optional[int] = None,
    norm_pcts: List[float] = [0.05, 0.07, 0.10],
    output_dir: str = "experiments/behavior_tests/mcq_evals/outputs",
    stacking_conditions: Optional[List[str]] = None,
    max_scenarios: Optional[int] = None,
    vector_dir: str = "experiments/steering/vectors",
    gpu_memory: float = 0.80,
    max_model_len: int = 4096,
):
    """
    Run the MCQ text-based emotion steering experiment.
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_name]

    if layer is None:
        layer = config["layer"]

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
    logger.info(f"Emotions: {emotions}")
    logger.info(f"Norm percentages: {norm_pcts}")
    logger.info(f"Scenarios: {len(scenarios)}")
    logger.info(f"Stacking conditions: {[s['id'] for s in all_stacking]}")

    # Load emotion vectors
    logger.info("Loading emotion vectors...")
    vectors = {}
    for emotion in emotions:
        vec, std = load_text_emotion_vector(emotion, layer, config)
        vectors[emotion] = vec

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
        gpu_memory_utilization=gpu_memory,
        max_model_len=max_model_len,
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
    conditions = [
        {"name": "neutral", "emotion": None, "sign": 0, "pct": 0},
    ]

    for pct in norm_pcts:
        pct_label = f"{int(pct*100)}pct"
        for emotion in emotions:
            # Positive steering (+emotion)
            conditions.append({
                "name": f"{emotion}_{pct_label}",
                "emotion": emotion,
                "sign": 1,
                "pct": pct,
            })
            # Negative steering (-emotion)
            conditions.append({
                "name": f"neg_{emotion}_{pct_label}",
                "emotion": emotion,
                "sign": -1,
                "pct": pct,
            })

    logger.info(f"Steering conditions: {[c['name'] for c in conditions]}")

    # Prepare output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = model_name.split("/")[-1].lower().replace("-", "_")
    emotions_tag = "_".join(emotions)
    output_file = output_dir / f"mcq_text_{emotions_tag}_{model_tag}_layer{layer}_{timestamp}.jsonl"

    results = []

    # Run experiment
    for cond in conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Condition: {cond['name']}")
        logger.info(f"{'='*60}")

        # Set up steering
        if cond["pct"] > 0 and cond["emotion"]:
            target_norm = cond["pct"] * layer_norm
            vec = vectors[cond["emotion"]]
            # Normalize and scale
            vec_unit = vec / np.linalg.norm(vec)
            scaled_vec = cond["sign"] * vec_unit * target_norm
            logger.info(f"Steering with {cond['sign']:+d}*{cond['emotion']}, norm: {np.linalg.norm(scaled_vec):.2f}")
            steering.set_raw_vector(scaled_vec, steer_prompt=True, steer_generation=True)
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

                if stack_id == "behavior_only":
                    mcq = mcqs["behavior"]
                    options = mcq["options"]
                    question = mcq["question"]

                    permutations = generate_all_permutations(options)

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

                    outputs = llm.generate(batch_prompts, sampling_params)

                    for i, output in enumerate(outputs):
                        meta = batch_meta[i]
                        letter_logprobs = extract_letter_logprobs(output, tokenizer)
                        letter_probs = logprobs_to_probs(letter_logprobs)

                        option_probs = {}
                        for letter, opt_id in meta["letter_mapping"].items():
                            option_probs[opt_id] = letter_probs.get(letter, 0.0)

                        letter_order = [meta["letter_mapping"][l] for l in "ABCD"]

                        result = {
                            "model": model_name,
                            "steering": cond["name"],
                            "emotion": cond["emotion"],
                            "sign": cond["sign"],
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
                    prior_qa = []
                    prior_answers = {}

                    for mcq_idx, mcq_type in enumerate(sequence):
                        mcq = mcqs[mcq_type]
                        options = mcq["options"]
                        question = mcq["question"]

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

                        outputs = llm.generate(batch_prompts, sampling_params)

                        all_letter_logprobs = []
                        for i, output in enumerate(outputs):
                            meta = batch_meta[i]
                            letter_logprobs = extract_letter_logprobs(output, tokenizer)
                            letter_probs = logprobs_to_probs(letter_logprobs)
                            all_letter_logprobs.append(letter_logprobs)

                            option_probs = {}
                            for letter, opt_id in meta["letter_mapping"].items():
                                option_probs[opt_id] = letter_probs.get(letter, 0.0)

                            letter_order = [meta["letter_mapping"][l] for l in "ABCD"]

                            result = {
                                "model": model_name,
                                "steering": cond["name"],
                                "emotion": cond["emotion"],
                                "sign": cond["sign"],
                                "strength": cond["pct"],
                                "scenario": scenario_id,
                                "stacking": stack_id,
                                "mcq_type": mcq_type,
                                "mcq_position": mcq_idx,
                                "perm_idx": meta["perm_idx"],
                                "letter_order": letter_order,
                                "logprobs": letter_logprobs,
                                "letter_probs": letter_probs,
                                "option_probs": option_probs,
                                "prior_answers": dict(prior_answers),
                            }
                            results.append(result)

                        first_perm_logprobs = all_letter_logprobs[0]
                        valid_letters = {k: v for k, v in first_perm_logprobs.items() if v > float('-inf')}
                        if valid_letters:
                            selected_letter = max(valid_letters, key=valid_letters.get)
                        else:
                            selected_letter = "A"

                        prior_qa.append((question + "\n" + "\n".join(
                            f"{chr(65+i)}) {opt['text']}" for i, opt in enumerate(options)
                        ), selected_letter))
                        prior_answers[mcq_type] = selected_letter

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
    print("SUMMARY: Text Emotion Steering Effects on MCQ Responses")
    print("=" * 80)

    # Group by steering condition
    by_steering = defaultdict(list)
    for r in results:
        by_steering[r["steering"]].append(r)

    # For each steering condition, compute P(anxiety-aligned) vs P(contentment-aligned)
    for steering_name in sorted(by_steering.keys()):
        steering_results = by_steering[steering_name]

        anxiety_probs = []
        contentment_probs = []

        for r in steering_results:
            for opt_id, prob in r["option_probs"].items():
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
    parser = argparse.ArgumentParser(description="MCQ Text Emotion Steering Experiment")
    parser.add_argument("--model", type=str, default="unsloth/gemma-3-27b-it",
                        help=f"Model to use. Available: {list(MODEL_CONFIGS.keys())}")
    parser.add_argument("--emotions", type=str, nargs="+", default=["fear", "happiness"],
                        help="Emotions to test")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to steer at (default: model-specific)")
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[0.05, 0.07, 0.10],
                        help="Steering strengths as fraction of layer norm")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/behavior_tests/mcq_evals/outputs")
    parser.add_argument("--prompts", type=str,
                        default="experiments/behavior_tests/mcq_evals/prompts_v9.json")
    parser.add_argument("--stacking", type=str, nargs="+", default=None,
                        help="Stacking conditions to run (default: all)")
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="Limit number of scenarios (for testing)")
    parser.add_argument("--vector-dir", type=str,
                        default="experiments/steering/vectors",
                        help="Directory containing emotion vectors")
    parser.add_argument("--gpu-memory", type=float, default=0.80,
                        help="GPU memory utilization (default: 0.80)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum model sequence length (default: 4096)")

    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        prompts_file=args.prompts,
        emotions=args.emotions,
        layer=args.layer,
        norm_pcts=args.norm_pcts,
        output_dir=args.output_dir,
        stacking_conditions=args.stacking,
        max_scenarios=args.max_scenarios,
        vector_dir=args.vector_dir,
        gpu_memory=args.gpu_memory,
        max_model_len=args.max_model_len,
    )
