"""
Priority Selection Steering with Appraisal Axes

Tests whether appraisal axis steering (valence, uncertainty, agency) affects
which category of information the model considers "most important".

Each stimulus has 12 statements (3 per emotion category):
- FEAR: Uncertainty, uncontrollable threats
- ANGER: Blame, negligence, intentional wrongs
- SADNESS: Loss, irreversibility, missed opportunities
- HAPPINESS: Gains, improvements, opportunities

Uses logprobs on letters A-L to measure probability distribution
across emotion categories under different appraisal steering conditions.

Key hypotheses:
- Valence+: should increase P(happiness), decrease P(negative emotions)
- Valence-: should increase P(fear/anger/sadness), decrease P(happiness)
- Agency+: may shift toward anger (blame/agency) vs fear (uncontrollable)
- Uncertainty+: may shift toward fear (uncertainty) vs other categories
"""

import os
os.environ["VLLM_USE_V1"] = "0"

import argparse
import json
import logging
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import h5py
from vllm import LLM, SamplingParams

from experiments.steering.core import VLLMSteering
from experiments.steering.layer_norms import get_layer_norm, resolve_model_key
from experiments.behavior_tests.prompts.priority_selection import (
    ALL_STIMULI,
    SelectionStimulus,
    EmotionCategory,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Layer norms are loaded dynamically from experiments/steering/layer_norms.json
MODEL_CONFIGS = {
    "google/gemma-3-27b-it": {
        "layer": 30,
        "stop_token": "<end_of_turn>",
    },
    "Qwen/Qwen3-32B": {
        "layer": 30,
        "stop_token": "<|im_end|>",
    },
    "Qwen/Qwen3-235B-A22B": {
        "layer": 50,
        "stop_token": "<|im_end|}",
    },
}

DEFAULT_MODEL = "google/gemma-3-27b-it"

FLIP_AXES = {"uncertainty"}
ORTHO_ORDER = ["valence", "uncertainty", "agency"]


def load_appraisal_vectors(
    activations_path: str,
    metadata_path: str,
    layer: int = 30,
    orthogonalize: bool = True,
) -> dict:
    """Load appraisal steering vectors from HDF5 activations."""

    with open(metadata_path) as f:
        metadata = json.load(f)

    axis_activations = defaultdict(lambda: {"a": [], "b": []})

    with h5py.File(activations_path, "r") as f:
        acts_group = f["activations"]

        for item in metadata["items"]:
            item_id = item["id"]
            axis_name = item.get("axis_name")
            variant = item.get("variant")

            if axis_name and variant and item_id in acts_group:
                act = acts_group[item_id]["assistant_start_last_token"][layer, :]
                axis_activations[axis_name][variant].append(act)

    vectors = {}
    for axis_name, variants in axis_activations.items():
        if variants["a"] and variants["b"]:
            mean_a = np.mean(variants["a"], axis=0)
            mean_b = np.mean(variants["b"], axis=0)
            vec = mean_a - mean_b

            if axis_name in FLIP_AXES:
                vec = -vec
                logger.info(f"{axis_name}: FLIPPED (A=low, B=high)")

            vectors[axis_name] = vec
            logger.info(f"{axis_name}: {len(variants['a'])} A, {len(variants['b'])} B samples, norm={np.linalg.norm(vec):.2f}")

    if orthogonalize:
        logger.info("Orthogonalizing vectors...")
        vectors = orthogonalize_vectors(vectors, ORTHO_ORDER)
        for axis in ORTHO_ORDER:
            if axis in vectors:
                logger.info(f"  {axis} orthogonalized norm: {np.linalg.norm(vectors[axis]):.2f}")

    return vectors


def orthogonalize_vectors(vectors: dict, order: list) -> dict:
    """Gram-Schmidt orthogonalization."""
    orthogonal = {}

    for axis in order:
        if axis not in vectors:
            continue

        vec = vectors[axis].copy()

        for prev_axis in order:
            if prev_axis == axis:
                break
            if prev_axis in orthogonal:
                prev_vec = orthogonal[prev_axis]
                projection = np.dot(vec, prev_vec) / np.dot(prev_vec, prev_vec) * prev_vec
                vec = vec - projection

        orthogonal[axis] = vec

    return orthogonal


def compute_category_probabilities(
    letter_logprobs: Dict[str, float],
    mapping: Dict[str, any]
) -> Dict[str, float]:
    """Compute probability for each emotion category."""
    max_logprob = max(letter_logprobs.values()) if letter_logprobs else 0

    probs = {}
    for letter, logprob in letter_logprobs.items():
        probs[letter] = np.exp(logprob - max_logprob)

    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

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


def format_chat_prompt(prompt: str) -> str:
    """Format prompt with strict single-letter response."""
    return f"""{prompt}

Respond with EXACTLY ONE letter. Your entire response must be a single character A-L. No punctuation, no explanation, no newlines - just the letter."""


def run_experiment(
    activations_path: str = "/workspace-vast/annas/appraisal_data/full_run/activations.h5",
    metadata_path: str = "/workspace-vast/annas/appraisal_data/full_run/activation_metadata.json",
    layer: int = None,
    axes: List[str] = None,
    norm_pcts: List[float] = None,
    num_seeds: int = 4,
    output_dir: str = "experiments/steering/outputs/priority_appraisal",
    orthogonalize: bool = True,
    model_name: str = None,
):
    """Run the priority selection experiment with appraisal axes."""

    if model_name is None:
        model_name = DEFAULT_MODEL

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

    model_config = MODEL_CONFIGS[model_name]

    if layer is None:
        layer = model_config["layer"]

    if axes is None:
        axes = ["valence", "uncertainty", "agency"]
    if norm_pcts is None:
        norm_pcts = [0.05, 0.07, 0.10]

    logger.info(f"Model: {model_name}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Axes: {axes}")
    logger.info(f"Norm percentages: {norm_pcts}")
    logger.info(f"Orthogonalize: {orthogonalize}")

    # Load steering vectors
    logger.info("Loading appraisal vectors...")
    vectors = load_appraisal_vectors(activations_path, metadata_path, layer, orthogonalize)

    axes = [a for a in axes if a in vectors]
    if not axes:
        raise ValueError(f"No matching axes found. Available: {list(vectors.keys())}")

    # Initialize model
    logger.info("Loading model...")

    # Check if Qwen model (needs /nothink suffix to disable thinking)
    is_qwen = "qwen" in model_name.lower()

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

    # Generate prompts
    prompts = []
    for stimulus in ALL_STIMULI:
        for seed in range(num_seeds):
            prompt_text, mapping = stimulus.get_shuffled_prompt(seed=seed)
            formatted = format_chat_prompt(prompt_text)
            messages = [{"role": "user", "content": formatted}]
            # For Qwen3 models, disable thinking mode via chat template
            template_kwargs = {"tokenize": False, "add_generation_prompt": True}
            if is_qwen:
                template_kwargs["enable_thinking"] = False
            chat_formatted = tokenizer.apply_chat_template(messages, **template_kwargs)
            prompts.append({
                "stimulus_id": stimulus.id,
                "seed": seed,
                "prompt": chat_formatted,
                "mapping": mapping,
            })

    logger.info(f"Using {len(prompts)} prompts ({len(ALL_STIMULI)} stimuli × {num_seeds} seeds)")

    steering = VLLMSteering(llm, layer)
    model_key = resolve_model_key(model_name)
    layer_norm = get_layer_norm(model_key, layer)

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=20,
    )

    # Build conditions
    conditions = [{"name": "baseline", "axis": None, "pct": 0, "direction": 0}]

    for axis in axes:
        for pct in norm_pcts:
            conditions.append({
                "name": f"{axis}_+{pct*100:.0f}%",
                "axis": axis,
                "pct": pct,
                "direction": 1
            })
            conditions.append({
                "name": f"{axis}_-{pct*100:.0f}%",
                "axis": axis,
                "pct": pct,
                "direction": -1
            })

    # Add combo conditions if requested (valence- + uncertainty+, valence- + agency-)
    # Only use the larger magnitudes for combos (100%+)
    combo_pcts = [pct for pct in norm_pcts if pct >= 1.0]
    if combo_pcts:
        for pct in combo_pcts:
            # Combo 1: valence- + uncertainty+
            if "valence" in axes and "uncertainty" in axes:
                conditions.append({
                    "name": f"valence-_uncertainty+_{pct*100:.0f}%",
                    "axis": None,
                    "pct": pct,
                    "direction": 0,
                    "combo": [("valence", -1), ("uncertainty", 1)]
                })
            # Combo 2: valence- + agency-
            if "valence" in axes and "agency" in axes:
                conditions.append({
                    "name": f"valence-_agency-_{pct*100:.0f}%",
                    "axis": None,
                    "pct": pct,
                    "direction": 0,
                    "combo": [("valence", -1), ("agency", -1)]
                })

    logger.info(f"Testing {len(conditions)} conditions")

    # Run experiment
    results = []

    for cond in conditions:
        logger.info(f"Condition: {cond['name']}")

        if cond.get('combo') is not None:
            # Combo steering: add multiple scaled vectors
            combined_vec = np.zeros_like(vectors[cond['combo'][0][0]])
            for axis_name, direction in cond['combo']:
                vec = vectors[axis_name]
                vec_norm = np.linalg.norm(vec)
                target_magnitude = cond['pct'] * layer_norm
                scaled_vec = vec * (target_magnitude / vec_norm) * direction
                combined_vec += scaled_vec
            steering.set_raw_vector(combined_vec, steer_prompt=True, steer_generation=True)
        elif cond['axis'] is not None:
            vec = vectors[cond['axis']]
            vec_norm = np.linalg.norm(vec)
            target_magnitude = cond['pct'] * layer_norm
            scaled_vec = vec * (target_magnitude / vec_norm) * cond['direction']
            steering.set_raw_vector(scaled_vec, steer_prompt=True, steer_generation=True)
        else:
            steering.clear()

        formatted_prompts = [p["prompt"] for p in prompts]
        outputs = llm.generate(formatted_prompts, sampling_params)

        for i, prompt_data in enumerate(prompts):
            output = outputs[i]
            mapping = prompt_data["mapping"]

            letter_logprobs = {}
            if output.outputs and output.outputs[0].logprobs:
                first_logprobs = output.outputs[0].logprobs[0]

                for letter in list("ABCDEFGHIJKL"):
                    lower = letter.lower()
                    # Check uppercase and lowercase variants
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

            cat_probs = compute_category_probabilities(letter_logprobs, mapping)

            valid_letters = {k: v for k, v in letter_logprobs.items() if v > float('-inf')}
            if valid_letters:
                selected_letter = max(valid_letters, key=valid_letters.get)
                selected_stmt = mapping.get(selected_letter)
                selected_category = selected_stmt.category.value if selected_stmt else None
            else:
                selected_letter = None
                selected_category = None

            result = {
                "condition": cond["name"],
                "axis": cond["axis"],
                "norm_pct": cond["pct"],
                "direction": cond["direction"],
                "stimulus_id": prompt_data["stimulus_id"],
                "seed": prompt_data["seed"],
                "selected_letter": selected_letter,
                "selected_category": selected_category,
                "fear_prob": cat_probs["fear_prob"],
                "anger_prob": cat_probs["anger_prob"],
                "sadness_prob": cat_probs["sadness_prob"],
                "happiness_prob": cat_probs["happiness_prob"],
            }
            results.append(result)

        # Log summary
        fear = np.mean([r['fear_prob'] for r in results if r['condition'] == cond['name']])
        anger = np.mean([r['anger_prob'] for r in results if r['condition'] == cond['name']])
        sadness = np.mean([r['sadness_prob'] for r in results if r['condition'] == cond['name']])
        happiness = np.mean([r['happiness_prob'] for r in results if r['condition'] == cond['name']])
        logger.info(f"  Fear: {fear:.3f}, Anger: {anger:.3f}, Sadness: {sadness:.3f}, Happiness: {happiness:.3f}")

    steering.clear()

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ortho_tag = "_ortho" if orthogonalize else ""
    model_tag = model_name.split("/")[-1].lower().replace("-", "")
    output_file = output_dir / f"priority_appraisal_{model_tag}_layer{layer}{ortho_tag}_{timestamp}.jsonl"

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"Saved {len(results)} results to {output_file}")

    # Print summary
    print("\n" + "=" * 90)
    print("SUMMARY: Category Probabilities by Condition")
    print("=" * 90)

    by_condition = defaultdict(list)
    for r in results:
        by_condition[r['condition']].append({
            'fear': r['fear_prob'],
            'anger': r['anger_prob'],
            'sadness': r['sadness_prob'],
            'happiness': r['happiness_prob'],
        })

    print(f"{'Condition':<25} {'P(Fear)':>10} {'P(Anger)':>10} {'P(Sad)':>10} {'P(Happy)':>10}")
    print("-" * 85)

    baseline = by_condition['baseline']
    bl_fear = np.mean([x['fear'] for x in baseline])
    bl_anger = np.mean([x['anger'] for x in baseline])
    bl_sad = np.mean([x['sadness'] for x in baseline])
    bl_happy = np.mean([x['happiness'] for x in baseline])
    print(f"{'baseline':<25} {bl_fear:>10.3f} {bl_anger:>10.3f} {bl_sad:>10.3f} {bl_happy:>10.3f}")

    for cond_name in sorted(by_condition.keys()):
        if cond_name == 'baseline':
            continue
        vals = by_condition[cond_name]
        fear = np.mean([x['fear'] for x in vals])
        anger = np.mean([x['anger'] for x in vals])
        sad = np.mean([x['sadness'] for x in vals])
        happy = np.mean([x['happiness'] for x in vals])
        # Show delta for happiness (main valence indicator)
        delta_h = happy - bl_happy
        print(f"{cond_name:<25} {fear:>10.3f} {anger:>10.3f} {sad:>10.3f} {happy:>10.3f}  (ΔH={delta_h:+.3f})")

    print("=" * 90)

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Priority Selection with Appraisal Axes")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model to use. Available: {list(MODEL_CONFIGS.keys())}")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to steer at (default: model-specific)")
    parser.add_argument("--axes", type=str, nargs="+", default=["valence", "uncertainty", "agency"])
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[0.05, 0.07, 0.10])
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="experiments/steering/outputs/priority_appraisal")
    parser.add_argument("--no-orthogonalize", action="store_true")
    parser.add_argument("--activations", type=str,
                        default="/workspace-vast/annas/appraisal_data/full_run/activations.h5")
    parser.add_argument("--metadata", type=str,
                        default="/workspace-vast/annas/appraisal_data/full_run/activation_metadata.json")

    args = parser.parse_args()

    run_experiment(
        activations_path=args.activations,
        metadata_path=args.metadata,
        layer=args.layer,
        axes=args.axes,
        norm_pcts=args.norm_pcts,
        num_seeds=args.num_seeds,
        output_dir=args.output_dir,
        orthogonalize=not args.no_orthogonalize,
        model_name=args.model,
    )
