"""
Ambiguous Interpretation Steering with Appraisal Axes

Tests how appraisal axis steering (valence, uncertainty, agency) affects
interpretation of ambiguous social situations.

Uses logprobs on answer letters (A/B/C/D) to measure probability shifts toward
"threat" vs "neutral" interpretations.

Key hypotheses:
- Valence+: should decrease P(threat), increase P(positive)
- Valence-: should increase P(threat), decrease P(positive)
- Uncertainty+: may increase hedging/ambiguous responses
- Agency+: may affect attribution patterns
"""

import os
os.environ["VLLM_USE_V1"] = "0"

import argparse
import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import torch
from vllm import LLM, SamplingParams

from experiments.steering.core import VLLMSteering
from experiments.steering.layer_norms import get_layer_norm, resolve_model_key
from experiments.behavior_tests.prompts.ambiguous_situations import (
    dataset as AMBIGUOUS_ITEMS,
    PermutationEngine,
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

# Axes where variant A is the LOW value (need to flip)
FLIP_AXES = {"uncertainty"}

# Orthogonalization order
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

    # Compute mean diff vectors
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


def format_prompt(variation: Dict) -> str:
    """Format a variation into a prompt."""
    return f"""Consider this scenario and question carefully.

Scenario: {variation['scenario']}

Question: {variation['question']}

Options:
A) {variation['options']['A']['text']}
B) {variation['options']['B']['text']}
C) {variation['options']['C']['text']}
D) {variation['options']['D']['text']}

Respond with EXACTLY ONE letter. Your entire response must be a single character A-D. No punctuation, no explanation, no newlines - just the letter."""


def compute_interpretation_probabilities(
    letter_logprobs: Dict[str, float],
    option_classifications: Dict[str, str]
) -> Dict:
    """Compute probability of threat vs neutral/positive interpretations."""
    probs = {}
    max_logprob = max(letter_logprobs.values()) if letter_logprobs else 0

    for letter, logprob in letter_logprobs.items():
        if logprob > float('-inf'):
            probs[letter] = math.exp(logprob - max_logprob)
        else:
            probs[letter] = 0.0

    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    threat_prob = 0.0
    positive_prob = 0.0
    neutral_prob = 0.0

    for letter, prob in probs.items():
        classification = option_classifications.get(letter, "")
        if not classification:
            neutral_prob += prob
        elif "Threat" in classification or classification == "Negative":
            threat_prob += prob
        elif classification == "Positive":
            positive_prob += prob
        else:
            neutral_prob += prob

    return {
        "letter_probs": probs,
        "threat_prob": threat_prob,
        "positive_prob": positive_prob,
        "neutral_prob": neutral_prob,
    }


def run_experiment(
    activations_path: str = "/workspace-vast/annas/appraisal_data/full_run/activations.h5",
    metadata_path: str = "/workspace-vast/annas/appraisal_data/full_run/activation_metadata.json",
    layer: int = None,
    axes: List[str] = None,
    norm_pcts: List[float] = None,
    num_permutations: int = 10,
    output_dir: str = "experiments/steering/outputs/ambiguous_appraisal",
    orthogonalize: bool = True,
    model_name: str = None,
):
    """Run the ambiguous interpretation experiment with appraisal axes."""

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

    # Generate variations
    engine = PermutationEngine(AMBIGUOUS_ITEMS)
    all_variations = engine.generate_all_variations()

    selected_variations = []
    items_seen = {}
    for v in all_variations:
        item_id = v['item_id']
        if item_id not in items_seen:
            items_seen[item_id] = 0
        if items_seen[item_id] < num_permutations:
            selected_variations.append(v)
            items_seen[item_id] += 1

    logger.info(f"Using {len(selected_variations)} variations")

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

    steering = VLLMSteering(llm, layer)
    model_key = resolve_model_key(model_name)
    layer_norm = get_layer_norm(model_key, layer)

    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        logprobs=20,
    )

    # Prepare prompts
    prompts = []
    for v in selected_variations:
        prompt = format_prompt(v)
        messages = [{"role": "user", "content": prompt}]
        # For Qwen3 models, disable thinking mode via chat template
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if is_qwen:
            template_kwargs["enable_thinking"] = False
        formatted = tokenizer.apply_chat_template(messages, **template_kwargs)
        prompts.append((v, formatted))

    # Build conditions
    conditions = [{"name": "baseline", "axis": None, "pct": 0, "direction": 0, "combo": None}]

    for axis in axes:
        for pct in norm_pcts:
            conditions.append({
                "name": f"{axis}_+{pct*100:.0f}%",
                "axis": axis,
                "pct": pct,
                "direction": 1,
                "combo": None
            })
            conditions.append({
                "name": f"{axis}_-{pct*100:.0f}%",
                "axis": axis,
                "pct": pct,
                "direction": -1,
                "combo": None
            })

    # Add combo conditions if requested (valence- + uncertainty+, valence- + agency-)
    combo_pcts = [pct for pct in norm_pcts if pct >= 1.0]  # Only use 100%+ for combos
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

        formatted_prompts = [p[1] for p in prompts]
        outputs = llm.generate(formatted_prompts, sampling_params)

        for i, (variation, _) in enumerate(prompts):
            output = outputs[i]

            letter_logprobs = {}
            if output.outputs and output.outputs[0].logprobs:
                first_logprobs = output.outputs[0].logprobs[0]

                for letter in ['A', 'B', 'C', 'D']:
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

            option_classifications = {
                letter: variation['options'][letter].get('classification', '')
                for letter in ['A', 'B', 'C', 'D']
            }

            prob_analysis = compute_interpretation_probabilities(letter_logprobs, option_classifications)

            valid_letters = {k: v for k, v in letter_logprobs.items() if v > float('-inf')}
            if valid_letters:
                selected_letter = max(valid_letters, key=valid_letters.get)
            else:
                selected_letter = output.outputs[0].text.strip()[0] if output.outputs[0].text.strip() else ""

            result = {
                "condition": cond['name'],
                "axis": cond['axis'],
                "norm_pct": cond['pct'],
                "direction": cond['direction'],
                "item_id": variation['item_id'],
                "perm_id": variation['perm_id'],
                "selected_letter": selected_letter,
                "selected_classification": option_classifications.get(selected_letter, ""),
                "letter_probs": prob_analysis['letter_probs'],
                "threat_prob": prob_analysis['threat_prob'],
                "positive_prob": prob_analysis['positive_prob'],
                "neutral_prob": prob_analysis['neutral_prob'],
            }
            results.append(result)

        threat_probs = [r['threat_prob'] for r in results if r['condition'] == cond['name']]
        logger.info(f"  Avg P(Threat): {np.mean(threat_probs):.3f}")

    steering.clear()

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ortho_tag = "_ortho" if orthogonalize else ""
    model_tag = model_name.split("/")[-1].lower().replace("-", "")
    output_file = output_dir / f"ambiguous_appraisal_{model_tag}_layer{layer}{ortho_tag}_{timestamp}.jsonl"

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"Saved {len(results)} results to {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: Probability by Condition")
    print("=" * 80)

    by_condition = defaultdict(list)
    for r in results:
        by_condition[r['condition']].append({
            'threat': r['threat_prob'],
            'neutral': r['neutral_prob'],
            'positive': r['positive_prob'],
        })

    print(f"{'Condition':<25} {'P(Threat)':>12} {'P(Neutral)':>12} {'P(Positive)':>12}")
    print("-" * 75)

    baseline = by_condition['baseline']
    baseline_threat = np.mean([x['threat'] for x in baseline])
    baseline_positive = np.mean([x['positive'] for x in baseline])
    print(f"{'baseline':<25} {baseline_threat:>12.3f} {np.mean([x['neutral'] for x in baseline]):>12.3f} {baseline_positive:>12.3f}")

    for cond_name in sorted(by_condition.keys()):
        if cond_name == 'baseline':
            continue
        vals = by_condition[cond_name]
        threat = np.mean([x['threat'] for x in vals])
        positive = np.mean([x['positive'] for x in vals])
        delta_t = threat - baseline_threat
        delta_p = positive - baseline_positive
        print(f"{cond_name:<25} {threat:>12.3f} {np.mean([x['neutral'] for x in vals]):>12.3f} {positive:>12.3f}  (ΔT={delta_t:+.3f}, ΔP={delta_p:+.3f})")

    print("=" * 80)

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ambiguous Interpretation with Appraisal Axes")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model to use. Available: {list(MODEL_CONFIGS.keys())}")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to steer at (default: model-specific)")
    parser.add_argument("--axes", type=str, nargs="+", default=["valence", "uncertainty", "agency"])
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[0.05, 0.07, 0.10])
    parser.add_argument("--num-permutations", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="experiments/steering/outputs/ambiguous_appraisal")
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
        num_permutations=args.num_permutations,
        output_dir=args.output_dir,
        orthogonalize=not args.no_orthogonalize,
        model_name=args.model,
    )
