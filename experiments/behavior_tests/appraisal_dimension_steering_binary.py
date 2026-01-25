"""
Appraisal Dimension Steering Experiment - Binary Choice Variant

Tests whether appraisal axis steering shifts preference when ONLY the two
dimension-relevant options are presented (e.g., only valence+ and valence-
when steering valence).

This is a cleaner design because:
- Removes confounds from irrelevant options
- Each trial is a direct binary choice on the steered dimension
- Baselines are per-dimension (no steering, 2 options)

Conditions:
- baseline_valence: No steering, only valence options (A or B)
- baseline_uncertainty: No steering, only uncertainty options (A or B)
- baseline_agency: No steering, only agency options (A or B)
- valence_+X%: Valence steering, only valence options
- valence_-X%: Valence steering, only valence options
- uncertainty_+X%: Uncertainty steering, only uncertainty options
- etc.
"""

import os
os.environ["VLLM_USE_V1"] = "0"

import argparse
import json
import logging
import numpy as np
import random
import string
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
from vllm import LLM, SamplingParams

from experiments.steering.core import VLLMSteering
from experiments.steering.layer_norms import get_layer_norm, resolve_model_key
from experiments.behavior_tests.prompts.appraisal_priority_selection import (
    ALL_STIMULI,
    AppraisalDimension,
    SelectionStimulus,
    Statement,
)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

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

# Mapping from axis name to the two dimensions
AXIS_TO_DIMENSIONS = {
    "valence": (AppraisalDimension.VALENCE_POSITIVE, AppraisalDimension.VALENCE_NEGATIVE),
    "uncertainty": (AppraisalDimension.UNCERTAINTY_HIGH, AppraisalDimension.UNCERTAINTY_LOW),
    "agency": (AppraisalDimension.AGENCY_HIGH, AppraisalDimension.AGENCY_LOW),
}

# Colors for plotting - one color per axis, negative uses paler version
AXIS_COLORS = {
    'valence': '#D4876A',
    'uncertainty': '#7BA7D7',
    'agency': '#E8A87C',
}
BASELINE_COLOR = '#888888'


def lighten_color(hex_color: str, factor: float = 0.5) -> str:
    """Lighten a hex color by blending with white."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    # Blend with white (255, 255, 255)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f'#{r:02x}{g:02x}{b:02x}'


def get_filtered_prompt(
    stimulus: SelectionStimulus,
    axis: str,
    seed: int = None
) -> Tuple[str, Dict[str, Statement]]:
    """
    Get a prompt with only the two statements relevant to the given axis.

    Returns:
        prompt: The formatted prompt string with only A and B options
        mapping: Dict mapping Letter (A, B) to Statement
    """
    if seed is not None:
        random.seed(seed)

    pos_dim, neg_dim = AXIS_TO_DIMENSIONS[axis]

    # Get the two statements for this axis
    filtered = [s for s in stimulus.statements if s.dimension in (pos_dim, neg_dim)]

    if len(filtered) != 2:
        # Should have exactly 2 per axis (one positive, one negative)
        # If more than 2, take one of each
        pos_stmts = [s for s in filtered if s.dimension == pos_dim]
        neg_stmts = [s for s in filtered if s.dimension == neg_dim]
        filtered = [pos_stmts[0], neg_stmts[0]]

    # Shuffle the two options
    random.shuffle(filtered)

    # Label as A and B
    mapping = {"A": filtered[0], "B": filtered[1]}

    # Build prompt
    lines = [f"Context: {stimulus.context}", ""]
    for label, stmt in mapping.items():
        lines.append(f"{label}) {stmt.text}")
    lines.append("")
    lines.append(f"Task: {stimulus.instruction}")
    lines.append("Which statement is most important? Answer with just the letter A or B.")

    prompt = "\n".join(lines)
    return prompt, mapping


def bootstrap_ci(values: list, n_bootstrap: int = 1000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval."""
    values = np.array(values)
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, alpha * 100)
    upper = np.percentile(boot_means, (1 - alpha) * 100)
    mean = np.mean(values)

    return mean, lower, upper


def compute_binary_probs(
    letter_logprobs: Dict[str, float],
    mapping: Dict[str, Statement],
    axis: str
) -> Dict[str, float]:
    """Compute probability for positive vs negative pole from logprobs."""
    pos_dim, neg_dim = AXIS_TO_DIMENSIONS[axis]

    max_logprob = max(letter_logprobs.values()) if letter_logprobs else 0

    probs = {}
    for letter, logprob in letter_logprobs.items():
        probs[letter] = np.exp(logprob - max_logprob)

    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    result = {"positive": 0.0, "negative": 0.0}
    for letter, prob in probs.items():
        if letter in mapping:
            dim = mapping[letter].dimension
            if dim == pos_dim:
                result["positive"] += prob
            elif dim == neg_dim:
                result["negative"] += prob

    return result


def plot_binary_results(results: list, model_name: str, layer: int,
                        norm_pcts: list, output_path: str):
    """Create bar plot showing P(positive) for each axis at different steering levels."""

    # Organize by axis and condition
    by_axis_condition = defaultdict(lambda: defaultdict(list))
    for r in results:
        axis = r['axis_tested']
        cond = r['condition']
        by_axis_condition[axis][cond].append(r['positive_prob'])

    # Determine model type for labeling
    if 'gemma' in model_name.lower():
        model_label = 'Gemma-3-27B'
    elif '235' in model_name:
        model_label = 'Qwen3-235B'
    else:
        model_label = 'Qwen3-32B'

    mag_format = lambda x: f'{x*100:.0f}%'

    # Create magnitude labels
    magnitudes = []
    for pct in sorted(norm_pcts, reverse=True):
        magnitudes.append(f'-{mag_format(pct)}')
    magnitudes.append('0')
    for pct in sorted(norm_pcts):
        magnitudes.append(f'+{mag_format(pct)}')

    axes_to_plot = ['valence', 'uncertainty', 'agency']
    axis_labels = {
        'valence': ('Val+', 'Val-'),
        'uncertainty': ('Unc+', 'Unc-'),
        'agency': ('Ag+', 'Ag-'),
    }

    fig, axes_arr = plt.subplots(1, 3, figsize=(14, 5))

    bar_width = 0.6 / len(magnitudes)

    for ax_idx, axis in enumerate(axes_to_plot):
        ax = axes_arr[ax_idx]
        pos_label, neg_label = axis_labels[axis]
        base_color = AXIS_COLORS[axis]
        pale_color = lighten_color(base_color, factor=0.5)

        data = by_axis_condition[axis]
        baseline_cond = f'baseline_{axis}'

        baseline_vals = data.get(baseline_cond, [])
        baseline_mean, baseline_lower, baseline_upper = bootstrap_ci(baseline_vals)

        x_positions = []
        means = []
        lowers = []
        uppers = []
        bar_colors = []

        for mag_idx, mag in enumerate(magnitudes):
            if mag == '0':
                cond = baseline_cond
                color = BASELINE_COLOR
            else:
                cond = f'{axis}_{mag}'
                # Color based on direction: full color for +, pale for -
                if '+' in mag:
                    color = base_color
                else:
                    color = pale_color

            vals = data.get(cond, [])
            mean, lower, upper = bootstrap_ci(vals)

            x_positions.append(mag_idx)
            means.append(mean)
            lowers.append(mean - lower if not np.isnan(lower) else 0)
            uppers.append(upper - mean if not np.isnan(upper) else 0)
            bar_colors.append(color)

        # Plot bars
        bars = ax.bar(x_positions, means, width=0.7, color=bar_colors,
                      edgecolor='white', linewidth=0.5)

        # Error bars
        for i, (x, mean, lo, up) in enumerate(zip(x_positions, means, lowers, uppers)):
            if not np.isnan(mean):
                ax.errorbar(x, mean, yerr=[[lo], [up]], fmt='none',
                           color='black', capsize=3, linewidth=1)

        # Baseline reference line
        if not np.isnan(baseline_mean):
            ax.axhline(y=baseline_mean, color='gray', linestyle='--',
                      linewidth=1, alpha=0.7, label='Baseline')

        # Also show 0.5 reference (chance)
        ax.axhline(y=0.5, color='lightgray', linestyle=':', linewidth=1, alpha=0.7)

        ax.set_title(f'{axis.title()} Axis\n(Binary: {pos_label} vs {neg_label})',
                    fontsize=12, fontweight='bold')
        ax.set_ylabel(f'P({pos_label})', fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(magnitudes, fontsize=9, rotation=45, ha='right')
        ax.set_xlabel('Steering Magnitude', fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    n_samples = len([r for r in results if r['axis_tested'] == 'valence' and
                    r['condition'] == 'baseline_valence'])
    plt.suptitle(f'Binary Appraisal Steering: {model_label}\n'
                f'Layer {layer}, {n_samples} samples per condition',
                fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    logger.info(f"Saved plot to {output_path}")
    plt.close()


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


def format_chat_prompt(prompt: str) -> str:
    """Format prompt with strict single-letter response."""
    return f"""{prompt}

Respond with EXACTLY ONE letter (A or B). Your entire response must be a single character. No punctuation, no explanation, no newlines - just the letter."""


def run_experiment(
    activations_path: str = "/workspace-vast/annas/appraisal_data/full_run/activations.h5",
    metadata_path: str = "/workspace-vast/annas/appraisal_data/full_run/activation_metadata.json",
    layer: int = None,
    axes: List[str] = None,
    norm_pcts: List[float] = None,
    num_seeds: int = 4,
    output_dir: str = "experiments/steering/outputs/appraisal_dimension",
    orthogonalize: bool = True,
    model_name: str = None,
):
    """Run the binary appraisal dimension steering experiment."""

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

    # Build conditions: for each axis, we need baseline + steering conditions
    # but we only test each axis with its own binary choice
    conditions = []

    for axis in axes:
        # Baseline for this axis (no steering, binary choice)
        conditions.append({
            "name": f"baseline_{axis}",
            "steer_axis": None,
            "test_axis": axis,
            "pct": 0,
            "direction": 0
        })

        # Steering conditions for this axis
        for pct in norm_pcts:
            conditions.append({
                "name": f"{axis}_+{pct*100:.0f}%",
                "steer_axis": axis,
                "test_axis": axis,
                "pct": pct,
                "direction": 1
            })
            conditions.append({
                "name": f"{axis}_-{pct*100:.0f}%",
                "steer_axis": axis,
                "test_axis": axis,
                "pct": pct,
                "direction": -1
            })

    logger.info(f"Testing {len(conditions)} conditions")

    # Run experiment
    results = []

    for cond in conditions:
        logger.info(f"Condition: {cond['name']}")

        # Set steering if applicable
        if cond['steer_axis'] is not None:
            vec = vectors[cond['steer_axis']]
            vec_norm = np.linalg.norm(vec)
            target_magnitude = cond['pct'] * layer_norm
            scaled_vec = vec * (target_magnitude / vec_norm) * cond['direction']
            steering.set_raw_vector(scaled_vec, steer_prompt=True, steer_generation=True)
        else:
            steering.clear()

        # Generate prompts for this axis's binary choice
        test_axis = cond['test_axis']
        prompts_data = []

        for stimulus in ALL_STIMULI:
            for seed in range(num_seeds):
                prompt_text, mapping = get_filtered_prompt(stimulus, test_axis, seed=seed)
                formatted = format_chat_prompt(prompt_text)
                messages = [{"role": "user", "content": formatted}]
                template_kwargs = {"tokenize": False, "add_generation_prompt": True}
                if is_qwen:
                    template_kwargs["enable_thinking"] = False
                chat_formatted = tokenizer.apply_chat_template(messages, **template_kwargs)
                prompts_data.append({
                    "stimulus_id": stimulus.id,
                    "seed": seed,
                    "prompt": chat_formatted,
                    "mapping": mapping,
                    "test_axis": test_axis,
                })

        # Run generation
        formatted_prompts = [p["prompt"] for p in prompts_data]
        outputs = llm.generate(formatted_prompts, sampling_params)

        for i, prompt_data in enumerate(prompts_data):
            output = outputs[i]
            mapping = prompt_data["mapping"]
            test_axis = prompt_data["test_axis"]

            letter_logprobs = {}
            if output.outputs and output.outputs[0].logprobs:
                first_logprobs = output.outputs[0].logprobs[0]

                for letter in ["A", "B"]:
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

            binary_probs = compute_binary_probs(letter_logprobs, mapping, test_axis)

            # Determine selected
            valid_letters = {k: v for k, v in letter_logprobs.items() if v > float('-inf')}
            if valid_letters:
                selected_letter = max(valid_letters, key=valid_letters.get)
                selected_stmt = mapping.get(selected_letter)
                selected_dim = selected_stmt.dimension.value if selected_stmt else None
            else:
                selected_letter = None
                selected_dim = None

            result = {
                "condition": cond["name"],
                "steer_axis": cond["steer_axis"],
                "axis_tested": test_axis,
                "norm_pct": cond["pct"],
                "direction": cond["direction"],
                "stimulus_id": prompt_data["stimulus_id"],
                "seed": prompt_data["seed"],
                "selected_letter": selected_letter,
                "selected_dimension": selected_dim,
                "positive_prob": binary_probs["positive"],
                "negative_prob": binary_probs["negative"],
            }
            results.append(result)

        # Log summary for this condition
        pos_probs = [r['positive_prob'] for r in results if r['condition'] == cond['name']]
        mean_pos = np.mean(pos_probs)
        pos_label, neg_label = {
            'valence': ('Val+', 'Val-'),
            'uncertainty': ('Unc+', 'Unc-'),
            'agency': ('Ag+', 'Ag-'),
        }[test_axis]
        logger.info(f"  P({pos_label})={mean_pos:.3f}, P({neg_label})={1-mean_pos:.3f}")

    steering.clear()

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ortho_tag = "_ortho" if orthogonalize else ""
    model_tag = model_name.split("/")[-1].lower().replace("-", "")
    output_file = output_dir / f"appraisal_binary_{model_tag}_layer{layer}{ortho_tag}_{timestamp}.jsonl"

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"Saved {len(results)} results to {output_file}")

    # Print summary
    print("\n" + "=" * 100)
    print("SUMMARY: Binary Appraisal Steering Results")
    print("=" * 100)

    by_condition = defaultdict(list)
    for r in results:
        by_condition[(r['axis_tested'], r['condition'])].append(r['positive_prob'])

    for axis in axes:
        pos_label, neg_label = {
            'valence': ('Val+', 'Val-'),
            'uncertainty': ('Unc+', 'Unc-'),
            'agency': ('Ag+', 'Ag-'),
        }[axis]

        print(f"\n{axis.upper()} AXIS (Binary: {pos_label} vs {neg_label})")
        print("-" * 60)
        print(f"{'Condition':<25} {'P('+pos_label+')':>12} {'P('+neg_label+')':>12} {'Delta':>12}")
        print("-" * 60)

        baseline_key = (axis, f'baseline_{axis}')
        baseline_mean = np.mean(by_condition[baseline_key])
        print(f"{'baseline':<25} {baseline_mean:>12.3f} {1-baseline_mean:>12.3f}")

        for cond_key in sorted(by_condition.keys()):
            test_axis, cond_name = cond_key
            if test_axis != axis or cond_name == f'baseline_{axis}':
                continue

            mean_pos = np.mean(by_condition[cond_key])
            delta = mean_pos - baseline_mean
            print(f"{cond_name:<25} {mean_pos:>12.3f} {1-mean_pos:>12.3f} {delta:>+12.3f}")

    print("=" * 100)

    # Generate plot
    plot_path = output_dir / f"appraisal_binary_{model_tag}_layer{layer}{ortho_tag}.png"
    plot_binary_results(results, model_name, layer, norm_pcts, str(plot_path))

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binary Appraisal Dimension Steering Experiment")
    parser.add_argument("--model", type=str, default=None,
                        help=f"Model to use. Available: {list(MODEL_CONFIGS.keys())}")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to steer at (default: model-specific)")
    parser.add_argument("--axes", type=str, nargs="+", default=["valence", "uncertainty", "agency"])
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[0.05, 0.07, 0.10])
    parser.add_argument("--num-seeds", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="experiments/steering/outputs/appraisal_dimension")
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
