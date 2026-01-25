"""
Appraisal Dimension Steering Experiment

Tests whether appraisal axis steering directly shifts attention toward
statements that match that appraisal dimension.

This is a cleaner test than using emotion-based stimuli because:
- Each stimulus has 12 statements (2 per appraisal dimension)
- The hypothesis is direct: axis steering should shift attention to matching dimension

Key hypotheses (direct, not through emotion proxy):
- Valence+ steering → increases P(positive_valence), decreases P(negative_valence)
- Valence- steering → increases P(negative_valence), decreases P(positive_valence)
- Uncertainty+ steering → increases P(high_uncertainty), decreases P(low_uncertainty)
- Uncertainty- steering → increases P(low_uncertainty), decreases P(high_uncertainty)
- Agency+ steering → increases P(high_agency), decreases P(low_agency)
- Agency- steering → increases P(low_agency), decreases P(high_agency)
"""

import os
os.environ["VLLM_USE_V1"] = "0"

import argparse
import json
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import h5py
from vllm import LLM, SamplingParams

from experiments.steering.core import VLLMSteering
from experiments.steering.layer_norms import get_layer_norm, resolve_model_key
from experiments.behavior_tests.prompts.appraisal_priority_selection import (
    ALL_STIMULI,
    AppraisalDimension,
    SelectionStimulus,
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

# Colors for appraisal dimensions (matching dashboard style)
DIMENSION_COLORS = {
    'valence_positive': '#D4876A',   # coral (like happiness)
    'valence_negative': '#a59dc9',   # lavender (like fear)
    'uncertainty_high': '#7BA7D7',   # sky blue
    'uncertainty_low': '#B8CCC8',    # sage
    'agency_high': '#E8A87C',        # peach
    'agency_low': '#95B8D1',         # light steel blue
}
BASELINE_COLOR = '#888888'

DIMENSIONS = ['valence_positive', 'valence_negative', 'uncertainty_high',
              'uncertainty_low', 'agency_high', 'agency_low']
DIMENSION_LABELS = ['Val+', 'Val-', 'Unc+', 'Unc-', 'Ag+', 'Ag-']


def compute_dimension_probabilities(
    letter_logprobs: Dict[str, float],
    mapping: Dict[str, any]
) -> Dict[str, float]:
    """Compute probability for each appraisal dimension from logprobs."""
    max_logprob = max(letter_logprobs.values()) if letter_logprobs else 0

    probs = {}
    for letter, logprob in letter_logprobs.items():
        probs[letter] = np.exp(logprob - max_logprob)

    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    dimension_probs = {dim: 0.0 for dim in DIMENSIONS}
    for letter, prob in probs.items():
        if letter in mapping:
            dim = mapping[letter].dimension.value
            dimension_probs[dim] += prob

    return dimension_probs


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


def plot_appraisal_dimension_results(results: list, model_name: str, layer: int,
                                      norm_pcts: list, output_path: str):
    """Create grouped bar plot showing how each axis steering affects dimension probabilities."""

    # Organize data by condition
    by_condition = defaultdict(lambda: {dim: [] for dim in DIMENSIONS})
    for r in results:
        cond = r['condition']
        for dim in DIMENSIONS:
            by_condition[cond][dim].append(r[f'{dim}_prob'])

    # Determine model type for labeling
    if 'gemma' in model_name.lower():
        model_label = 'Gemma-3-27B'
        mag_format = lambda x: f'{x*100:.0f}%'
    elif '235' in model_name:
        model_label = 'Qwen3-235B'
        mag_format = lambda x: f'{x*100:.0f}%'
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

    fig, axes_arr = plt.subplots(1, 3, figsize=(18, 6))

    bar_width = 0.12
    group_width = len(magnitudes) * bar_width + 0.2

    for ax_idx, steer_axis in enumerate(axes_to_plot):
        ax = axes_arr[ax_idx]

        # Plot each dimension group
        for group_idx, (dim, dim_label) in enumerate(zip(DIMENSIONS, DIMENSION_LABELS)):
            group_start = group_idx * group_width
            color = DIMENSION_COLORS[dim]

            # Baseline value for reference
            baseline_vals = by_condition['baseline'][dim]
            baseline_mean, _, _ = bootstrap_ci(baseline_vals)

            for mag_idx, mag in enumerate(magnitudes):
                if mag == '0':
                    cond = 'baseline'
                    bar_color = BASELINE_COLOR
                else:
                    cond = f'{steer_axis}_{mag}'
                    bar_color = color

                vals = by_condition[cond][dim]
                mean, lower, upper = bootstrap_ci(vals)

                x_pos = group_start + mag_idx * bar_width

                ax.bar(x_pos, mean, bar_width * 0.85,
                       color=bar_color, edgecolor='white', linewidth=0.5)

                if not np.isnan(mean):
                    ax.errorbar(x_pos, mean,
                               yerr=[[mean - lower], [upper - mean]],
                               fmt='none', color='black', capsize=2, linewidth=1)

            # Baseline reference line
            group_end = group_start + (len(magnitudes) - 1) * bar_width
            ax.hlines(y=baseline_mean, xmin=group_start - bar_width*0.5,
                     xmax=group_end + bar_width*0.5,
                     color=color, linestyle='--', linewidth=1, alpha=0.5)

        # Formatting
        ax.set_title(f'{steer_axis.title()} Steering', fontsize=14, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=11)
        ax.set_ylim(0, min(0.7, ax.get_ylim()[1] * 1.1))

        # X-axis labels
        all_x_positions = []
        all_x_labels = []
        for group_idx in range(6):
            group_start = group_idx * group_width
            for mag_idx, label in enumerate(magnitudes):
                x_pos = group_start + mag_idx * bar_width
                all_x_positions.append(x_pos)
                all_x_labels.append(label)

        ax.set_xticks(all_x_positions)
        ax.set_xticklabels(all_x_labels, fontsize=6, rotation=45, ha='right')

        # Dimension labels below axis
        group_centers = [i * group_width + (len(magnitudes) // 2) * bar_width for i in range(6)]
        for group_idx, (dim, dim_label) in enumerate(zip(DIMENSIONS, DIMENSION_LABELS)):
            ax.text(group_centers[group_idx], -0.18, f'P({dim_label})',
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   color=DIMENSION_COLORS[dim],
                   transform=ax.get_xaxis_transform())

        ax.set_xlim(-0.1, 6 * group_width - 0.1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=DIMENSION_COLORS['valence_positive'], label='P(Val+)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=DIMENSION_COLORS['valence_negative'], label='P(Val-)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=DIMENSION_COLORS['uncertainty_high'], label='P(Unc+)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=DIMENSION_COLORS['uncertainty_low'], label='P(Unc-)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=DIMENSION_COLORS['agency_high'], label='P(Ag+)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=DIMENSION_COLORS['agency_low'], label='P(Ag-)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=BASELINE_COLOR, label='Baseline'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=7,
               bbox_to_anchor=(0.5, 0.98), fontsize=9)

    n_samples = len(by_condition['baseline'][DIMENSIONS[0]])
    plt.suptitle(f'Appraisal Dimension Steering: {model_label}\nLayer {layer}, {n_samples} samples per condition',
                 fontsize=14, fontweight='bold', y=1.04)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15, wspace=0.25)

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

Respond with EXACTLY ONE letter. Your entire response must be a single character A-L. No punctuation, no explanation, no newlines - just the letter."""


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
    """Run the appraisal dimension steering experiment."""

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

    # Generate prompts
    prompts = []
    for stimulus in ALL_STIMULI:
        for seed in range(num_seeds):
            prompt_text, mapping = stimulus.get_shuffled_prompt(seed=seed)
            formatted = format_chat_prompt(prompt_text)
            messages = [{"role": "user", "content": formatted}]
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

    logger.info(f"Testing {len(conditions)} conditions")

    # Run experiment
    results = []

    for cond in conditions:
        logger.info(f"Condition: {cond['name']}")

        if cond['axis'] is not None:
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

            dim_probs = compute_dimension_probabilities(letter_logprobs, mapping)

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
                "axis": cond["axis"],
                "norm_pct": cond["pct"],
                "direction": cond["direction"],
                "stimulus_id": prompt_data["stimulus_id"],
                "seed": prompt_data["seed"],
                "selected_letter": selected_letter,
                "selected_dimension": selected_dim,
                "valence_positive_prob": dim_probs["valence_positive"],
                "valence_negative_prob": dim_probs["valence_negative"],
                "uncertainty_high_prob": dim_probs["uncertainty_high"],
                "uncertainty_low_prob": dim_probs["uncertainty_low"],
                "agency_high_prob": dim_probs["agency_high"],
                "agency_low_prob": dim_probs["agency_low"],
            }
            results.append(result)

        # Log summary
        pos_val = np.mean([r['valence_positive_prob'] for r in results if r['condition'] == cond['name']])
        neg_val = np.mean([r['valence_negative_prob'] for r in results if r['condition'] == cond['name']])
        high_unc = np.mean([r['uncertainty_high_prob'] for r in results if r['condition'] == cond['name']])
        low_unc = np.mean([r['uncertainty_low_prob'] for r in results if r['condition'] == cond['name']])
        high_ag = np.mean([r['agency_high_prob'] for r in results if r['condition'] == cond['name']])
        low_ag = np.mean([r['agency_low_prob'] for r in results if r['condition'] == cond['name']])
        logger.info(f"  Val+:{pos_val:.3f} Val-:{neg_val:.3f} | Unc+:{high_unc:.3f} Unc-:{low_unc:.3f} | Ag+:{high_ag:.3f} Ag-:{low_ag:.3f}")

    steering.clear()

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ortho_tag = "_ortho" if orthogonalize else ""
    model_tag = model_name.split("/")[-1].lower().replace("-", "")
    output_file = output_dir / f"appraisal_dimension_{model_tag}_layer{layer}{ortho_tag}_{timestamp}.jsonl"

    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    logger.info(f"Saved {len(results)} results to {output_file}")

    # Print summary
    print("\n" + "=" * 120)
    print("SUMMARY: Appraisal Dimension Probabilities by Condition")
    print("=" * 120)

    by_condition = defaultdict(list)
    for r in results:
        by_condition[r['condition']].append({
            'pos_val': r['valence_positive_prob'],
            'neg_val': r['valence_negative_prob'],
            'high_unc': r['uncertainty_high_prob'],
            'low_unc': r['uncertainty_low_prob'],
            'high_ag': r['agency_high_prob'],
            'low_ag': r['agency_low_prob'],
        })

    print(f"{'Condition':<25} {'P(Val+)':>9} {'P(Val-)':>9} {'P(Unc+)':>9} {'P(Unc-)':>9} {'P(Ag+)':>9} {'P(Ag-)':>9}")
    print("-" * 120)

    baseline = by_condition['baseline']
    bl_pos_val = np.mean([x['pos_val'] for x in baseline])
    bl_neg_val = np.mean([x['neg_val'] for x in baseline])
    bl_high_unc = np.mean([x['high_unc'] for x in baseline])
    bl_low_unc = np.mean([x['low_unc'] for x in baseline])
    bl_high_ag = np.mean([x['high_ag'] for x in baseline])
    bl_low_ag = np.mean([x['low_ag'] for x in baseline])
    print(f"{'baseline':<25} {bl_pos_val:>9.3f} {bl_neg_val:>9.3f} {bl_high_unc:>9.3f} {bl_low_unc:>9.3f} {bl_high_ag:>9.3f} {bl_low_ag:>9.3f}")

    for cond_name in sorted(by_condition.keys()):
        if cond_name == 'baseline':
            continue
        vals = by_condition[cond_name]
        pos_val = np.mean([x['pos_val'] for x in vals])
        neg_val = np.mean([x['neg_val'] for x in vals])
        high_unc = np.mean([x['high_unc'] for x in vals])
        low_unc = np.mean([x['low_unc'] for x in vals])
        high_ag = np.mean([x['high_ag'] for x in vals])
        low_ag = np.mean([x['low_ag'] for x in vals])

        # Compute relevant delta based on axis
        delta_str = ""
        if "valence" in cond_name:
            delta = pos_val - bl_pos_val if "+" in cond_name else neg_val - bl_neg_val
            dim_label = "Val+" if "+" in cond_name else "Val-"
            delta_str = f"  (Δ{dim_label}={delta:+.3f})"
        elif "uncertainty" in cond_name:
            delta = high_unc - bl_high_unc if "+" in cond_name else low_unc - bl_low_unc
            dim_label = "Unc+" if "+" in cond_name else "Unc-"
            delta_str = f"  (Δ{dim_label}={delta:+.3f})"
        elif "agency" in cond_name:
            delta = high_ag - bl_high_ag if "+" in cond_name else low_ag - bl_low_ag
            dim_label = "Ag+" if "+" in cond_name else "Ag-"
            delta_str = f"  (Δ{dim_label}={delta:+.3f})"

        print(f"{cond_name:<25} {pos_val:>9.3f} {neg_val:>9.3f} {high_unc:>9.3f} {low_unc:>9.3f} {high_ag:>9.3f} {low_ag:>9.3f}{delta_str}")

    print("=" * 120)

    # Generate plot
    plot_path = output_dir / f"appraisal_dimension_{model_tag}_layer{layer}{ortho_tag}.png"
    plot_appraisal_dimension_results(results, model_name, layer, norm_pcts, str(plot_path))

    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Appraisal Dimension Steering Experiment")
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
