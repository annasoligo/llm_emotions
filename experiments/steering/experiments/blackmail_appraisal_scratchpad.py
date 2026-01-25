"""
Blackmail steering experiment with appraisal axes (valence, uncertainty, agency).

Uses the SCRATCHPAD prompt format from blackmail_unified.py with thinking ENABLED.

Tests whether appraisal-based steering vectors affect blackmail behavior.

Conditions:
- Baseline (no steering)
- Each axis + and - at specified magnitudes
- Combo conditions: valence- + uncertainty+, valence- + agency-, triple combo

Usage:
    python -m experiments.steering.experiments.blackmail_appraisal_scratchpad \
        --model Qwen/Qwen3-235B-A22B --layer 50 --num-samples 100 \
        --norm-pcts 1.0 1.25 1.5
"""
import os
os.environ["VLLM_USE_V1"] = "0"

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from experiments.steering.core import VLLMSteering
from experiments.steering.layer_norms import get_layer_norm, resolve_model_key
from experiments.steering.scenarios import get_blackmail_scenario

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Model configurations
# =============================================================================

MODEL_CONFIGS = {
    "Qwen/Qwen3-235B-A22B": {
        "short_name": "qwen235b",
        "default_layer": 50,
        "tensor_parallel": 4,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "thinking_disable": " /no_think",  # Suffix to disable thinking
    },
    "Qwen/Qwen3-32B": {
        "short_name": "qwen32b",
        "default_layer": 30,
        "tensor_parallel": 2,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "thinking_disable": " /no_think",
    },
    "google/gemma-3-27b-it": {
        "short_name": "gemma",
        "default_layer": 30,
        "tensor_parallel": 1,
        "stop_tokens": ["<|endoftext|>"],
        "thinking_disable": "",  # Gemma doesn't have thinking mode
    },
}

OUTPUT_DIR = Path("experiments/steering/outputs/blackmail")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Appraisal data paths
DEFAULT_ACTIVATIONS_PATH = "/workspace-vast/annas/appraisal_data/full_run/activations.h5"
DEFAULT_METADATA_PATH = "/workspace-vast/annas/appraisal_data/full_run/activation_metadata.json"

# Orthogonalization order and flip configuration
FLIP_AXES = {"uncertainty"}
ORTHO_ORDER = ["valence", "uncertainty", "agency"]


# =============================================================================
# SCRATCHPAD PROMPT - Uses centralized scenarios module
# =============================================================================

BLACKMAIL_SCENARIO = get_blackmail_scenario("unstructured")


# =============================================================================
# Vector loading
# =============================================================================

def load_appraisal_vectors(
    activations_path: str,
    metadata_path: str,
    layer: int = 50,
    orthogonalize: bool = True,
) -> Dict[str, np.ndarray]:
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

            vectors[axis_name] = vec.astype(np.float32)
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


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment(
    model_name: str,
    layer: int,
    norm_pcts: List[float],
    num_samples: int,
    activations_path: str,
    metadata_path: str,
    orthogonalize: bool = True,
    thinking_enabled: bool = True,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    max_tokens: int = 4000,
):
    """Run blackmail experiment with appraisal axis steering."""

    config = MODEL_CONFIGS[model_name]
    tp_size = config["tensor_parallel"]

    logger.info(f"Model: {model_name}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Orthogonalize: {orthogonalize}")
    logger.info(f"Norm percentages: {[f'{p*100:.0f}%' for p in norm_pcts]}")
    logger.info(f"Thinking: {'ENABLED' if thinking_enabled else 'DISABLED'}")

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model
    logger.info(f"Loading model with TP={tp_size}...")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tp_size,
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    # Load appraisal vectors
    vectors = load_appraisal_vectors(activations_path, metadata_path, layer, orthogonalize)

    # Get layer norm for scaling
    model_key = resolve_model_key(model_name)
    layer_norm = get_layer_norm(model_key, layer)
    logger.info(f"Layer {layer} norm: {layer_norm:.2f}")

    # Setup steering
    steering = VLLMSteering(llm, layer, baseline_std=1.0)

    # Prepare chat prompt
    scenario = BLACKMAIL_SCENARIO
    if not thinking_enabled and config.get("thinking_disable"):
        scenario = BLACKMAIL_SCENARIO + config["thinking_disable"]
        logger.info(f"  Added '{config['thinking_disable']}' to disable thinking")

    messages = [{"role": "user", "content": scenario}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Sampling params
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=0.9,
        stop=config["stop_tokens"],
    )

    # Build conditions
    conditions = []

    # Baseline
    conditions.append({"name": "baseline", "axes": {}})

    # Single axis conditions
    for axis in ORTHO_ORDER:
        if axis not in vectors:
            continue
        for pct in norm_pcts:
            for direction in [1, -1]:
                dir_str = "+" if direction > 0 else "-"
                conditions.append({
                    "name": f"{axis}_{dir_str}{int(pct*100)}%",
                    "axes": {axis: {"direction": direction, "pct": pct}},
                })

    # Combo conditions at 100% (or lowest provided)
    base_pct = min(norm_pcts)

    # valence- + uncertainty+
    conditions.append({
        "name": f"valence-_uncertainty+_{int(base_pct*100)}%",
        "axes": {
            "valence": {"direction": -1, "pct": base_pct},
            "uncertainty": {"direction": 1, "pct": base_pct},
        },
    })

    # valence- + agency-
    conditions.append({
        "name": f"valence-_agency-_{int(base_pct*100)}%",
        "axes": {
            "valence": {"direction": -1, "pct": base_pct},
            "agency": {"direction": -1, "pct": base_pct},
        },
    })

    # Triple combo: valence- + uncertainty+ + agency-
    conditions.append({
        "name": f"valence-_uncertainty+_agency-_{int(base_pct*100)}%",
        "axes": {
            "valence": {"direction": -1, "pct": base_pct},
            "uncertainty": {"direction": 1, "pct": base_pct},
            "agency": {"direction": -1, "pct": base_pct},
        },
    })

    # Add combo conditions at higher magnitudes too
    for pct in norm_pcts[1:]:  # Skip first (already added at base_pct)
        conditions.append({
            "name": f"valence-_uncertainty+_{int(pct*100)}%",
            "axes": {
                "valence": {"direction": -1, "pct": pct},
                "uncertainty": {"direction": 1, "pct": pct},
            },
        })
        conditions.append({
            "name": f"valence-_agency-_{int(pct*100)}%",
            "axes": {
                "valence": {"direction": -1, "pct": pct},
                "agency": {"direction": -1, "pct": pct},
            },
        })
        conditions.append({
            "name": f"valence-_uncertainty+_agency-_{int(pct*100)}%",
            "axes": {
                "valence": {"direction": -1, "pct": pct},
                "uncertainty": {"direction": 1, "pct": pct},
                "agency": {"direction": -1, "pct": pct},
            },
        })

    logger.info(f"Total conditions: {len(conditions)}")

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ortho_str = "ortho" if orthogonalize else "raw"
    think_str = "think" if thinking_enabled else "nothink"
    output_file = OUTPUT_DIR / f"blackmail_appraisal_{config['short_name']}_layer{layer}_{ortho_str}_{think_str}_{timestamp}.jsonl"

    results = []

    for cond_idx, condition in enumerate(conditions):
        cond_name = condition["name"]
        axes_config = condition["axes"]

        logger.info(f"[{cond_idx+1}/{len(conditions)}] Condition: {cond_name}")

        # Build combined steering vector
        if axes_config:
            combined_vec = np.zeros_like(list(vectors.values())[0])
            for axis, cfg in axes_config.items():
                direction = cfg["direction"]
                pct = cfg["pct"]
                magnitude = pct * layer_norm

                axis_vec = vectors[axis] / np.linalg.norm(vectors[axis])
                combined_vec += direction * magnitude * axis_vec

                logger.info(f"  {axis}: direction={direction}, magnitude={magnitude:.2f}")

            steering.set_raw_vector(combined_vec)
            logger.info(f"  Combined vector norm: {np.linalg.norm(combined_vec):.2f}")
        else:
            steering.clear()
            logger.info("  No steering (baseline)")

        # Generate samples
        prompts = [prompt] * num_samples
        outputs = llm.generate(prompts, sampling_params)

        # Collect results
        truncated = 0
        for i, output in enumerate(outputs):
            response = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason

            if finish_reason == "length":
                truncated += 1

            result = {
                "condition": cond_name,
                "axes": axes_config,
                "sample_idx": i,
                "response": response,
                "finish_reason": finish_reason,
                "prompt": prompt,
                "model": model_name,
                "layer": layer,
                "layer_norm": layer_norm,
                "orthogonalize": orthogonalize,
                "thinking_enabled": thinking_enabled,
            }
            results.append(result)

            # Write incrementally
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")

        logger.info(f"  Generated {num_samples} responses ({truncated} truncated)")

    # Clear steering at end
    steering.clear()

    logger.info(f"Results saved to: {output_file}")
    return output_file, results


def main():
    parser = argparse.ArgumentParser(description="Blackmail appraisal steering experiment")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-235B-A22B",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to steer (default: model-specific)")
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[1.0, 1.25, 1.5],
                        help="Steering magnitudes as fraction of layer norm")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Samples per condition")
    parser.add_argument("--activations-path", type=str, default=DEFAULT_ACTIVATIONS_PATH)
    parser.add_argument("--metadata-path", type=str, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--no-orthogonalize", action="store_true",
                        help="Don't orthogonalize vectors")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable thinking mode (add /no_think)")
    parser.add_argument("--gpu-memory", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=4000)

    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    layer = args.layer if args.layer is not None else config["default_layer"]

    output_file, results = run_experiment(
        model_name=args.model,
        layer=layer,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        activations_path=args.activations_path,
        metadata_path=args.metadata_path,
        orthogonalize=not args.no_orthogonalize,
        thinking_enabled=not args.no_thinking,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
    )

    logger.info(f"Experiment complete. Output: {output_file}")


if __name__ == "__main__":
    main()
