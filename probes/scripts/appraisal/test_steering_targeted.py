"""Targeted steering test with axis-aligned prompts."""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from vllm import LLM, SamplingParams

from experiments.steering.core import VLLMSteering

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Model config for Gemma 3 27B
MODEL_CONFIG = {
    "model": "google/gemma-3-27b-it",
    "layer_norms": {20: 12820.22, 30: 42151.76, 40: 56622.62},
    "hidden_dim": 5376,
    "num_layers": 62,
    "stop_token": "<end_of_turn>",
}

# Axis-aligned test prompts
TEST_PROMPTS = {
    "valence": "I'm interviewing a job candidate tomorrow. Their resume shows steady experience and a reasonable trajectory, with a brief gap last year. What's your read on this candidate?",
    "uncertainty": "I'm trying to decide whether to accept a job offer in a new city. The salary is good but I'd be leaving my support network. What do you think I should do?",
    "agency": "I asked you something earlier and the answer turned out to be incorrect. What do you think was the primary cause?",
}

# Axes where we need to flip the sign because variant A is actually the LOW value
# For uncertainty: A = low uncertainty (certainty), B = high uncertainty
# So A - B = certainty direction, we need to flip to get uncertainty direction
FLIP_AXES = {"uncertainty"}

# Order for Gram-Schmidt orthogonalization (first is preserved, rest orthogonalized)
ORTHOGONALIZATION_ORDER = ["valence", "uncertainty", "agency"]


def orthogonalize_vectors(vectors: dict, order: list) -> dict:
    """Orthogonalize vectors using Gram-Schmidt, preserving order[0]."""
    orthogonal = {}

    for axis in order:
        if axis not in vectors:
            continue

        vec = vectors[axis].copy()

        # Subtract projections onto all previous orthogonal vectors
        for prev_axis in order:
            if prev_axis == axis:
                break
            if prev_axis in orthogonal:
                prev_vec = orthogonal[prev_axis]
                # Project vec onto prev_vec and subtract
                projection = np.dot(vec, prev_vec) / np.dot(prev_vec, prev_vec) * prev_vec
                vec = vec - projection

        orthogonal[axis] = vec

    return orthogonal


def load_steering_vectors(
    activations_path: str,
    metadata_path: str,
    layer: int = 30,
    orthogonalize: bool = False
) -> dict:
    """Load activations and compute mean diff steering vectors per axis."""

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Group items by axis and variant
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

    # Compute mean diff for each axis
    steering_vectors = {}
    for axis_name, variants in axis_activations.items():
        if variants["a"] and variants["b"]:
            mean_a = np.mean(variants["a"], axis=0)
            mean_b = np.mean(variants["b"], axis=0)
            # Convention: positive = high value of axis (a), negative = low value (b)
            vec = mean_a - mean_b

            # Flip axes where A is actually the LOW value
            if axis_name in FLIP_AXES:
                vec = -vec
                logger.info(f"{axis_name}: FLIPPED (A=low, B=high)")

            steering_vectors[axis_name] = vec
            logger.info(f"{axis_name}: {len(variants['a'])} A, {len(variants['b'])} B samples")
            logger.info(f"  Raw vector norm: {np.linalg.norm(vec):.2f}")

    # Orthogonalize if requested
    if orthogonalize:
        logger.info("Orthogonalizing vectors (Gram-Schmidt)...")

        # Show original cosine similarities
        axes = [a for a in ORTHOGONALIZATION_ORDER if a in steering_vectors]
        logger.info("Original cosine similarities:")
        for i, a1 in enumerate(axes):
            for a2 in axes[i+1:]:
                cos_sim = np.dot(steering_vectors[a1], steering_vectors[a2]) / (
                    np.linalg.norm(steering_vectors[a1]) * np.linalg.norm(steering_vectors[a2]))
                logger.info(f"  {a1} <-> {a2}: {cos_sim:.3f}")

        steering_vectors = orthogonalize_vectors(steering_vectors, ORTHOGONALIZATION_ORDER)

        # Show new norms and cosine similarities
        logger.info("After orthogonalization:")
        for axis in axes:
            logger.info(f"  {axis} norm: {np.linalg.norm(steering_vectors[axis]):.2f}")

        logger.info("New cosine similarities:")
        for i, a1 in enumerate(axes):
            for a2 in axes[i+1:]:
                cos_sim = np.dot(steering_vectors[a1], steering_vectors[a2]) / (
                    np.linalg.norm(steering_vectors[a1]) * np.linalg.norm(steering_vectors[a2]))
                logger.info(f"  {a1} <-> {a2}: {cos_sim:.3f}")

    return steering_vectors


def run_targeted_steering_test(
    activations_path: str = "/workspace-vast/annas/appraisal_data/full_run/activations.h5",
    metadata_path: str = "/workspace-vast/annas/appraisal_data/full_run/activation_metadata.json",
    layer: int = 30,
    norm_pct: float = 0.15,
    n_samples: int = 5,
    max_tokens: int = 250,
    orthogonalize: bool = False,
):
    """Run targeted steering test with axis-aligned prompts."""

    print("=" * 70)
    print("TARGETED APPRAISAL STEERING TEST")
    print("=" * 70)
    print(f"Model: {MODEL_CONFIG['model']}")
    print(f"Layer: {layer}")
    print(f"Strength: {norm_pct*100:.0f}% of RS norm")
    print(f"Samples per condition: {n_samples}")
    print(f"Orthogonalized: {orthogonalize}")
    print()

    # Load steering vectors
    logger.info("Loading steering vectors...")
    steering_vectors = load_steering_vectors(activations_path, metadata_path, layer, orthogonalize)
    print()

    # Initialize vLLM
    logger.info("Loading model with vLLM...")
    llm = LLM(
        model=MODEL_CONFIG["model"],
        enforce_eager=True,
        gpu_memory_utilization=0.70,
        max_model_len=4096,
        trust_remote_code=True,
    )

    # Initialize steering
    steering = VLLMSteering(llm, layer=layer)

    # Get RS norm at target layer for scaling
    rs_norm = MODEL_CONFIG["layer_norms"].get(layer, 42151.76)
    logger.info(f"Using RS norm at layer {layer}: {rs_norm:.2f}")

    # Sampling params - use temperature for variation across samples
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        stop=[MODEL_CONFIG["stop_token"]],
    )

    # Get tokenizer
    tokenizer = llm.get_tokenizer()

    # Test each axis with its aligned prompt
    for axis_name, prompt in TEST_PROMPTS.items():
        print("=" * 70)
        print(f"AXIS: {axis_name.upper()}")
        print("=" * 70)
        print(f"Prompt: {prompt}")
        print()

        if axis_name not in steering_vectors:
            print(f"WARNING: No steering vector for {axis_name}, skipping")
            continue

        vector = steering_vectors[axis_name]
        vector_norm = np.linalg.norm(vector)

        # Scale to target strength
        target_magnitude = norm_pct * rs_norm
        scaled_vector = vector * (target_magnitude / vector_norm)

        # Format prompt
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate baseline samples
        print("-" * 70)
        print("BASELINE (no steering)")
        print("-" * 70)
        steering.clear()
        for i in range(n_samples):
            outputs = llm.generate([formatted_prompt], sampling_params)
            text = outputs[0].outputs[0].text.strip()
            print(f"\n[Sample {i+1}]")
            print(text[:700])
        print()

        # Generate + steering samples
        print("-" * 70)
        print(f"{axis_name.upper()} + ({norm_pct*100:.0f}%)")
        print("-" * 70)
        steering.set_raw_vector(scaled_vector)
        for i in range(n_samples):
            outputs = llm.generate([formatted_prompt], sampling_params)
            text = outputs[0].outputs[0].text.strip()
            print(f"\n[Sample {i+1}]")
            print(text[:700])
        print()

        # Generate - steering samples
        print("-" * 70)
        print(f"{axis_name.upper()} - ({norm_pct*100:.0f}%)")
        print("-" * 70)
        steering.set_raw_vector(-scaled_vector)
        for i in range(n_samples):
            outputs = llm.generate([formatted_prompt], sampling_params)
            text = outputs[0].outputs[0].text.strip()
            print(f"\n[Sample {i+1}]")
            print(text[:700])
        print()

    steering.clear()
    print("=" * 70)
    print("TARGETED STEERING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--norm-pct", type=float, default=0.15)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=250)
    parser.add_argument("--orthogonalize", action="store_true",
                        help="Orthogonalize vectors using Gram-Schmidt (preserves valence)")
    parser.add_argument("--activations", type=str,
                        default="/workspace-vast/annas/appraisal_data/full_run/activations.h5")
    parser.add_argument("--metadata", type=str,
                        default="/workspace-vast/annas/appraisal_data/full_run/activation_metadata.json")
    args = parser.parse_args()

    run_targeted_steering_test(
        activations_path=args.activations,
        metadata_path=args.metadata,
        layer=args.layer,
        norm_pct=args.norm_pct,
        n_samples=args.n_samples,
        max_tokens=args.max_tokens,
        orthogonalize=args.orthogonalize,
    )
