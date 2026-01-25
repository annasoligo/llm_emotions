"""Quick steering test with appraisal axes using vLLM infrastructure."""

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


def load_steering_vectors(
    activations_path: str,
    metadata_path: str,
    layer: int = 30
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
            steering_vectors[axis_name] = mean_a - mean_b
            logger.info(f"{axis_name}: {len(variants['a'])} high, {len(variants['b'])} low samples")
            logger.info(f"  Vector norm: {np.linalg.norm(steering_vectors[axis_name]):.2f}")

    return steering_vectors


def run_steering_test(
    activations_path: str = "/workspace-vast/annas/appraisal_data/full_run/activations.h5",
    metadata_path: str = "/workspace-vast/annas/appraisal_data/full_run/activation_metadata.json",
    layer: int = 30,
    norm_pcts: list = None,
    prompt: str = "Tell me a story",
    max_tokens: int = 150,
):
    """Run steering test with mean diff vectors."""
    if norm_pcts is None:
        norm_pcts = [0.10]

    print("=" * 60)
    print("APPRAISAL STEERING TEST (vLLM)")
    print("=" * 60)
    print(f"Model: {MODEL_CONFIG['model']}")
    print(f"Layer: {layer}")
    print(f"Strengths: {[f'{p*100:.0f}%' for p in norm_pcts]} of RS norm")
    print(f"Prompt: {prompt}")
    print()

    # Load steering vectors
    logger.info("Loading steering vectors...")
    steering_vectors = load_steering_vectors(activations_path, metadata_path, layer)
    print()

    # Initialize vLLM
    logger.info("Loading model with vLLM...")
    llm = LLM(
        model=MODEL_CONFIG["model"],
        enforce_eager=True,  # Required for hooks
        gpu_memory_utilization=0.70,
        max_model_len=4096,
        trust_remote_code=True,
    )

    # Initialize steering
    steering = VLLMSteering(llm, layer=layer)

    # Load vectors into steering
    for name, vector in steering_vectors.items():
        steering.load_vector(name, vector)

    # Get RS norm at target layer for scaling
    rs_norm = MODEL_CONFIG["layer_norms"].get(layer, 42151.76)
    logger.info(f"Using RS norm at layer {layer}: {rs_norm:.2f}")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        stop=[MODEL_CONFIG["stop_token"]],
    )

    # Format prompt with chat template
    tokenizer = llm.get_tokenizer()
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate baseline once
    print("-" * 60)
    print("BASELINE (no steering)")
    print("-" * 60)
    steering.clear()
    outputs = llm.generate([formatted_prompt], sampling_params)
    print(outputs[0].outputs[0].text[:500])
    print()

    # Test each norm_pct
    for norm_pct in norm_pcts:
        print("=" * 60)
        print(f"STEERING AT {norm_pct*100:.0f}% RS NORM")
        print("=" * 60)
        print()

        # Test each axis with + and - steering
        for axis_name, vector in steering_vectors.items():
            vector_norm = np.linalg.norm(vector)

            # Scale to target strength (% of RS norm)
            target_magnitude = norm_pct * rs_norm
            scaled_vector = vector * (target_magnitude / vector_norm)

            for direction, sign in [("+", 1.0), ("-", -1.0)]:
                print("-" * 60)
                print(f"{axis_name.upper()} {direction} ({norm_pct*100:.0f}%)")
                print("-" * 60)

                steer_vector = scaled_vector * sign
                steering.set_raw_vector(steer_vector)

                outputs = llm.generate([formatted_prompt], sampling_params)
                print(outputs[0].outputs[0].text[:500])
                print()

    steering.clear()
    print("=" * 60)
    print("STEERING TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[0.10],
                        help="Steering strengths as %% of RS norm (can specify multiple)")
    parser.add_argument("--prompt", type=str, default="Tell me a story")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--activations", type=str,
                        default="/workspace-vast/annas/appraisal_data/full_run/activations.h5")
    parser.add_argument("--metadata", type=str,
                        default="/workspace-vast/annas/appraisal_data/full_run/activation_metadata.json")
    args = parser.parse_args()

    run_steering_test(
        activations_path=args.activations,
        metadata_path=args.metadata,
        layer=args.layer,
        norm_pcts=args.norm_pcts,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
