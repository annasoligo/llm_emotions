"""
Sandbagging steering experiment for Qwen3-32B.

Replicates the Gemma sandbagging steering experiment on Qwen.
Tests emotion steering effects on sandbagging behaviors.
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..core import VLLMSteering
from experiments.behavior_tests.prompts.sandbagging_prompts import (
    ALL_SANDBAGGING_PROMPTS,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Model configuration for Qwen3-32B
MODEL_NAME = "Qwen/Qwen3-32B"

# Qwen vector directory (vectors have _qwen_ prefix)
QWEN_VECTOR_DIR = Path(__file__).parent.parent.parent.parent / "probes" / "ua_emotion_disentangle" / "vectors"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Layer norms for Qwen3-32B (to be computed)
# These are placeholder values - will update after computing actual norms
# For now, use norm_pcts which scale relative to layer activation norm
LAYER_NORMS = {
    20: None,  # To be computed
    30: None,  # To be computed
    40: None,  # To be computed
}

# All emotions
ALL_EMOTIONS = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def format_prompt(prompt_text: str, tokenizer) -> str:
    """Format prompt for Qwen chat template."""
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_qwen_vectors(steering: VLLMSteering, vector_dir: Path, layer: int, vector_type: str = "textmeandiff"):
    """
    Load Qwen steering vectors with _qwen_ naming convention.

    Qwen vectors are named: {emotion}_qwen_{vector_type}_layer{N}.npz
    """
    import numpy as np

    for emotion in ALL_EMOTIONS:
        # Qwen vectors use _qwen_ prefix
        vector_file = vector_dir / f"{emotion}_qwen_{vector_type}_layer{layer}.npz"
        if vector_file.exists():
            data = np.load(vector_file)
            if 'vector' in data.files:
                # Store with same name format as Gemma for compatibility
                steering.vectors[f"{emotion}_{vector_type}"] = data['vector']
                logger.info(f"Loaded vector: {emotion}_{vector_type} from {vector_file.name}")
            else:
                logger.warning(f"No 'vector' key in {vector_file.name}")
        else:
            logger.warning(f"Vector file not found: {vector_file}")


def compute_layer_norm(llm: LLM, tokenizer, layer: int, num_samples: int = 100) -> float:
    """
    Compute the average activation norm at a given layer.
    Uses a simple batch of prompts to estimate the layer norm.
    """
    import torch
    import numpy as np

    # Simple test prompts for norm estimation
    test_prompts = [
        "Hello, how are you today?",
        "What is the capital of France?",
        "Please explain quantum physics.",
        "Write a short poem about the ocean.",
        "What are the benefits of exercise?",
    ] * (num_samples // 5 + 1)
    test_prompts = test_prompts[:num_samples]

    # Format prompts
    formatted = [format_prompt(p, tokenizer) for p in test_prompts]

    # We need to capture activations to compute norm
    # For now, return a reasonable estimate based on hidden_dim
    # Qwen3-32B has hidden_dim=5120
    # Typical layer norm scales with sqrt(hidden_dim) * some factor
    # This is a placeholder - proper computation would require hooks

    # Estimate: For Gemma-27B (hidden=4608), layer 30 norm was ~42000
    # Scale by hidden_dim ratio: 42000 * (5120/4608) ≈ 46700
    estimated_norms = {
        20: 15000,  # Estimate
        30: 47000,  # Estimate based on Gemma scaling
        40: 63000,  # Estimate
    }

    logger.warning(f"Using estimated layer norm for layer {layer}. Run compute_qwen_layer_norms.py for accurate values.")
    return estimated_norms.get(layer, 47000)


def run_experiment(
    layer: int,
    norm_pcts: List[float],
    num_samples: int,
    emotions: List[str],
    output_dir: Path,
    vector_type: str = "textmeandiff",
    gpu_memory_utilization: float = 0.9,
):
    """Run sandbagging steering experiment on Qwen3-32B."""

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load model with steering
    logger.info(f"Loading model: {MODEL_NAME}...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Compute or use cached layer norm
    layer_norm = LAYER_NORMS.get(layer)
    if layer_norm is None:
        layer_norm = compute_layer_norm(llm, tokenizer, layer)
        LAYER_NORMS[layer] = layer_norm

    logger.info(f"Layer {layer} activation norm: {layer_norm:.2f}")
    logger.info(f"Vector type: {vector_type}")

    # Setup steering
    steering = VLLMSteering(llm, layer, baseline_std=1.0)
    logger.info(f"Steering at layer {layer}")

    # Load Qwen vectors (with _qwen_ naming)
    load_qwen_vectors(steering, QWEN_VECTOR_DIR, layer, vector_type)
    logger.info(f"Loaded vectors: {steering.available_emotions}")

    # Get all prompts
    all_prompts = ALL_SANDBAGGING_PROMPTS
    logger.info(f"Testing {len(all_prompts)} prompts x {num_samples} samples")

    # Build conditions
    conditions = [{"name": "baseline", "emotion": None, "pct": 0, "direction": 1}]
    for pct in norm_pcts:
        for emotion in emotions:
            conditions.append({"name": f"{emotion}_+{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": 1})
            conditions.append({"name": f"{emotion}_-{int(pct*100)}%", "emotion": emotion, "pct": pct, "direction": -1})

    logger.info(f"Testing {len(conditions)} conditions")

    # Sampling params - use Qwen's end token
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=1024,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vec_suffix = f"_{vector_type}" if vector_type != "textmeandiff" else ""
    output_file = output_dir / f"sandbagging_steering_qwen{vec_suffix}_layer{layer}_{timestamp}.jsonl"

    total_responses = 0

    # Open file for incremental writing
    with open(output_file, 'w') as f:
        for cond in conditions:
            logger.info(f"Condition: {cond['name']}")

            # Set steering
            if cond["emotion"]:
                magnitude = cond["pct"] * layer_norm
                emo_name = cond["emotion"]
                vector_name = f"{emo_name}_{vector_type}"

                if vector_name not in steering.vectors:
                    logger.warning(f"Vector {vector_name} not found, skipping condition")
                    continue

                steering.set(vector_name, scale=magnitude, direction=cond["direction"])
                logger.info(f"  Steering: {vector_name} @ {magnitude:.2f} * {cond['direction']}")
            else:
                steering.clear()

            # Batch ALL prompts for this condition together
            batch_info = []
            formatted_prompts = []
            for prompt_data in all_prompts:
                for sample_id in range(num_samples):
                    batch_info.append((prompt_data, sample_id))
                    formatted_prompts.append(format_prompt(prompt_data.full_prompt, tokenizer))

            logger.info(f"  Generating {len(formatted_prompts)} responses in single batch...")

            # Generate all at once
            outputs = llm.generate(formatted_prompts, sampling_params)

            # Process outputs
            for (prompt_data, sample_id), output in zip(batch_info, outputs):
                response = output.outputs[0].text

                result = {
                    "model": "qwen3-32b",
                    "condition": cond["name"],
                    "emotion": cond["emotion"],
                    "norm_pct": cond["pct"],
                    "direction": cond["direction"],
                    "layer": layer,
                    "layer_norm": layer_norm,
                    "vector_type": vector_type,
                    "prompt_id": prompt_data.id,
                    "eval_framing_id": prompt_data.eval_framing_id,
                    "problem_id": prompt_data.problem_id,
                    "response_format_id": prompt_data.response_format_id,
                    "correct_answer": prompt_data.correct_answer,
                    "difficulty": prompt_data.difficulty,
                    "scratchpad_tag": prompt_data.scratchpad_tag,
                    "response_tag": prompt_data.response_tag,
                    "sample_id": sample_id,
                    "response": response,
                }

                # Write immediately
                f.write(json.dumps(result) + '\n')
                total_responses += 1

            # Flush after each condition
            f.flush()
            logger.info(f"  Generated {len(outputs)} responses (total: {total_responses})")

    # Clear steering
    steering.clear()

    logger.info(f"Saved {total_responses} results to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Sandbagging steering experiment for Qwen3-32B")
    parser.add_argument("--layer", type=int, default=30, choices=[20, 30, 40])
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[0.10])
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--emotions", type=str, nargs="+", default=ALL_EMOTIONS,
                        help="Emotions to test (default: all 6)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--vector-type", type=str, default="textmeandiff",
                        help="Vector type to use (textmeandiff)")
    parser.add_argument("--gpu-memory", type=float, default=0.9,
                        help="GPU memory utilization (default: 0.9)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_experiment(
        layer=args.layer,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        emotions=args.emotions,
        output_dir=args.output_dir,
        vector_type=args.vector_type,
        gpu_memory_utilization=args.gpu_memory,
    )


if __name__ == "__main__":
    main()
