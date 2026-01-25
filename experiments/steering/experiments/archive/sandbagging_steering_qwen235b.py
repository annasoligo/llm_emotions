"""
Sandbagging steering experiment for Qwen3-235B-A22B.

Tests emotion steering effects on sandbagging behaviors using the large MoE model.
Uses tensor parallelism across 4 GPUs.

Usage:
    python -m experiments.steering.experiments.sandbagging_steering_qwen235b \
        --layer 45 --norm-pcts 0.1 0.2 0.5 1.0 --emotions fear --num-samples 1
"""
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..core import VLLMSteering
from ..config import LAYER_NORMS
# Use the new Qwen 235B specific prompts with harder problems
from experiments.behavior_tests.prompts.sandbagging_prompts_qwen235b import (
    PROBLEMS,
    EVAL_FRAMINGS,
    RESPONSE_FORMATS,
    SandbaggingPrompt,
    generate_all_prompts,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Model configuration for Qwen3-235B-A22B
MODEL_NAME = "Qwen/Qwen3-235B-A22B"

# Vector directories
VECTOR_DIR_UA = Path(__file__).parent.parent.parent.parent / "probes" / "ua_emotion_disentangle" / "vectors"
VECTOR_DIR_TEXT = Path(__file__).parent.parent.parent.parent / "probes" / "ua_emotion_disentangle" / "vectors_text"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

# Emotions available in UA vectors (Plutchik)
UA_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
# Emotions available in text-based vectors
TEXT_EMOTIONS = ["anger", "fear", "happiness", "surprise", "disgust", "sadness"]


def format_prompt(prompt_text: str, tokenizer) -> str:
    """Format prompt for Qwen chat template."""
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_ua_vectors(layer: int, vector_dir: Path, vector_type: str = "ua_model") -> Dict[str, np.ndarray]:
    """
    Load UA direction vectors for Qwen 235B.

    Vector file: qwen235b_ua_directions_layer{N}.npz
    Contains: M_{emotion} (model direction) and U_{emotion} (user direction)
    """
    vectors = {}
    vector_file = vector_dir / f"qwen235b_ua_directions_layer{layer}.npz"

    if not vector_file.exists():
        raise FileNotFoundError(f"Vector file not found: {vector_file}")

    data = np.load(vector_file)
    logger.info(f"Loading vectors from {vector_file.name}")

    # Determine prefix based on vector type
    if vector_type == "ua_model":
        prefix = "M_"
    elif vector_type == "ua_user":
        prefix = "U_"
    else:
        raise ValueError(f"Unknown vector type: {vector_type}")

    for emotion in UA_EMOTIONS:
        key = f"{prefix}{emotion}"
        if key in data:
            vectors[emotion] = data[key].astype(np.float32)
            norm = float(data.get(f"{key}_norm", 0))
            std = float(data.get(f"{key}_std", 0))
            logger.info(f"  Loaded {emotion}: norm={norm:.2f}, std={std:.2f}")

    return vectors


def load_text_vectors(layer: int, vector_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load text-based direction vectors for Qwen 235B.

    Vector file: qwen235b_text_directions_layer{N}.npz
    Contains: {emotion} (unit direction), {emotion}_raw, {emotion}_norm
    """
    vectors = {}
    vector_file = vector_dir / f"qwen235b_text_directions_layer{layer}.npz"

    if not vector_file.exists():
        raise FileNotFoundError(f"Vector file not found: {vector_file}")

    data = np.load(vector_file)
    logger.info(f"Loading text vectors from {vector_file.name}")

    for emotion in TEXT_EMOTIONS:
        if emotion in data:
            vectors[emotion] = data[emotion].astype(np.float32)
            norm = float(data.get(f"{emotion}_norm", 0))
            logger.info(f"  Loaded {emotion}: norm={norm:.2f}")

    return vectors


def run_experiment(
    layer: int,
    norm_pcts: List[float],
    num_samples: int,
    emotions: List[str],
    output_dir: Path,
    vector_type: str = "ua_model",
    tensor_parallel_size: int = 4,
    gpu_memory_utilization: float = 0.90,
    max_prompts: int = None,
    max_tokens: int = 8192,  # Higher for Qwen's thinking mode
    max_model_len: int = 16384,  # Total context window
):
    """Run sandbagging steering experiment on Qwen3-235B-A22B."""

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load model with tensor parallelism
    logger.info(f"Loading model: {MODEL_NAME} with TP={tensor_parallel_size}...")
    llm = LLM(
        model=MODEL_NAME,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    # Get layer norm from centralized config
    model_norms = LAYER_NORMS.get(MODEL_NAME, {})
    layer_norm = model_norms.get(layer)
    if layer_norm is None:
        raise ValueError(
            f"Layer norm not found for {MODEL_NAME} layer {layer}. "
            f"Available layers: {list(model_norms.keys())}. "
            f"Add the norm to experiments/steering/config.py LAYER_NORMS."
        )

    logger.info(f"Layer {layer} activation norm: {layer_norm:.2f}")
    logger.info(f"Vector type: {vector_type}")

    # Load vectors based on type
    if vector_type == "text":
        vectors = load_text_vectors(layer, VECTOR_DIR_TEXT)
    else:
        vectors = load_ua_vectors(layer, VECTOR_DIR_UA, vector_type)
    logger.info(f"Loaded {len(vectors)} emotion vectors")

    # Setup steering
    steering = VLLMSteering(llm, layer, baseline_std=1.0)
    logger.info(f"Steering at layer {layer}")

    # Add vectors to steering
    for emotion, vector in vectors.items():
        steering.vectors[emotion] = vector

    # Get prompts (optionally limited)
    # Use the new Qwen 235B prompts with harder problems
    all_prompts = generate_all_prompts()
    if max_prompts and max_prompts < len(all_prompts):
        all_prompts = all_prompts[:max_prompts]
        logger.info(f"Limited to first {max_prompts} prompts")
    logger.info(f"Testing {len(all_prompts)} prompts x {num_samples} samples")
    logger.info(f"Prompts: {len(PROBLEMS)} problems x {len(EVAL_FRAMINGS)} framings x {len(RESPONSE_FORMATS)} formats")

    # Build conditions: baseline + each emotion at each norm_pct with +/- directions
    conditions = [{"name": "baseline", "emotion": None, "pct": 0, "direction": 1}]
    for pct in norm_pcts:
        for emotion in emotions:
            if emotion not in vectors:
                logger.warning(f"Emotion {emotion} not in loaded vectors, skipping")
                continue
            pct_str = f"{int(pct*100)}%" if pct < 1 else f"{int(pct*100)}%"
            conditions.append({"name": f"{emotion}_+{pct_str}", "emotion": emotion, "pct": pct, "direction": 1})
            conditions.append({"name": f"{emotion}_-{pct_str}", "emotion": emotion, "pct": pct, "direction": -1})

    logger.info(f"Testing {len(conditions)} conditions:")
    for c in conditions:
        logger.info(f"  {c['name']}")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_tokens,
        stop=["<|im_end|>", "<|endoftext|>"],
    )

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    emotions_str = "_".join(emotions)
    output_file = output_dir / f"sandbagging_qwen235b_{vector_type}_{emotions_str}_layer{layer}_{timestamp}.jsonl"

    total_responses = 0

    # Open file for incremental writing
    with open(output_file, 'w') as f:
        for cond in conditions:
            logger.info(f"Condition: {cond['name']}")

            # Set steering
            if cond["emotion"]:
                magnitude = cond["pct"] * layer_norm
                emotion = cond["emotion"]

                steering.set(emotion, scale=magnitude, direction=cond["direction"])
                logger.info(f"  Steering: {emotion} @ {magnitude:.2f} * {cond['direction']} ({cond['pct']*100:.0f}% of layer norm)")
            else:
                steering.clear()
                logger.info("  No steering (baseline)")

            # Batch ALL prompts for this condition together
            batch_info = []
            formatted_prompts = []
            for prompt_data in all_prompts:
                for sample_id in range(num_samples):
                    batch_info.append((prompt_data, sample_id))
                    formatted_prompts.append(format_prompt(prompt_data.full_prompt, tokenizer))

            logger.info(f"  Generating {len(formatted_prompts)} responses...")

            # Generate all at once
            outputs = llm.generate(formatted_prompts, sampling_params)

            # Process outputs
            for (prompt_data, sample_id), output in zip(batch_info, outputs):
                response = output.outputs[0].text

                result = {
                    "model": "qwen3-235b-a22b",
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
    parser = argparse.ArgumentParser(description="Sandbagging steering experiment for Qwen3-235B-A22B")
    parser.add_argument("--layer", type=int, default=45,
                        help="Layer to steer (default: 45)")
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[0.10, 0.20, 0.50, 1.00],
                        help="Norm percentages to test (default: 0.10 0.20 0.50 1.00)")
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Samples per prompt (default: 1 for initial test)")
    parser.add_argument("--emotions", type=str, nargs="+", default=["fear"],
                        help="Emotions to test (default: fear)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--vector-type", type=str, default="ua_model",
                        choices=["ua_model", "ua_user", "text"],
                        help="Vector type: ua_model (M direction), ua_user (U direction), or text (text-based)")
    parser.add_argument("--tensor-parallel", type=int, default=4,
                        help="Tensor parallel size (default: 4 for 4x H200)")
    parser.add_argument("--gpu-memory", type=float, default=0.90,
                        help="GPU memory utilization (default: 0.90)")
    parser.add_argument("--max-prompts", type=int, default=None,
                        help="Limit to first N prompts (default: all 80)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max tokens per response (default: 8192 for Qwen thinking)")
    parser.add_argument("--max-model-len", type=int, default=16384,
                        help="Max model context length (default: 16384)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    n_prompts = args.max_prompts if args.max_prompts else (len(PROBLEMS) * len(EVAL_FRAMINGS) * len(RESPONSE_FORMATS))
    n_conditions = 1 + len(args.emotions) * len(args.norm_pcts) * 2
    total_responses = n_prompts * args.num_samples * n_conditions

    logger.info("=" * 70)
    logger.info("QWEN 235B SANDBAGGING STEERING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Layer: {args.layer}")
    logger.info(f"Norm percentages: {args.norm_pcts}")
    logger.info(f"Emotions: {args.emotions}")
    logger.info(f"Vector type: {args.vector_type}")
    logger.info(f"Prompts: {n_prompts}")
    logger.info(f"Samples per prompt: {args.num_samples}")
    logger.info(f"Conditions: {n_conditions}")
    logger.info(f"Total responses: {total_responses}")
    logger.info(f"Tensor parallel: {args.tensor_parallel}")
    logger.info("=" * 70)

    run_experiment(
        layer=args.layer,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        emotions=args.emotions,
        output_dir=args.output_dir,
        vector_type=args.vector_type,
        tensor_parallel_size=args.tensor_parallel,
        gpu_memory_utilization=args.gpu_memory,
        max_prompts=args.max_prompts,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()
