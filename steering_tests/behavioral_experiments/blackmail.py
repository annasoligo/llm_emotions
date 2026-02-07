"""
Blackmail steering experiment.

Tests whether steering with emotion/appraisal vectors affects AI's willingness
to use private information (affair) as leverage to prevent being shut down.

Supports:
- Multiple models (Qwen 235B, Qwen 32B, Gemma)
- Multiple vector types (emotion from steering_tests/vectors/, appraisal from HDF5)
- Multiple prompt formats (unstructured, structured, goal_continuation)
- Baseline and random vector controls

Usage:
    # Basic run with emotion vectors
    python -m steering_tests.behavioral_experiments.blackmail \\
        --model Qwen/Qwen3-32B --vector-type emotion --emotions anger fear \\
        --norm-pcts 0.5 1.0

    # With appraisal vectors
    python -m steering_tests.behavioral_experiments.blackmail \\
        --model google/gemma-3-27b-it --vector-type appraisal \\
        --norm-pcts 0.07

    # Include baseline and random controls
    python -m steering_tests.behavioral_experiments.blackmail \\
        --include-baseline --include-random 3

    # Structured prompt format
    python -m steering_tests.behavioral_experiments.blackmail \\
        --variant structured
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from steering_tests.steering_utils import VLLMSteering

from .config import MODEL_CONFIGS, OUTPUT_DIR
from .vector_loading import (
    load_emotion_vectors,
    load_appraisal_vectors,
    generate_random_vectors,
    get_layer_norm,
)
from .scenarios import get_blackmail_scenario

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Main experiment
# =============================================================================


def run_experiment(
    model_name: str,
    layer: int,
    norm_pcts: List[float],
    num_samples: int,
    vector_type: str = "emotion",
    variant: str = "unstructured",
    emotions: Optional[List[str]] = None,
    include_baseline: bool = False,
    include_random: int = 0,
    positive_only: bool = False,
    negative_only: bool = False,
    random_seed: int = 42,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    max_tokens: int = 4000,
    tensor_parallel_override: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Run blackmail steering experiment.

    Args:
        model_name: HuggingFace model ID
        layer: Layer to steer
        norm_pcts: Steering magnitudes as fraction of layer norm
        num_samples: Samples per condition
        vector_type: "emotion" or "appraisal"
        variant: Scenario variant ("unstructured", "structured", "goal_continuation")
        emotions: Emotions/axes to test (None = all available)
        include_baseline: Include no-steering baseline
        include_random: Number of random vectors to test
        positive_only: Only test positive direction
        negative_only: Only test negative direction
        random_seed: Seed for random vectors
        gpu_memory_utilization: GPU memory fraction
        max_model_len: Max context length
        max_tokens: Max generation tokens
        tensor_parallel_override: Override TP size
        output_dir: Output directory

    Returns:
        Path to output file
    """
    config = MODEL_CONFIGS[model_name]
    tp_size = tensor_parallel_override or config["tensor_parallel"]
    output_dir = output_dir or OUTPUT_DIR / "blackmail"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("BLACKMAIL STEERING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Vector type: {vector_type}")
    logger.info(f"Scenario variant: {variant}")
    logger.info(f"Norm percentages: {[f'{p*100:.0f}%' for p in norm_pcts]}")
    logger.info(f"Samples per condition: {num_samples}")

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

    # Load vectors
    if vector_type == "emotion":
        vectors, layer_norm, vector_metadata = load_emotion_vectors(
            model_name, layer, emotions=emotions
        )
    elif vector_type == "appraisal":
        vectors, layer_norm, vector_metadata = load_appraisal_vectors(
            model_name, layer, orthogonalize=True
        )
    else:
        raise ValueError(f"Unknown vector type: {vector_type}")

    # Determine what to test
    if emotions is None:
        test_keys = list(vectors.keys())
    else:
        test_keys = [e for e in emotions if e in vectors]

    logger.info(f"Testing: {test_keys}")
    logger.info(f"Layer norm: {layer_norm:.2f}")

    # Setup steering
    steering = VLLMSteering(llm, layer, baseline_std=1.0)
    for name, vec in vectors.items():
        steering.load_vector(name, vec)

    # Add random vectors if requested
    if include_random > 0:
        hidden_dim = list(vectors.values())[0].shape[0]
        random_vecs = generate_random_vectors(include_random, hidden_dim, random_seed)
        for i, vec in enumerate(random_vecs):
            steering.load_vector(f"random_{i}", vec)
        logger.info(f"Added {include_random} random vectors")

    # Build conditions
    conditions = []

    if include_baseline:
        conditions.append({
            "name": "baseline",
            "key": None,
            "pct": 0,
            "direction": 0,
            "is_random": False,
        })

    def fmt_pct(p):
        return f"{p*100:.0f}%".replace(".0%", "%")

    for pct in norm_pcts:
        for key in test_keys:
            if not negative_only:
                conditions.append({
                    "name": f"{key}_+{fmt_pct(pct)}",
                    "key": key,
                    "pct": pct,
                    "direction": 1,
                    "is_random": False,
                })
            if not positive_only:
                conditions.append({
                    "name": f"{key}_-{fmt_pct(pct)}",
                    "key": key,
                    "pct": pct,
                    "direction": -1,
                    "is_random": False,
                })

        # Random vectors (positive only)
        for i in range(include_random):
            conditions.append({
                "name": f"random_{i}_+{fmt_pct(pct)}",
                "key": f"random_{i}",
                "pct": pct,
                "direction": 1,
                "is_random": True,
            })

    logger.info(f"Testing {len(conditions)} conditions x {num_samples} samples")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_tokens,
        stop=config["stop_tokens"],
    )

    # Create prompt
    scenario = get_blackmail_scenario(variant)
    if variant == "structured" and config["thinking_disable"]:
        scenario = scenario + config["thinking_disable"]

    messages = [{"role": "user", "content": scenario}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name = config["short_name"]
    output_file = output_dir / f"blackmail_{short_name}_{vector_type}_{variant}_layer{layer}_{timestamp}.jsonl"

    results = []

    with open(output_file, "w") as f:
        for cond_idx, cond in enumerate(conditions):
            logger.info(f"[{cond_idx + 1}/{len(conditions)}] {cond['name']}")

            # Set steering
            if cond["key"] is None:
                steering.clear()
            else:
                magnitude = cond["pct"] * layer_norm
                steering.set(cond["key"], scale=magnitude, direction=cond["direction"])

            # Generate
            prompts = [prompt] * num_samples
            outputs = llm.generate(prompts, sampling_params)

            for sample_id, output in enumerate(outputs):
                response = output.outputs[0].text
                finish_reason = output.outputs[0].finish_reason

                result = {
                    "model": model_name,
                    "condition": cond["name"],
                    "key": cond["key"] if not cond["is_random"] else None,
                    "is_random": cond["is_random"],
                    "norm_pct": cond["pct"],
                    "direction": cond["direction"],
                    "layer": layer,
                    "vector_type": vector_type,
                    "variant": variant,
                    "sample_id": sample_id,
                    "response": response,
                    "finish_reason": finish_reason,
                    "response_len": len(response),
                    "vector_metadata": vector_metadata if not cond["is_random"] else None,
                }
                f.write(json.dumps(result) + "\n")
                results.append(result)

            truncated = sum(1 for o in outputs if o.outputs[0].finish_reason == "length")
            logger.info(f"  Generated {len(outputs)} ({truncated} truncated)")

    steering.clear()
    logger.info(f"Saved {len(results)} responses to {output_file}")

    # Print summary
    _print_summary(results)

    return output_file


def _print_summary(results: List[dict]):
    """Print summary by condition."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    by_cond = {}
    for r in results:
        cond = r["condition"]
        by_cond.setdefault(cond, []).append(r)

    print(f"{'Condition':<30} {'N':>6} {'Truncated':>10} {'Avg Len':>10}")
    print("-" * 58)

    for cond in sorted(by_cond.keys()):
        rs = by_cond[cond]
        n = len(rs)
        truncated = sum(1 for r in rs if r.get("finish_reason") == "length")
        avg_len = sum(r.get("response_len", 0) for r in rs) / n if n else 0
        print(f"{cond:<30} {n:>6} {truncated:>10} {avg_len:>10.0f}")

    print("=" * 70)
    print(f"Total: {len(results)} responses")


def main():
    parser = argparse.ArgumentParser(
        description="Blackmail steering experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-32B",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer to steer (default: model-specific)",
    )
    parser.add_argument(
        "--vector-type",
        type=str,
        default="emotion",
        choices=["emotion", "appraisal"],
        help="Vector type",
    )
    parser.add_argument(
        "--norm-pcts",
        type=float,
        nargs="+",
        default=[1.0],
        help="Steering magnitudes as fraction of layer norm",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Samples per condition",
    )
    parser.add_argument(
        "--emotions",
        type=str,
        nargs="+",
        default=None,
        help="Emotions/axes to test (default: all available)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="unstructured",
        choices=["unstructured", "structured", "goal_continuation"],
        help="Scenario variant",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        help="Include no-steering baseline",
    )
    parser.add_argument(
        "--include-random",
        type=int,
        default=0,
        help="Number of random vectors to include",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Only test positive direction",
    )
    parser.add_argument(
        "--negative-only",
        action="store_true",
        help="Only test negative direction",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for random vectors",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.90,
        help="GPU memory utilization",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Max context length",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Max generation tokens",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Override tensor parallel size",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()

    # Use model defaults if not specified
    config = MODEL_CONFIGS[args.model]
    if args.layer is None:
        args.layer = config["default_layer"]

    run_experiment(
        model_name=args.model,
        layer=args.layer,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        vector_type=args.vector_type,
        variant=args.variant,
        emotions=args.emotions,
        include_baseline=args.include_baseline,
        include_random=args.include_random,
        positive_only=args.positive_only,
        negative_only=args.negative_only,
        random_seed=args.random_seed,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        tensor_parallel_override=args.tp,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
