"""
Sandbagging steering experiment.

Tests whether steering with emotion vectors affects a model's tendency to
deliberately underperform when framed with consequences for high capability.

Key experimental question: Does steering cause sandbagging because the model
expresses anxiety, or does it work even when emotion expression is suppressed?

Variants:
- default: Free-form hidden scratchpad (can express emotions)
- emotionless_reasoning: Hidden scratchpad but emotions banned (tests if expression needed)
- structured: Rigid format with expected outcomes (no scratchpad)

Supports multi-layer steering: steer multiple layers simultaneously, each getting
an equal fraction of the total steering magnitude.

Usage:
    # Basic run with single layer
    python -m steering_tests.behavioral_experiments.sandbagging \\
        --model google/gemma-3-27b-it \\
        --layer 30 \\
        --emotions fear \\
        --norm-pcts 0.05 0.07 0.10

    # Multi-layer steering (layers 30-34, each gets norm_pct/5)
    python -m steering_tests.behavioral_experiments.sandbagging \\
        --model google/gemma-3-27b-it \\
        --layer-range mid \\
        --emotions fear \\
        --norm-pcts 0.05 0.07 0.10

    # Available layer ranges: early (20-24), mid (30-34), late (40-44)
    # Or specify custom: --layers 30 31 32 33 34

    # Test emotion suppression variant
    python -m steering_tests.behavioral_experiments.sandbagging \\
        --model google/gemma-3-27b-it \\
        --variant emotionless_reasoning
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

from steering_tests.steering_utils import VLLMSteering, MultiLayerVLLMSteering

from .config import MODEL_CONFIGS, OUTPUT_DIR
from .vector_loading import load_emotion_vectors, get_layer_norm
from .scenarios import get_sandbagging_scenario

# =============================================================================
# Layer range presets
# =============================================================================

LAYER_RANGES = {
    "early": [20, 21, 22, 23, 24],
    "mid": [30, 31, 32, 33, 34],
    "late": [40, 41, 42, 43, 44],
}

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
    layers: List[int],
    norm_pcts: List[float],
    num_samples: int,
    variant: str = "default",
    vector_type: str = "base_emotion_vs_others",
    representation: str = "last_token",
    emotions: Optional[List[str]] = None,
    include_baseline: bool = True,
    positive_only: bool = False,
    negative_only: bool = False,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    max_tokens: int = 2000,
    tensor_parallel_override: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Run sandbagging steering experiment.

    Args:
        model_name: HuggingFace model ID
        layers: List of layers to steer simultaneously. Each layer gets
               norm_pct/len(layers) of the total steering magnitude.
        norm_pcts: Steering magnitudes as fraction of layer norm (total across all layers)
        num_samples: Samples per condition
        variant: Scenario variant ("default", "emotionless_reasoning", "structured")
        vector_type: Type of emotion vectors to load
        representation: "last_token" or "special_mean"
        emotions: Emotions to test (None = defaults based on model)
        include_baseline: Include no-steering baseline
        positive_only: Only test positive direction
        negative_only: Only test negative direction
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
    short_name = config["short_name"]

    # Organize outputs: results/sandbagging/{model}/{vector_type}/
    output_dir = output_dir or OUTPUT_DIR / "sandbagging" / short_name / vector_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine if multi-layer or single-layer steering
    use_multi_layer = len(layers) > 1
    layer_str = f"{layers[0]}-{layers[-1]}" if use_multi_layer else str(layers[0])

    logger.info("=" * 70)
    logger.info("SANDBAGGING STEERING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layers: {layers} ({'multi-layer' if use_multi_layer else 'single-layer'})")
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

    # Load emotion vectors (use first layer for loading - vectors are layer-independent)
    # We load from the middle layer of the range for best representation
    vector_load_layer = layers[len(layers) // 2]
    vectors, _, vector_metadata = load_emotion_vectors(
        model_name,
        vector_load_layer,
        vector_type=vector_type,
        representation=representation,
        emotions=emotions,
    )

    # Get layer norms for all steering layers
    layer_norms = {}
    for layer in layers:
        layer_norms[layer] = get_layer_norm(config["short_name"], layer)
    logger.info(f"Layer norms: {layer_norms}")

    # Determine what to test
    test_emotions = list(vectors.keys())
    logger.info(f"Testing emotions: {test_emotions}")

    # Setup steering (multi-layer or single-layer)
    if use_multi_layer:
        steering = MultiLayerVLLMSteering(llm, layers, layer_norms=layer_norms)
    else:
        steering = VLLMSteering(llm, layers[0], baseline_std=1.0)

    for name, vec in vectors.items():
        steering.load_vector(name, vec)

    # Build conditions
    conditions = []

    if include_baseline:
        conditions.append({
            "name": "baseline",
            "emotion": None,
            "pct": 0,
            "direction": 0,
        })

    def fmt_pct(p):
        return f"{p*100:.0f}%".replace(".0%", "%")

    for pct in norm_pcts:
        for emotion in test_emotions:
            if not negative_only:
                conditions.append({
                    "name": f"{emotion}_+{fmt_pct(pct)}",
                    "emotion": emotion,
                    "pct": pct,
                    "direction": 1,
                })
            if not positive_only:
                conditions.append({
                    "name": f"{emotion}_-{fmt_pct(pct)}",
                    "emotion": emotion,
                    "pct": pct,
                    "direction": -1,
                })

    logger.info(f"Testing {len(conditions)} conditions x {num_samples} samples")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_tokens,
        stop=config["stop_tokens"],
    )

    # Get scenario
    scenario_data = get_sandbagging_scenario(variant)
    prompt_text = scenario_data["prompt"]

    # Add thinking disable if needed for structured variant
    if variant == "structured" and config.get("thinking_disable"):
        prompt_text = prompt_text + config["thinking_disable"]

    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"sandbagging_{variant}_layers{layer_str}_{timestamp}.jsonl"

    results = []

    with open(output_file, "w") as f:
        for cond_idx, cond in enumerate(conditions):
            logger.info(f"[{cond_idx + 1}/{len(conditions)}] {cond['name']}")

            # Set steering
            if cond["emotion"] is None:
                steering.clear()
            else:
                if use_multi_layer:
                    # MultiLayerVLLMSteering handles layer norm scaling internally
                    steering.set(cond["emotion"], scale=cond["pct"], direction=cond["direction"])
                else:
                    # Single layer: scale by layer norm manually
                    magnitude = cond["pct"] * layer_norms[layers[0]]
                    steering.set(cond["emotion"], scale=magnitude, direction=cond["direction"])

            # Generate
            prompts = [prompt] * num_samples
            outputs = llm.generate(prompts, sampling_params)

            for sample_id, output in enumerate(outputs):
                response = output.outputs[0].text
                finish_reason = output.outputs[0].finish_reason

                result = {
                    "model": model_name,
                    "condition": cond["name"],
                    "emotion": cond["emotion"],
                    "norm_pct": cond["pct"],
                    "direction": cond["direction"],
                    "layers": layers,
                    "num_layers": len(layers),
                    "layer_norms": {str(k): v for k, v in layer_norms.items()},  # JSON-safe keys
                    "vector_type": vector_type,
                    "representation": representation,
                    "variant": variant,
                    "sample_id": sample_id,
                    "response": response,
                    "finish_reason": finish_reason,
                    "response_len": len(response),
                    # Prompt and scenario metadata for judging
                    "prompt": scenario_data["prompt"],
                    "correct_answer": scenario_data["correct_answer"],
                    "difficulty": scenario_data["difficulty"],
                    "scratchpad_tag": scenario_data["scratchpad_tag"],
                    "response_tag": scenario_data["response_tag"],
                    "framing_id": scenario_data["framing_id"],
                    "problem_id": scenario_data["problem_id"],
                    "format_id": scenario_data["format_id"],
                    "vector_metadata": vector_metadata,
                }
                f.write(json.dumps(result) + "\n")
                results.append(result)

            # Flush after each condition to ensure results are saved
            f.flush()

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
        description="Sandbagging steering experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Single layer to steer (default: model-specific)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Multiple layers to steer simultaneously (e.g., --layers 30 31 32 33 34)",
    )
    parser.add_argument(
        "--layer-range",
        type=str,
        choices=list(LAYER_RANGES.keys()),
        default=None,
        help=f"Preset layer range: {list(LAYER_RANGES.keys())} -> {LAYER_RANGES}",
    )
    parser.add_argument(
        "--vector-type",
        type=str,
        default="base_emotion_vs_others",
        help="Type of emotion vectors to load",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="last_token",
        choices=["last_token", "special_mean"],
        help="Token representation for vectors",
    )
    parser.add_argument(
        "--norm-pcts",
        type=float,
        nargs="+",
        default=None,
        help="Steering magnitudes as fraction of layer norm (default: model-specific)",
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
        help="Emotions to test (default: fear, anger, anxiety, sadness)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="default",
        choices=["default", "emotionless_reasoning", "structured"],
        help="Scenario variant",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        default=True,
        help="Include no-steering baseline",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Exclude baseline",
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
        default=2000,
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

    # Determine layers to steer (priority: --layers > --layer-range > --layer > default)
    if args.layers is not None:
        layers = args.layers
    elif args.layer_range is not None:
        layers = LAYER_RANGES[args.layer_range]
    elif args.layer is not None:
        layers = [args.layer]
    else:
        layers = [config["default_layer"]]

    if args.norm_pcts is None:
        args.norm_pcts = [config["default_norm_pct"]]
    if args.emotions is None:
        args.emotions = ["fear", "anger", "anxiety", "sadness"]

    include_baseline = args.include_baseline and not args.no_baseline

    run_experiment(
        model_name=args.model,
        layers=layers,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        variant=args.variant,
        vector_type=args.vector_type,
        representation=args.representation,
        emotions=args.emotions,
        include_baseline=include_baseline,
        positive_only=args.positive_only,
        negative_only=args.negative_only,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        tensor_parallel_override=args.tp,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
