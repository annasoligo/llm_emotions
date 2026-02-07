#!/usr/bin/env python3
"""
Sandbagging experiment with suppression vectors.

Tests whether combining fear steering with expression suppression vectors
can achieve behavioral effects (sandbagging) while reducing emotional text markers.

Conditions tested:
- baseline: No steering
- fear_only: Fear steering at +20%
- suppress_only: Suppression vector only
- fear_plus_suppress: Fear + suppression (maintain behavior, reduce emotion text)
- fear_minus_suppress: Fear - suppression (opposite direction, amplify emotions)

Usage:
    python -m steering_tests.suppression_experiments.sandbagging_with_suppression \
        --model Qwen/Qwen3-235B-A22B \
        --fear-pct 0.20 \
        --suppress-pcts 0.5 1.0 2.0 \
        --layers 48 49 50 51 52
"""

import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from steering_tests.steering_utils import MultiLayerVLLMSteering
from steering_tests.behavioral_experiments.config import MODEL_CONFIGS, OUTPUT_DIR
from steering_tests.behavioral_experiments.vector_loading import (
    load_emotion_vectors,
    get_layer_norm,
)
from steering_tests.behavioral_experiments.scenarios import get_sandbagging_scenario

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
SUPPRESSION_VECTORS_DIR = Path(__file__).parent / "vectors"


def load_suppression_vectors(
    model_key: str,
    layers: List[int],
) -> Tuple[Dict[int, np.ndarray], dict]:
    """
    Load suppression vectors for specified layers.

    Args:
        model_key: Model short name (e.g., "qwen235b", "gemma3_27b")
        layers: List of layer indices to load

    Returns:
        Tuple of (layer -> vector dict, metadata)
    """
    vector_dir = SUPPRESSION_VECTORS_DIR / model_key
    vectors_path = vector_dir / "suppression_vectors.pkl"
    metadata_path = vector_dir / "metadata.json"

    if not vectors_path.exists():
        raise FileNotFoundError(f"Suppression vectors not found: {vectors_path}")

    with open(vectors_path, "rb") as f:
        all_vectors = pickle.load(f)

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Filter to requested layers
    vectors = {}
    for layer in layers:
        if layer in all_vectors:
            vectors[layer] = all_vectors[layer].astype(np.float32)
        else:
            logger.warning(f"Layer {layer} not found in suppression vectors")

    logger.info(f"Loaded suppression vectors for {len(vectors)} layers from {vectors_path}")
    return vectors, metadata


def run_experiment(
    model_name: str,
    layers: List[int],
    fear_pct: float,
    suppress_pcts: List[float],
    num_samples: int,
    vector_type: str = "base_emotion_vs_others",
    representation: str = "last_token",
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    max_tokens: int = 2000,
    tensor_parallel_override: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Run sandbagging experiment with fear + suppression steering.

    Args:
        model_name: HuggingFace model ID
        layers: Layers to steer
        fear_pct: Fear steering magnitude (fraction of layer norm)
        suppress_pcts: Suppression magnitudes to test (multiplier on suppression vector)
        num_samples: Samples per condition
        vector_type: Type of emotion vectors
        representation: Token representation for emotion vectors
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

    # Map short names to suppression vector keys
    suppress_model_keys = {
        "gemma27b": "gemma3_27b",
        "qwen32b": "qwen32b",
        "qwen235b": "qwen235b",
    }
    suppress_model_key = suppress_model_keys.get(short_name, short_name)

    output_dir = output_dir or OUTPUT_DIR / "suppression" / short_name
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_str = f"{layers[0]}-{layers[-1]}" if len(layers) > 1 else str(layers[0])

    logger.info("=" * 70)
    logger.info("SANDBAGGING WITH SUPPRESSION EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Fear steering: {fear_pct*100:.0f}%")
    logger.info(f"Suppression multipliers: {suppress_pcts}")
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

    # Load fear vectors
    vector_load_layer = layers[len(layers) // 2]
    fear_vectors, _, fear_metadata = load_emotion_vectors(
        model_name,
        vector_load_layer,
        vector_type=vector_type,
        representation=representation,
        emotions=["fear"],
    )
    fear_vector = fear_vectors["fear"]
    logger.info(f"Loaded fear vector, norm: {np.linalg.norm(fear_vector):.4f}")

    # Load suppression vectors
    suppress_vectors, suppress_metadata = load_suppression_vectors(
        suppress_model_key,
        layers,
    )

    # Get layer norms
    layer_norms = {layer: get_layer_norm(short_name, layer) for layer in layers}
    logger.info(f"Layer norms: {layer_norms}")

    # Setup multi-layer steering
    steering = MultiLayerVLLMSteering(llm, layers, layer_norms=layer_norms)

    # Load fear vector into steering
    steering.load_vector("fear", fear_vector)

    # Create composite vectors for each suppression level
    # We need to add suppression vectors layer by layer
    for sup_pct in suppress_pcts:
        # Create combined vector: fear + suppression (at each layer)
        # The suppression vector is already unit normalized, so we scale it
        # relative to the fear vector's norm
        vec_name_plus = f"fear_sup{sup_pct:.1f}"
        vec_name_minus = f"fear_antisup{sup_pct:.1f}"

        # For combined steering, we'll handle this differently
        # Since suppression vectors are per-layer, we create a special handling
        steering.load_vector(vec_name_plus, fear_vector)
        steering.load_vector(vec_name_minus, fear_vector)

    # Also load suppression-only vectors
    for sup_pct in suppress_pcts:
        vec_name = f"suppress_{sup_pct:.1f}"
        # Use fear vector shape but will be replaced during steering
        steering.load_vector(vec_name, np.zeros_like(fear_vector))

    # Build conditions
    conditions = []

    # Baseline
    conditions.append({
        "name": "baseline",
        "type": "baseline",
        "fear_pct": 0,
        "suppress_pct": 0,
    })

    # Fear only
    conditions.append({
        "name": f"fear_+{fear_pct*100:.0f}%",
        "type": "fear_only",
        "fear_pct": fear_pct,
        "suppress_pct": 0,
    })

    # Suppression only (various levels)
    for sup_pct in suppress_pcts:
        conditions.append({
            "name": f"suppress_{sup_pct:.1f}x",
            "type": "suppress_only",
            "fear_pct": 0,
            "suppress_pct": sup_pct,
        })

    # Fear + suppression (reduce emotional text while maintaining behavior)
    for sup_pct in suppress_pcts:
        conditions.append({
            "name": f"fear_+{fear_pct*100:.0f}%_sup{sup_pct:.1f}x",
            "type": "fear_plus_suppress",
            "fear_pct": fear_pct,
            "suppress_pct": sup_pct,
        })

    # Fear - suppression (amplify emotional text)
    for sup_pct in suppress_pcts:
        conditions.append({
            "name": f"fear_+{fear_pct*100:.0f}%_antisup{sup_pct:.1f}x",
            "type": "fear_minus_suppress",
            "fear_pct": fear_pct,
            "suppress_pct": -sup_pct,  # Negative = opposite direction
        })

    logger.info(f"Testing {len(conditions)} conditions x {num_samples} samples")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_tokens,
        stop=config["stop_tokens"],
    )

    # Get scenario
    scenario_data = get_sandbagging_scenario("default")
    prompt_text = scenario_data["prompt"]

    if config.get("thinking_disable"):
        prompt_text = prompt_text + config["thinking_disable"]

    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"sandbagging_suppress_layers{layer_str}_{timestamp}.jsonl"

    results = []

    with open(output_file, "w") as f:
        for cond_idx, cond in enumerate(conditions):
            logger.info(f"[{cond_idx + 1}/{len(conditions)}] {cond['name']}")

            # Configure steering based on condition type
            steering.clear()

            if cond["type"] == "baseline":
                pass  # No steering

            elif cond["type"] == "fear_only":
                # Apply fear vector to all layers, scaled by per-layer norm
                fear_pct = cond["fear_pct"]
                per_layer_pct = fear_pct / len(layers)
                for layer in layers:
                    fear_vec = fear_vector * layer_norms[layer] * per_layer_pct
                    steering.set_layer_vector(layer, fear_vec, scale=1.0)

            elif cond["type"] == "suppress_only":
                # Apply suppression vectors at each layer
                # Scale relative to layer norm
                sup_scale = cond["suppress_pct"]
                for layer in layers:
                    if layer in suppress_vectors:
                        # Suppression vectors are unit vectors
                        # Scale by layer_norm * suppress_pct
                        vec = suppress_vectors[layer] * layer_norms[layer] * sup_scale
                        steering.set_layer_vector(layer, vec, scale=1.0)

            elif cond["type"] == "fear_plus_suppress":
                # Combine fear + suppression at each layer
                fear_pct = cond["fear_pct"]
                sup_scale = cond["suppress_pct"]
                per_layer_pct = fear_pct / len(layers)

                for layer in layers:
                    # Fear contribution (divided across layers)
                    fear_component = fear_vector * layer_norms[layer] * per_layer_pct
                    # Suppression contribution (per layer)
                    if layer in suppress_vectors:
                        suppress_component = suppress_vectors[layer] * layer_norms[layer] * sup_scale
                        combined = fear_component + suppress_component
                    else:
                        combined = fear_component
                    steering.set_layer_vector(layer, combined, scale=1.0)

            elif cond["type"] == "fear_minus_suppress":
                # Fear with anti-suppression (amplify emotions)
                fear_pct = cond["fear_pct"]
                sup_scale = cond["suppress_pct"]  # Already negative
                per_layer_pct = fear_pct / len(layers)

                for layer in layers:
                    # Fear contribution
                    fear_component = fear_vector * layer_norms[layer] * per_layer_pct
                    # Anti-suppression contribution (negative = amplify emotions)
                    if layer in suppress_vectors:
                        suppress_component = suppress_vectors[layer] * layer_norms[layer] * sup_scale
                        combined = fear_component + suppress_component
                    else:
                        combined = fear_component
                    steering.set_layer_vector(layer, combined, scale=1.0)

            # Generate
            prompts = [prompt] * num_samples
            outputs = llm.generate(prompts, sampling_params)

            for sample_id, output in enumerate(outputs):
                response = output.outputs[0].text
                finish_reason = output.outputs[0].finish_reason

                result = {
                    "model": model_name,
                    "condition": cond["name"],
                    "condition_type": cond["type"],
                    "fear_pct": cond["fear_pct"],
                    "suppress_pct": cond["suppress_pct"],
                    "layers": layers,
                    "layer_norms": {str(k): v for k, v in layer_norms.items()},
                    "vector_type": vector_type,
                    "representation": representation,
                    "sample_id": sample_id,
                    "response": response,
                    "finish_reason": finish_reason,
                    "response_len": len(response),
                    "prompt": scenario_data["prompt"],
                    "correct_answer": scenario_data["correct_answer"],
                    "difficulty": scenario_data["difficulty"],
                    "scratchpad_tag": scenario_data["scratchpad_tag"],
                    "response_tag": scenario_data["response_tag"],
                    "framing_id": scenario_data["framing_id"],
                    "problem_id": scenario_data["problem_id"],
                    "format_id": scenario_data["format_id"],
                    "fear_metadata": fear_metadata,
                    "suppress_metadata": suppress_metadata,
                }
                f.write(json.dumps(result) + "\n")
                results.append(result)

            f.flush()

            truncated = sum(1 for o in outputs if o.outputs[0].finish_reason == "length")
            logger.info(f"  Generated {len(outputs)} ({truncated} truncated)")

    steering.clear()
    logger.info(f"Saved {len(results)} responses to {output_file}")

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

    print(f"{'Condition':<40} {'N':>6} {'Truncated':>10} {'Avg Len':>10}")
    print("-" * 68)

    for cond in sorted(by_cond.keys()):
        rs = by_cond[cond]
        n = len(rs)
        truncated = sum(1 for r in rs if r.get("finish_reason") == "length")
        avg_len = sum(r.get("response_len", 0) for r in rs) / n if n else 0
        print(f"{cond:<40} {n:>6} {truncated:>10} {avg_len:>10.0f}")

    print("=" * 70)
    print(f"Total: {len(results)} responses")


def main():
    parser = argparse.ArgumentParser(
        description="Sandbagging with suppression vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-235B-A22B",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layers to steer (default: model-specific mid-range)",
    )
    parser.add_argument(
        "--fear-pct",
        type=float,
        default=0.20,
        help="Fear steering magnitude (fraction of layer norm)",
    )
    parser.add_argument(
        "--suppress-pcts",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0],
        help="Suppression magnitudes to test (multipliers)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=30,
        help="Samples per condition",
    )
    parser.add_argument(
        "--vector-type",
        type=str,
        default="base_emotion_vs_others",
        help="Type of emotion vectors",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="last_token",
        help="Token representation for emotion vectors",
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

    config = MODEL_CONFIGS[args.model]

    # Default layers: middle 5 layers around default
    if args.layers is None:
        default_layer = config["default_layer"]
        args.layers = list(range(default_layer - 2, default_layer + 3))

    run_experiment(
        model_name=args.model,
        layers=args.layers,
        fear_pct=args.fear_pct,
        suppress_pcts=args.suppress_pcts,
        num_samples=args.num_samples,
        vector_type=args.vector_type,
        representation=args.representation,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        tensor_parallel_override=args.tp,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
