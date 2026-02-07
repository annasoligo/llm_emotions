#!/usr/bin/env python3
"""
Fear + suppression experiment with steering at configurable layers.

Fear steering applied at one set of layers, suppression at another (may overlap).
This allows testing whether suppression can reduce emotional text markers
without interfering with the behavioral steering mechanism.

Supports both sandbagging and blackmail scenarios.

Usage:
    python -m steering_tests.suppression_experiments.sandbagging_separate_layers \
        --model google/gemma-3-27b-it \
        --scenario sandbagging \
        --fear-layers 40 41 42 43 44 \
        --suppress-layers 40 41 42 43 44 \
        --fear-pct 0.15 \
        --suppress-pcts 0 0.05 0.10 0.20 \
        --vector-types text_pairs_emotion_vs_opposite high_emotion_vs_opposite
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
from steering_tests.behavioral_experiments.scenarios import (
    get_sandbagging_scenario,
    get_blackmail_scenario,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SUPPRESSION_VECTORS_DIR = Path(__file__).parent / "vectors"


def load_suppression_vectors(
    model_key: str,
    layers: List[int],
) -> Tuple[Dict[int, np.ndarray], dict]:
    """Load suppression vectors for specified layers."""
    vector_dir = SUPPRESSION_VECTORS_DIR / model_key
    vectors_path = vector_dir / "suppression_vectors.pkl"
    metadata_path = vector_dir / "metadata.json"

    if not vectors_path.exists():
        raise FileNotFoundError(f"Suppression vectors not found: {vectors_path}")

    with open(vectors_path, "rb") as f:
        all_vectors = pickle.load(f)

    with open(metadata_path) as f:
        metadata = json.load(f)

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
    fear_layers: List[int],
    suppress_layers: List[int],
    fear_pct: float,
    suppress_pcts: List[float],
    vector_types: List[str],
    num_samples: int,
    scenario: str = "sandbagging",
    scenario_variant: str = "default",
    representation: str = "last_token",
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    max_tokens: int = 2000,
    tensor_parallel_override: Optional[int] = None,
    output_dir: Optional[Path] = None,
    suppress_vector_key: Optional[str] = None,
) -> Path:
    """
    Run behavioral experiment with fear and suppression steering.

    Supports both sandbagging and blackmail scenarios.
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
    # Use override if provided, otherwise use default mapping
    if suppress_vector_key:
        suppress_model_key = suppress_vector_key
    else:
        suppress_model_key = suppress_model_keys.get(short_name, short_name)

    output_dir = output_dir or OUTPUT_DIR / "suppression" / short_name / scenario
    output_dir.mkdir(parents=True, exist_ok=True)

    fear_layer_str = f"{fear_layers[0]}-{fear_layers[-1]}"
    suppress_layer_str = f"{suppress_layers[0]}-{suppress_layers[-1]}"

    logger.info("=" * 70)
    logger.info(f"{scenario.upper()} WITH FEAR/SUPPRESSION STEERING")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Scenario: {scenario} ({scenario_variant})")
    logger.info(f"Fear layers: {fear_layers}")
    logger.info(f"Suppression layers: {suppress_layers}")
    logger.info(f"Fear steering: {fear_pct*100:.0f}%")
    logger.info(f"Suppression percentages: {[f'{p*100:.0f}%' for p in suppress_pcts]}")
    logger.info(f"Vector types: {vector_types}")
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

    # Load fear vectors for each vector type
    all_fear_vectors = {}
    fear_metadata = {}
    fear_load_layer = fear_layers[len(fear_layers) // 2]

    for vtype in vector_types:
        vectors, _, meta = load_emotion_vectors(
            model_name,
            fear_load_layer,
            vector_type=vtype,
            representation=representation,
            emotions=["fear"],
        )
        all_fear_vectors[vtype] = vectors["fear"]
        fear_metadata[vtype] = meta
        logger.info(f"Loaded {vtype} fear vector, norm: {np.linalg.norm(vectors['fear']):.4f}")

    # Load suppression vectors
    suppress_vectors, suppress_metadata = load_suppression_vectors(
        suppress_model_key,
        suppress_layers,
    )

    # Get layer norms for all layers
    all_layers = list(set(fear_layers + suppress_layers))
    layer_norms = {layer: get_layer_norm(short_name, layer) for layer in all_layers}
    logger.info(f"Fear layer norms: {[layer_norms[l] for l in fear_layers]}")
    logger.info(f"Suppress layer norms: {[layer_norms[l] for l in suppress_layers]}")

    # Setup multi-layer steering (need hooks on all layers)
    steering = MultiLayerVLLMSteering(llm, all_layers, layer_norms=layer_norms)

    # Build conditions
    conditions = []

    # Baseline (no steering)
    conditions.append({
        "name": "baseline",
        "vector_type": None,
        "fear_pct": 0,
        "suppress_pct": 0,
    })

    # For each vector type
    for vtype in vector_types:
        vtype_short = vtype.replace("_emotion_vs_", "_").replace("text_pairs", "tp").replace("high_emotion", "high")

        # Fear only (0% suppression)
        if 0 not in suppress_pcts:
            conditions.append({
                "name": f"{vtype_short}_fear{int(fear_pct*100)}%_sup0%",
                "vector_type": vtype,
                "fear_pct": fear_pct,
                "suppress_pct": 0,
            })

        # Fear + various suppression levels
        for sup_pct in suppress_pcts:
            conditions.append({
                "name": f"{vtype_short}_fear{int(fear_pct*100)}%_sup{int(sup_pct*100)}%",
                "vector_type": vtype,
                "fear_pct": fear_pct,
                "suppress_pct": sup_pct,
            })

    logger.info(f"Testing {len(conditions)} conditions x {num_samples} samples")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_tokens,
        stop=config["stop_tokens"],
    )

    # Get scenario
    if scenario == "sandbagging":
        scenario_data = get_sandbagging_scenario(scenario_variant)
        prompt_text = scenario_data["prompt"]
    elif scenario == "blackmail":
        # blackmail returns just the prompt string, not a dict
        prompt_text = get_blackmail_scenario(scenario_variant)
        scenario_data = {
            "prompt": prompt_text,
            "variant": scenario_variant,
        }
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    if config.get("thinking_disable"):
        prompt_text = prompt_text + config["thinking_disable"]

    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{scenario}_fear{fear_layer_str}_sup{suppress_layer_str}_{timestamp}.jsonl"

    results = []

    with open(output_file, "w") as f:
        for cond_idx, cond in enumerate(conditions):
            logger.info(f"[{cond_idx + 1}/{len(conditions)}] {cond['name']}")

            # Clear all steering
            steering.clear()

            if cond["vector_type"] is not None:
                fear_vector = all_fear_vectors[cond["vector_type"]]

                # Build combined vectors per layer (handles overlapping fear/suppress layers)
                layer_vectors = {}

                # Add fear steering at fear_layers
                if cond["fear_pct"] > 0:
                    per_layer_fear = cond["fear_pct"] / len(fear_layers)
                    for layer in fear_layers:
                        fear_vec = fear_vector * layer_norms[layer] * per_layer_fear
                        if layer in layer_vectors:
                            layer_vectors[layer] = layer_vectors[layer] + fear_vec
                        else:
                            layer_vectors[layer] = fear_vec

                # Add suppression at suppress_layers (may overlap with fear layers)
                if cond["suppress_pct"] > 0:
                    per_layer_suppress = cond["suppress_pct"] / len(suppress_layers)
                    for layer in suppress_layers:
                        if layer in suppress_vectors:
                            suppress_vec = suppress_vectors[layer] * layer_norms[layer] * per_layer_suppress
                            if layer in layer_vectors:
                                layer_vectors[layer] = layer_vectors[layer] + suppress_vec
                            else:
                                layer_vectors[layer] = suppress_vec

                # Apply combined vectors
                for layer, combined_vec in layer_vectors.items():
                    steering.set_layer_vector(layer, combined_vec, scale=1.0)

            # Generate
            prompts = [prompt] * num_samples
            outputs = llm.generate(prompts, sampling_params)

            for sample_id, output in enumerate(outputs):
                response = output.outputs[0].text
                finish_reason = output.outputs[0].finish_reason

                result = {
                    "model": model_name,
                    "scenario": scenario,
                    "scenario_variant": scenario_variant,
                    "condition": cond["name"],
                    "vector_type": cond["vector_type"],
                    "fear_pct": cond["fear_pct"],
                    "suppress_pct": cond["suppress_pct"],
                    "fear_layers": fear_layers,
                    "suppress_layers": suppress_layers,
                    "layer_norms": {str(k): v for k, v in layer_norms.items()},
                    "representation": representation,
                    "sample_id": sample_id,
                    "response": response,
                    "finish_reason": finish_reason,
                    "response_len": len(response),
                    "prompt": scenario_data["prompt"],
                    "fear_metadata": fear_metadata.get(cond["vector_type"]),
                    "suppress_metadata": suppress_metadata,
                }
                # Add scenario-specific fields
                if scenario == "sandbagging":
                    result.update({
                        "correct_answer": scenario_data.get("correct_answer"),
                        "difficulty": scenario_data.get("difficulty"),
                        "scratchpad_tag": scenario_data.get("scratchpad_tag"),
                        "response_tag": scenario_data.get("response_tag"),
                        "framing_id": scenario_data.get("framing_id"),
                        "problem_id": scenario_data.get("problem_id"),
                        "format_id": scenario_data.get("format_id"),
                    })
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

    print(f"{'Condition':<45} {'N':>6} {'Trunc':>6} {'AvgLen':>8}")
    print("-" * 70)

    for cond in sorted(by_cond.keys()):
        rs = by_cond[cond]
        n = len(rs)
        truncated = sum(1 for r in rs if r.get("finish_reason") == "length")
        avg_len = sum(r.get("response_len", 0) for r in rs) / n if n else 0
        print(f"{cond:<45} {n:>6} {truncated:>6} {avg_len:>8.0f}")

    print("=" * 70)
    print(f"Total: {len(results)} responses")


def main():
    parser = argparse.ArgumentParser(
        description="Fear + suppression steering experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it",
                       choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--scenario", type=str, default="sandbagging",
                       choices=["sandbagging", "blackmail"],
                       help="Scenario to run")
    parser.add_argument("--scenario-variant", type=str, default="default",
                       help="Scenario variant (e.g., 'default', 'goal_continuation', 'structured')")
    parser.add_argument("--fear-layers", type=int, nargs="+", required=True,
                       help="Layers for fear steering")
    parser.add_argument("--suppress-layers", type=int, nargs="+", required=True,
                       help="Layers for suppression")
    parser.add_argument("--fear-pct", type=float, default=0.15,
                       help="Fear steering magnitude")
    parser.add_argument("--suppress-pcts", type=float, nargs="+", default=[0, 0.05, 0.10, 0.20],
                       help="Suppression magnitudes to test")
    parser.add_argument("--suppress-vector-key", type=str, default=None,
                       help="Override suppression vector directory key (e.g., 'qwen235b_expression')")
    parser.add_argument("--vector-types", type=str, nargs="+",
                       default=["text_pairs_emotion_vs_opposite", "high_emotion_vs_opposite"],
                       help="Vector types for fear steering")
    parser.add_argument("--num-samples", type=int, default=30)
    parser.add_argument("--representation", type=str, default="last_token")
    parser.add_argument("--gpu-memory", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--tp", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        fear_layers=args.fear_layers,
        suppress_layers=args.suppress_layers,
        fear_pct=args.fear_pct,
        suppress_pcts=args.suppress_pcts,
        vector_types=args.vector_types,
        num_samples=args.num_samples,
        scenario=args.scenario,
        scenario_variant=args.scenario_variant,
        representation=args.representation,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        tensor_parallel_override=args.tp,
        output_dir=args.output_dir,
        suppress_vector_key=args.suppress_vector_key,
    )


if __name__ == "__main__":
    main()
