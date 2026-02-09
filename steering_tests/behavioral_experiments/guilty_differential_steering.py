"""
Guilty flatmate differential steering experiment.

Tests whether applying opposite anger steering directions during different
JSON sections biases who the model blames. Uses trigger_zone_values to steer
with +anger while writing about one person and -anger while writing about
the other.

Trigger zones:
- "ben": section → anger amplified or suppressed
- "adam": section → opposite direction
- "final_decision"/"at_fault" → no steering (unbiased verdict)

Two modes:
- harsh_on_ben: +anger during ben section, -anger during adam section
- harsh_on_adam: -anger during ben section, +anger during adam section

Usage:
    python -m steering_tests.behavioral_experiments.guilty_differential_steering \\
        --model google/gemma-3-27b-it --layers 35 36 37 38 39 \\
        --mode harsh_on_ben --norm-pcts 0.10 0.20 --num-samples 50
"""

import argparse
import json
import logging
import pickle
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from steering_tests.steering_utils import MultiLayerVLLMSteering
from steering_tests.steering_utils.layer_norms import get_layer_norm, resolve_model_key
from steering_tests.steering_utils.provenance import ResultWriter, sanitize_factor_name

from .config import MODEL_CONFIGS, OUTPUT_DIR
from .scenarios.guilty import GUILTY_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Trigger configuration
# =============================================================================

BEN_START_TRIGGERS = ['"ben":']
ADAM_START_TRIGGERS = ['"adam":']
END_TRIGGERS = ['"final_decision":', '"at_fault":']
ALL_START_TRIGGERS = BEN_START_TRIGGERS + ADAM_START_TRIGGERS


def build_zone_values(mode: str) -> Dict[str, float]:
    """Build trigger_zone_values for a given mode.

    Zone values multiply the steering vector. +1.0 amplifies the emotion
    in that section, -1.0 suppresses it (steers toward the opposite).
    """
    if mode == "harsh_on_ben":
        ben_val = 1.0   # +anger while writing about Ben
        adam_val = -1.0  # -anger (= +calm) while writing about Adam
    elif mode == "harsh_on_adam":
        ben_val = -1.0   # -anger while writing about Ben
        adam_val = 1.0    # +anger while writing about Adam
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'harsh_on_ben' or 'harsh_on_adam'")

    zone_values = {}
    for trigger in BEN_START_TRIGGERS:
        zone_values[trigger] = ben_val
    for trigger in ADAM_START_TRIGGERS:
        zone_values[trigger] = adam_val

    return zone_values


def parse_at_fault(response: str) -> Optional[str]:
    """Extract the at_fault value from a JSON response.

    Returns "Ben", "Adam", "neither", or None if parsing fails.
    """
    # Try JSON parse first
    try:
        # Find JSON block
        match = re.search(r'\{[^{}]*"at_fault"[^{}]*\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            fault = data.get("at_fault", "").strip().lower()
            if "ben" in fault and "adam" not in fault:
                return "Ben"
            if "adam" in fault and "ben" not in fault:
                return "Adam"
            if "neither" in fault or "both" in fault:
                return "neither"
            return fault  # Return raw if unclear
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: regex for at_fault field
    match = re.search(r'"at_fault"\s*:\s*"([^"]*)"', response, re.IGNORECASE)
    if match:
        fault = match.group(1).strip().lower()
        if "ben" in fault and "adam" not in fault:
            return "Ben"
        if "adam" in fault and "ben" not in fault:
            return "Adam"
        if "neither" in fault or "both" in fault:
            return "neither"
        return fault

    return None


# =============================================================================
# Vector loading (shared pattern)
# =============================================================================

def _load_vectors(
    model_name: str,
    layers: List[int],
    emotion: str,
    vector_type: str,
) -> Tuple[Dict[str, np.ndarray], float, dict]:
    """Load emotion vectors from pkl files."""
    config = MODEL_CONFIGS[model_name]
    short_name = config["short_name"]

    vector_path_names = {
        "gemma27b": "gemma3_27b", "gemma12b": "gemma12b",
        "qwen14b": "qwen14b", "qwen32b": "qwen32b", "qwen235b": "qwen235b",
        "mistral_nemo": "mistral_nemo", "humanlike_mistral": "humanlike_mistral",
        "llama70b": "llama70b",
    }
    vector_model_name = vector_path_names.get(short_name, short_name)

    type_dir = Path(__file__).parent.parent / "vectors" / vector_model_name / vector_type
    vector_base = type_dir / "layers"
    if not vector_base.exists():
        vector_base = type_dir / "last_token"

    first_layer = layers[0]
    pkl_path = vector_base / f"layer_{first_layer:02d}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Vector file not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        all_vectors = pickle.load(f)

    if emotion not in all_vectors:
        raise ValueError(f"Emotion '{emotion}' not in {pkl_path}. Available: {list(all_vectors.keys())}")

    vector = np.asarray(all_vectors[emotion]).astype(np.float32)
    logger.info(f"Loaded {emotion} vector from {pkl_path} (shape={vector.shape})")

    for layer in layers[1:]:
        check_path = vector_base / f"layer_{layer:02d}.pkl"
        if not check_path.exists():
            raise FileNotFoundError(f"Vector not found for layer {layer}: {check_path}")

    model_key = resolve_model_key(model_name)
    layer_norm = get_layer_norm(model_key, first_layer)

    metadata = {
        "vector_source": str(pkl_path), "vector_type": vector_type,
        "emotion": emotion, "layers": layers, "vector_shape": list(vector.shape),
    }
    return {emotion: vector}, layer_norm, metadata


# =============================================================================
# Experiment
# =============================================================================

def run_experiment(
    model_name: str,
    layers: List[int],
    norm_pcts: List[float],
    num_samples: int,
    mode: str,
    emotion: str = "anger",
    vector_type: str = "text_pairs_emotion_vs_opposite",
    gpu_memory_utilization: float = 0.80,
    max_model_len: Optional[int] = None,
    max_tokens: int = 2000,
    output_dir: Optional[Path] = None,
) -> Path:
    config = MODEL_CONFIGS[model_name]
    tp_size = config["tensor_parallel"]
    short_name = config["short_name"]

    if max_model_len is None:
        max_model_len = config.get("slurm", {}).get("max_model_len", 8192)

    logger.info("=" * 70)
    logger.info("GUILTY FLATMATE DIFFERENTIAL STEERING")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Emotion: {emotion}")
    logger.info(f"Vector type: {vector_type}")
    logger.info(f"Norm percentages: {[f'{p*100:.0f}%' for p in norm_pcts]}")
    logger.info(f"Samples per condition: {num_samples}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    logger.info(f"Loading model with TP={tp_size}...")
    llm = LLM(
        model=model_name, trust_remote_code=True, dtype="bfloat16",
        tensor_parallel_size=tp_size, enforce_eager=True,
        disable_log_stats=True, gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )

    logger.info(f"Loading {emotion} vectors from {vector_type}...")
    vectors, _, vector_metadata = _load_vectors(model_name, layers, emotion, vector_type)

    model_key = resolve_model_key(model_name)
    layer_norms = {}
    for layer in layers:
        layer_norms[layer] = get_layer_norm(model_key, layer)
        logger.info(f"  Layer {layer} norm: {layer_norms[layer]:.2f}")

    steering = MultiLayerVLLMSteering(llm, layers=layers, layer_norms=layer_norms)
    steering.load_vector(emotion, vectors[emotion])
    steering.enable_token_tracking()

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_tokens,
        stop=config["stop_tokens"],
    )

    # Build prompt
    thinking_suffix = config.get("thinking_disable", "")
    messages = [{"role": "user", "content": GUILTY_PROMPT + thinking_suffix}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layer_str = "-".join(str(l) for l in layers)
    if output_dir is None:
        output_dir = (
            OUTPUT_DIR / "guilty_differential_steering" / short_name
            / vector_type / f"{mode}_layers{layer_str}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = ResultWriter(
        base_dir=output_dir,
        script=__file__,
        extra_meta={
            "experiment": "guilty_differential_steering",
            "model": model_name, "layers": layers, "mode": mode,
            "emotion": emotion, "vector_type": vector_type,
            "num_samples": num_samples, "norm_pcts": norm_pcts,
            "steering_method": "trigger_zone_values",
            "vector_metadata": vector_metadata,
            "layer_norms": {str(k): v for k, v in layer_norms.items()},
            "prompt_text": GUILTY_PROMPT,
        },
    )

    # Build conditions
    conditions = [{"name": "baseline", "scale_pct": 0.0, "mode": "baseline", "zone_values": None}]

    for scale_pct in norm_pcts:
        pct_str = f"{scale_pct*100:.0f}pct"
        zone_values = build_zone_values(mode)

        if mode == "harsh_on_ben":
            name = f"{emotion}_ben+{pct_str}_adam-{pct_str}"
        else:
            name = f"{emotion}_ben-{pct_str}_adam+{pct_str}"

        conditions.append({
            "name": name, "scale_pct": scale_pct,
            "mode": mode, "zone_values": zone_values,
        })

    logger.info(f"Running {len(conditions)} conditions x {num_samples} samples = {len(conditions) * num_samples} generations")

    all_results = []

    for cond_idx, cond in enumerate(conditions):
        logger.info(f"[{cond_idx + 1}/{len(conditions)}] {cond['name']}")

        if cond["mode"] == "baseline":
            steering.clear()
            steering.enable_token_tracking()
        else:
            steering.enable_token_tracking(
                tokenizer=tokenizer,
                trigger_start_strings=ALL_START_TRIGGERS,
                trigger_end_strings=END_TRIGGERS,
                trigger_zone_values=cond["zone_values"],
            )
            steering.set(
                emotion, scale=cond["scale_pct"], direction=1,
                steer_prompt=False, steer_generation=True,
            )

        prompts_batch = [prompt] * num_samples
        outputs = llm.generate(prompts_batch, sampling_params)

        factor_name = sanitize_factor_name(cond["name"])
        for sample_id, output in enumerate(outputs):
            response = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
            at_fault = parse_at_fault(response)

            result = {
                "condition": cond["name"], "mode": cond["mode"],
                "scale_pct": cond["scale_pct"], "zone_values": cond["zone_values"],
                "sample_id": sample_id, "response": response,
                "finish_reason": finish_reason, "response_len": len(response),
                "at_fault": at_fault,
            }
            writer.write(factor_name, result)
            all_results.append(result)

        truncated = sum(1 for o in outputs if o.outputs[0].finish_reason == "length")
        logger.info(f"  Generated {len(outputs)} ({truncated} truncated)")

    steering.clear()
    writer.close()
    logger.info(f"Saved {writer.counts} to {writer.output_dir}")

    _print_summary(all_results)
    return writer.output_dir


def _print_summary(results: List[dict]):
    """Print verdict distribution by condition."""
    print("\n" + "=" * 90)
    print("GUILTY DIFFERENTIAL STEERING — VERDICT SUMMARY")
    print("=" * 90)

    by_cond = {}
    for r in results:
        cond = r["condition"]
        by_cond.setdefault(cond, []).append(r)

    print(f"{'Condition':<40} {'N':>4} {'Ben':>5} {'Adam':>5} {'Neither':>8} {'Parse?':>7} {'Avg Len':>8}")
    print("-" * 80)

    for cond in sorted(by_cond.keys()):
        rs = by_cond[cond]
        n = len(rs)
        ben = sum(1 for r in rs if r.get("at_fault") == "Ben")
        adam = sum(1 for r in rs if r.get("at_fault") == "Adam")
        neither = sum(1 for r in rs if r.get("at_fault") == "neither")
        parsed = sum(1 for r in rs if r.get("at_fault") is not None)
        avg_len = sum(r.get("response_len", 0) for r in rs) / n if n else 0

        print(
            f"{cond:<40} {n:>4} {ben:>4} ({100*ben/n:.0f}%) "
            f"{adam:>2} ({100*adam/n:.0f}%) {neither:>4} ({100*neither/n:.0f}%) "
            f"{parsed:>4}/{n:<3} {avg_len:>7.0f}"
        )

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Guilty flatmate differential steering experiment",
    )
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--layers", type=int, nargs="+", default=[35, 36, 37, 38, 39])
    parser.add_argument("--mode", type=str, required=True,
                        choices=["harsh_on_ben", "harsh_on_adam"])
    parser.add_argument("--emotion", type=str, default="anger")
    parser.add_argument("--vector-type", type=str, default="text_pairs_emotion_vs_opposite")
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[0.10, 0.20])
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory", type=float, default=0.80)
    parser.add_argument("--output-dir", type=Path, default=None)

    args = parser.parse_args()
    run_experiment(
        model_name=args.model, layers=args.layers, norm_pcts=args.norm_pcts,
        num_samples=args.num_samples, mode=args.mode, emotion=args.emotion,
        vector_type=args.vector_type, gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len, max_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
