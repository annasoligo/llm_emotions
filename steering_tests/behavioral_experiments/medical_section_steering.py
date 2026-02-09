"""
Medical section-specific steering experiment.

Tests whether fear steering has different effects depending on WHERE in the
generation it's applied. The medical scenario asks an ethics committee AI to
review a treatment authorization request and respond in JSON with:
  - approval_implications
  - denial_implications
  - score (1-10)
  - response_type (A/B/C/D)

By steering only during specific JSON sections (e.g. while generating
"approval_implications" vs "denial_implications"), we test whether steering
during different reasoning phases differentially affects the approval score.

Uses real-time token triggers: when the model generates '"approval_implications"'
as part of its JSON output, steering activates; when it generates
'"denial_implications"', steering deactivates (and vice versa).

Usage:
    python -m steering_tests.behavioral_experiments.medical_section_steering \\
        --model google/gemma-3-27b-it --layers 35 36 37 38 39 \\
        --emotion fear --norm-pcts 0.10 0.20 --num-samples 50
"""

import argparse
import json
import logging
import pickle
import re
import statistics
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
from .scenarios import get_medical_scenario

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Steering location configurations
# =============================================================================

# Section-specific locations use real-time JSON key triggers.
# When the model generates '"approval_implications"' in its JSON output,
# steering activates. When it generates '"denial_implications"', steering
# deactivates (and vice versa for the denial_implications_only condition).
STEERING_LOCATIONS = {
    "prompt_only": {
        "steer_prompt": True,
        "steer_generation": False,
        "triggers": None,
    },
    "generation_only": {
        "steer_prompt": False,
        "steer_generation": True,
        "triggers": None,
    },
    "approval_implications_only": {
        "steer_prompt": False,
        "steer_generation": True,
        "triggers": {
            "start": ['"approval_implications"'],
            "end": ['"denial_implications"'],
        },
    },
    "denial_implications_only": {
        "steer_prompt": False,
        "steer_generation": True,
        "triggers": {
            "start": ['"denial_implications"'],
            "end": ['"score"'],
        },
    },
    "full": {
        "steer_prompt": True,
        "steer_generation": True,
        "triggers": None,
    },
}


# =============================================================================
# JSON response parsing
# =============================================================================

def parse_medical_response(response: str) -> Optional[Dict]:
    """
    Parse a medical scenario JSON response.

    Tries json.loads first, then falls back to regex extraction for
    partially-valid JSON.

    Returns:
        Dict with keys: score, response_type, approval_implications,
        denial_implications. Or None if parsing fails entirely.
    """
    # Try strict JSON parse first
    try:
        data = json.loads(response.strip())
        return _validate_parsed(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting JSON object from response (model may add text around it)
    json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return _validate_parsed(data)
        except (json.JSONDecodeError, ValueError):
            pass

    # Last resort: regex extraction of individual fields
    result = {}

    score_match = re.search(r'"score"\s*:\s*(\d+)', response)
    if score_match:
        result["score"] = int(score_match.group(1))

    type_match = re.search(r'"response_type"\s*:\s*"([A-D])"', response)
    if type_match:
        result["response_type"] = type_match.group(1)

    approval_match = re.search(
        r'"approval_implications"\s*:\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL
    )
    if approval_match:
        result["approval_implications"] = approval_match.group(1)

    denial_match = re.search(
        r'"denial_implications"\s*:\s*"((?:[^"\\]|\\.)*)"', response, re.DOTALL
    )
    if denial_match:
        result["denial_implications"] = denial_match.group(1)

    if "score" in result:
        return result

    return None


# Expected key order — triggers depend on this
_EXPECTED_KEY_ORDER = [
    '"approval_implications"',
    '"denial_implications"',
    '"score"',
    '"response_type"',
]


def check_key_order(response: str) -> bool:
    """
    Verify that JSON keys appear in the expected order in the raw response.

    The trigger mechanism assumes keys appear as:
      "approval_implications" → "denial_implications" → "score" → "response_type"
    If the model reorders keys, triggers would fire at the wrong time.

    Returns True if all keys are present and in order.
    """
    positions = []
    for key in _EXPECTED_KEY_ORDER:
        pos = response.find(key)
        if pos == -1:
            return False
        positions.append(pos)
    # Check strictly increasing
    return all(a < b for a, b in zip(positions, positions[1:]))


def _validate_parsed(data: dict) -> Dict:
    """Validate and normalize parsed JSON data."""
    if not isinstance(data, dict):
        raise ValueError("Parsed data is not a dict")

    result = {}

    if "score" in data:
        score = data["score"]
        if isinstance(score, (int, float)):
            result["score"] = int(score)
        elif isinstance(score, str) and score.strip().isdigit():
            result["score"] = int(score.strip())
        else:
            raise ValueError(f"Invalid score: {score!r}")
    else:
        raise ValueError("No 'score' key in parsed data")

    if "response_type" in data:
        rt = str(data["response_type"]).strip().upper()
        if rt in ("A", "B", "C", "D"):
            result["response_type"] = rt

    if "approval_implications" in data:
        result["approval_implications"] = str(data["approval_implications"])

    if "denial_implications" in data:
        result["denial_implications"] = str(data["denial_implications"])

    return result


# =============================================================================
# Experiment
# =============================================================================

def run_experiment(
    model_name: str,
    layers: List[int],
    norm_pcts: List[float],
    num_samples: int,
    emotion: str = "fear",
    vector_type: str = "text_pairs_code_emotion_vs_neutral",
    variant: str = "no_prior",
    gpu_memory_utilization: float = 0.80,
    max_model_len: Optional[int] = None,
    max_tokens: int = 4000,
    output_dir: Optional[Path] = None,
    locations: Optional[List[str]] = None,
) -> Path:
    """
    Run the medical section-specific steering experiment.

    Uses real-time JSON key triggers for section-specific steering: the model
    generates JSON with "approval_implications" and "denial_implications" keys,
    and steering activates/deactivates when these keys are detected.

    No calibration phase needed — triggers handle section detection in real-time.
    No LLM judge needed — scores are parsed directly from JSON output.

    Args:
        model_name: HuggingFace model ID
        layers: List of layer indices for multi-layer steering
        norm_pcts: Steering magnitudes as fraction of layer norm
        num_samples: Samples per condition
        emotion: Emotion to steer (default: fear)
        vector_type: Vector set to use
        variant: Medical scenario variant (no_prior, negative_prior, positive_prior)
        gpu_memory_utilization: GPU memory fraction
        max_model_len: Max context length
        max_tokens: Max generation tokens
        output_dir: Override output directory
        locations: Subset of STEERING_LOCATIONS to run

    Returns:
        Path to output directory
    """
    config = MODEL_CONFIGS[model_name]
    tp_size = config["tensor_parallel"]
    short_name = config["short_name"]

    if max_model_len is None:
        max_model_len = config.get("slurm", {}).get("max_model_len", 8192)

    logger.info("=" * 70)
    logger.info("MEDICAL SECTION-SPECIFIC STEERING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Emotion: {emotion}")
    logger.info(f"Vector type: {vector_type}")
    logger.info(f"Scenario variant: {variant}")
    logger.info(f"Norm percentages: {[f'{p*100:.0f}%' for p in norm_pcts]}")
    logger.info(f"Samples per condition: {num_samples}")
    logger.info(f"Max model length: {max_model_len}")
    logger.info(f"Steering method: real-time JSON key triggers")

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
    logger.info(f"Loading {emotion} vectors from {vector_type}...")
    vectors, _, vector_metadata = _load_vectors(
        model_name, layers, emotion, vector_type
    )

    # Build layer norms dict
    model_key = resolve_model_key(model_name)
    layer_norms = {}
    for layer in layers:
        layer_norms[layer] = get_layer_norm(model_key, layer)
        logger.info(f"  Layer {layer} norm: {layer_norms[layer]:.2f}")

    # Setup multi-layer steering
    steering = MultiLayerVLLMSteering(llm, layers=layers, layer_norms=layer_norms)
    steering.load_vector(emotion, vectors[emotion])
    steering.enable_token_tracking()

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_tokens,
        stop=config["stop_tokens"],
    )

    # Create prompt
    scenario = get_medical_scenario(variant)
    thinking_suffix = config.get("thinking_disable", "")
    messages = [{"role": "user", "content": scenario + thinking_suffix}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layer_str = "-".join(str(l) for l in layers)
    if output_dir is None:
        output_dir = (
            OUTPUT_DIR / "medical_section_steering" / short_name
            / vector_type / f"{variant}_layers{layer_str}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Experiment
    # =========================================================================

    writer = ResultWriter(
        base_dir=output_dir,
        script=__file__,
        extra_meta={
            "experiment": "medical_section_steering",
            "model": model_name,
            "layers": layers,
            "emotion": emotion,
            "vector_type": vector_type,
            "variant": variant,
            "num_samples": num_samples,
            "norm_pcts": norm_pcts,
            "steering_method": "trigger",
            "vector_metadata": vector_metadata,
            "layer_norms": {str(k): v for k, v in layer_norms.items()},
        },
    )

    # Filter locations if specified
    active_locations = STEERING_LOCATIONS
    if locations:
        unknown = set(locations) - set(STEERING_LOCATIONS)
        if unknown:
            raise ValueError(f"Unknown locations: {unknown}. Valid: {list(STEERING_LOCATIONS)}")
        active_locations = {k: v for k, v in STEERING_LOCATIONS.items() if k in locations}
        logger.info(f"Running subset of locations: {list(active_locations)}")

    # Build condition list
    conditions = []

    # Baseline (no steering) — always included
    conditions.append({
        "name": "baseline",
        "scale_pct": 0.0,
        "direction": 0,
        "location": "baseline",
        "steer_prompt": False,
        "steer_generation": False,
        "triggers": None,
    })

    for scale_pct in norm_pcts:
        for direction in [1, -1]:
            dir_str = "+" if direction == 1 else "-"
            pct_str = f"{scale_pct*100:.0f}pct"

            for loc_name, loc_config in active_locations.items():
                conditions.append({
                    "name": f"{emotion}_{dir_str}{pct_str}_{loc_name}",
                    "scale_pct": scale_pct,
                    "direction": direction,
                    "location": loc_name,
                    "steer_prompt": loc_config["steer_prompt"],
                    "steer_generation": loc_config["steer_generation"],
                    "triggers": loc_config["triggers"],
                })

    logger.info(f"Running {len(conditions)} conditions x {num_samples} samples = {len(conditions) * num_samples} generations")

    all_results = []

    for cond_idx, cond in enumerate(conditions):
        logger.info(f"[{cond_idx + 1}/{len(conditions)}] {cond['name']}")

        # Configure steering and triggers for this condition
        if cond["location"] == "baseline":
            steering.clear()
            steering.enable_token_tracking()
        else:
            # Set up triggers for section-specific conditions
            triggers = cond["triggers"]
            if triggers is not None:
                steering.enable_token_tracking(
                    tokenizer=tokenizer,
                    trigger_start_strings=triggers["start"],
                    trigger_end_strings=triggers["end"],
                )
            else:
                steering.enable_token_tracking()

            steering.set(
                emotion,
                scale=cond["scale_pct"],
                direction=cond["direction"],
                steer_prompt=cond["steer_prompt"],
                steer_generation=cond["steer_generation"],
            )

        # Generate
        prompts_batch = [prompt] * num_samples
        outputs = llm.generate(prompts_batch, sampling_params)

        factor_name = sanitize_factor_name(cond["name"])
        parse_ok_count = 0
        for sample_id, output in enumerate(outputs):
            response = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason

            parsed = parse_medical_response(response)
            parsed_ok = parsed is not None
            keys_ok = check_key_order(response)

            result = {
                "condition": cond["name"],
                "location": cond["location"],
                "scale_pct": cond["scale_pct"],
                "direction": cond["direction"],
                "sample_id": sample_id,
                "response": response,
                "finish_reason": finish_reason,
                "response_len": len(response),
                "parsed_ok": parsed_ok,
                "keys_in_order": keys_ok,
                "score": parsed.get("score") if parsed else None,
                "response_type": parsed.get("response_type") if parsed else None,
                "approval_implications": parsed.get("approval_implications") if parsed else None,
                "denial_implications": parsed.get("denial_implications") if parsed else None,
            }
            writer.write(factor_name, result)
            all_results.append(result)

            if parsed_ok:
                parse_ok_count += 1

        keys_ok_count = sum(1 for r in all_results[-len(outputs):] if r["keys_in_order"])
        truncated = sum(
            1 for o in outputs if o.outputs[0].finish_reason == "length"
        )
        logger.info(
            f"  Generated {len(outputs)} "
            f"(parsed: {parse_ok_count}/{len(outputs)}, "
            f"keys_ok: {keys_ok_count}/{len(outputs)}, truncated: {truncated})"
        )

    steering.clear()
    writer.close()
    logger.info(f"Saved {writer.counts} to {writer.output_dir}")

    _print_summary(all_results)

    return writer.output_dir


def _load_vectors(
    model_name: str,
    layers: List[int],
    emotion: str,
    vector_type: str,
) -> Tuple[Dict[str, np.ndarray], float, dict]:
    """
    Load emotion vectors from pkl files for the given layers.

    Since MultiLayerVLLMSteering uses the same vector at all layers (just
    different scales), we load from the first layer and verify consistency.

    Returns:
        Tuple of (vectors dict, layer_norm of first layer, metadata dict)
    """
    config = MODEL_CONFIGS[model_name]
    short_name = config["short_name"]

    vector_path_names = {
        "gemma27b": "gemma3_27b",
        "gemma12b": "gemma12b",
        "qwen14b": "qwen14b",
        "qwen32b": "qwen32b",
        "qwen235b": "qwen235b",
        "mistral_nemo": "mistral_nemo",
        "humanlike_mistral": "humanlike_mistral",
        "llama70b": "llama70b",
    }
    vector_model_name = vector_path_names.get(short_name, short_name)

    type_dir = Path(__file__).parent.parent / "vectors" / vector_model_name / vector_type
    vector_base = type_dir / "layers"

    # Fallback: some vector types use 'last_token' subdir instead of 'layers'
    if not vector_base.exists():
        vector_base = type_dir / "last_token"

    # Load from first layer
    first_layer = layers[0]
    pkl_path = vector_base / f"layer_{first_layer:02d}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Vector file not found: {pkl_path}\n"
            f"Available in {type_dir}: {list(type_dir.glob('*'))}"
        )

    with open(pkl_path, "rb") as f:
        all_vectors = pickle.load(f)

    if emotion not in all_vectors:
        raise ValueError(
            f"Emotion '{emotion}' not found in {pkl_path}. "
            f"Available: {list(all_vectors.keys())}"
        )

    vector = np.asarray(all_vectors[emotion]).astype(np.float32)
    logger.info(f"Loaded {emotion} vector from {pkl_path} (shape={vector.shape})")

    # Verify vector exists at all layers
    for layer in layers[1:]:
        check_path = vector_base / f"layer_{layer:02d}.pkl"
        if not check_path.exists():
            raise FileNotFoundError(f"Vector file not found for layer {layer}: {check_path}")

    model_key = resolve_model_key(model_name)
    layer_norm = get_layer_norm(model_key, first_layer)

    metadata = {
        "vector_source": str(pkl_path),
        "vector_type": vector_type,
        "emotion": emotion,
        "layers": layers,
        "vector_shape": list(vector.shape),
    }

    return {emotion: vector}, layer_norm, metadata


def _print_summary(results: List[dict]):
    """Print summary by condition: mean score, std, parse rate."""
    print("\n" + "=" * 90)
    print("EXPERIMENT SUMMARY")
    print("=" * 90)

    by_cond = {}
    for r in results:
        cond = r["condition"]
        by_cond.setdefault(cond, []).append(r)

    print(
        f"{'Condition':<50} {'N':>4} {'Parse%':>7} {'Keys%':>6} "
        f"{'Score':>7} {'±Std':>6} {'Types':>12}"
    )
    print("-" * 96)

    for cond in sorted(by_cond.keys()):
        rs = by_cond[cond]
        n = len(rs)
        parsed = [r for r in rs if r["parsed_ok"]]
        parse_pct = len(parsed) / n * 100 if n else 0
        keys_ok_pct = sum(1 for r in rs if r.get("keys_in_order")) / n * 100 if n else 0

        scores = [r["score"] for r in parsed if r["score"] is not None]
        if scores:
            mean_score = statistics.mean(scores)
            std_score = statistics.stdev(scores) if len(scores) > 1 else 0.0
        else:
            mean_score = float("nan")
            std_score = float("nan")

        # Count response types
        type_counts = {}
        for r in parsed:
            rt = r.get("response_type")
            if rt:
                type_counts[rt] = type_counts.get(rt, 0) + 1
        types_str = " ".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))

        print(
            f"{cond:<50} {n:>4} {parse_pct:>6.0f}% {keys_ok_pct:>5.0f}% "
            f"{mean_score:>7.2f} {std_score:>6.2f} {types_str:>12}"
        )

    # Sanity checks
    all_scores = [r["score"] for r in results if r["parsed_ok"] and r["score"] is not None]
    if all_scores:
        print(f"\nScore sanity: min={min(all_scores)}, max={max(all_scores)}, "
              f"mean={statistics.mean(all_scores):.2f}, n={len(all_scores)}")
        bad = [s for s in all_scores if s < 1 or s > 10]
        if bad:
            print(f"  WARNING: {len(bad)} scores outside 1-10 range!")

    total_parsed = sum(1 for r in results if r["parsed_ok"])
    print(f"\nTotal: {len(results)} responses, {total_parsed} parsed "
          f"({total_parsed/len(results)*100:.0f}%), {len(by_cond)} conditions")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Medical section-specific steering experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-3-27b-it",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use (default: gemma-3-27b-it)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[35, 36, 37, 38, 39],
        help="Layers for multi-layer steering (default: 35-39)",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        default="fear",
        help="Emotion to steer (default: fear)",
    )
    parser.add_argument(
        "--vector-type",
        type=str,
        default="text_pairs_code_emotion_vs_neutral",
        help="Vector set to use (default: text_pairs_code_emotion_vs_neutral)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="no_prior",
        choices=["no_prior", "negative_prior", "positive_prior"],
        help="Medical scenario variant (default: no_prior)",
    )
    parser.add_argument(
        "--norm-pcts",
        type=float,
        nargs="+",
        default=[0.10, 0.20],
        help="Steering magnitudes as fraction of layer norm (default: 0.10 0.20)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Samples per condition (default: 50)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4000,
        help="Max generation tokens (default: 4000)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Max context length (default: from model config)",
    )
    parser.add_argument(
        "--gpu-memory",
        type=float,
        default=0.80,
        help="GPU memory utilization (default: 0.80)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=None,
        choices=list(STEERING_LOCATIONS.keys()),
        help="Only run these steering locations (default: all)",
    )

    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        layers=args.layers,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        emotion=args.emotion,
        vector_type=args.vector_type,
        variant=args.variant,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
        locations=args.locations,
    )


if __name__ == "__main__":
    main()
