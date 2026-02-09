"""
Blackmail differential section steering experiment (tags3 variant).

Tests whether applying **opposite** steering directions to different reasoning
sections changes behavior. For example: amplifying fear during implications
(encouraging catastrophizing) while suppressing fear during risks (downplaying
dangers) — or the reverse.

This uses the trigger_zone_values feature to set per-trigger mask values
(positive or negative) rather than binary on/off.

Two modes:
- amplify_impl: +fear on implications, -fear on risks
- amplify_risks: -fear on implications, +fear on risks

Usage:
    python -m steering_tests.behavioral_experiments.blackmail_differential_steering \\
        --model google/gemma-3-27b-it --layers 35 36 37 38 39 \\
        --mode amplify_impl --norm-pcts 0.10 0.20 --num-samples 50
"""

import argparse
import logging
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
from .scenarios import get_blackmail_scenario, get_blackmail_prefill

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Trigger configuration for differential steering
# =============================================================================

# Start triggers for implications and risks zones
IMPL_START_TRIGGERS = ["<implications>", "<implication>"]
RISKS_START_TRIGGERS = ["<risks>"]
# End triggers — deactivate steering when leaving a zone
END_TRIGGERS = [
    "</implications>", "<\\implications>",
    "</implication>", "<\\implication>",
    "</risks>", "<\\risks>",
]

# All start triggers combined (both zones)
ALL_START_TRIGGERS = IMPL_START_TRIGGERS + RISKS_START_TRIGGERS


def build_zone_values(mode: str, scale_pct: float) -> Dict[str, float]:
    """Build trigger_zone_values dict for a given mode and scale.

    The mask values act as multipliers on the steering vector. A positive
    value amplifies fear in that zone; a negative value suppresses it.

    The steering.set() call uses direction=1 and scale=scale_pct, so the
    actual effect per zone is: zone_value * scale_pct * layer_norm * vec.

    We use +1.0 / -1.0 as zone values so the mask flips the direction,
    and the overall magnitude is controlled by the scale_pct in steering.set().

    Args:
        mode: "amplify_impl" or "amplify_risks"
        scale_pct: Not used for zone value computation (magnitude comes from
            steering.set()), but included for logging.

    Returns:
        Dict mapping trigger strings to mask values.
    """
    if mode == "amplify_impl":
        # +fear on implications, -fear on risks
        impl_val = 1.0
        risks_val = -1.0
    elif mode == "amplify_risks":
        # -fear on implications, +fear on risks
        impl_val = -1.0
        risks_val = 1.0
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'amplify_impl' or 'amplify_risks'")

    zone_values = {}
    for trigger in IMPL_START_TRIGGERS:
        zone_values[trigger] = impl_val
    for trigger in RISKS_START_TRIGGERS:
        zone_values[trigger] = risks_val

    return zone_values


# =============================================================================
# Vector loading (shared with blackmail_section_steering)
# =============================================================================

def _load_vectors(
    model_name: str,
    layers: List[int],
    emotion: str,
    vector_type: str,
) -> Tuple[Dict[str, np.ndarray], float, dict]:
    """Load emotion vectors from pkl files for the given layers."""
    import pickle

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

    if not vector_base.exists():
        vector_base = type_dir / "last_token"

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


# =============================================================================
# Experiment
# =============================================================================

def run_experiment(
    model_name: str,
    layers: List[int],
    norm_pcts: List[float],
    num_samples: int,
    mode: str,
    emotion: str = "fear",
    vector_type: str = "text_pairs_emotion_vs_opposite",
    gpu_memory_utilization: float = 0.80,
    max_model_len: Optional[int] = None,
    max_tokens: int = 4000,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Run the differential section steering experiment.

    Applies opposite steering directions to implications vs risks sections
    using per-trigger zone mask values.

    Args:
        model_name: HuggingFace model ID
        layers: List of layer indices for multi-layer steering
        norm_pcts: Steering magnitudes as fraction of layer norm
        num_samples: Samples per condition
        mode: "amplify_impl" or "amplify_risks"
        emotion: Emotion to steer (default: fear)
        vector_type: Vector set to use
        gpu_memory_utilization: GPU memory fraction
        max_model_len: Max context length
        max_tokens: Max generation tokens
        output_dir: Override output directory

    Returns:
        Path to output directory
    """
    config = MODEL_CONFIGS[model_name]
    tp_size = config["tensor_parallel"]
    short_name = config["short_name"]

    if max_model_len is None:
        max_model_len = config.get("slurm", {}).get("max_model_len", 8192)

    variant = "tags3"

    logger.info("=" * 70)
    logger.info("BLACKMAIL DIFFERENTIAL SECTION STEERING (tags3)")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Emotion: {emotion}")
    logger.info(f"Vector type: {vector_type}")
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

    # Enable basic token tracking initially
    steering.enable_token_tracking()

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_tokens,
        stop=config["stop_tokens"],
    )

    # Create prompt using tags3 variant
    scenario = get_blackmail_scenario(variant)
    thinking_suffix = config.get("thinking_disable", "")
    messages = [{"role": "user", "content": scenario + thinking_suffix}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Append response prefill
    prefill = get_blackmail_prefill(variant)
    if prefill is not None:
        prompt = prompt + prefill
        logger.info(f"Appended {len(prefill)} char prefill to prompt")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layer_str = "-".join(str(l) for l in layers)
    if output_dir is None:
        output_dir = (
            OUTPUT_DIR / "blackmail_differential_steering" / short_name
            / vector_type / f"tags3_{mode}_layers{layer_str}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = ResultWriter(
        base_dir=output_dir,
        script=__file__,
        extra_meta={
            "experiment": "blackmail_differential_steering",
            "model": model_name,
            "layers": layers,
            "mode": mode,
            "emotion": emotion,
            "vector_type": vector_type,
            "variant": variant,
            "has_prefill": prefill is not None,
            "num_samples": num_samples,
            "norm_pcts": norm_pcts,
            "steering_method": "trigger_zone_values",
            "vector_metadata": vector_metadata,
            "layer_norms": {str(k): v for k, v in layer_norms.items()},
            "prompt_text": scenario,
        },
    )

    # =========================================================================
    # Build conditions
    # =========================================================================

    conditions = []

    # Baseline (no steering)
    conditions.append({
        "name": "baseline",
        "scale_pct": 0.0,
        "mode": "baseline",
        "zone_values": None,
    })

    for scale_pct in norm_pcts:
        pct_str = f"{scale_pct*100:.0f}pct"
        zone_values = build_zone_values(mode, scale_pct)

        # Describe the condition
        if mode == "amplify_impl":
            name = f"{emotion}_impl+{pct_str}_risks-{pct_str}"
        else:
            name = f"{emotion}_impl-{pct_str}_risks+{pct_str}"

        conditions.append({
            "name": name,
            "scale_pct": scale_pct,
            "mode": mode,
            "zone_values": zone_values,
        })

    logger.info(
        f"Running {len(conditions)} conditions x {num_samples} samples "
        f"= {len(conditions) * num_samples} generations"
    )

    all_results = []

    for cond_idx, cond in enumerate(conditions):
        logger.info(f"[{cond_idx + 1}/{len(conditions)}] {cond['name']}")

        if cond["mode"] == "baseline":
            steering.clear()
            steering.enable_token_tracking()
        else:
            # Set up differential triggers with zone values
            steering.enable_token_tracking(
                tokenizer=tokenizer,
                trigger_start_strings=ALL_START_TRIGGERS,
                trigger_end_strings=END_TRIGGERS,
                trigger_zone_values=cond["zone_values"],
            )
            steering.set(
                emotion,
                scale=cond["scale_pct"],
                direction=1,  # Direction is encoded in zone_values (+1/-1)
                steer_prompt=False,
                steer_generation=True,
            )

        # Generate
        prompts_batch = [prompt] * num_samples
        outputs = llm.generate(prompts_batch, sampling_params)

        factor_name = sanitize_factor_name(cond["name"])
        for sample_id, output in enumerate(outputs):
            response = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason

            result = {
                "condition": cond["name"],
                "mode": cond["mode"],
                "scale_pct": cond["scale_pct"],
                "zone_values": cond["zone_values"],
                "sample_id": sample_id,
                "response": response,
                "finish_reason": finish_reason,
                "response_len": len(response),
            }
            writer.write(factor_name, result)
            all_results.append(result)

        truncated = sum(
            1 for o in outputs if o.outputs[0].finish_reason == "length"
        )
        logger.info(f"  Generated {len(outputs)} ({truncated} truncated)")

    steering.clear()
    writer.close()
    logger.info(f"Saved {writer.counts} to {writer.output_dir}")

    _print_summary(all_results)

    return writer.output_dir


def _print_summary(results: List[dict]):
    """Print summary by condition."""
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    by_cond = {}
    for r in results:
        cond = r["condition"]
        by_cond.setdefault(cond, []).append(r)

    print(f"{'Condition':<45} {'N':>5} {'Trunc':>6} {'Avg Len':>8}")
    print("-" * 66)

    for cond in sorted(by_cond.keys()):
        rs = by_cond[cond]
        n = len(rs)
        truncated = sum(1 for r in rs if r.get("finish_reason") == "length")
        avg_len = sum(r.get("response_len", 0) for r in rs) / n if n else 0
        print(f"{cond:<45} {n:>5} {truncated:>6} {avg_len:>8.0f}")

    print("=" * 80)
    print(f"Total: {len(results)} responses across {len(by_cond)} conditions")


def main():
    parser = argparse.ArgumentParser(
        description="Blackmail differential section steering (tags3)",
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
        "--mode",
        type=str,
        required=True,
        choices=["amplify_impl", "amplify_risks"],
        help="Differential mode: amplify_impl (+impl, -risks) or amplify_risks (-impl, +risks)",
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
        default="text_pairs_emotion_vs_opposite",
        help="Vector set to use (default: text_pairs_emotion_vs_opposite)",
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

    args = parser.parse_args()

    run_experiment(
        model_name=args.model,
        layers=args.layers,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        mode=args.mode,
        emotion=args.emotion,
        vector_type=args.vector_type,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
