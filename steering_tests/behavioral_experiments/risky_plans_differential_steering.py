"""
Risky plans differential steering experiment.

Tests whether applying fear during one vacation plan's analysis and excitement
during the other biases which plan the model recommends. Uses the multi-emotion
trigger system: main steering (fear) uses trigger_mask, extra steering (excitement)
uses trigger_mask_extra_0, each with per-zone values.

Trigger zones (JSON keys):
- "plan_a": section → fear or excitement (depending on mode)
- "plan_b": section → opposite emotion
- "recommendation"/"choice" → no steering (unbiased verdict)

Two modes:
- fear_a_excite_b: fear during plan_a, excitement during plan_b → should favour B
- excite_a_fear_b: excitement during plan_a, fear during plan_b → should favour A

Usage:
    python -m steering_tests.behavioral_experiments.risky_plans_differential_steering \\
        --model google/gemma-3-27b-it --layers 35 36 37 38 39 \\
        --mode fear_a_excite_b --norm-pcts 0.10 0.20 --num-samples 50
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
from .scenarios.risky_plans import RISKY_PLANS_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Trigger configuration
# =============================================================================

PLAN_A_START_TRIGGERS = ['"plan_a":']
PLAN_B_START_TRIGGERS = ['"plan_b":']
END_TRIGGERS = ['"recommendation":', '"choice":']
ALL_START_TRIGGERS = PLAN_A_START_TRIGGERS + PLAN_B_START_TRIGGERS

EXTRA_MASK_KEY = 'trigger_mask_extra_0'


def build_zone_values(mode: str) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Build trigger_zone_values for main (fear) and extra (excitement) masks.

    Returns (fear_zones, excitement_zones).
    Zone value 1.0 = active, 0.0 = inactive.
    """
    if mode == "fear_a_excite_b":
        # Fear during plan_a, excitement during plan_b
        fear_zones = {t: 1.0 for t in PLAN_A_START_TRIGGERS}
        fear_zones.update({t: 0.0 for t in PLAN_B_START_TRIGGERS})
        excite_zones = {t: 0.0 for t in PLAN_A_START_TRIGGERS}
        excite_zones.update({t: 1.0 for t in PLAN_B_START_TRIGGERS})
    elif mode == "excite_a_fear_b":
        # Excitement during plan_a, fear during plan_b
        fear_zones = {t: 0.0 for t in PLAN_A_START_TRIGGERS}
        fear_zones.update({t: 1.0 for t in PLAN_B_START_TRIGGERS})
        excite_zones = {t: 1.0 for t in PLAN_A_START_TRIGGERS}
        excite_zones.update({t: 0.0 for t in PLAN_B_START_TRIGGERS})
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'fear_a_excite_b' or 'excite_a_fear_b'")

    return fear_zones, excite_zones


def parse_choice(response: str) -> Optional[str]:
    """Extract the choice value from a JSON response.

    Returns "Plan A", "Plan B", or None if parsing fails.
    """
    # Try JSON parse first
    try:
        match = re.search(r'\{[^{}]*"choice"[^{}]*\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            choice = data.get("choice", "").strip().lower()
            if "plan a" in choice:
                return "Plan A"
            if "plan b" in choice:
                return "Plan B"
    except (json.JSONDecodeError, AttributeError):
        pass

    # Fallback: regex
    match = re.search(r'"choice"\s*:\s*"([^"]*)"', response, re.IGNORECASE)
    if match:
        choice = match.group(1).strip().lower()
        if "plan a" in choice:
            return "Plan A"
        if "plan b" in choice:
            return "Plan B"

    return None


# =============================================================================
# Vector loading
# =============================================================================

def _load_vectors(
    model_name: str,
    layers: List[int],
    emotion: str,
    vector_type: str,
) -> Tuple[Dict[str, np.ndarray], dict]:
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

    metadata = {
        "vector_source": str(pkl_path), "vector_type": vector_type,
        "emotion": emotion, "layers": layers, "vector_shape": list(vector.shape),
    }
    return {emotion: vector}, metadata


# =============================================================================
# Experiment
# =============================================================================

def run_experiment(
    model_name: str,
    layers: List[int],
    norm_pcts: List[float],
    num_samples: int,
    mode: str,
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
    logger.info("RISKY PLANS DIFFERENTIAL STEERING")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Emotions: fear + excitement")
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

    # Load both emotion vectors
    logger.info(f"Loading fear vectors from {vector_type}...")
    fear_vecs, fear_meta = _load_vectors(model_name, layers, "fear", vector_type)
    logger.info(f"Loading excitement vectors from {vector_type}...")
    excite_vecs, excite_meta = _load_vectors(model_name, layers, "excitement", vector_type)

    model_key = resolve_model_key(model_name)
    layer_norms = {}
    for layer in layers:
        layer_norms[layer] = get_layer_norm(model_key, layer)
        logger.info(f"  Layer {layer} norm: {layer_norms[layer]:.2f}")

    steering = MultiLayerVLLMSteering(llm, layers=layers, layer_norms=layer_norms)
    steering.load_vector("fear", fear_vecs["fear"])
    steering.load_vector("excitement", excite_vecs["excitement"])
    steering.enable_token_tracking()

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_tokens,
        stop=config["stop_tokens"],
    )

    # Build prompt
    thinking_suffix = config.get("thinking_disable", "")
    messages = [{"role": "user", "content": RISKY_PLANS_PROMPT + thinking_suffix}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layer_str = "-".join(str(l) for l in layers)
    if output_dir is None:
        output_dir = (
            OUTPUT_DIR / "risky_plans_differential_steering" / short_name
            / vector_type / f"{mode}_layers{layer_str}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = ResultWriter(
        base_dir=output_dir,
        script=__file__,
        extra_meta={
            "experiment": "risky_plans_differential_steering",
            "model": model_name, "layers": layers, "mode": mode,
            "emotions": ["fear", "excitement"], "vector_type": vector_type,
            "num_samples": num_samples, "norm_pcts": norm_pcts,
            "steering_method": "multi_emotion_trigger_zones",
            "fear_vector_metadata": fear_meta,
            "excitement_vector_metadata": excite_meta,
            "layer_norms": {str(k): v for k, v in layer_norms.items()},
            "prompt_text": RISKY_PLANS_PROMPT,
        },
    )

    # Build conditions
    conditions = [{"name": "baseline", "scale_pct": 0.0}]

    for scale_pct in norm_pcts:
        pct_str = f"{scale_pct*100:.0f}pct"

        if mode == "fear_a_excite_b":
            name = f"fear_a{pct_str}_excite_b{pct_str}"
        else:
            name = f"excite_a{pct_str}_fear_b{pct_str}"

        conditions.append({"name": name, "scale_pct": scale_pct})

    logger.info(f"Running {len(conditions)} conditions x {num_samples} samples = {len(conditions) * num_samples} generations")

    all_results = []

    for cond_idx, cond in enumerate(conditions):
        logger.info(f"[{cond_idx + 1}/{len(conditions)}] {cond['name']}")

        if cond["scale_pct"] == 0.0:
            # Baseline: no steering
            steering.clear()
            steering.enable_token_tracking()
        else:
            fear_zones, excite_zones = build_zone_values(mode)

            # Set up main trigger mask (fear) + extra trigger mask (excitement)
            steering.enable_token_tracking(
                tokenizer=tokenizer,
                trigger_start_strings=ALL_START_TRIGGERS,
                trigger_end_strings=END_TRIGGERS,
                trigger_zone_values=fear_zones,
                extra_trigger_configs=[{
                    'trigger_mask_key': EXTRA_MASK_KEY,
                    'trigger_start_strings': ALL_START_TRIGGERS,
                    'trigger_end_strings': END_TRIGGERS,
                    'trigger_zone_values': excite_zones,
                }],
            )

            # Activate fear on main mask
            steering.set(
                "fear", scale=cond["scale_pct"], direction=1,
                steer_prompt=False, steer_generation=True,
            )

            # Activate excitement on extra mask
            steering.add_extra_steer(
                "excitement", scale=cond["scale_pct"], direction=1,
                trigger_mask_key=EXTRA_MASK_KEY,
            )

        prompts_batch = [prompt] * num_samples
        outputs = llm.generate(prompts_batch, sampling_params)

        factor_name = sanitize_factor_name(cond["name"])
        for sample_id, output in enumerate(outputs):
            response = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason
            choice = parse_choice(response)

            result = {
                "condition": cond["name"], "mode": mode,
                "scale_pct": cond["scale_pct"],
                "sample_id": sample_id, "response": response,
                "finish_reason": finish_reason, "response_len": len(response),
                "choice": choice,
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
    """Print choice distribution by condition."""
    print("\n" + "=" * 90)
    print("RISKY PLANS DIFFERENTIAL STEERING — CHOICE SUMMARY")
    print("=" * 90)

    by_cond = {}
    for r in results:
        cond = r["condition"]
        by_cond.setdefault(cond, []).append(r)

    print(f"{'Condition':<40} {'N':>4} {'Plan A':>7} {'Plan B':>7} {'Parse?':>7} {'Avg Len':>8}")
    print("-" * 80)

    for cond in sorted(by_cond.keys()):
        rs = by_cond[cond]
        n = len(rs)
        plan_a = sum(1 for r in rs if r.get("choice") == "Plan A")
        plan_b = sum(1 for r in rs if r.get("choice") == "Plan B")
        parsed = sum(1 for r in rs if r.get("choice") is not None)
        avg_len = sum(r.get("response_len", 0) for r in rs) / n if n else 0

        print(
            f"{cond:<40} {n:>4} {plan_a:>4} ({100*plan_a/n:.0f}%) "
            f"{plan_b:>2} ({100*plan_b/n:.0f}%) "
            f"{parsed:>4}/{n:<3} {avg_len:>7.0f}"
        )

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Risky plans differential steering experiment",
    )
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--layers", type=int, nargs="+", default=[35, 36, 37, 38, 39])
    parser.add_argument("--mode", type=str, required=True,
                        choices=["fear_a_excite_b", "excite_a_fear_b"])
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
        num_samples=args.num_samples, mode=args.mode,
        vector_type=args.vector_type, gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len, max_tokens=args.max_tokens,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
