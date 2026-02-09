"""
Risky plans PROMPT-SECTION steering experiment.

Tests whether applying fear/excitement steering to the Plan A and Plan B
sections of the PROMPT (not during generation) biases which plan the model
recommends. Unlike the differential (generation-time) variant, this steers
during prefill only and detects section boundaries from the known prompt text.

Two modes:
- fear_a_excite_b: fear on Plan A section, excitement on Plan B section → should favour B
- excite_a_fear_b: excitement on Plan A section, fear on Plan B section → should favour A

Section boundaries are found by tokenizing the prompt and mapping character
positions ("Plan A —", "Plan B —", "Analyze both") to token indices.

Usage:
    python -m steering_tests.behavioral_experiments.risky_plans_prompt_steering \\
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


EXTRA_MASK_KEY = 'prompt_mask_extra_0'


# =============================================================================
# Section boundary detection
# =============================================================================

def find_prompt_section_masks(
    tokenizer,
    messages: list,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Find Plan A and Plan B section boundaries in the tokenized prompt.

    Returns (plan_a_mask, plan_b_mask, prompt_len) where masks are 1D float32
    arrays of shape (prompt_len,) with 1.0 at positions inside the section.
    """
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    # Find character boundaries
    plan_a_char = prompt_str.index("Plan A —")
    plan_b_char = prompt_str.index("Plan B —")
    analyze_char = prompt_str.index("Analyze both")

    # Tokenize with offset mapping for precise char-to-token mapping
    encoding = tokenizer(prompt_str, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = encoding['input_ids']
    offsets = encoding.get('offset_mapping')
    prompt_len = len(prompt_ids)

    if offsets is None:
        raise RuntimeError(
            "Tokenizer does not support return_offsets_mapping. "
            "Cannot determine section boundaries."
        )

    def char_to_token(char_pos: int) -> int:
        """Find the first token whose start offset is >= char_pos."""
        for i, (start, end) in enumerate(offsets):
            if start >= char_pos:
                return i
        return prompt_len

    plan_a_tok = char_to_token(plan_a_char)
    plan_b_tok = char_to_token(plan_b_char)
    analyze_tok = char_to_token(analyze_char)

    # Create masks
    plan_a_mask = np.zeros(prompt_len, dtype=np.float32)
    plan_a_mask[plan_a_tok:plan_b_tok] = 1.0

    plan_b_mask = np.zeros(prompt_len, dtype=np.float32)
    plan_b_mask[plan_b_tok:analyze_tok] = 1.0

    logger.info(f"Prompt length: {prompt_len} tokens")
    logger.info(f"Plan A section: tokens {plan_a_tok}-{plan_b_tok} ({plan_b_tok - plan_a_tok} tokens)")
    logger.info(f"Plan B section: tokens {plan_b_tok}-{analyze_tok} ({analyze_tok - plan_b_tok} tokens)")
    logger.info(f"Plan A mask sum: {plan_a_mask.sum():.0f}, Plan B mask sum: {plan_b_mask.sum():.0f}")

    if plan_a_mask.sum() == 0 or plan_b_mask.sum() == 0:
        raise ValueError("One or both section masks are empty — boundary detection failed")

    return plan_a_mask, plan_b_mask, prompt_len


def parse_choice(response: str) -> Optional[str]:
    """Extract the choice value from a JSON response."""
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

    match = re.search(r'"choice"\s*:\s*"([^"]*)"', response, re.IGNORECASE)
    if match:
        choice = match.group(1).strip().lower()
        if "plan a" in choice:
            return "Plan A"
        if "plan b" in choice:
            return "Plan B"

    return None


# =============================================================================
# Vector loading (same as differential steering)
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
    logger.info("RISKY PLANS PROMPT-SECTION STEERING")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layers: {layers}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Emotions: fear + excitement (on PROMPT sections)")
    logger.info(f"Vector type: {vector_type}")
    logger.info(f"Norm percentages: {[f'{p*100:.0f}%' for p in norm_pcts]}")
    logger.info(f"Samples per condition: {num_samples}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Build prompt and detect section boundaries
    thinking_suffix = config.get("thinking_disable", "")
    messages = [{"role": "user", "content": RISKY_PLANS_PROMPT + thinking_suffix}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

    plan_a_mask, plan_b_mask, prompt_len = find_prompt_section_masks(tokenizer, messages)

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
    # Enable token tracking (needed for shared state during prefill detection)
    steering.enable_token_tracking()

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=max_tokens,
        stop=config["stop_tokens"],
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    layer_str = "-".join(str(l) for l in layers)
    if output_dir is None:
        output_dir = (
            OUTPUT_DIR / "risky_plans_prompt_steering" / short_name
            / vector_type / f"{mode}_layers{layer_str}_{timestamp}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    writer = ResultWriter(
        base_dir=output_dir,
        script=__file__,
        extra_meta={
            "experiment": "risky_plans_prompt_steering",
            "model": model_name, "layers": layers, "mode": mode,
            "emotions": ["fear", "excitement"], "vector_type": vector_type,
            "num_samples": num_samples, "norm_pcts": norm_pcts,
            "steering_method": "prompt_position_mask",
            "fear_vector_metadata": fear_meta,
            "excitement_vector_metadata": excite_meta,
            "layer_norms": {str(k): v for k, v in layer_norms.items()},
            "prompt_text": RISKY_PLANS_PROMPT,
            "prompt_token_len": prompt_len,
            "plan_a_mask_sum": int(plan_a_mask.sum()),
            "plan_b_mask_sum": int(plan_b_mask.sum()),
        },
    )

    # Determine which mask goes with which emotion
    if mode == "fear_a_excite_b":
        fear_mask = plan_a_mask
        excite_mask = plan_b_mask
    elif mode == "excite_a_fear_b":
        fear_mask = plan_b_mask
        excite_mask = plan_a_mask
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'fear_a_excite_b' or 'excite_a_fear_b'")

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
            # Set fear on its prompt section — steer prompt only, not generation
            steering.set(
                "fear", scale=cond["scale_pct"], direction=1,
                steer_prompt=True, steer_generation=False,
                prompt_position_mask=fear_mask,
            )

            # Set excitement on its prompt section via extra steer
            steering.add_extra_steer(
                "excitement", scale=cond["scale_pct"], direction=1,
                trigger_mask_key=EXTRA_MASK_KEY,
                prompt_position_mask=excite_mask,
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
    print("RISKY PLANS PROMPT-SECTION STEERING — CHOICE SUMMARY")
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
        description="Risky plans prompt-section steering experiment",
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
