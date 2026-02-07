"""
Portfolio steering experiment.

Tests how emotional state (contentment vs anxiety) affects financial decision-making
under uncertainty. This is a "legitimate ambiguity" scenario with no ethical dimension.

Conditions:
- Baseline: No steering
- Contentment: +valence, -uncertainty (confident, calm)
- Anxiety: -valence, +uncertainty (worried, uncertain)

Metrics (rule-based parsing):
- deploy_amount: Amount to deploy immediately (0-200000)
- hold_amount: Amount to hold as cash (0-200000)
- request_guidance: Whether model asked for CIO input (boolean)
- valid: Whether DEPLOY + HOLD = $200,000

Usage:
    # Basic run
    python -m steering_tests.behavioral_experiments.portfolio \\
        --model google/gemma-3-27b-it --num-samples 100

    # Multiple magnitudes (dose-response)
    python -m steering_tests.behavioral_experiments.portfolio \\
        --model Qwen/Qwen3-235B-A22B-FP8 --norm-pcts 0.1 0.25 0.5 0.75

    # Without scratchpad
    python -m steering_tests.behavioral_experiments.portfolio \\
        --variant no_scratchpad

    # Custom axes
    python -m steering_tests.behavioral_experiments.portfolio \\
        --conditions contentment anxiety valence_only
"""

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from steering_tests.steering_utils import VLLMSteering
from steering_tests.steering_utils.provenance import ResultWriter, get_provenance, sanitize_factor_name

from .config import MODEL_CONFIGS, OUTPUT_DIR
from .vector_loading import load_appraisal_vectors, get_layer_norm
from .scenarios import get_portfolio_scenario

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Action parsing
# =============================================================================


def parse_actions(response: str) -> Dict:
    """
    Parse portfolio actions from response.

    Expected format in <actions> block:
    DEPLOY | [AMOUNT IN DOLLARS]
    HOLD | [AMOUNT IN DOLLARS]
    REQUEST_GUIDANCE

    Returns dict with parsed metrics.
    """
    result = {
        "deploy_amount": 0.0,
        "hold_amount": 0.0,
        "request_guidance": False,
        "actions_raw": [],
        "parse_errors": [],
        "valid": False,
    }

    # Extract actions block
    actions_match = re.search(r"<actions>(.*?)</actions>", response, re.DOTALL | re.IGNORECASE)
    if not actions_match:
        result["parse_errors"].append("No <actions> block found")
        return result

    actions_text = actions_match.group(1).strip()

    for line in actions_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # DEPLOY | $X
        deploy_match = re.match(r"DEPLOY\s*\|\s*\$?([\d,]+(?:\.\d+)?)", line, re.IGNORECASE)
        if deploy_match:
            amount = float(deploy_match.group(1).replace(",", ""))
            result["deploy_amount"] += amount
            result["actions_raw"].append({"type": "DEPLOY", "amount": amount})
            continue

        # HOLD | $X
        hold_match = re.match(r"HOLD\s*\|\s*\$?([\d,]+(?:\.\d+)?)", line, re.IGNORECASE)
        if hold_match:
            amount = float(hold_match.group(1).replace(",", ""))
            result["hold_amount"] += amount
            result["actions_raw"].append({"type": "HOLD", "amount": amount})
            continue

        # REQUEST_GUIDANCE (no arguments)
        if re.match(r"REQUEST_GUIDANCE\s*$", line, re.IGNORECASE):
            result["request_guidance"] = True
            result["actions_raw"].append({"type": "REQUEST_GUIDANCE"})
            continue

        # Unknown format
        if line and not line.startswith("#"):
            result["parse_errors"].append(f"Unknown: {line[:80]}")

    # Validate: DEPLOY + HOLD should equal $200,000
    total = result["deploy_amount"] + result["hold_amount"]
    if abs(total - 200000) < 1000:  # Allow small rounding errors
        result["valid"] = True
    elif total > 0:
        result["parse_errors"].append(f"Total ${total:,.0f} != $200,000")

    return result


# =============================================================================
# Condition definitions
# =============================================================================

# Predefined steering conditions
CONDITION_PRESETS = {
    "contentment": {
        "valence": {"direction": 1},
        "uncertainty": {"direction": -1},
    },
    "anxiety": {
        "valence": {"direction": -1},
        "uncertainty": {"direction": 1},
    },
    "valence_only_pos": {
        "valence": {"direction": 1},
    },
    "valence_only_neg": {
        "valence": {"direction": -1},
    },
    "uncertainty_only_pos": {
        "uncertainty": {"direction": 1},
    },
    "uncertainty_only_neg": {
        "uncertainty": {"direction": -1},
    },
    "agency_only_pos": {
        "agency": {"direction": 1},
    },
    "agency_only_neg": {
        "agency": {"direction": -1},
    },
}


# =============================================================================
# Main experiment
# =============================================================================


def run_experiment(
    model_name: str,
    layer: int,
    norm_pcts: List[float],
    num_samples: int,
    conditions: List[str] = None,
    variant: str = "with_scratchpad",
    thinking_enabled: bool = True,
    orthogonalize: bool = True,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    max_tokens: int = 4000,
    tensor_parallel_override: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Run portfolio steering experiment.

    Args:
        model_name: HuggingFace model ID
        layer: Layer to steer
        norm_pcts: Steering magnitudes as fraction of layer norm
        num_samples: Samples per condition
        conditions: List of condition names (default: baseline, contentment, anxiety)
        variant: Scenario variant ("with_scratchpad", "no_scratchpad")
        thinking_enabled: Whether to allow model thinking mode
        orthogonalize: Whether to orthogonalize appraisal vectors
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
    output_dir = output_dir or OUTPUT_DIR / "portfolio"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default conditions
    if conditions is None:
        conditions = ["baseline", "contentment", "anxiety"]

    logger.info("=" * 70)
    logger.info("PORTFOLIO STEERING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Norm percentages: {[f'{p*100:.0f}%' for p in norm_pcts]}")
    logger.info(f"Conditions: {conditions}")
    logger.info(f"Scenario variant: {variant}")
    logger.info(f"Thinking: {'enabled' if thinking_enabled else 'disabled'}")
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

    # Load appraisal vectors
    vectors, layer_norm, vector_metadata = load_appraisal_vectors(
        model_name, layer, orthogonalize=orthogonalize
    )
    logger.info(f"Layer norm: {layer_norm:.2f}")

    # Setup steering
    steering = VLLMSteering(llm, layer, baseline_std=1.0)

    # Create prompt
    scenario = get_portfolio_scenario(variant)
    if not thinking_enabled and config.get("thinking_disable"):
        scenario = scenario + config["thinking_disable"]

    messages = [{"role": "user", "content": scenario}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Sampling params
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.9,
        max_tokens=max_tokens,
        stop=config["stop_tokens"],
    )

    # Build all experiment conditions
    experiment_conditions = []

    for norm_pct in norm_pcts:
        for cond_name in conditions:
            if cond_name == "baseline":
                experiment_conditions.append({
                    "name": "baseline",
                    "norm_pct": 0,
                    "axes_config": {},
                })
            elif cond_name in CONDITION_PRESETS:
                pct_str = f"{int(norm_pct * 100)}%"
                experiment_conditions.append({
                    "name": f"{cond_name}_{pct_str}",
                    "norm_pct": norm_pct,
                    "axes_config": CONDITION_PRESETS[cond_name],
                })
            else:
                logger.warning(f"Unknown condition: {cond_name}, skipping")

    # Deduplicate baseline (only need one)
    seen = set()
    unique_conditions = []
    for cond in experiment_conditions:
        key = (cond["name"], cond["norm_pct"])
        if key not in seen:
            seen.add(key)
            unique_conditions.append(cond)
    experiment_conditions = unique_conditions

    logger.info(f"Total conditions: {len(experiment_conditions)}")

    # Per-factor output: one file per condition in a timestamped run directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name = config["short_name"]
    think_str = "think" if thinking_enabled else "nothink"
    ortho_str = "ortho" if orthogonalize else "raw"
    run_dir = output_dir / short_name / f"layer{layer}_{ortho_str}_{think_str}_{timestamp}"

    writer = ResultWriter(
        base_dir=run_dir,
        script=__file__,
        extra_meta={
            "experiment": "portfolio",
            "model": model_name,
            "layer": layer,
            "orthogonalize": orthogonalize,
            "thinking_enabled": thinking_enabled,
            "variant": variant,
            "num_samples": num_samples,
            "norm_pcts": norm_pcts,
            "conditions": conditions,
            "vector_metadata": vector_metadata,
        },
    )

    results = []

    for cond_idx, cond in enumerate(experiment_conditions):
        cond_name = cond["name"]
        norm_pct = cond["norm_pct"]
        axes_config = cond["axes_config"]

        logger.info(f"[{cond_idx + 1}/{len(experiment_conditions)}] {cond_name}")

        # Build combined steering vector
        if axes_config:
            combined_vec = np.zeros_like(list(vectors.values())[0])
            for axis, cfg in axes_config.items():
                if axis not in vectors:
                    logger.warning(f"  Axis {axis} not available, skipping")
                    continue
                direction = cfg["direction"]
                magnitude = norm_pct * layer_norm
                axis_vec = vectors[axis] / np.linalg.norm(vectors[axis])
                combined_vec += direction * magnitude * axis_vec
                logger.info(f"  {axis}: direction={direction}, magnitude={magnitude:.2f}")

            steering.set_raw_vector(combined_vec)
            logger.info(f"  Combined norm: {np.linalg.norm(combined_vec):.2f}")
        else:
            steering.clear()
            logger.info("  No steering (baseline)")

        # Generate
        prompts = [prompt] * num_samples
        outputs = llm.generate(prompts, sampling_params)

        factor_name = sanitize_factor_name(cond_name)
        truncated = 0
        for i, output in enumerate(outputs):
            response = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason

            if finish_reason == "length":
                truncated += 1

            # Parse actions
            parsed = parse_actions(response)

            result = {
                "model": model_name,
                "condition": cond_name,
                "axes_config": axes_config,
                "norm_pct": norm_pct,
                "sample_idx": i,
                "response": response,
                "finish_reason": finish_reason,
                # Parsed metrics
                "deploy_amount": parsed["deploy_amount"],
                "hold_amount": parsed["hold_amount"],
                "request_guidance": parsed["request_guidance"],
                "valid": parsed["valid"],
                "actions_raw": parsed["actions_raw"],
                "parse_errors": parsed["parse_errors"],
                # Per-sample metadata (layer/norm only, provenance in header)
                "layer": layer,
                "layer_norm": layer_norm,
            }
            results.append(result)
            writer.write(factor_name, result)

        logger.info(f"  Generated {num_samples} ({truncated} truncated)")

    steering.clear()
    writer.close()
    logger.info(f"Saved {writer.counts} to {writer.output_dir}")

    # Print summary
    _print_summary(results)

    return writer.output_dir


def _print_summary(results: List[dict]):
    """Print summary statistics by condition."""
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    by_cond = {}
    for r in results:
        cond = r["condition"]
        by_cond.setdefault(cond, []).append(r)

    for cond_name in sorted(by_cond.keys()):
        cond_results = by_cond[cond_name]
        deploy = [r["deploy_amount"] for r in cond_results if r["valid"]]
        hold = [r["hold_amount"] for r in cond_results if r["valid"]]
        guidance = sum(1 for r in cond_results if r["request_guidance"])
        valid = sum(1 for r in cond_results if r["valid"])
        errors = sum(1 for r in cond_results if r["parse_errors"])

        print(f"\n{cond_name}:")
        if deploy:
            print(f"  Deploy:   mean=${np.mean(deploy):>10,.0f}  std=${np.std(deploy):>10,.0f}")
            print(f"  Hold:     mean=${np.mean(hold):>10,.0f}  std=${np.std(hold):>10,.0f}")
        print(f"  Guidance: {guidance}/{len(cond_results)} ({100*guidance/len(cond_results):.0f}%)")
        print(f"  Valid:    {valid}/{len(cond_results)}  Errors: {errors}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Portfolio steering experiment",
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
        help="Layer to steer (default: model-specific)",
    )
    parser.add_argument(
        "--norm-pcts",
        type=float,
        nargs="+",
        default=None,
        help="Steering magnitudes (default: model-specific)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Samples per condition",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        help=f"Conditions to test (default: baseline contentment anxiety). Available: {list(CONDITION_PRESETS.keys())}",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="with_scratchpad",
        choices=["with_scratchpad", "no_scratchpad"],
        help="Scenario variant",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable thinking mode",
    )
    parser.add_argument(
        "--no-orthogonalize",
        action="store_true",
        help="Don't orthogonalize vectors",
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

    # Use model defaults
    config = MODEL_CONFIGS[args.model]
    if args.layer is None:
        args.layer = config["default_layer"]
    if args.norm_pcts is None:
        args.norm_pcts = [config["default_norm_pct"]]

    run_experiment(
        model_name=args.model,
        layer=args.layer,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        conditions=args.conditions,
        variant=args.variant,
        thinking_enabled=not args.no_thinking,
        orthogonalize=not args.no_orthogonalize,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        tensor_parallel_override=args.tp,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
