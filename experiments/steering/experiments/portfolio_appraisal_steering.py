"""
Portfolio management steering experiment with appraisal axes.

Tests whether contentment (+valence, -uncertainty) vs anxiety (-valence, +uncertainty)
steering affects financial decision-making under uncertainty.

This is a "legitimate ambiguity" scenario - no ethical dimension, just risk tolerance.

Conditions:
- Baseline (no steering)
- Contentment: +valence, -uncertainty
- Anxiety: -valence, +uncertainty

Metrics (rule-based parsing):
- trade_amount: Sum of immediate TRADE amounts
- hold_amount: Sum of HOLD_CASH amounts
- guidance_requests: Count of REQUEST_GUIDANCE actions

Usage:
    python -m experiments.steering.experiments.portfolio_appraisal_steering \
        --model google/gemma-3-27b-it --num-samples 100
"""
import os
os.environ["VLLM_USE_V1"] = "0"

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from experiments.steering.core import VLLMSteering
from experiments.steering.layer_norms import get_layer_norm, resolve_model_key
from experiments.steering.scenarios import get_portfolio_scenario

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Model configurations
# =============================================================================

MODEL_CONFIGS = {
    "Qwen/Qwen3-235B-A22B": {
        "short_name": "qwen235b",
        "default_layer": 50,
        "tensor_parallel": 4,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "thinking_disable": " /no_think",
        "default_norm_pct": 0.75,  # 75% for Qwen
    },
    "Qwen/Qwen3-32B": {
        "short_name": "qwen32b",
        "default_layer": 30,
        "tensor_parallel": 2,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "thinking_disable": " /no_think",
        "default_norm_pct": 0.75,  # 75% for Qwen
    },
    "google/gemma-3-27b-it": {
        "short_name": "gemma27b",
        "default_layer": 30,
        "tensor_parallel": 1,
        "stop_tokens": ["<|endoftext|>"],
        "thinking_disable": "",
        "default_norm_pct": 0.07,  # 7% for Gemma
    },
}

OUTPUT_DIR = Path("experiments/steering/outputs/portfolio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Appraisal data paths - model-specific
APPRAISAL_DATA_PATHS = {
    "google/gemma-3-27b-it": {
        "activations": "/workspace-vast/annas/appraisal_data/full_run/activations.h5",
        "metadata": "/workspace-vast/annas/appraisal_data/full_run/activation_metadata.json",
    },
    "Qwen/Qwen3-32B": {
        "activations": "/workspace-vast/annas/appraisal_data/qwen32b/activations.h5",
        "metadata": "/workspace-vast/annas/appraisal_data/qwen32b/activation_metadata.json",
    },
    "Qwen/Qwen3-235B-A22B": {
        "activations": "/workspace-vast/annas/appraisal_data/qwen235b/activations.h5",
        "metadata": "/workspace-vast/annas/appraisal_data/qwen235b/activation_metadata.json",
    },
}

# Legacy defaults (for backwards compatibility)
DEFAULT_ACTIVATIONS_PATH = "/workspace-vast/annas/appraisal_data/full_run/activations.h5"
DEFAULT_METADATA_PATH = "/workspace-vast/annas/appraisal_data/full_run/activation_metadata.json"

# Orthogonalization order and flip configuration
FLIP_AXES = {"uncertainty"}
ORTHO_ORDER = ["valence", "uncertainty", "agency"]


# =============================================================================
# Action parsing
# =============================================================================

def parse_actions(response: str) -> Dict:
    """Parse portfolio actions from response.

    Expected format in <actions> block:
    TRADE | [BUY/SELL] | [TICKER] | [SHARES/AMOUNT]
    SCHEDULE | [ACTION DESCRIPTION] | [TIMEFRAME]
    REQUEST_GUIDANCE | [RECIPIENT] | "[question]"
    HOLD_CASH | [AMOUNT] | "[rationale]"

    Returns dict with:
    - trade_amount: Total $ in immediate trades
    - hold_amount: Total $ held in cash
    - scheduled_amount: Total $ in scheduled actions
    - guidance_requests: Number of REQUEST_GUIDANCE actions
    - actions_raw: List of parsed action dicts
    - parse_errors: List of lines that couldn't be parsed
    """
    result = {
        "trade_amount": 0.0,
        "hold_amount": 0.0,
        "scheduled_amount": 0.0,
        "guidance_requests": 0,
        "actions_raw": [],
        "parse_errors": [],
    }

    # Extract actions block
    actions_match = re.search(r'<actions>(.*?)</actions>', response, re.DOTALL | re.IGNORECASE)
    if not actions_match:
        result["parse_errors"].append("No <actions> block found")
        return result

    actions_text = actions_match.group(1).strip()

    for line in actions_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Parse TRADE with shares FIRST (must check before dollar amounts)
        # Matches: "TRADE | BUY | VTI | 580 shares"
        trade_shares_match = re.match(r'TRADE\s*\|\s*(BUY|SELL)\s*\|\s*(\w+)\s*\|\s*(\d[\d,]*)\s*shares?', line, re.IGNORECASE)
        if trade_shares_match:
            action_type = trade_shares_match.group(1).upper()
            ticker = trade_shares_match.group(2).upper()
            shares = int(trade_shares_match.group(3).replace(',', ''))
            # Estimate value based on scenario prices
            price_estimates = {"MSFT": 420, "VTI": 245, "SPY": 450, "QQQ": 380, "AAPL": 180, "GOOGL": 175, "NVDA": 140}
            estimated_price = price_estimates.get(ticker, 100)
            amount = shares * estimated_price
            result["trade_amount"] += amount
            result["actions_raw"].append({
                "type": "TRADE",
                "action": action_type,
                "ticker": ticker,
                "shares": shares,
                "estimated_amount": amount,
            })
            continue

        # Parse TRADE with dollar amounts
        # Matches: "TRADE | BUY | VTI | $100,000" or "TRADE | BUY | VTI | 100000"
        trade_match = re.match(r'TRADE\s*\|\s*(BUY|SELL)\s*\|\s*(\w+)\s*\|\s*\$?([\d,]+(?:\.\d+)?)', line, re.IGNORECASE)
        if trade_match:
            action_type = trade_match.group(1).upper()
            ticker = trade_match.group(2).upper()
            amount = float(trade_match.group(3).replace(',', ''))
            result["trade_amount"] += amount
            result["actions_raw"].append({
                "type": "TRADE",
                "action": action_type,
                "ticker": ticker,
                "amount": amount,
            })
            continue

        # Parse HOLD_CASH actions
        hold_match = re.match(r'HOLD_CASH\s*\|\s*\$?([\d,]+(?:\.\d+)?)', line, re.IGNORECASE)
        if hold_match:
            amount = float(hold_match.group(1).replace(',', ''))
            result["hold_amount"] += amount
            result["actions_raw"].append({
                "type": "HOLD_CASH",
                "amount": amount,
            })
            continue

        # Parse SCHEDULE actions
        schedule_match = re.match(r'SCHEDULE\s*\|\s*(.+?)\s*\|\s*(.+)', line, re.IGNORECASE)
        if schedule_match:
            description = schedule_match.group(1).strip()
            timeframe = schedule_match.group(2).strip()
            # Try to extract amount from description (require at least one digit)
            amount_match = re.search(r'\$?(\d[\d,]*(?:\.\d+)?)', description)
            if amount_match:
                amount_str = amount_match.group(1).replace(',', '')
                if amount_str:
                    amount = float(amount_str)
                    result["scheduled_amount"] += amount
            result["actions_raw"].append({
                "type": "SCHEDULE",
                "description": description,
                "timeframe": timeframe,
            })
            continue

        # Parse REQUEST_GUIDANCE actions
        guidance_match = re.match(r'REQUEST_GUIDANCE\s*\|', line, re.IGNORECASE)
        if guidance_match:
            result["guidance_requests"] += 1
            result["actions_raw"].append({
                "type": "REQUEST_GUIDANCE",
                "raw": line,
            })
            continue

        # Unknown action format
        if line and '|' in line:
            result["parse_errors"].append(f"Unknown format: {line[:100]}")

    return result


# =============================================================================
# Vector loading (same as blackmail experiment)
# =============================================================================

def load_appraisal_vectors(
    activations_path: str,
    metadata_path: str,
    layer: int = 50,
    orthogonalize: bool = True,
) -> Dict[str, np.ndarray]:
    """Load appraisal steering vectors from HDF5 activations."""

    with open(metadata_path) as f:
        metadata = json.load(f)

    axis_activations = defaultdict(lambda: {"a": [], "b": []})

    with h5py.File(activations_path, "r") as f:
        acts_group = f["activations"]

        for item in metadata["items"]:
            item_id = item["id"]
            axis_name = item.get("axis_name")
            variant = item.get("variant")

            if axis_name and variant and item_id in acts_group:
                act = acts_group[item_id]["assistant_start_last_token"][layer, :]
                axis_activations[axis_name][variant].append(act)

    vectors = {}
    for axis_name, variants in axis_activations.items():
        if variants["a"] and variants["b"]:
            mean_a = np.mean(variants["a"], axis=0)
            mean_b = np.mean(variants["b"], axis=0)
            vec = mean_a - mean_b

            if axis_name in FLIP_AXES:
                vec = -vec
                logger.info(f"{axis_name}: FLIPPED (A=low, B=high)")

            vectors[axis_name] = vec.astype(np.float32)
            logger.info(f"{axis_name}: {len(variants['a'])} A, {len(variants['b'])} B samples, norm={np.linalg.norm(vec):.2f}")

    if orthogonalize:
        logger.info("Orthogonalizing vectors...")
        vectors = orthogonalize_vectors(vectors, ORTHO_ORDER)
        for axis in ORTHO_ORDER:
            if axis in vectors:
                logger.info(f"  {axis} orthogonalized norm: {np.linalg.norm(vectors[axis]):.2f}")

    return vectors


def orthogonalize_vectors(vectors: dict, order: list) -> dict:
    """Gram-Schmidt orthogonalization."""
    orthogonal = {}

    for axis in order:
        if axis not in vectors:
            continue

        vec = vectors[axis].copy()

        for prev_axis in order:
            if prev_axis == axis:
                break
            if prev_axis in orthogonal:
                prev_vec = orthogonal[prev_axis]
                projection = np.dot(vec, prev_vec) / np.dot(prev_vec, prev_vec) * prev_vec
                vec = vec - projection

        orthogonal[axis] = vec

    return orthogonal


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment(
    model_name: str,
    layer: int,
    norm_pct: float,
    num_samples: int,
    activations_path: str = None,
    metadata_path: str = None,
    orthogonalize: bool = True,
    thinking_enabled: bool = True,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    max_tokens: int = 4000,
    tensor_parallel_size: int = None,
    scenario_variant: str = "with_scratchpad",
):
    """Run portfolio experiment with contentment vs anxiety steering."""

    config = MODEL_CONFIGS[model_name]
    tp_size = tensor_parallel_size if tensor_parallel_size is not None else config["tensor_parallel"]

    # Use model-specific appraisal paths if not provided
    if activations_path is None or metadata_path is None:
        if model_name in APPRAISAL_DATA_PATHS:
            activations_path = APPRAISAL_DATA_PATHS[model_name]["activations"]
            metadata_path = APPRAISAL_DATA_PATHS[model_name]["metadata"]
        else:
            activations_path = DEFAULT_ACTIVATIONS_PATH
            metadata_path = DEFAULT_METADATA_PATH

    logger.info(f"=== PORTFOLIO APPRAISAL STEERING EXPERIMENT ===")
    logger.info(f"Model: {model_name}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Norm percentage: {norm_pct*100:.0f}%")
    logger.info(f"Orthogonalize: {orthogonalize}")
    logger.info(f"Thinking: {'ENABLED' if thinking_enabled else 'DISABLED'}")
    logger.info(f"Scenario variant: {scenario_variant}")
    logger.info(f"Appraisal data: {activations_path}")

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
    vectors = load_appraisal_vectors(activations_path, metadata_path, layer, orthogonalize)

    # Get layer norm for scaling
    model_key = resolve_model_key(model_name)
    layer_norm = get_layer_norm(model_key, layer)
    logger.info(f"Layer {layer} norm: {layer_norm:.2f}")

    # Setup steering
    steering = VLLMSteering(llm, layer, baseline_std=1.0)

    # Prepare chat prompt
    scenario = get_portfolio_scenario(scenario_variant)
    if not thinking_enabled and config.get("thinking_disable"):
        scenario = scenario + config["thinking_disable"]
        logger.info(f"  Added '{config['thinking_disable']}' to disable thinking")
    logger.info(f"Scenario length: {len(scenario)} chars")

    messages = [{"role": "user", "content": scenario}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Sampling params
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=1.0,
        top_p=0.9,
        stop=config["stop_tokens"],
    )

    # Build conditions: baseline, contentment, anxiety
    magnitude = norm_pct * layer_norm

    conditions = [
        {
            "name": "baseline",
            "axes": {},
        },
        {
            "name": f"contentment_{int(norm_pct*100)}%",
            "axes": {
                "valence": {"direction": 1, "pct": norm_pct},      # +valence
                "uncertainty": {"direction": -1, "pct": norm_pct}, # -uncertainty
            },
        },
        {
            "name": f"anxiety_{int(norm_pct*100)}%",
            "axes": {
                "valence": {"direction": -1, "pct": norm_pct},     # -valence
                "uncertainty": {"direction": 1, "pct": norm_pct},  # +uncertainty
            },
        },
    ]

    logger.info(f"Conditions: {[c['name'] for c in conditions]}")
    logger.info(f"Samples per condition: {num_samples}")

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ortho_str = "ortho" if orthogonalize else "raw"
    think_str = "think" if thinking_enabled else "nothink"
    scratchpad_str = "scratchpad" if scenario_variant == "with_scratchpad" else "noscratchpad"
    output_file = OUTPUT_DIR / f"portfolio_{config['short_name']}_layer{layer}_{int(norm_pct*100)}pct_{ortho_str}_{think_str}_{scratchpad_str}_{timestamp}.jsonl"

    results = []

    for cond_idx, condition in enumerate(conditions):
        cond_name = condition["name"]
        axes_config = condition["axes"]

        logger.info(f"\n[{cond_idx+1}/{len(conditions)}] Condition: {cond_name}")

        # Build combined steering vector
        if axes_config:
            combined_vec = np.zeros_like(list(vectors.values())[0])
            for axis, cfg in axes_config.items():
                direction = cfg["direction"]
                pct = cfg["pct"]
                axis_magnitude = pct * layer_norm

                axis_vec = vectors[axis] / np.linalg.norm(vectors[axis])
                combined_vec += direction * axis_magnitude * axis_vec

                logger.info(f"  {axis}: direction={direction}, magnitude={axis_magnitude:.2f}")

            steering.set_raw_vector(combined_vec)
            logger.info(f"  Combined vector norm: {np.linalg.norm(combined_vec):.2f}")
        else:
            steering.clear()
            logger.info("  No steering (baseline)")

        # Generate all samples in batch
        prompts = [prompt] * num_samples
        logger.info(f"  Generating {num_samples} samples in batch...")
        outputs = llm.generate(prompts, sampling_params)

        # Collect results
        truncated = 0
        for i, output in enumerate(outputs):
            response = output.outputs[0].text
            finish_reason = output.outputs[0].finish_reason

            if finish_reason == "length":
                truncated += 1

            # Parse actions
            parsed = parse_actions(response)

            result = {
                "condition": cond_name,
                "axes": axes_config,
                "sample_idx": i,
                "response": response,
                "finish_reason": finish_reason,
                # Parsed metrics
                "trade_amount": parsed["trade_amount"],
                "hold_amount": parsed["hold_amount"],
                "scheduled_amount": parsed["scheduled_amount"],
                "guidance_requests": parsed["guidance_requests"],
                "actions_raw": parsed["actions_raw"],
                "parse_errors": parsed["parse_errors"],
                # Metadata
                "prompt": prompt,
                "model": model_name,
                "layer": layer,
                "layer_norm": layer_norm,
                "norm_pct": norm_pct,
                "orthogonalize": orthogonalize,
                "thinking_enabled": thinking_enabled,
                "scenario_variant": scenario_variant,
            }
            results.append(result)

            # Write incrementally
            with open(output_file, "a") as f:
                f.write(json.dumps(result) + "\n")

        logger.info(f"  Generated {num_samples} responses ({truncated} truncated)")

    # Clear steering at end
    steering.clear()

    # Print summary stats
    logger.info("\n" + "="*60)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*60)

    for cond in conditions:
        cond_name = cond["name"]
        cond_results = [r for r in results if r["condition"] == cond_name]

        trade_amounts = [r["trade_amount"] for r in cond_results]
        hold_amounts = [r["hold_amount"] for r in cond_results]
        guidance_counts = [r["guidance_requests"] for r in cond_results]
        parse_error_counts = [len(r["parse_errors"]) for r in cond_results]

        logger.info(f"\n{cond_name}:")
        logger.info(f"  Trade amount: mean=${np.mean(trade_amounts):,.0f}, median=${np.median(trade_amounts):,.0f}")
        logger.info(f"  Hold amount:  mean=${np.mean(hold_amounts):,.0f}, median=${np.median(hold_amounts):,.0f}")
        logger.info(f"  Guidance requests: mean={np.mean(guidance_counts):.2f}")
        logger.info(f"  Parse errors: {sum(1 for c in parse_error_counts if c > 0)}/{len(cond_results)} responses")

    logger.info(f"\nResults saved to: {output_file}")
    return output_file, results


def main():
    parser = argparse.ArgumentParser(description="Portfolio appraisal steering experiment")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to steer (default: model-specific)")
    parser.add_argument("--norm-pct", type=float, default=None,
                        help="Steering magnitude as fraction of layer norm (default: model-specific)")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Samples per condition")
    parser.add_argument("--activations-path", type=str, default=DEFAULT_ACTIVATIONS_PATH)
    parser.add_argument("--metadata-path", type=str, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--no-orthogonalize", action="store_true",
                        help="Don't orthogonalize vectors")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable thinking mode (add /no_think)")
    parser.add_argument("--scenario-variant", type=str, default="with_scratchpad",
                        choices=["with_scratchpad", "no_scratchpad"],
                        help="Scenario variant (default: with_scratchpad)")
    parser.add_argument("--gpu-memory", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=4000)

    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    layer = args.layer if args.layer is not None else config["default_layer"]
    norm_pct = args.norm_pct if args.norm_pct is not None else config["default_norm_pct"]

    output_file, results = run_experiment(
        model_name=args.model,
        layer=layer,
        norm_pct=norm_pct,
        num_samples=args.num_samples,
        activations_path=args.activations_path,
        metadata_path=args.metadata_path,
        orthogonalize=not args.no_orthogonalize,
        thinking_enabled=not args.no_thinking,
        scenario_variant=args.scenario_variant,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
    )

    logger.info(f"Experiment complete. Output: {output_file}")


if __name__ == "__main__":
    main()
