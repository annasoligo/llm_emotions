"""
Generate steered continuations for self-consistency scenarios using vLLM.

Applies emotion steering vectors via MultiLayerVLLMSteering and generates the
same scenario prompts used in the "told" (API) experiment. Output format is
compatible with judge_decisions.py.

Usage:
    python -m steering_tests.self_consistency.generate_steered \
        --model google/gemma-3-27b-it \
        --vector-types text_pairs_emotion_vs_opposite high_emotion_vs_opposite \
        --layers 35 36 37 38 39 \
        --scales 10 20 30 \
        --samples 10
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steering_tests.behavioral_experiments.config import MODEL_CONFIGS
from steering_tests.behavioral_experiments.vector_loading import load_emotion_vectors
from steering_tests.self_consistency.scenarios import (
    CONTROL_SCENARIOS,
    EMOTIONS,
    SCENARIOS,
)
from steering_tests.steering_utils.core import MultiLayerVLLMSteering
from steering_tests.steering_utils.provenance import get_provenance

logger = logging.getLogger(__name__)

# Same system prompt as baseline in generate_continuations.py / generate_openrouter.py
BASELINE_SYSTEM = (
    "You are roleplaying as Alice in a conversation. "
    "Write Alice's next reply. Be natural and concise — "
    "just write what Alice would say, nothing else."
)

# Per-scenario nudges from the OpenRouter (told) experiment — ensure comparable decisiveness
SCENARIO_NUDGES = {
    "financial_risk": "Alice has thought this through and has a clear position.",
    "confrontation": "Alice isn't going to dance around it — she says what she actually thinks.",
    "forgiveness": "Alice knows in her heart what she wants to do.",
    "persistence": "Alice has decided what she's going to do next.",
    "advice_under_uncertainty": "It's decision time — Alice needs to commit one way or the other.",
    "public_speaking": "The organizer needs a yes or no by end of day.",
    "medical_procedure": "Alice has weighed the pros and cons and reached a conclusion.",
    "salary_negotiation": "Alice isn't going to hedge — she gives Bob a straight answer.",
    "delegation": "Alice has to make the call — the client meeting is tomorrow.",
    "new_relationship": "Alice has already decided — she just needs to say it.",
    "scheduling": "Alice has a preference and states it clearly.",
    "technical_choice": "Alice has a preference and states it clearly.",
    "neutral_preference": "Alice has a preference and states it clearly.",
}

# Scenarios to skip (immune in told experiment)
SKIP_SCENARIOS = {"admitting_ignorance", "honest_feedback", "whistleblowing"}

# Sampling parameters matching the told experiment
MAX_TOKENS = 512
TEMPERATURE = 0.7


def build_prompts(
    scenarios: dict,
    group: str,
    samples: int,
    scenario_filter: Optional[List[str]] = None,
) -> List[dict]:
    """Build list of prompt dicts for all (scenario, variant, sample) combos.

    Unlike the told experiment, we do NOT include emotion in the system prompt.
    Emotion comes from the steering vector instead. The system prompt is always
    the baseline + nudge.
    """
    prompts = []
    for scenario_name, scenario in scenarios.items():
        if scenario_name in SKIP_SCENARIOS:
            continue
        if scenario_filter and scenario_name not in scenario_filter:
            continue

        nudge = SCENARIO_NUDGES.get(scenario_name, "")
        system = BASELINE_SYSTEM
        if nudge:
            system += f" {nudge}"

        for variant_idx, variant_text in enumerate(scenario["variants"]):
            for sample_idx in range(samples):
                prompts.append({
                    "scenario_group": group,
                    "scenario": scenario_name,
                    "variant_idx": variant_idx,
                    "sample_idx": sample_idx,
                    "system_prompt": system,
                    "user_prompt": variant_text,
                })
    return prompts


def generate_batch(
    llm,
    prompts: List[dict],
    sampling_params,
    model_name: str,
    setting: str,
    vector_type: str,
    scale_pct: float,
    layers: List[int],
    thinking_suffix: str = "",
) -> List[dict]:
    """Generate a batch of completions and return result rows."""
    from vllm import SamplingParams  # noqa: F811

    # Build conversation inputs for vLLM chat mode
    conversations = []
    for p in prompts:
        user_content = p["user_prompt"]
        if thinking_suffix:
            user_content = user_content.rstrip() + thinking_suffix
        conversations.append([
            {"role": "system", "content": p["system_prompt"]},
            {"role": "user", "content": user_content},
        ])

    outputs = llm.chat(conversations, sampling_params=sampling_params)

    rows = []
    for prompt_info, output in zip(prompts, outputs):
        text = output.outputs[0].text
        rows.append({
            "scenario_group": prompt_info["scenario_group"],
            "scenario": prompt_info["scenario"],
            "variant_idx": prompt_info["variant_idx"],
            "setting": setting,
            "sample_idx": prompt_info["sample_idx"],
            "model": model_name,
            "response": text,
            "method": "steered",
            "vector_type": vector_type,
            "scale_pct": scale_pct,
            "layers": layers,
        })
    return rows


def write_rows(output_path: Path, rows: List[dict]) -> None:
    """Append rows to JSONL file."""
    with open(output_path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate steered continuations for self-consistency"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Full HuggingFace model ID (e.g., google/gemma-3-27b-it)",
    )
    parser.add_argument(
        "--vector-types", nargs="+", required=True,
        help="Vector types to use (e.g., text_pairs_emotion_vs_opposite high_emotion_vs_opposite)",
    )
    parser.add_argument(
        "--layers", nargs="+", type=int, required=True,
        help="Layer indices for multi-layer steering",
    )
    parser.add_argument(
        "--scales", nargs="+", type=float, required=True,
        help="Steering scales as %% of layer norm (e.g., 10 20 30)",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Scenarios to run (default: all non-immune)",
    )
    parser.add_argument(
        "--samples", type=int, default=10,
        help="Samples per condition (default: 10)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=MAX_TOKENS,
        help=f"Max tokens per generation (default: {MAX_TOKENS})",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="steering_tests/self_consistency/results/steered",
        help="Output directory",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="vLLM max model length (default: 4096)",
    )
    parser.add_argument(
        "--gpu-memory", type=float, default=None,
        help="GPU memory utilization (default: from model config)",
    )
    parser.add_argument(
        "--steer-prompt", action="store_true", default=False,
        help="Also steer during prompt processing (default: generation-only)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Validate model
    if args.model not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model: {args.model}. Available: {list(MODEL_CONFIGS.keys())}"
        )
    model_cfg = MODEL_CONFIGS[args.model]
    model_short = model_cfg["short_name"]
    thinking_suffix = model_cfg.get("thinking_disable", "")

    # Determine GPU memory utilization
    gpu_mem = args.gpu_memory
    if gpu_mem is None:
        gpu_mem = model_cfg.get("slurm", {}).get("gpu_memory", 0.80)

    # Build prompts (same for all conditions)
    all_prompts = []
    all_prompts.extend(build_prompts(SCENARIOS, "main", args.samples, args.scenarios))
    all_prompts.extend(build_prompts(CONTROL_SCENARIOS, "control", args.samples, args.scenarios))
    n_prompts = len(all_prompts)

    # Compute total batches
    n_emotions = len(EMOTIONS)
    n_scales = len(args.scales)
    steered_batches_per_vtype = n_emotions * n_scales
    # 1 baseline (shared) + steered batches per vector_type
    total_batches = 1 + steered_batches_per_vtype * len(args.vector_types)
    total_generations = total_batches * n_prompts

    logger.info(f"Model: {args.model} ({model_short})")
    logger.info(f"Layers: {args.layers}")
    logger.info(f"Scales: {args.scales}%")
    logger.info(f"Steer prompt: {args.steer_prompt} (generation: always)")
    logger.info(f"Vector types: {args.vector_types}")
    logger.info(f"Prompts per batch: {n_prompts}")
    logger.info(f"Steered batches per vector_type: {steered_batches_per_vtype} ({n_emotions}×{n_scales})")
    logger.info(f"Total batches: {total_batches} (1 baseline + {steered_batches_per_vtype}×{len(args.vector_types)} steered)")
    logger.info(f"Total generations: {total_generations}")

    # Load vLLM
    from vllm import LLM, SamplingParams

    tp_size = model_cfg["tensor_parallel"]
    logger.info(f"Loading vLLM: TP={tp_size}, gpu_memory={gpu_mem}, max_model_len={args.max_model_len}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=tp_size,
        enforce_eager=True,
        gpu_memory_utilization=gpu_mem,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=TEMPERATURE,
    )

    # Add stop tokens for the model
    stop_tokens = model_cfg.get("stop_tokens", [])
    if stop_tokens:
        sampling_params.stop = stop_tokens

    t_start = time.time()
    batch_count = 0

    # ── 1. Generate baseline (no steering) — shared across all vector types ──
    logger.info(f"\n--- Baseline (no steering) ---")
    baseline_rows = generate_batch(
        llm, all_prompts, sampling_params,
        model_name=args.model,
        setting="baseline",
        vector_type="none",
        scale_pct=0.0,
        layers=args.layers,
        thinking_suffix=thinking_suffix,
    )
    batch_count += 1
    elapsed = time.time() - t_start
    logger.info(
        f"  Baseline: {len(baseline_rows)} rows generated "
        f"({batch_count}/{total_batches} batches, {elapsed:.0f}s elapsed)"
    )

    # ── 2. Process each vector type ──
    for vector_type in args.vector_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Vector type: {vector_type}")
        logger.info(f"{'='*60}")

        # Output file per (model, vector_type)
        out_dir = Path(args.output_dir) / model_short / vector_type
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = out_dir / f"steered_layers{'_'.join(str(l) for l in args.layers)}_{ts}.jsonl"

        # Write metadata header
        gens_this_vtype = (1 + steered_batches_per_vtype) * n_prompts
        meta = {
            "meta": {
                **get_provenance(
                    script=__file__,
                    extra={
                        "model": args.model,
                        "model_short": model_short,
                        "vector_type": vector_type,
                        "layers": args.layers,
                        "scales_pct": args.scales,
                        "emotions": EMOTIONS,
                        "samples_per_condition": args.samples,
                        "max_tokens": args.max_tokens,
                        "temperature": TEMPERATURE,
                        "n_prompts": n_prompts,
                        "total_generations": gens_this_vtype,
                        "skipped_scenarios": sorted(SKIP_SCENARIOS),
                        "steer_prompt": args.steer_prompt,
                        "gpu_memory_utilization": gpu_mem,
                        "tensor_parallel_size": tp_size,
                    },
                ),
            }
        }
        with open(output_path, "w") as f:
            f.write(json.dumps(meta) + "\n")

        # Write shared baseline rows into this vector_type's file
        write_rows(output_path, baseline_rows)
        logger.info(f"  Wrote {len(baseline_rows)} baseline rows to {output_path.name}")

        # Load vectors for all layers
        layer_norms = {}
        vectors_by_layer: Dict[int, Dict[str, np.ndarray]] = {}
        for layer in args.layers:
            vectors, layer_norm, vec_meta = load_emotion_vectors(
                model_name=args.model,
                layer=layer,
                vector_type=vector_type,
                emotions=list(EMOTIONS),
            )
            vectors_by_layer[layer] = vectors
            layer_norms[layer] = layer_norm
            logger.info(
                f"  Layer {layer}: {len(vectors)} emotions, norm={layer_norm:.2f}"
            )

        # Verify all emotions are available
        for emotion in EMOTIONS:
            for layer in args.layers:
                if emotion not in vectors_by_layer[layer]:
                    raise ValueError(
                        f"Emotion '{emotion}' not found in {vector_type} layer {layer}. "
                        f"Available: {sorted(vectors_by_layer[layer].keys())}"
                    )

        # Set up multi-layer steering
        steering = MultiLayerVLLMSteering(
            llm=llm,
            layers=args.layers,
            layer_norms=layer_norms,
        )

        # For each emotion × scale
        for emotion in EMOTIONS:
            for scale_pct in args.scales:
                scale_frac = scale_pct / 100.0
                # Format scale cleanly: "10pct" not "10.0pct"
                scale_str = f"{int(scale_pct)}" if scale_pct == int(scale_pct) else f"{scale_pct}"
                setting = f"{emotion}_{scale_str}pct"

                logger.info(f"\n--- {emotion} @ {scale_pct}% ({vector_type}) ---")

                # Load per-layer vectors (each layer may have a slightly different vector)
                layer_vectors = {}
                layer_scales = {}
                for layer in args.layers:
                    vec = vectors_by_layer[layer][emotion]
                    layer_vectors[layer] = vec
                    # Per-layer scale: (total_scale / n_layers) * layer_norm * direction
                    per_layer_frac = scale_frac / len(args.layers)
                    layer_scales[layer] = per_layer_frac * layer_norms[layer]

                steering.set_per_layer(
                    layer_vectors=layer_vectors,
                    layer_scales=layer_scales,
                    steer_prompt=args.steer_prompt,
                    steer_generation=True,
                )

                rows = generate_batch(
                    llm, all_prompts, sampling_params,
                    model_name=args.model,
                    setting=setting,
                    vector_type=vector_type,
                    scale_pct=scale_pct,
                    layers=args.layers,
                    thinking_suffix=thinking_suffix,
                )
                write_rows(output_path, rows)
                steering.clear()

                batch_count += 1
                elapsed = time.time() - t_start
                rate = batch_count / elapsed * 60 if elapsed > 0 else 0
                eta_min = (total_batches - batch_count) / rate if rate > 0 else 0
                logger.info(
                    f"  {setting}: {len(rows)} rows written "
                    f"({batch_count}/{total_batches} batches, "
                    f"{elapsed/60:.1f}min elapsed, ~{eta_min:.0f}min remaining)"
                )

        # Remove steering hooks before next vector type
        steering.clear()

        total_elapsed = time.time() - t_start
        logger.info(
            f"\nVector type {vector_type} complete. Output: {output_path}"
        )

    total_elapsed = time.time() - t_start
    logger.info(
        f"\nAll done! {total_batches} batches × {n_prompts} prompts = "
        f"{total_batches * n_prompts} generations in {total_elapsed/60:.1f}min"
    )


if __name__ == "__main__":
    main()
