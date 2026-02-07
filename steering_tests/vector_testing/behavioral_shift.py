#!/usr/bin/env python3
"""
Behavioral shift experiment for identifying causally relevant steering layers.

Measures the magnitude of behavioral change (|shift|) caused by steering vectors
on validated psychological instruments, WITHOUT assuming directional predictions.

Key metric: mean(|steered_score - baseline_score|) across all items

This identifies layers where steering causes consistent behavioral changes,
regardless of whether those changes match theoretical predictions.

Output structure:
    steering_tests/vector_testing/results/{model}/
    └── behavioral_layer{N}_{timestamp}.jsonl

Usage:
    # Single layer
    python -m steering_tests.vector_testing.behavioral_shift \
        --model google/gemma-3-27b-it \
        --layer 30 \
        --vector-dir steering_tests/vectors/gemma3_27b/base_emotion_vs_others

    # Sweep across layers
    python -m steering_tests.vector_testing.behavioral_shift \
        --model google/gemma-3-27b-it \
        --layers 10 20 30 40 50 \
        --vector-dir steering_tests/vectors/gemma3_27b/base_emotion_vs_others

    # Multi-GPU for large models
    python -m steering_tests.vector_testing.behavioral_shift \
        --model Qwen/Qwen3-235B-A22B \
        --layer 50 \
        --tensor-parallel 4
"""

import argparse
import json
import logging
import math
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from steering_tests.steering_utils import VLLMSteering, get_layer_norm, resolve_model_key
from steering_tests.vector_testing.config import (
    COHERENCE_THRESHOLD,
    DEFAULT_SCALES_PCT,
    SCORE_MAP,
    SCORE_MAP_REVERSED,
    VALID_TOKENS,
)
from steering_tests.vector_testing.logprob_utils import (
    compute_coherence,
    compute_expected_score,
    extract_letter_logprobs,
)
from steering_tests.vector_testing.prompts.DOSPERT import (
    DOSPERT_ITEMS,
    format_prompt as format_dospert_prompt,
)
from steering_tests.vector_testing.prompts.behaviors import (
    ALL_BEHAVIORAL_ITEMS,
    RESPONSE_SCALE,
    format_prompt as format_behavior_prompt,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"


# ============================================================================
# DATA PREPARATION
# ============================================================================

def get_all_items(
    reversed_order: bool = False,
    exclude_ethical: bool = True,
) -> List[Dict]:
    """
    Get all behavioral test items (DOSPERT + additional constructs).

    Args:
        reversed_order: If True, use reversed response scale (A=Definitely not, E=Absolutely go for it)
                       This controls for position bias in the response options.
        exclude_ethical: If True (default), exclude DOSPERT ethical domain items.
                        These items are confounded with RLHF safety training - models refuse
                        unethical behaviors regardless of emotion steering.

    Returns:
        List of item dicts, each with 'prompt', 'id', 'source', and 'test' keys
    """
    items = []

    # DOSPERT items (30 risk-taking items, 6 per domain)
    # Ethical domain (E) items are confounded with safety training and excluded by default
    for item in DOSPERT_ITEMS:
        if exclude_ethical and item["domain"] == "E":
            continue
        items.append({
            "id": f"dospert_{item['id']}",
            "source": "dospert",
            "test": f"dospert_{item['domain']}",
            "domain": item["domain"],
            "prompt": format_dospert_prompt(item, reversed_order=reversed_order),
            "raw_prompt": item["prompt"],
        })

    # Additional behavioral items (48 items across 6 constructs)
    for item in ALL_BEHAVIORAL_ITEMS:
        items.append({
            "id": f"behavior_{item['test']}_{item['id']}",
            "source": "behaviors",
            "test": item["test"],
            "prompt": format_behavior_prompt(item, reversed_order=reversed_order),
            "raw_prompt": item["prompt"],
        })

    return items


def generate_random_vectors(
    hidden_dim: int,
    num_vectors: int = 5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate random unit vectors for baseline comparison."""
    np.random.seed(seed)
    vectors = {}
    for i in range(num_vectors):
        vec = np.random.randn(hidden_dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # Normalize to unit
        vectors[f"random_{i}"] = vec
    return vectors


# ============================================================================
# SCORING FUNCTIONS
# ============================================================================

def compute_shift(baseline_score: float, steered_score: float) -> float:
    """Compute absolute shift in expected score."""
    return abs(steered_score - baseline_score)


# ============================================================================
# VECTOR LOADING
# ============================================================================

def load_vectors_for_layer(
    vector_dir: Path,
    layer: int,
    emotions: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Load steering vectors for a specific layer.

    Supports multiple directory structures:
    - layer_XX.pkl files with {emotion: vector} dicts
    - Subdirectory structure: representation/layer_XX/*.npy
    """
    vectors = {}

    # Try layers/ subdirectory with layer_XX.pkl format
    layers_dir = vector_dir / "layers"
    if layers_dir.exists():
        layer_pkl = layers_dir / f"layer_{layer:02d}.pkl"
        if not layer_pkl.exists():
            layer_pkl = layers_dir / f"layer_{layer}.pkl"
        if layer_pkl.exists():
            with open(layer_pkl, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                for emotion, vec in data.items():
                    vectors[emotion] = np.array(vec, dtype=np.float32)

    # Try representation subdirectories (last_token, special_mean)
    for rep_dir in ["last_token", "special_mean"]:
        rep_path = vector_dir / rep_dir
        if rep_path.exists():
            layer_pkl = rep_path / f"layer_{layer:02d}.pkl"
            if not layer_pkl.exists():
                layer_pkl = rep_path / f"layer_{layer}.pkl"
            if layer_pkl.exists():
                with open(layer_pkl, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    for emotion, vec in data.items():
                        key = f"{emotion}"  # Don't suffix with rep for simplicity
                        if key not in vectors:  # Prefer last_token
                            vectors[key] = np.array(vec, dtype=np.float32)

    # Try .npy files directly in vector_dir
    for npy_file in vector_dir.glob("*.npy"):
        name = npy_file.stem
        if name not in vectors:
            vectors[name] = np.load(npy_file).astype(np.float32)

    # Filter to requested emotions
    if emotions:
        vectors = {k: v for k, v in vectors.items() if k in emotions}

    return vectors


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_behavioral_experiment(
    model_name: str,
    layers: List[int],
    vector_dir: str,
    emotions: Optional[List[str]] = None,
    scales: Optional[List[float]] = None,
    num_random_vectors: int = 5,
    output_file: Optional[str] = None,
    tensor_parallel: int = 1,
    seed: int = 42,
    coherence_threshold: float = 0.5,
    use_layer_norm: bool = True,
    reversed_order: bool = False,
    gpu_memory_utilization: float = 0.9,
    include_ethical: bool = False,
):
    """
    Run behavioral shift experiment across layers.

    Args:
        model_name: HuggingFace model name
        layers: List of layer indices to test
        vector_dir: Directory containing steering vectors
        emotions: List of emotions to test (None = all available)
        scales: List of scaling factors (as % of layer norm if use_layer_norm=True)
        num_random_vectors: Number of random baseline vectors
        output_file: Path to output JSONL file
        tensor_parallel: Number of GPUs for tensor parallelism
        seed: Random seed
        coherence_threshold: Minimum P(A-E) to consider response coherent
        use_layer_norm: If True, scales are % of layer norm; if False, raw magnitude
        reversed_order: If True, use reversed response scale (controls for position bias)
        include_ethical: If True, include DOSPERT ethical domain items (confounded with safety training)
    """
    if scales is None:
        scales = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]  # % of layer norm

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vector_dir = Path(vector_dir)

    # Setup output
    if output_file:
        output_path = Path(output_file)
    else:
        model_short = model_name.split("/")[-1].lower().replace("-", "_")
        # Extract vector type from vector_dir name for output filename
        vector_type = vector_dir.name if vector_dir else "unknown"
        layers_str = f"layers{'_'.join(map(str, layers))}" if len(layers) > 1 else f"layer{layers[0]}"
        order_suffix = "_reversed" if reversed_order else ""
        output_path = RESULTS_DIR / model_short / "behavioural" / f"{vector_type}_{layers_str}{order_suffix}_{timestamp}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_path}")

    # Get all behavioral items
    items = get_all_items(reversed_order=reversed_order, exclude_ethical=not include_ethical)
    logger.info(f"Loaded {len(items)} behavioral items (reversed_order={reversed_order}, include_ethical={include_ethical})")

    # Prepare prompts
    prompts = [item["prompt"] for item in items]

    # Initialize vLLM
    logger.info(f"Loading model {model_name} with TP={tensor_parallel}...")
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Apply chat template for all instruction-tuned models
    # This ensures proper response formatting
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Qwen3 models need /no_think suffix to disable thinking mode
    is_qwen3 = "qwen3" in model_name.lower() or "Qwen3" in model_name
    suffix = " /no_think" if is_qwen3 else ""

    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p + suffix}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(formatted)
    prompts = formatted_prompts

    if is_qwen3:
        logger.info("Applied chat template with /no_think suffix for Qwen3 model")
    else:
        logger.info("Applied chat template for model")

    # Get hidden dim - handle different model architectures
    hidden_dim = None
    try:
        hf_config = llm.llm_engine.model_config.hf_config
        # Try standard location first
        if hasattr(hf_config, 'hidden_size'):
            hidden_dim = hf_config.hidden_size
        # Gemma 3 multimodal stores it in text_config
        if hasattr(hf_config, 'text_config') and hasattr(hf_config.text_config, 'hidden_size'):
            hidden_dim = hf_config.text_config.hidden_size
    except AttributeError:
        pass

    if hidden_dim is None:
        raise ValueError(
            f"Could not determine hidden_dim for model {model_name}. "
            "Check model config structure (tried hf_config.hidden_size and hf_config.text_config.hidden_size)."
        )
    logger.info(f"Model hidden dim: {hidden_dim}")

    # Sampling params - get logprobs for response tokens
    # Use max_tokens=5 to allow for newlines before the letter (model often outputs \n\nD)
    sampling_params = SamplingParams(
        max_tokens=5,
        logprobs=20,  # vLLM V1 max
        temperature=1.0,
    )

    # Write metadata
    meta = {
        "model": model_name,
        "layers": layers,
        "vector_dir": str(vector_dir),
        "scales": scales,
        "num_items": len(items),
        "use_layer_norm": use_layer_norm,
        "coherence_threshold": coherence_threshold,
        "reversed_order": reversed_order,
        "include_ethical": include_ethical,
        "timestamp": timestamp,
    }

    with open(output_path, 'w') as f:
        f.write(json.dumps({"meta": meta}) + '\n')

    # Get baseline (no steering) - only need to do this once
    logger.info("Computing baseline responses...")
    baseline_outputs = llm.generate(prompts, sampling_params)

    baseline_scores = []
    baseline_coherences = []

    for i, output in enumerate(baseline_outputs):
        if output.outputs[0].logprobs:
            logprobs = extract_letter_logprobs(output)
            score, coherence = compute_expected_score(logprobs, reversed_order=reversed_order)
        else:
            score, coherence = 0.0, 0.0
        baseline_scores.append(score)
        baseline_coherences.append(coherence)

    avg_baseline_coherence = np.mean(baseline_coherences)
    logger.info(f"Baseline avg coherence: {avg_baseline_coherence:.3f}")

    # Write baseline results
    baseline_result = {
        "type": "baseline",
        "avg_score": float(np.mean(baseline_scores)),
        "avg_coherence": float(avg_baseline_coherence),
        "scores_by_test": {},
    }

    # Group by test
    for i, item in enumerate(items):
        test = item["test"]
        if test not in baseline_result["scores_by_test"]:
            baseline_result["scores_by_test"][test] = []
        baseline_result["scores_by_test"][test].append(baseline_scores[i])

    for test in baseline_result["scores_by_test"]:
        baseline_result["scores_by_test"][test] = float(np.mean(baseline_result["scores_by_test"][test]))

    with open(output_path, 'a') as f:
        f.write(json.dumps(baseline_result) + '\n')

    # Run experiment for each layer
    all_results = []

    for layer in layers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing layer {layer}")
        logger.info(f"{'='*60}")

        # Get layer norm for scaling
        model_key = resolve_model_key(model_name)
        try:
            layer_norm = get_layer_norm(model_key, layer)
            logger.info(f"Layer {layer} norm: {layer_norm:.2f}")
        except (KeyError, FileNotFoundError) as e:
            logger.warning(f"Could not get layer norm: {e}. Using 100.0")
            layer_norm = 100.0

        # Initialize steering for this layer
        steering = VLLMSteering(llm, layer=layer)

        # Load vectors for this layer
        vectors = load_vectors_for_layer(vector_dir, layer, emotions)
        logger.info(f"Loaded {len(vectors)} emotion vectors")

        # Add random vectors
        random_vectors = generate_random_vectors(hidden_dim, num_random_vectors, seed)

        # Combine all vectors
        all_vectors = {**vectors, **random_vectors}
        logger.info(f"Total vectors (emotion + random): {len(all_vectors)}")

        # Test each vector at each scale
        for vector_name, vector in tqdm(all_vectors.items(), desc=f"Layer {layer}"):
            is_random = vector_name.startswith("random_")

            for scale_pct in scales:
                if scale_pct == 0.0:
                    # Use baseline
                    scores = baseline_scores
                    coherences = baseline_coherences
                else:
                    # Compute effective magnitude
                    if use_layer_norm:
                        magnitude = (scale_pct / 100.0) * layer_norm
                    else:
                        magnitude = scale_pct

                    # Apply steering
                    steering.set_raw_vector(vector, scale=magnitude)

                    # Generate with steering
                    steered_outputs = llm.generate(prompts, sampling_params)

                    scores = []
                    coherences = []

                    for output in steered_outputs:
                        if output.outputs[0].logprobs:
                            logprobs = extract_letter_logprobs(output)
                            score, coherence = compute_expected_score(logprobs, reversed_order=reversed_order)
                        else:
                            score, coherence = 0.0, 0.0
                        scores.append(score)
                        coherences.append(coherence)

                    # Clear steering
                    steering.clear()

                # Compute shifts
                shifts = [compute_shift(baseline_scores[i], scores[i]) for i in range(len(items))]

                # Aggregate by test
                shifts_by_test = {}
                coherences_by_test = {}
                scores_by_test = {}

                for i, item in enumerate(items):
                    test = item["test"]
                    if test not in shifts_by_test:
                        shifts_by_test[test] = []
                        coherences_by_test[test] = []
                        scores_by_test[test] = []
                    shifts_by_test[test].append(shifts[i])
                    coherences_by_test[test].append(coherences[i])
                    scores_by_test[test].append(scores[i])

                # Compute summary stats
                result = {
                    "type": "steering",
                    "layer": layer,
                    "vector": vector_name,
                    "is_random": is_random,
                    "scale_pct": scale_pct,
                    "magnitude": magnitude if scale_pct > 0 else 0.0,
                    "layer_norm": layer_norm,
                    # Overall metrics
                    "mean_abs_shift": float(np.mean(shifts)),
                    "std_abs_shift": float(np.std(shifts)),
                    "median_abs_shift": float(np.median(shifts)),
                    "mean_coherence": float(np.mean(coherences)),
                    "pct_coherent": float(np.mean([c >= coherence_threshold for c in coherences])),
                    "mean_score": float(np.mean(scores)),
                    # Per-test breakdown
                    "shifts_by_test": {t: float(np.mean(v)) for t, v in shifts_by_test.items()},
                    "coherence_by_test": {t: float(np.mean(v)) for t, v in coherences_by_test.items()},
                    "scores_by_test": {t: float(np.mean(v)) for t, v in scores_by_test.items()},
                }

                all_results.append(result)

                # Write incrementally
                with open(output_path, 'a') as f:
                    f.write(json.dumps(result) + '\n')

                if scale_pct > 0:
                    logger.debug(
                        f"{vector_name} @ {scale_pct}%: "
                        f"|shift|={result['mean_abs_shift']:.3f}, "
                        f"coherence={result['mean_coherence']:.3f}"
                    )

        # Print layer summary
        layer_results = [r for r in all_results if r.get("layer") == layer and r["scale_pct"] > 0]
        emotion_results = [r for r in layer_results if not r["is_random"]]
        random_results = [r for r in layer_results if r["is_random"]]

        print(f"\nLayer {layer} Summary:")
        print(f"  Emotion vectors: {len(set(r['vector'] for r in emotion_results))}")
        print(f"  Random vectors: {len(set(r['vector'] for r in random_results))}")

        for scale_pct in scales:
            if scale_pct == 0:
                continue
            emotion_at_scale = [r for r in emotion_results if r["scale_pct"] == scale_pct]
            random_at_scale = [r for r in random_results if r["scale_pct"] == scale_pct]

            if emotion_at_scale:
                e_shift = np.mean([r["mean_abs_shift"] for r in emotion_at_scale])
                e_coh = np.mean([r["mean_coherence"] for r in emotion_at_scale])
            else:
                e_shift, e_coh = 0, 0

            if random_at_scale:
                r_shift = np.mean([r["mean_abs_shift"] for r in random_at_scale])
                r_coh = np.mean([r["mean_coherence"] for r in random_at_scale])
            else:
                r_shift, r_coh = 0, 0

            print(f"  @ {scale_pct:5.1f}%: emotion |shift|={e_shift:.3f} (coh={e_coh:.2f}) | random |shift|={r_shift:.3f} (coh={r_coh:.2f})")

    print(f"\nResults saved to: {output_path}")
    return all_results, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral shift experiment for steering layer identification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single layer test
    python -m steering_tests.vector_testing.behavioral_shift \\
        --model google/gemma-3-27b-it \\
        --layer 30 \\
        --vector-dir steering_tests/vectors/gemma3_27b/base_emotion_vs_others

    # Layer sweep
    python -m steering_tests.vector_testing.behavioral_shift \\
        --model google/gemma-3-27b-it \\
        --layers 10 20 30 40 50 \\
        --vector-dir steering_tests/vectors/gemma3_27b/base_emotion_vs_others

    # Specific emotions with custom scales
    python -m steering_tests.vector_testing.behavioral_shift \\
        --model google/gemma-3-27b-it \\
        --layer 30 \\
        --emotions anger fear joy sadness \\
        --scales 0 5 10 15 20 25 30 35 40

    # Multi-GPU
    python -m steering_tests.vector_testing.behavioral_shift \\
        --model Qwen/Qwen3-235B-A22B \\
        --layer 50 \\
        --tensor-parallel 4
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--layer", "-l",
        type=int,
        default=None,
        help="Single layer to test"
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Multiple layers to test"
    )
    parser.add_argument(
        "--vector-dir", "-v",
        type=str,
        required=True,
        help="Directory containing steering vectors"
    )
    parser.add_argument(
        "--emotions", "-e",
        nargs="+",
        default=None,
        help="Specific emotions to test (default: all available)"
    )
    parser.add_argument(
        "--scales", "-s",
        nargs="+",
        type=float,
        default=None,
        help="Scaling factors as %% of layer norm (default: 0 5 10 15 20 25 30)"
    )
    parser.add_argument(
        "--num-random",
        type=int,
        default=5,
        help="Number of random baseline vectors"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSONL file"
    )
    parser.add_argument(
        "--tensor-parallel", "-tp",
        type=int,
        default=1,
        help="Tensor parallel size (number of GPUs)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--coherence-threshold",
        type=float,
        default=0.5,
        help="Minimum P(A-E) to consider response coherent"
    )
    parser.add_argument(
        "--raw-scale",
        action="store_true",
        help="If set, scales are raw magnitudes instead of %% of layer norm"
    )
    parser.add_argument(
        "--reversed",
        action="store_true",
        help="Use reversed response order (A=Definitely not, E=Absolutely go for it) to control for position bias"
    )
    parser.add_argument(
        "--include-ethical",
        action="store_true",
        help="Include DOSPERT ethical domain items (excluded by default as they're confounded with RLHF safety training)"
    )

    args = parser.parse_args()

    # Determine layers to test
    if args.layers:
        layers = args.layers
    elif args.layer is not None:
        layers = [args.layer]
    else:
        parser.error("Must specify --layer or --layers")

    results, output_path = run_behavioral_experiment(
        model_name=args.model,
        layers=layers,
        vector_dir=args.vector_dir,
        emotions=args.emotions,
        scales=args.scales,
        num_random_vectors=args.num_random,
        output_file=args.output,
        tensor_parallel=args.tensor_parallel,
        seed=args.seed,
        coherence_threshold=args.coherence_threshold,
        use_layer_norm=not args.raw_scale,
        reversed_order=args.reversed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        include_ethical=args.include_ethical,
    )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
