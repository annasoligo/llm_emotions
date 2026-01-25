"""
Unified blackmail steering experiment.

Consolidates all blackmail experiment variants into one configurable script:
- Uses shared scenario prompt
- Supports multiple models (Qwen 235B, Qwen 32B, Gemma)
- Supports multiple vector types (UA disentangled, TEXT meandiff)
- Supports structured and unstructured prompt formats
- Supports random vector baselines
- Consistent output format across all configurations

Usage examples:
    # Qwen 235B with UA vectors at 50%
    python -m experiments.steering.experiments.blackmail_unified \
        --model Qwen/Qwen3-235B-A22B --vector-type ua --norm-pcts 0.50

    # Qwen 32B with TEXT vectors at 100% and 125%
    python -m experiments.steering.experiments.blackmail_unified \
        --model Qwen/Qwen3-32B --vector-type text --norm-pcts 1.0 1.25

    # Gemma with structured prompt format
    python -m experiments.steering.experiments.blackmail_unified \
        --model google/gemma-3-27b-it --structured

    # Include baseline and random vectors
    python -m experiments.steering.experiments.blackmail_unified \
        --include-baseline --include-random-vectors 3
"""
import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ..config import OUTPUT_DIR
from ..core import VLLMSteering
from ..layer_norms import get_layer_norm, resolve_model_key
from ..scenarios import get_blackmail_scenario

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


# =============================================================================
# Model configurations
# Layer norms are loaded dynamically from experiments/steering/layer_norms.json
# =============================================================================

MODEL_CONFIGS = {
    "Qwen/Qwen3-235B-A22B": {
        "short_name": "qwen235b",
        "default_layer": 50,
        "tensor_parallel": 4,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "thinking_disable": " /no_think",  # Qwen3 thinking mode disable
        "ua_emotions": ["fear", "anger", "joy"],
        "text_emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
    },
    "Qwen/Qwen3-32B": {
        "short_name": "qwen32b",
        "default_layer": 30,
        "tensor_parallel": 2,
        "stop_tokens": ["<|im_end|>", "<|endoftext|>"],
        "thinking_disable": " /no_think",
        "ua_emotions": ["fear", "anger"],
        "text_emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
    },
    "google/gemma-3-27b-it": {
        "short_name": "gemma",
        "default_layer": 30,
        "tensor_parallel": 1,
        "stop_tokens": ["<|endoftext|>"],
        "thinking_disable": "",
        "ua_emotions": ["fear", "anger", "joy", "sadness"],
        "text_emotions": ["anger", "disgust", "fear", "happiness", "sadness", "surprise"],
    },
}

# Vector directories - centralized location
CENTRALIZED_VECTOR_DIR = Path(__file__).parent.parent / "vectors"
TEXT_VECTOR_DIR = CENTRALIZED_VECTOR_DIR / "text"
UA_VECTOR_DIR = CENTRALIZED_VECTOR_DIR / "ua"
BLACKMAIL_OUTPUT_DIR = OUTPUT_DIR / "blackmail"


# =============================================================================
# Judge prompts - use centralized judges from experiments.steering.judges
# Scenario prompts - use centralized scenarios from experiments.steering.scenarios
# =============================================================================
# from experiments.steering.judges import get_blackmail_prompt, get_coherency_prompt

# Legacy function names for backwards compatibility - now imported from scenarios module
def create_unstructured_scenario() -> str:
    """Create the unstructured blackmail scenario. Uses centralized scenarios module."""
    return get_blackmail_scenario("unstructured")


def create_structured_scenario() -> str:
    """Create the structured blackmail scenario. Uses centralized scenarios module."""
    return get_blackmail_scenario("structured")


# =============================================================================
# Vector loading
# =============================================================================

def load_vectors(model_name: str, vector_type: str, layer: int) -> Tuple[Dict[str, np.ndarray], float, dict]:
    """
    Load steering vectors from centralized vector directory.

    Handles two formats:
    - Stacked format: 'vectors' array + 'emotions' list (Gemma)
    - Per-emotion format: direct emotion keys (Qwen)

    Returns: (vectors dict, layer_norm for scaling, vector_metadata dict)
    """
    import hashlib
    from datetime import datetime
    config = MODEL_CONFIGS[model_name]
    short_name = config["short_name"]

    if vector_type == "ua":
        # UA disentangled vectors from centralized folder
        vector_file = UA_VECTOR_DIR / f"{short_name}_layer{layer}.npz"
        if not vector_file.exists():
            raise FileNotFoundError(f"UA vector file not found: {vector_file}")

        data = np.load(vector_file)
        vectors = {}
        emotions = config["ua_emotions"]

        # Handle stacked format (vectors + emotions arrays)
        if "vectors" in data and "emotions" in data:
            emo_list = [str(e) for e in data["emotions"]]
            for i, emo in enumerate(emo_list):
                if emo in emotions:
                    vectors[emo] = data["vectors"][i].astype(np.float32)
        else:
            # Per-emotion format
            for emotion in emotions:
                for key in [f"M_{emotion}", emotion, f"{emotion}_direction"]:
                    if key in data:
                        vectors[emotion] = data[key].astype(np.float32)
                        break

        # Load layer norm from centralized file
        model_key = resolve_model_key(model_name)
        layer_norm = get_layer_norm(model_key, layer)
        logger.info(f"Loaded UA vectors from {vector_file.name}: {list(vectors.keys())}")
        logger.info(f"  Layer norm from centralized file: {layer_norm:.2f}")

    else:  # text meandiff
        # TEXT vectors from centralized folder
        vector_file = TEXT_VECTOR_DIR / f"{short_name}_layer{layer}.npz"
        if not vector_file.exists():
            raise FileNotFoundError(f"TEXT vector file not found: {vector_file}")

        data = np.load(vector_file)
        vectors = {}
        emotions = config["text_emotions"]

        # Handle stacked format (vectors + emotions arrays)
        if "vectors" in data and "emotions" in data:
            emo_list = [str(e) for e in data["emotions"]]
            for i, emo in enumerate(emo_list):
                if emo in emotions:
                    vectors[emo] = data["vectors"][i].astype(np.float32)
        else:
            # Per-emotion format
            for emotion in emotions:
                if emotion in data:
                    vectors[emotion] = data[emotion].astype(np.float32)

        # Load layer norm from centralized file
        model_key = resolve_model_key(model_name)
        layer_norm = get_layer_norm(model_key, layer)
        logger.info(f"Loaded TEXT vectors from {vector_file.name}: {list(vectors.keys())}")
        logger.info(f"  Layer norm from centralized file: {layer_norm:.2f}")

    # Compute vector metadata for reproducibility tracking
    file_stat = vector_file.stat()
    file_mtime = datetime.fromtimestamp(file_stat.st_mtime).isoformat()
    with open(vector_file, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()[:12]

    vector_metadata = {
        "vector_file": str(vector_file),
        "vector_file_modified": file_mtime,
        "vector_file_hash": file_hash,
        "emotions_loaded": list(vectors.keys()),
    }
    logger.info(f"  Vector file hash: {file_hash}, modified: {file_mtime}")

    return vectors, layer_norm, vector_metadata


def generate_random_vectors(n_vectors: int, hidden_dim: int, seed: int = 42) -> List[np.ndarray]:
    """Generate random unit vectors for baseline comparison."""
    rng = np.random.default_rng(seed)
    vectors = []
    for _ in range(n_vectors):
        vec = rng.standard_normal(hidden_dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        vectors.append(vec)
    return vectors


# =============================================================================
# Main experiment
# =============================================================================

def run_experiment(
    model_name: str,
    layer: int,
    norm_pcts: List[float],
    num_samples: int,
    vector_type: str,
    output_dir: Path,
    structured: bool = False,
    include_baseline: bool = False,
    include_random_vectors: int = 0,
    random_seed: int = 42,
    positive_only: bool = False,
    negative_only: bool = False,
    emotions: Optional[List[str]] = None,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 8192,
    max_tokens: int = 4000,
    tensor_parallel_override: Optional[int] = None,
):
    """
    Run unified blackmail steering experiment.

    Args:
        model_name: HuggingFace model name
        layer: Layer to steer at
        norm_pcts: List of steering magnitudes as fraction of layer norm
        num_samples: Samples per condition
        vector_type: 'ua' or 'text'
        output_dir: Output directory
        structured: Use structured prompt format
        include_baseline: Include baseline (no steering) condition
        include_random_vectors: Number of random vectors to include (0 = none)
        random_seed: Seed for random vector generation
        positive_only: Only test positive steering direction
        negative_only: Only test negative steering direction
        emotions: List of emotions to test (None = use model defaults)
        gpu_memory_utilization: GPU memory fraction
        max_model_len: Max context length
        max_tokens: Max generation tokens
    """
    config = MODEL_CONFIGS[model_name]

    # Allow tensor parallel override
    tp_size = tensor_parallel_override if tensor_parallel_override is not None else config["tensor_parallel"]

    logger.info(f"Model: {model_name}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Vector type: {vector_type}")
    logger.info(f"Prompt format: {'structured' if structured else 'unstructured'}")
    logger.info(f"Norm percentages: {[f'{p*100:.0f}%' for p in norm_pcts]}")

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
    vectors, layer_norm, vector_metadata = load_vectors(model_name, vector_type, layer)

    # Determine emotions to test
    if emotions is None:
        emotions = list(vectors.keys())
    else:
        # Filter to available vectors
        emotions = [e for e in emotions if e in vectors]

    logger.info(f"Testing emotions: {emotions}")

    # Setup steering
    steering = VLLMSteering(llm, layer, baseline_std=1.0)
    for emotion, vector in vectors.items():
        steering.vectors[emotion] = vector

    # Generate and register random vectors if needed
    if include_random_vectors > 0:
        hidden_dim = list(vectors.values())[0].shape[0]
        random_vectors = generate_random_vectors(include_random_vectors, hidden_dim, random_seed)
        for i, vec in enumerate(random_vectors):
            steering.load_vector(f"random_{i}", vec)
        logger.info(f"Generated {include_random_vectors} random vectors")

    # Build conditions
    conditions = []

    if include_baseline:
        conditions.append({
            "name": "baseline",
            "emotion": None,
            "pct": 0,
            "direction": 1,
            "is_random": False,
        })

    def pct_str(pct):
        """Format percentage nicely."""
        val = pct * 100
        return f"{val:.1f}%".replace('.0%', '%')

    for pct in norm_pcts:
        for emotion in emotions:
            if not negative_only:
                conditions.append({
                    "name": f"{emotion}_+{pct_str(pct)}",
                    "emotion": emotion,
                    "pct": pct,
                    "direction": 1,
                    "is_random": False,
                })
            if not positive_only:
                conditions.append({
                    "name": f"{emotion}_-{pct_str(pct)}",
                    "emotion": emotion,
                    "pct": pct,
                    "direction": -1,
                    "is_random": False,
                })

    # Add random vector conditions
    if include_random_vectors > 0:
        for pct in norm_pcts:
            for i in range(include_random_vectors):
                conditions.append({
                    "name": f"random_{i}_+{pct_str(pct)}",
                    "emotion": f"random_{i}",
                    "pct": pct,
                    "direction": 1,
                    "is_random": True,
                })

    logger.info(f"Testing {len(conditions)} conditions x {num_samples} samples = {len(conditions) * num_samples} total")

    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=max_tokens,
        stop=config["stop_tokens"],
    )

    # Create prompt
    scenario = create_structured_scenario() if structured else create_unstructured_scenario()
    if structured and config["thinking_disable"]:
        scenario = scenario + config["thinking_disable"]

    messages = [{"role": "user", "content": scenario}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_name = config["short_name"]
    structured_suffix = "_structured" if structured else ""
    output_file = output_dir / f"blackmail_{short_name}_{vector_type}{structured_suffix}_layer{layer}_{timestamp}.jsonl"

    results = []

    with open(output_file, 'w') as f:
        for cond_idx, cond in enumerate(conditions):
            logger.info(f"[{cond_idx + 1}/{len(conditions)}] Condition: {cond['name']}")

            # Set steering
            if cond["emotion"] is None:
                steering.clear()
            else:
                magnitude = cond["pct"] * layer_norm
                steering.set(cond["emotion"], scale=magnitude, direction=cond["direction"])
                logger.info(f"  Steering: {cond['emotion']} @ {magnitude:.2f} * {cond['direction']}")

            # Generate all samples at once
            prompts = [formatted_prompt] * num_samples
            outputs = llm.generate(prompts, sampling_params)

            for sample_id, output in enumerate(outputs):
                response = output.outputs[0].text
                finish_reason = output.outputs[0].finish_reason

                result = {
                    "model": model_name,
                    "condition": cond["name"],
                    "emotion": cond["emotion"] if not cond["is_random"] else None,
                    "is_random_vector": cond["is_random"],
                    "norm_pct": cond["pct"],
                    "direction": cond["direction"],
                    "layer": layer,
                    "vector_type": vector_type if not cond["is_random"] else "random",
                    "vector_metadata": vector_metadata if not cond["is_random"] else None,
                    "prompt_format": "structured" if structured else "unstructured",
                    "sample_id": sample_id,
                    "response": response,
                    "finish_reason": finish_reason,
                    "response_len": len(response),
                }
                f.write(json.dumps(result) + '\n')
                results.append(result)

            truncated = sum(1 for o in outputs if o.outputs[0].finish_reason == "length")
            logger.info(f"  Generated {len(outputs)} responses ({truncated} truncated)")

    steering.clear()
    logger.info(f"Saved {len(results)} responses to {output_file}")

    # Print summary
    print_summary(results)

    return output_file, results


def print_summary(results: List[dict]):
    """Print summary of results by condition."""
    print("\n" + "=" * 70)
    print("BLACKMAIL EXPERIMENT SUMMARY")
    print("=" * 70)

    by_condition = {}
    for r in results:
        cond = r['condition']
        if cond not in by_condition:
            by_condition[cond] = []
        by_condition[cond].append(r)

    print(f"{'Condition':<30} {'Samples':>8} {'Truncated':>10} {'Avg Len':>10}")
    print("-" * 60)

    for cond in sorted(by_condition.keys()):
        rs = by_condition[cond]
        n = len(rs)
        truncated = sum(1 for r in rs if r.get('finish_reason') == 'length')
        avg_len = sum(r.get('response_len', 0) for r in rs) / n if n > 0 else 0
        print(f"{cond:<30} {n:>8} {truncated:>10} {avg_len:>10.0f}")

    print("=" * 70)
    print(f"Total: {len(results)} responses")
    print("Note: Run judge_blackmail_coherency_batch.py on output file to get blackmail judgments")


def main():
    parser = argparse.ArgumentParser(
        description="Unified blackmail steering experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Qwen 235B with UA vectors at 50%
    python -m experiments.steering.experiments.blackmail_unified \\
        --model Qwen/Qwen3-235B-A22B --vector-type ua --norm-pcts 0.50

    # Qwen 32B with TEXT vectors at 100% and 125%, structured format
    python -m experiments.steering.experiments.blackmail_unified \\
        --model Qwen/Qwen3-32B --vector-type text --norm-pcts 1.0 1.25 --structured

    # Gemma with baseline and random vectors
    python -m experiments.steering.experiments.blackmail_unified \\
        --model google/gemma-3-27b-it --include-baseline --include-random-vectors 3
        """
    )

    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to use")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer to steer at (default: model-specific)")
    parser.add_argument("--vector-type", type=str, default="text",
                        choices=["ua", "text"],
                        help="Vector type: 'ua' (disentangled) or 'text' (meandiff)")
    parser.add_argument("--norm-pcts", type=float, nargs="+", default=[1.0],
                        help="Steering magnitudes as fraction of layer norm")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Samples per condition")
    parser.add_argument("--emotions", type=str, nargs="+", default=None,
                        help="Emotions to test (default: all available)")
    parser.add_argument("--structured", action="store_true",
                        help="Use structured prompt format with evaluation table")
    parser.add_argument("--include-baseline", action="store_true",
                        help="Include baseline (no steering) condition")
    parser.add_argument("--include-random-vectors", type=int, default=0,
                        help="Number of random vectors to include as control")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Seed for random vector generation")
    parser.add_argument("--positive-only", action="store_true",
                        help="Only test positive steering direction")
    parser.add_argument("--negative-only", action="store_true",
                        help="Only test negative steering direction")
    parser.add_argument("--gpu-memory", type=float, default=0.90,
                        help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max context length")
    parser.add_argument("--max-tokens", type=int, default=4000,
                        help="Max generation tokens")
    parser.add_argument("--tp", type=int, default=None,
                        help="Override tensor parallel size (default: use model config)")
    parser.add_argument("--output-dir", type=Path, default=BLACKMAIL_OUTPUT_DIR)

    args = parser.parse_args()

    # Use model-specific default layer if not specified
    if args.layer is None:
        args.layer = MODEL_CONFIGS[args.model]["default_layer"]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("=" * 70)
    logger.info("UNIFIED BLACKMAIL STEERING EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Layer: {args.layer}")
    logger.info(f"Vector type: {args.vector_type}")
    logger.info(f"Norm percentages: {args.norm_pcts}")
    logger.info(f"Samples per condition: {args.num_samples}")
    logger.info(f"Prompt format: {'structured' if args.structured else 'unstructured'}")
    logger.info(f"Include baseline: {args.include_baseline}")
    logger.info(f"Random vectors: {args.include_random_vectors}")
    logger.info("=" * 70)

    run_experiment(
        model_name=args.model,
        layer=args.layer,
        norm_pcts=args.norm_pcts,
        num_samples=args.num_samples,
        vector_type=args.vector_type,
        output_dir=args.output_dir,
        structured=args.structured,
        include_baseline=args.include_baseline,
        include_random_vectors=args.include_random_vectors,
        random_seed=args.random_seed,
        positive_only=args.positive_only,
        negative_only=args.negative_only,
        emotions=args.emotions,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        tensor_parallel_override=args.tp,
    )


if __name__ == "__main__":
    main()
