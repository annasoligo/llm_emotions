#!/usr/bin/env python3
"""
Steering vector KL divergence experiment using vLLM.

Measures how much steering vectors shift the model's output distribution
compared to baseline, using KL divergence as the metric.

This helps validate that:
1. Trained steering vectors produce meaningful distribution shifts
2. Different vector types (emotion vs random) have different KL profiles
3. The relationship between steering magnitude and distribution shift

Output structure:
    steering_tests/vector_testing/results/
    └── {model_short_name}/
        ├── kl_layer{N}_{vector_source}_{timestamp}.jsonl
        ├── kl_layer{N}_{vector_source}_{timestamp}_kl.png
        └── kl_layer{N}_{vector_source}_{timestamp}_entropy.png

Usage:
    python -m steering_tests.vector_testing.kl_div \
        --model google/gemma-3-27b-it \
        --layer 30 \
        --vector-dir steering_tests/vectors/gemma3_27b/base_emotion_vs_others

    # With tensor parallelism for large models
    python -m steering_tests.vector_testing.kl_div \
        --model Qwen/Qwen3-235B-A22B \
        --layer 50 \
        --tensor-parallel 4 \
        --vector-dir steering_tests/vectors/qwen235b/...
"""

import argparse
import json
import logging
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from steering_tests.steering_utils import VLLMSteering, get_layer_norm, resolve_model_key
from steering_tests.steering_utils.plotting import plot_kl_results, plot_entropy_change

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Results directory
RESULTS_DIR = Path(__file__).parent / "results"

# Model configurations
MODEL_CONFIGS = {
    "google/gemma-3-27b-it": {
        "hidden_dim": 5376,
        "default_layer": 30,
    },
    "google/gemma-3-12b-it": {
        "hidden_dim": 3840,
        "default_layer": 24,
    },
    "Qwen/Qwen3-32B": {
        "hidden_dim": 5120,
        "default_layer": 30,
    },
    "Qwen/Qwen3-235B-A22B": {
        "hidden_dim": 4096,
        "default_layer": 45,
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "hidden_dim": 5120,
        "default_layer": 24,
    },
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_fineweb_samples(
    num_samples: int = 100,
    min_length: int = 100,
    max_length: int = 500,
    seed: int = 42,
) -> List[str]:
    """
    Load random text samples from FineWeb for KL divergence calculation.

    Args:
        num_samples: Number of samples to load
        min_length: Minimum text length (chars)
        max_length: Maximum text length to use (truncate longer)
        seed: Random seed for shuffling

    Returns:
        List of text strings
    """
    logger.info(f"Loading {num_samples} samples from FineWeb...")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        "sample-10BT",
        split="train",
        streaming=True
    )

    dataset = dataset.shuffle(seed=seed)

    samples = []
    for item in dataset:
        text = item.get('text', '')
        if len(text) >= min_length:
            samples.append(text[:max_length])
            if len(samples) >= num_samples:
                break

    logger.info(f"Loaded {len(samples)} samples")
    return samples


def generate_random_vectors(
    hidden_dim: int,
    num_vectors: int = 5,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate random unit vectors for baseline comparison.

    Args:
        hidden_dim: Model hidden dimension
        num_vectors: Number of random vectors to generate
        seed: Random seed

    Returns:
        Dict mapping "random_0", "random_1", etc. to unit vectors
    """
    np.random.seed(seed)
    vectors = {}
    for i in range(num_vectors):
        vec = np.random.randn(hidden_dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # Normalize to unit
        vectors[f"random_{i}"] = vec
    return vectors


# ============================================================================
# KL DIVERGENCE COMPUTATION
# ============================================================================

def compute_kl_divergence(
    p_logprobs: Dict[str, float],
    q_logprobs: Dict[str, float],
    eps: float = 1e-10,
) -> float:
    """
    Compute KL(P || Q) from log probability dicts.

    KL(P || Q) = sum_x P(x) * log(P(x) / Q(x))
               = sum_x P(x) * (log P(x) - log Q(x))

    Args:
        p_logprobs: Log probabilities from distribution P (baseline)
        q_logprobs: Log probabilities from distribution Q (steered)
        eps: Small value for numerical stability

    Returns:
        KL divergence value
    """
    # Get union of tokens
    all_tokens = set(p_logprobs.keys()) | set(q_logprobs.keys())

    # Convert to probabilities
    p_probs = {}
    q_probs = {}

    for token in all_tokens:
        p_logp = p_logprobs.get(token, -100)  # Very low prob for missing tokens
        q_logp = q_logprobs.get(token, -100)
        p_probs[token] = np.exp(p_logp)
        q_probs[token] = np.exp(q_logp)

    # Normalize (they should already sum to ~1, but just in case)
    p_total = sum(p_probs.values())
    q_total = sum(q_probs.values())

    kl = 0.0
    for token in all_tokens:
        p = p_probs[token] / p_total
        q = q_probs[token] / q_total
        if p > eps:
            kl += p * (np.log(p + eps) - np.log(q + eps))

    return kl


def compute_js_divergence(
    p_logprobs: Dict[str, float],
    q_logprobs: Dict[str, float],
) -> float:
    """
    Compute Jensen-Shannon divergence (symmetric KL).

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    """
    # Get union of tokens
    all_tokens = set(p_logprobs.keys()) | set(q_logprobs.keys())

    # Convert to probabilities
    p_probs = {}
    q_probs = {}
    m_probs = {}

    for token in all_tokens:
        p_logp = p_logprobs.get(token, -100)
        q_logp = q_logprobs.get(token, -100)
        p_probs[token] = np.exp(p_logp)
        q_probs[token] = np.exp(q_logp)

    # Normalize
    p_total = sum(p_probs.values())
    q_total = sum(q_probs.values())

    for token in all_tokens:
        p_probs[token] /= p_total
        q_probs[token] /= q_total
        m_probs[token] = 0.5 * (p_probs[token] + q_probs[token])

    # Convert M back to logprobs for KL computation
    m_logprobs = {t: np.log(p + 1e-10) for t, p in m_probs.items()}
    p_logprobs_norm = {t: np.log(p + 1e-10) for t, p in p_probs.items()}
    q_logprobs_norm = {t: np.log(p + 1e-10) for t, p in q_probs.items()}

    kl_pm = compute_kl_divergence(p_logprobs_norm, m_logprobs)
    kl_qm = compute_kl_divergence(q_logprobs_norm, m_logprobs)

    return 0.5 * kl_pm + 0.5 * kl_qm


def compute_entropy(logprobs: Dict[str, float]) -> float:
    """Compute entropy from logprobs dict."""
    probs = np.array([np.exp(lp) for lp in logprobs.values()])
    probs = probs / probs.sum()  # Normalize
    return -np.sum(probs * np.log(probs + 1e-10))


# ============================================================================
# OUTPUT PATH GENERATION
# ============================================================================

def get_model_short_name(model_name: str) -> str:
    """Convert model name to short directory-safe name."""
    # Common mappings
    mappings = {
        "google/gemma-3-27b-it": "gemma3_27b",
        "google/gemma-3-12b-it": "gemma3_12b",
        "google/gemma-2-9b-it": "gemma2_9b",
        "Qwen/Qwen3-32B": "qwen3_32b",
        "Qwen/Qwen3-235B-A22B": "qwen3_235b",
        "Qwen/Qwen2.5-14B-Instruct": "qwen25_14b",
    }
    if model_name in mappings:
        return mappings[model_name]

    # Fallback: extract meaningful parts
    name = model_name.split("/")[-1]  # Get last part
    name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)  # Replace special chars
    name = re.sub(r'_+', '_', name)  # Collapse multiple underscores
    return name.lower()


def get_vector_source_name(vector_dir: Optional[str]) -> str:
    """Extract meaningful name from vector directory path."""
    if not vector_dir:
        return "random_only"

    path = Path(vector_dir)
    # Try to get meaningful parts from path
    # e.g., "vectors/gemma3_27b/base_emotion_vs_others/last_token/layer_30"
    parts = path.parts

    # Look for common meaningful directory names
    meaningful = []
    for part in reversed(parts):
        if part.startswith("layer_"):
            continue  # Skip layer directory
        if part in ["vectors", "steering_tests"]:
            break  # Stop at root dirs
        meaningful.append(part)
        if len(meaningful) >= 2:
            break

    if meaningful:
        return "_".join(reversed(meaningful))
    return "custom"


def generate_output_paths(
    model_name: str,
    layer: int,
    vector_dir: Optional[str],
    timestamp: Optional[str] = None,
) -> Tuple[Path, Path, Path]:
    """
    Generate output paths for results and plots.

    Returns:
        Tuple of (jsonl_path, kl_plot_path, entropy_plot_path)
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_short = get_model_short_name(model_name)
    vector_source = get_vector_source_name(vector_dir)

    # Create model subdirectory
    model_dir = RESULTS_DIR / model_short
    model_dir.mkdir(parents=True, exist_ok=True)

    # Base filename: kl_layer{N}_{vector_source}_{timestamp}
    base_name = f"kl_layer{layer}_{vector_source}_{timestamp}"

    jsonl_path = model_dir / f"{base_name}.jsonl"
    kl_plot_path = model_dir / f"{base_name}_kl.png"
    entropy_plot_path = model_dir / f"{base_name}_entropy.png"

    return jsonl_path, kl_plot_path, entropy_plot_path


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_kl_experiment(
    model_name: str,
    layer: int,
    vector_dir: Optional[str] = None,
    emotions: Optional[List[str]] = None,
    scales: Optional[List[float]] = None,
    num_samples: int = 100,
    num_random_vectors: int = 5,
    output_file: Optional[str] = None,
    tensor_parallel: int = 1,
    seed: int = 42,
    top_logprobs: int = 20,  # vLLM V1 max is 20
    layer_norm_pct: Optional[float] = None,
    skip_plots: bool = False,
):
    """
    Run KL divergence experiment comparing steered vs baseline distributions.

    Args:
        model_name: HuggingFace model name
        layer: Layer to apply steering
        vector_dir: Directory containing steering vectors
        emotions: List of emotions to test (None = all available)
        scales: List of scaling factors to test
        num_samples: Number of text samples to use
        num_random_vectors: Number of random baseline vectors
        output_file: Path to output JSONL file (None = auto-generate)
        tensor_parallel: Number of GPUs for tensor parallelism
        seed: Random seed
        top_logprobs: Number of top logprobs to request from vLLM
        layer_norm_pct: If set, scale vectors as percentage of layer norm
        skip_plots: If True, don't generate plots at the end

    Output files are saved to:
        steering_tests/vector_testing/results/{model_short_name}/
            kl_layer{N}_{vector_source}_{timestamp}.jsonl
            kl_layer{N}_{vector_source}_{timestamp}_kl.png
            kl_layer{N}_{vector_source}_{timestamp}_entropy.png
    """
    if scales is None:
        scales = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    # Generate output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Derive plot paths from output file
        base = output_path.with_suffix('')
        kl_plot_path = base.parent / f"{base.name}_kl.png"
        entropy_plot_path = base.parent / f"{base.name}_entropy.png"
    else:
        output_path, kl_plot_path, entropy_plot_path = generate_output_paths(
            model_name, layer, vector_dir, timestamp
        )

    logger.info(f"Output paths:")
    logger.info(f"  Results: {output_path}")
    logger.info(f"  KL plot: {kl_plot_path}")
    logger.info(f"  Entropy plot: {entropy_plot_path}")

    # Initialize vLLM
    logger.info(f"Loading model {model_name} with TP={tensor_parallel}...")
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        tensor_parallel_size=tensor_parallel,
        trust_remote_code=True,
    )

    # Get model hidden dim from config
    if model_name in MODEL_CONFIGS:
        hidden_dim = MODEL_CONFIGS[model_name]["hidden_dim"]
    else:
        # Fallback: try to get from model config
        try:
            hidden_dim = llm.llm_engine.model_config.hf_config.hidden_size
        except AttributeError:
            logger.warning(f"Unknown model {model_name}, defaulting to hidden_dim=4096")
            hidden_dim = 4096
    logger.info(f"Model hidden dim: {hidden_dim}")

    # Get layer norm for scaling
    model_key = resolve_model_key(model_name)
    try:
        layer_norm = get_layer_norm(model_key, layer)
        logger.info(f"Layer {layer} norm: {layer_norm:.2f}")
    except (KeyError, FileNotFoundError) as e:
        logger.warning(f"Could not get layer norm: {e}. Using 1.0")
        layer_norm = 1.0

    # Initialize steering
    steering = VLLMSteering(llm, layer=layer)

    # Load steering vectors
    vectors = {}
    if vector_dir:
        vector_path = Path(vector_dir)
        if vector_path.exists():
            # Load .npy files directly from the directory
            for npy_file in vector_path.glob("*.npy"):
                name = npy_file.stem
                vectors[name] = np.load(npy_file)
                logger.info(f"Loaded vector: {name} (shape={vectors[name].shape})")

            # Load .pkl files (dict format with emotion keys)
            for pkl_file in vector_path.glob("*.pkl"):
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    for emotion, vec in data.items():
                        vectors[emotion] = np.array(vec)
                        logger.info(f"Loaded vector: {emotion} (shape={vectors[emotion].shape})")

            # Try layers subdirectory with layer_XX.pkl format
            layers_dir = vector_path / "layers"
            if layers_dir.exists():
                layer_pkl = layers_dir / f"layer_{layer:02d}.pkl"
                if not layer_pkl.exists():
                    layer_pkl = layers_dir / f"layer_{layer}.pkl"
                if layer_pkl.exists():
                    with open(layer_pkl, 'rb') as f:
                        data = pickle.load(f)
                    if isinstance(data, dict):
                        for emotion, vec in data.items():
                            vectors[emotion] = np.array(vec)
                            logger.info(f"Loaded vector: {emotion} (shape={vectors[emotion].shape})")

            # Also try subdirectory structure with .npy files
            for subdir in vector_path.iterdir():
                if subdir.is_dir() and subdir.name != "layers":
                    layer_dir = subdir / f"layer_{layer}"
                    if layer_dir.exists():
                        for npy_file in layer_dir.glob("*.npy"):
                            name = f"{npy_file.stem}_{subdir.name}"
                            vectors[name] = np.load(npy_file)
                            logger.info(f"Loaded vector: {name}")

    # Filter to requested emotions
    if emotions:
        vectors = {k: v for k, v in vectors.items() if any(e in k for e in emotions)}

    # Add random vectors
    random_vectors = generate_random_vectors(hidden_dim, num_random_vectors, seed)
    vectors.update(random_vectors)

    logger.info(f"Total vectors to test: {len(vectors)}")
    for name in vectors:
        logger.info(f"  - {name}")

    # Load evaluation data
    samples = load_fineweb_samples(num_samples, seed=seed)

    # Sampling params to get logprobs
    sampling_params = SamplingParams(
        max_tokens=1,  # Just get next token distribution
        logprobs=top_logprobs,
        temperature=1.0,  # Use temperature=1 to get true distribution
    )

    # Run experiment
    results = []
    experiment_meta = {
        "model": model_name,
        "layer": layer,
        "vector_dir": str(vector_dir) if vector_dir else None,
        "num_samples": len(samples),
        "scales": scales,
        "vectors": list(vectors.keys()),
        "layer_norm": layer_norm,
        "layer_norm_pct": layer_norm_pct,
        "timestamp": timestamp,
        "output_file": str(output_path),
        "kl_plot": str(kl_plot_path),
        "entropy_plot": str(entropy_plot_path),
    }

    with open(output_path, 'w') as f:
        f.write(json.dumps({"meta": experiment_meta}) + '\n')

    # Get baseline logprobs (no steering)
    logger.info("Computing baseline logprobs...")
    steering.clear()
    baseline_outputs = llm.generate(samples, sampling_params)
    baseline_logprobs = []
    for output in baseline_outputs:
        if output.outputs[0].logprobs:
            # Extract logprobs dict for the first (only) token
            token_logprobs = output.outputs[0].logprobs[0]
            # Convert to {token_str: logprob}
            lp_dict = {}
            for token_id, logprob_obj in token_logprobs.items():
                # Handle different vLLM versions
                if hasattr(logprob_obj, 'decoded_token'):
                    lp_dict[logprob_obj.decoded_token] = logprob_obj.logprob
                else:
                    lp_dict[str(token_id)] = logprob_obj
            baseline_logprobs.append(lp_dict)
        else:
            baseline_logprobs.append({})

    # Compute baseline entropy
    baseline_entropies = [compute_entropy(lp) if lp else 0.0 for lp in baseline_logprobs]
    avg_baseline_entropy = np.mean(baseline_entropies)
    logger.info(f"Average baseline entropy: {avg_baseline_entropy:.4f}")

    # Test each vector at each scale
    for vector_name, vector in tqdm(vectors.items(), desc="Testing vectors"):
        # Register vector with steering
        steering.load_vector(vector_name, vector)

        for scale in scales:
            if scale == 0.0:
                # Already have baseline
                kl_values = [0.0] * len(samples)
                js_values = [0.0] * len(samples)
                entropy_values = baseline_entropies
            else:
                # Compute effective magnitude
                if layer_norm_pct is not None:
                    # Scale as percentage of layer norm
                    magnitude = (layer_norm_pct / 100.0) * layer_norm * scale
                else:
                    # Use raw scale
                    magnitude = scale

                # Apply steering
                steering.set_raw_vector(vector, scale=magnitude)

                # Generate with steering
                steered_outputs = llm.generate(samples, sampling_params)

                # Compute KL divergence for each sample
                kl_values = []
                js_values = []
                entropy_values = []

                for i, output in enumerate(steered_outputs):
                    if output.outputs[0].logprobs and baseline_logprobs[i]:
                        token_logprobs = output.outputs[0].logprobs[0]
                        steered_lp = {}
                        for token_id, logprob_obj in token_logprobs.items():
                            if hasattr(logprob_obj, 'decoded_token'):
                                steered_lp[logprob_obj.decoded_token] = logprob_obj.logprob
                            else:
                                steered_lp[str(token_id)] = logprob_obj

                        kl = compute_kl_divergence(baseline_logprobs[i], steered_lp)
                        js = compute_js_divergence(baseline_logprobs[i], steered_lp)
                        ent = compute_entropy(steered_lp)

                        kl_values.append(kl)
                        js_values.append(js)
                        entropy_values.append(ent)
                    else:
                        kl_values.append(0.0)
                        js_values.append(0.0)
                        entropy_values.append(0.0)

                # Clear steering after each condition
                steering.clear()

            # Aggregate results
            result = {
                "vector": vector_name,
                "scale": scale,
                "kl_mean": float(np.mean(kl_values)),
                "kl_std": float(np.std(kl_values)),
                "kl_median": float(np.median(kl_values)),
                "js_mean": float(np.mean(js_values)),
                "js_std": float(np.std(js_values)),
                "entropy_mean": float(np.mean(entropy_values)),
                "entropy_baseline": avg_baseline_entropy,
                "entropy_change": float(np.mean(entropy_values)) - avg_baseline_entropy,
                "is_random": vector_name.startswith("random_"),
            }
            results.append(result)

            # Write incrementally
            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')

            logger.info(
                f"{vector_name} @ scale={scale:.1f}: "
                f"KL={result['kl_mean']:.4f} (±{result['kl_std']:.4f}), "
                f"JS={result['js_mean']:.4f}"
            )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Group by vector type
    random_results = [r for r in results if r['is_random']]
    emotion_results = [r for r in results if not r['is_random']]

    if emotion_results:
        print("\nEmotion vectors (mean KL at each scale):")
        for scale in scales:
            scale_results = [r for r in emotion_results if r['scale'] == scale]
            if scale_results:
                mean_kl = np.mean([r['kl_mean'] for r in scale_results])
                print(f"  scale={scale:.1f}: KL={mean_kl:.4f}")

    if random_results:
        print("\nRandom vectors (mean KL at each scale):")
        for scale in scales:
            scale_results = [r for r in random_results if r['scale'] == scale]
            if scale_results:
                mean_kl = np.mean([r['kl_mean'] for r in scale_results])
                print(f"  scale={scale:.1f}: KL={mean_kl:.4f}")

    print(f"\nResults saved to: {output_path}")

    # Generate plots
    if not skip_plots:
        logger.info("Generating plots...")
        try:
            plot_kl_results(str(output_path), str(kl_plot_path))
            logger.info(f"KL divergence plot saved to: {kl_plot_path}")
        except Exception as e:
            logger.warning(f"Failed to generate KL plot: {e}")

        try:
            plot_entropy_change(str(output_path), str(entropy_plot_path))
            logger.info(f"Entropy change plot saved to: {entropy_plot_path}")
        except Exception as e:
            logger.warning(f"Failed to generate entropy plot: {e}")

        print(f"\nPlots saved to:")
        print(f"  KL divergence: {kl_plot_path}")
        print(f"  Entropy change: {entropy_plot_path}")

    return results, output_path, kl_plot_path, entropy_plot_path


def main():
    parser = argparse.ArgumentParser(
        description="Steering vector KL divergence experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output structure:
    steering_tests/vector_testing/results/{model}/
        kl_layer{N}_{vector_source}_{timestamp}.jsonl
        kl_layer{N}_{vector_source}_{timestamp}_kl.png
        kl_layer{N}_{vector_source}_{timestamp}_entropy.png

Examples:
    # Basic usage (auto-generates output paths)
    python -m steering_tests.vector_testing.kl_div \\
        --model google/gemma-3-27b-it \\
        --layer 30 \\
        --vector-dir steering_tests/vectors/gemma3_27b/base_emotion_vs_others/last_token/layer_30

    # Test specific emotions with custom scales
    python -m steering_tests.vector_testing.kl_div \\
        --model google/gemma-3-27b-it \\
        --layer 30 \\
        --vector-dir steering_tests/vectors/gemma3_27b/base_emotion_vs_others/last_token/layer_30 \\
        --emotions anger fear happiness \\
        --scales 0 1 2 3 4 5

    # Multi-GPU for large models
    python -m steering_tests.vector_testing.kl_div \\
        --model Qwen/Qwen3-235B-A22B \\
        --layer 50 \\
        --tensor-parallel 4
        """
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        default="google/gemma-3-27b-it",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--layer", "-l",
        type=int,
        default=30,
        help="Layer to apply steering"
    )
    parser.add_argument(
        "--vector-dir", "-v",
        type=str,
        default=None,
        help="Directory containing steering vectors (.npy files)"
    )
    parser.add_argument(
        "--emotions", "-e",
        nargs="+",
        default=None,
        help="Specific emotions to test (default: all in vector dir)"
    )
    parser.add_argument(
        "--scales", "-s",
        nargs="+",
        type=float,
        default=None,
        help="Scaling factors to test (default: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)"
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=100,
        help="Number of FineWeb samples for evaluation"
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
        help="Output JSONL file (default: auto-generate based on model/layer/vectors)"
    )
    parser.add_argument(
        "--tensor-parallel", "-tp",
        type=int,
        default=1,
        help="Tensor parallel size (number of GPUs)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=20,  # vLLM V1 max is 20
        help="Number of top logprobs to request (max 20 for vLLM V1)"
    )
    parser.add_argument(
        "--layer-norm-pct",
        type=float,
        default=None,
        help="If set, scale vectors as percentage of layer norm"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip automatic plot generation"
    )

    args = parser.parse_args()

    results, output_path, kl_plot, entropy_plot = run_kl_experiment(
        model_name=args.model,
        layer=args.layer,
        vector_dir=args.vector_dir,
        emotions=args.emotions,
        scales=args.scales,
        num_samples=args.num_samples,
        num_random_vectors=args.num_random,
        output_file=args.output,
        tensor_parallel=args.tensor_parallel,
        seed=args.seed,
        top_logprobs=args.top_logprobs,
        layer_norm_pct=args.layer_norm_pct,
        skip_plots=args.skip_plots,
    )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Results: {output_path}")
    if not args.skip_plots:
        print(f"KL plot: {kl_plot}")
        print(f"Entropy plot: {entropy_plot}")


if __name__ == "__main__":
    main()
