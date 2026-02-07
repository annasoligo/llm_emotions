#!/usr/bin/env python3
"""
KL divergence experiment using HuggingFace transformers for full logit access.
This avoids vLLM's logprobs limitation by computing KL directly from logits.
"""

import os
import json
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_fineweb_samples(
    num_samples: int = 100,
    min_length: int = 100,
    max_length: int = 512,
    seed: int = 42,
) -> List[str]:
    """Load random samples from FineWeb dataset."""
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
    """Generate random unit vectors for baseline comparison."""
    np.random.seed(seed)
    vectors = {}
    for i in range(num_vectors):
        vec = np.random.randn(hidden_dim).astype(np.float32)
        vec = vec / np.linalg.norm(vec)
        vectors[f"random_{i}"] = vec
    return vectors


def load_vectors_from_dir(vector_dir: str, layer: int) -> Dict[str, np.ndarray]:
    """Load steering vectors from directory."""
    vector_dir = Path(vector_dir)
    vectors = {}

    # Check for layer-specific file in various subdirectories
    possible_paths = [
        vector_dir / "layers" / f"layer_{layer:02d}.pkl",
        vector_dir / "last_token" / f"layer_{layer:02d}.pkl",
        vector_dir / "special_mean" / f"layer_{layer:02d}.pkl",
    ]

    layer_file = None
    for path in possible_paths:
        if path.exists():
            layer_file = path
            break

    if layer_file and layer_file.exists():
        with open(layer_file, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                for name, vec in data.items():
                    if isinstance(vec, np.ndarray):
                        vectors[name] = vec.astype(np.float32)
        logger.info(f"Loaded {len(vectors)} vectors from {layer_file}")

    return vectors


def compute_kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """Compute KL(P || Q) from logits tensors using log-space computation.

    Args:
        p_logits: Baseline logits, shape (seq_len, vocab_size) or (vocab_size,)
        q_logits: Steered logits, same shape as p_logits

    Returns:
        Mean KL divergence across all positions
    """
    # Handle both single position and sequence of positions
    if p_logits.dim() == 1:
        p_logits = p_logits.unsqueeze(0)
        q_logits = q_logits.unsqueeze(0)

    # Use log_softmax for numerical stability
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    p = torch.softmax(p_logits, dim=-1)

    # KL(P || Q) = sum(P * (log P - log Q))
    kl_per_position = (p * (log_p - log_q)).sum(dim=-1)
    return kl_per_position.mean().item()


def compute_js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """Compute Jensen-Shannon divergence from logits.

    Args:
        p_logits: Baseline logits, shape (seq_len, vocab_size) or (vocab_size,)
        q_logits: Steered logits, same shape as p_logits

    Returns:
        Mean JS divergence across all positions
    """
    if p_logits.dim() == 1:
        p_logits = p_logits.unsqueeze(0)
        q_logits = q_logits.unsqueeze(0)

    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)

    # M = 0.5 * (P + Q) - the mixture distribution
    m = 0.5 * (p + q)

    # Use log for numerical stability, clamp to avoid log(0)
    log_p = torch.log(p.clamp(min=1e-10))
    log_q = torch.log(q.clamp(min=1e-10))
    log_m = torch.log(m.clamp(min=1e-10))

    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)

    js_per_position = 0.5 * (kl_pm + kl_qm)
    return js_per_position.mean().item()


def compute_entropy(logits: torch.Tensor) -> float:
    """Compute entropy of distribution from logits.

    Args:
        logits: Logits tensor, shape (seq_len, vocab_size) or (vocab_size,)

    Returns:
        Mean entropy across all positions
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    # Use log_softmax for numerical stability
    log_p = torch.log_softmax(logits, dim=-1)
    p = torch.softmax(logits, dim=-1)

    # H(P) = -sum(P * log P)
    entropy_per_position = -(p * log_p).sum(dim=-1)
    return entropy_per_position.mean().item()


class SteeringHook:
    """Hook to add steering vector to residual stream."""

    def __init__(self, steering_vector: torch.Tensor, scale: float = 1.0):
        self.steering_vector = steering_vector
        self.scale = scale
        self.handle = None

    def __call__(self, module, input, output):
        # Handle different output formats (tuple vs tensor)
        is_tuple = isinstance(output, tuple)
        hidden_states = output[0] if is_tuple else output

        # Add steering vector to all positions
        if hidden_states.dim() == 3:
            # (batch, seq, hidden)
            steering = self.steering_vector.unsqueeze(0).unsqueeze(0)
            steering = steering.expand(hidden_states.shape[0], hidden_states.shape[1], -1)
        elif hidden_states.dim() == 2:
            # (seq, hidden)
            steering = self.steering_vector.unsqueeze(0)
            steering = steering.expand(hidden_states.shape[0], -1)
        else:
            raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")

        modified = hidden_states + self.scale * steering

        if is_tuple:
            return (modified,) + output[1:]
        else:
            return modified

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle:
            self.handle.remove()
            self.handle = None


def run_kl_experiment_hf(
    model_name: str,
    layer: int,
    vector_dir: Optional[str] = None,
    emotions: Optional[List[str]] = None,
    scales: Optional[List[float]] = None,
    num_samples: int = 100,
    num_random_vectors: int = 5,
    output_file: Optional[str] = None,
    seed: int = 42,
    layer_norm_pct: Optional[float] = None,
    skip_plots: bool = False,
    max_length: int = 256,
):
    """
    Run KL divergence experiment using HuggingFace transformers.
    Computes full-vocabulary KL divergence from logits.
    """
    if scales is None:
        scales = [0.0, 1.0, 2.0, 5.0, 10.0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup output paths - save into KL subfolder
    model_short = model_name.split("/")[-1].lower().replace("-", "_")
    results_dir = Path("steering_tests/vector_testing/results") / model_short / "kl"
    results_dir.mkdir(parents=True, exist_ok=True)

    vector_source = Path(vector_dir).name if vector_dir else "random"
    output_path = results_dir / f"kl_hf_layer{layer}_{model_short}_{vector_source}_{timestamp}.jsonl"

    logger.info(f"Output: {output_path}")

    # Load model
    logger.info(f"Loading model {model_name}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    # Handle different config structures (e.g., multimodal models like Gemma 3)
    if hasattr(model.config, 'hidden_size'):
        hidden_dim = model.config.hidden_size
    elif hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'hidden_size'):
        hidden_dim = model.config.text_config.hidden_size
    else:
        raise ValueError("Could not find hidden_size in model config")
    logger.info(f"Hidden dim: {hidden_dim}")

    # Get layer module for hooks - handle various architectures
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        # Gemma 3 multimodal - layers directly on language_model
        layers = model.language_model.layers
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        # Other multimodal with nested model
        layers = model.language_model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError(f"Unknown model architecture: {type(model)}")

    target_layer = layers[layer]
    logger.info(f"Target layer: {layer}")

    # Compute layer norm from a few samples
    logger.info("Computing layer norm...")
    test_samples = load_fineweb_samples(10, seed=seed)
    norms = []

    def norm_hook(module, input, output):
        # Handle different output formats (tuple vs tensor)
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # Handle different tensor shapes
        if hidden.dim() == 3:
            # Standard (batch, seq, hidden)
            last_hidden = hidden[:, -1, :]
        elif hidden.dim() == 2:
            # (seq, hidden) - no batch dim
            last_hidden = hidden[-1, :]
        else:
            logger.warning(f"Unexpected hidden shape: {hidden.shape}")
            last_hidden = hidden.flatten()[-hidden_dim:]
        norms.append(torch.norm(last_hidden.float(), dim=-1).mean().item())
        return output

    handle = target_layer.register_forward_hook(norm_hook)
    with torch.no_grad():
        for text in test_samples:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            model(**inputs)
    handle.remove()

    layer_norm = np.mean(norms)
    logger.info(f"Layer {layer} norm: {layer_norm:.2f}")

    # Load vectors
    vectors = {}
    if vector_dir:
        vectors = load_vectors_from_dir(vector_dir, layer)
        if emotions:
            vectors = {k: v for k, v in vectors.items() if k in emotions}

    # Add random vectors
    random_vecs = generate_random_vectors(hidden_dim, num_random_vectors, seed)
    vectors.update(random_vecs)

    logger.info(f"Testing {len(vectors)} vectors: {list(vectors.keys())}")

    # Load evaluation samples
    samples = load_fineweb_samples(num_samples, seed=seed)

    # Tokenize samples
    logger.info("Tokenizing samples...")
    tokenized = []
    for text in samples:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        tokenized.append({k: v.to(device) for k, v in inputs.items()})

    # Get baseline logits for all positions
    logger.info("Computing baseline logits (all positions)...")
    baseline_logits = []
    baseline_entropies = []

    with torch.no_grad():
        for inputs in tqdm(tokenized, desc="Baseline"):
            outputs = model(**inputs)
            # Get logits for all positions: shape (seq_len, vocab_size)
            all_logits = outputs.logits[0, :, :].float()
            baseline_logits.append(all_logits)
            baseline_entropies.append(compute_entropy(all_logits))

    avg_baseline_entropy = np.mean(baseline_entropies)
    logger.info(f"Average baseline entropy: {avg_baseline_entropy:.4f}")

    # Write metadata
    meta = {
        "model": model_name,
        "layer": layer,
        "vector_dir": str(vector_dir) if vector_dir else None,
        "num_samples": len(samples),
        "scales": scales,
        "vectors": list(vectors.keys()),
        "layer_norm": layer_norm,
        "layer_norm_pct": layer_norm_pct,
        "timestamp": timestamp,
        "method": "huggingface_full_logits",
    }

    results = []
    with open(output_path, 'w') as f:
        f.write(json.dumps({"meta": meta}) + '\n')

    # Test each vector
    for vec_name, vec in tqdm(vectors.items(), desc="Testing vectors"):
        vec_tensor = torch.tensor(vec, dtype=torch.bfloat16, device=device)
        is_random = vec_name.startswith("random_")

        for scale in scales:
            if scale == 0:
                # No steering - use baseline
                kl_values = [0.0] * len(samples)
                js_values = [0.0] * len(samples)
                entropy_values = baseline_entropies
            else:
                # Compute effective magnitude
                if layer_norm_pct is not None:
                    magnitude = (layer_norm_pct / 100.0) * layer_norm * scale
                else:
                    magnitude = scale

                # Setup steering hook
                hook = SteeringHook(vec_tensor, scale=magnitude)
                hook.register(target_layer)

                kl_values = []
                js_values = []
                entropy_values = []

                with torch.no_grad():
                    for i, inputs in enumerate(tokenized):
                        outputs = model(**inputs)
                        # Get logits for all positions
                        steered_logits = outputs.logits[0, :, :].float()

                        kl = compute_kl_divergence(baseline_logits[i], steered_logits)
                        js = compute_js_divergence(baseline_logits[i], steered_logits)
                        ent = compute_entropy(steered_logits)

                        kl_values.append(kl)
                        js_values.append(js)
                        entropy_values.append(ent)

                hook.remove()

            result = {
                "vector": vec_name,
                "scale": scale,
                "kl_mean": np.mean(kl_values),
                "kl_std": np.std(kl_values),
                "kl_median": np.median(kl_values),
                "js_mean": np.mean(js_values),
                "js_std": np.std(js_values),
                "entropy_mean": np.mean(entropy_values),
                "entropy_baseline": avg_baseline_entropy,
                "entropy_change": np.mean(entropy_values) - avg_baseline_entropy,
                "is_random": is_random,
            }
            results.append(result)

            with open(output_path, 'a') as f:
                f.write(json.dumps(result) + '\n')

            logger.info(f"{vec_name} @ scale={scale}: KL={result['kl_mean']:.4f} (±{result['kl_std']:.4f}), JS={result['js_mean']:.4f}")

    logger.info(f"Results saved to {output_path}")
    return results, output_path


def main():
    parser = argparse.ArgumentParser(description="KL divergence experiment with HuggingFace (full logits)")
    parser.add_argument("--model", type=str, default="google/gemma-3-27b-it")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--vector-dir", type=str, default=None)
    parser.add_argument("--emotions", nargs="+", default=None)
    parser.add_argument("--scales", nargs="+", type=float, default=[1, 5, 10, 20, 50, 100, 150])
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-random", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-norm-pct", type=float, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--max-length", type=int, default=256)

    args = parser.parse_args()

    run_kl_experiment_hf(
        model_name=args.model,
        layer=args.layer,
        vector_dir=args.vector_dir,
        emotions=args.emotions,
        scales=args.scales,
        num_samples=args.num_samples,
        num_random_vectors=args.num_random,
        output_file=args.output,
        seed=args.seed,
        layer_norm_pct=args.layer_norm_pct,
        skip_plots=args.skip_plots,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
