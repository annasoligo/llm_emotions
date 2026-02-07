#!/usr/bin/env python3
"""
Extract "expression suppression" vectors from Human-Like-DPO-Dataset.

The dataset contains:
- chosen: emotional, expressive AI responses
- rejected: neutral, professional AI responses

We extract: rejected - chosen = suppression direction (neutral - emotional)

Vectors are saved as unit vectors with original norms in metadata.
"""

import argparse
import json
import logging
import os
import pickle
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

# Model configs
MODEL_CONFIGS = {
    "gemma3_27b": {
        "model_id": "google/gemma-3-27b-it",
        "num_layers": 62,
        "hidden_dim": 4608,
    },
    "gemma12b": {
        "model_id": "google/gemma-3-12b-it",
        "num_layers": 48,
        "hidden_dim": 3840,
    },
    "qwen235b": {
        "model_id": "Qwen/Qwen3-235B-A22B",
        "num_layers": 94,
        "hidden_dim": 4096,
    },
    "qwen32b": {
        "model_id": "Qwen/Qwen2.5-32B-Instruct",
        "num_layers": 64,
        "hidden_dim": 5120,
    },
    "qwen14b": {
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "num_layers": 48,
        "hidden_dim": 5120,
    },
}


def load_and_sample_dataset(n_samples: int = 100, seed: int = 42) -> List[dict]:
    """Load Human-Like-DPO-Dataset and sample n examples."""
    logger.info(f"Loading HumanLLMs/Human-Like-DPO-Dataset...")
    dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset", split="train")

    logger.info(f"Dataset size: {len(dataset)}")

    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    samples = [dataset[i] for i in indices]

    logger.info(f"Sampled {len(samples)} examples")
    return samples


def get_response_token_mask(
    tokenizer,
    full_ids: torch.Tensor,
    prompt: str,
) -> torch.Tensor:
    """Get mask for response tokens (everything after the prompt)."""
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    # Create mask: 0 for prompt, 1 for response
    mask = torch.zeros(full_ids.shape[-1], dtype=torch.bool)
    mask[prompt_len:] = True

    return mask


def extract_activations_for_text(
    model,
    tokenizer,
    text: str,
    prompt: str,
    device: str = "cuda",
) -> Dict[int, np.ndarray]:
    """
    Extract mean activation across response tokens for each layer.

    Returns:
        Dict mapping layer index to mean activation vector
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get response token mask
    response_mask = get_response_token_mask(tokenizer, inputs["input_ids"], prompt)
    response_mask = response_mask.to(device)

    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Extract mean activation per layer (only response tokens)
    layer_activations = {}
    for layer_idx, hidden_state in enumerate(outputs.hidden_states):
        # hidden_state shape: (batch, seq_len, hidden_dim)
        hidden = hidden_state[0]  # Remove batch dim

        # Apply response mask
        response_hidden = hidden[response_mask]

        if response_hidden.shape[0] > 0:
            mean_act = response_hidden.mean(dim=0).float().cpu().numpy()
        else:
            # Fallback to last token if mask is empty
            mean_act = hidden[-1].float().cpu().numpy()

        layer_activations[layer_idx] = mean_act

    return layer_activations


def extract_suppression_vectors(
    model_key: str,
    samples: List[dict],
    output_dir: Path,
    device: str = "cuda",
    tensor_parallel: int = 1,
):
    """Extract suppression vectors for a model."""
    from transformers import AutoModelForCausalLM

    config = MODEL_CONFIGS[model_key]
    model_id = config["model_id"]
    num_layers = config["num_layers"]

    logger.info(f"Loading model {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Load model
    if tensor_parallel > 1:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)

    model.eval()

    # Collect activations
    emotional_activations = {i: [] for i in range(num_layers + 1)}  # +1 for embeddings
    neutral_activations = {i: [] for i in range(num_layers + 1)}

    for sample in tqdm(samples, desc=f"Extracting {model_key}"):
        prompt = sample["prompt"]
        chosen = sample["chosen"]  # emotional
        rejected = sample["rejected"]  # neutral

        # Build full text (prompt + response)
        emotional_text = prompt + chosen
        neutral_text = prompt + rejected

        try:
            # Extract activations
            emo_acts = extract_activations_for_text(model, tokenizer, emotional_text, prompt, device)
            neu_acts = extract_activations_for_text(model, tokenizer, neutral_text, prompt, device)

            for layer_idx in emo_acts.keys():
                emotional_activations[layer_idx].append(emo_acts[layer_idx])
                neutral_activations[layer_idx].append(neu_acts[layer_idx])

        except Exception as e:
            logger.warning(f"Error processing sample: {e}")
            continue

    # Compute mean difference vectors
    logger.info("Computing suppression vectors...")

    vectors = {}
    metadata = {
        "model_key": model_key,
        "model_id": model_id,
        "n_samples": len(samples),
        "extraction_date": datetime.now().isoformat(),
        "direction": "neutral - emotional (rejected - chosen)",
        "layers": {},
    }

    for layer_idx in tqdm(range(num_layers + 1), desc="Computing vectors"):
        if not emotional_activations[layer_idx]:
            continue

        # Stack and compute means
        emo_stack = np.stack(emotional_activations[layer_idx])
        neu_stack = np.stack(neutral_activations[layer_idx])

        emo_mean = emo_stack.mean(axis=0)
        neu_mean = neu_stack.mean(axis=0)

        # Suppression direction: neutral - emotional
        diff = neu_mean - emo_mean

        # Compute norm before normalizing
        norm = float(np.linalg.norm(diff))

        # Normalize to unit vector
        if norm > 0:
            unit_vec = diff / norm
        else:
            unit_vec = diff

        vectors[layer_idx] = unit_vec.astype(np.float32)

        metadata["layers"][layer_idx] = {
            "norm": norm,
            "emotional_mean_norm": float(np.linalg.norm(emo_mean)),
            "neutral_mean_norm": float(np.linalg.norm(neu_mean)),
        }

    # Save vectors
    model_output_dir = output_dir / model_key
    model_output_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = model_output_dir / "suppression_vectors.pkl"
    with open(vectors_path, "wb") as f:
        pickle.dump(vectors, f)

    metadata_path = model_output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved vectors to {vectors_path}")
    logger.info(f"Saved metadata to {metadata_path}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return vectors, metadata


def main():
    parser = argparse.ArgumentParser(description="Extract suppression vectors")
    parser.add_argument("--model", "-m", type=str, required=True,
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Model to extract from")
    parser.add_argument("--n_samples", "-n", type=int, default=100,
                       help="Number of samples to use")
    parser.add_argument("--seed", "-s", type=int, default=42,
                       help="Random seed for sampling")
    parser.add_argument("--output_dir", "-o", type=str,
                       default="steering_tests/suppression_experiments/vectors",
                       help="Output directory")
    parser.add_argument("--tensor_parallel", "-tp", type=int, default=1,
                       help="Tensor parallel size (for large models)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and sample dataset
    samples = load_and_sample_dataset(args.n_samples, args.seed)

    # Extract vectors
    vectors, metadata = extract_suppression_vectors(
        args.model,
        samples,
        output_dir,
        tensor_parallel=args.tensor_parallel,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE: {args.model}")
    print(f"{'='*60}")
    print(f"Samples: {len(samples)}")
    print(f"Layers: {len(vectors)}")

    # Show norm distribution
    norms = [metadata["layers"][l]["norm"] for l in sorted(metadata["layers"].keys())]
    print(f"Vector norms: min={min(norms):.3f}, max={max(norms):.3f}, mean={np.mean(norms):.3f}")


if __name__ == "__main__":
    main()
