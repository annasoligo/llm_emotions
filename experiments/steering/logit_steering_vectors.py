"""
Compute steering vectors from emotion token embeddings using least squares.

Method (Inverse Logit Steering):
1. Define target distribution D:
   - D[i] = +1/n for all n tokens in target emotion
   - D[i] = -1/m for all m other tokens

2. Model computes logits as: logits = W_U @ v
   where W_U is unembedding matrix (vocab_size, hidden_size)

3. Find v such that W_U @ v ≈ D using least squares:
   v = lstsq(W_U, D)

This gives a steering vector that increases logits for emotion tokens
and decreases logits for all other tokens.

Usage:
    python -m experiments.steering.logit_steering_vectors --layer 30
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from emotion_logit_lens.emotion_tokens import EmotionTokenManager
from .config import MODEL_NAME, VECTOR_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_unembedding_matrix(model) -> torch.Tensor:
    """
    Get the unembedding matrix W_U from the model.

    The lm_head maps hidden states to logits: logits = W_U @ hidden
    W_U has shape (vocab_size, hidden_size).

    Returns:
        Tensor of shape (vocab_size, hidden_size)
    """
    if hasattr(model, 'lm_head'):
        return model.lm_head.weight.detach().float()
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        # Gemma 3 multimodal structure
        return model.language_model.lm_head.weight.detach().float()
    else:
        raise ValueError("Could not find lm_head in model")


def compute_lstsq_steering_vector(
    W_U: torch.Tensor,
    target_token_ids: List[int],
    vocab_size: int,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute steering vector using least squares to match target distribution.

    Target distribution D:
    - D[i] = +1/n for tokens in target set (n tokens)
    - D[i] = -1/m for all other tokens (m tokens)

    We solve: W_U @ v ≈ D  =>  v = lstsq(W_U, D)

    Args:
        W_U: Unembedding matrix (vocab_size, hidden_size)
        target_token_ids: Token IDs for target emotion
        vocab_size: Total vocabulary size
        device: Device to compute on

    Returns:
        Steering vector of shape (hidden_size,)
    """
    n = len(target_token_ids)
    m = vocab_size - n

    # Create target distribution
    D = torch.full((vocab_size,), -1.0 / m, device=device)
    D[target_token_ids] = 1.0 / n

    # Move W_U to device
    W_U = W_U.to(device)

    # Solve least squares: W_U @ v = D
    # W_U is (vocab_size, hidden_size), D is (vocab_size,)
    # We want v of shape (hidden_size,)
    result = torch.linalg.lstsq(W_U, D.unsqueeze(1))
    v = result.solution.squeeze()

    return v.cpu().numpy()


def compute_emotion_steering_vectors_lstsq(
    model,
    emotion_token_ids: Dict[str, List[int]],
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """
    Compute steering vectors for each emotion using least squares method.

    Args:
        model: HuggingFace model
        emotion_token_ids: Dict mapping emotion -> list of token IDs
        device: Device for computation

    Returns:
        Dict mapping emotion -> steering vector (hidden_size,)
    """
    logger.info("Getting unembedding matrix...")
    W_U = get_unembedding_matrix(model)
    vocab_size, hidden_size = W_U.shape
    logger.info(f"  W_U shape: {W_U.shape}")

    emotion_vectors = {}

    for emotion, token_ids in emotion_token_ids.items():
        if not token_ids:
            logger.warning(f"No tokens for emotion: {emotion}")
            continue

        logger.info(f"Computing vector for {emotion} ({len(token_ids)} tokens)...")

        vector = compute_lstsq_steering_vector(
            W_U, token_ids, vocab_size, device=device
        )

        norm = np.linalg.norm(vector)
        emotion_vectors[emotion] = vector

        logger.info(f"  {emotion}: norm={norm:.4f}")

    return emotion_vectors


def save_steering_vectors(
    vectors: Dict[str, np.ndarray],
    output_dir: Path,
    layer: int,
    method: str,
    normalize: bool = True,
):
    """
    Save steering vectors in the standard format.

    Args:
        vectors: Dict mapping emotion -> vector
        output_dir: Output directory
        layer: Layer number (for filename)
        method: Method used (for metadata)
        normalize: Whether to normalize vectors before saving
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for emotion, vector in vectors.items():
        if normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm

        # Save in same format as existing vectors
        filename = f"{emotion}_logit_layer{layer}.npz"
        output_path = output_dir / filename

        np.savez(output_path, vector=vector)
        logger.info(f"Saved: {output_path}")

    # Save metadata
    meta_path = output_dir / f"logit_vectors_layer{layer}_metadata.json"
    metadata = {
        "method": method,
        "layer": layer,
        "emotions": list(vectors.keys()),
        "normalized": normalize,
        "hidden_size": vectors[list(vectors.keys())[0]].shape[0],
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Compute logit-based steering vectors using least squares")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--layer", type=int, default=30, help="Layer number for filename")
    parser.add_argument("--output-dir", type=Path, default=VECTOR_DIR, help="Output directory")
    parser.add_argument("--no-normalize", action="store_true", help="Don't normalize vectors")
    parser.add_argument("--device", type=str, default="cuda", help="Device for computation")
    args = parser.parse_args()

    logger.info(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    logger.info("Loading emotion token IDs...")
    token_manager = EmotionTokenManager(model_name=args.model.replace("/", "_").replace("-", "_"))
    emotion_token_ids = token_manager.load_emotion_token_ids()

    logger.info(f"\nEmotions available: {list(emotion_token_ids.keys())}")
    for emotion, tokens in emotion_token_ids.items():
        logger.info(f"  {emotion}: {len(tokens)} tokens")

    logger.info(f"\nComputing least-squares steering vectors...")
    vectors = compute_emotion_steering_vectors_lstsq(
        model,
        emotion_token_ids,
        device=args.device,
    )

    logger.info(f"\nSaving to {args.output_dir}...")
    save_steering_vectors(
        vectors,
        args.output_dir,
        layer=args.layer,
        method="lstsq",
        normalize=not args.no_normalize,
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
