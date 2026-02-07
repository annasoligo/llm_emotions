#!/usr/bin/env python3
"""
Extract "internal suppression" vectors from contrastive pairs.

Pairs: "I feel X and will express it" vs "I feel X but won't show it"
Both acknowledge the emotion, differ only in expression.

Direction: expressed - internal
Adding the vector should push toward suppression (hiding emotion in output).
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent / "data" / "internal_vs_expressed_pairs.json"


def get_last_token_activation(
    model,
    tokenizer,
    text: str,
    layers: List[int],
    device: str = "cuda",
) -> Dict[int, np.ndarray]:
    """Get activation at last token for specified layers."""
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    activations = {}
    for layer_idx in layers:
        # +1 because hidden_states[0] is embeddings
        hidden = outputs.hidden_states[layer_idx + 1]
        # Last token - convert to float32 for numpy compatibility
        last_token_act = hidden[0, -1, :].float().cpu().numpy()
        activations[layer_idx] = last_token_act

    return activations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       help="HuggingFace model ID")
    parser.add_argument("--layers", type=str, required=True,
                       help="Layer range, e.g., '55-65' or '88-92'")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--tp", type=int, default=1,
                       help="Tensor parallel size")
    args = parser.parse_args()

    # Parse layers
    if "-" in args.layers:
        start, end = map(int, args.layers.split("-"))
        layers = list(range(start, end + 1))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    logger.info(f"Loading model: {args.model}")
    logger.info(f"Layers: {layers}")

    # Load data
    with open(DATA_PATH) as f:
        data = json.load(f)
    pairs = data["pairs"]
    logger.info(f"Loaded {len(pairs)} contrastive pairs")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()

    # Get hidden dim
    hidden_dim = model.config.hidden_size
    logger.info(f"Hidden dim: {hidden_dim}")

    # Collect activations
    expressed_acts = {layer: [] for layer in layers}
    internal_acts = {layer: [] for layer in layers}

    for pair in tqdm(pairs, desc="Processing pairs"):
        # Get activations for both
        expr_act = get_last_token_activation(model, tokenizer, pair["expressed"], layers)
        int_act = get_last_token_activation(model, tokenizer, pair["internal"], layers)

        for layer in layers:
            expressed_acts[layer].append(expr_act[layer])
            internal_acts[layer].append(int_act[layer])

    # Compute difference vectors: expressed - internal
    # So ADDING pushes toward internal (suppression)
    suppression_vectors = {}

    for layer in layers:
        expr_mean = np.mean(expressed_acts[layer], axis=0)
        int_mean = np.mean(internal_acts[layer], axis=0)

        # Direction: expressed - internal
        # NEGATE so adding pushes toward suppression
        diff = int_mean - expr_mean  # internal - expressed = suppression direction

        norm = np.linalg.norm(diff)
        logger.info(f"Layer {layer}: norm = {norm:.2f}")

        suppression_vectors[layer] = diff.astype(np.float32)

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "suppression_vectors.pkl", "wb") as f:
        pickle.dump(suppression_vectors, f)

    # Get model short name
    model_short = args.model.split("/")[-1].lower().replace("-", "_")

    metadata = {
        "model": args.model,
        "layers": layers,
        "hidden_dim": hidden_dim,
        "num_pairs": len(pairs),
        "description": "Internal suppression vectors (internal - expressed). ADD to suppress emotional expression while maintaining internal state.",
        "usage": "ADD to suppress emotional expression (already oriented for suppression)",
        "data_file": str(DATA_PATH),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
