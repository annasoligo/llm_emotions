#!/usr/bin/env python3
"""
Analyze text mean-diff steering vectors using Gemma Scope SAE features.

Computes cosine similarity between steering vectors and SAE decoder columns
to identify which SAE features align with each emotion direction.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download


def load_steering_vectors(vectors_path: str) -> dict:
    """Load steering vectors from npz file."""
    data = np.load(vectors_path)
    return {
        'vectors': data['vectors'],  # (n_emotions, d_model)
        'emotions': list(data['emotions']),
        'layer': int(data['layer']),
        'stds': data.get('stds', None),
    }


def load_sae_decoder(
    model_size: str = "27b",
    layer: int = 20,
    width: str = "16k",
    l0: str = "small",
    use_all_layers: bool = True,
) -> tuple:
    """Load SAE decoder weights from HuggingFace."""
    if use_all_layers:
        repo_id = f"google/gemma-scope-2-{model_size}-it"
        sae_path = f"resid_post_all/layer_{layer}_width_{width}_l0_{l0}"
    else:
        repo_id = f"google/gemma-scope-2-{model_size}-it"
        sae_path = f"resid_post/layer_{layer}_width_{width}_l0_{l0}"

    print(f"Loading SAE from {repo_id}/{sae_path}/params.safetensors")

    params_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{sae_path}/params.safetensors",
    )

    from safetensors import safe_open
    with safe_open(params_path, framework="pt", device="cpu") as f:
        W_dec = f.get_tensor("w_dec")  # (d_sae, d_model)

    print(f"  SAE dimensions: d_model={W_dec.shape[1]}, d_sae={W_dec.shape[0]}")
    return W_dec


def load_sae_example_data(
    model_size: str = "27b",
    layer: int = 20,
    width: str = "16k",
    l0: str = "small",
    use_all_layers: bool = True,
) -> dict:
    """Load SAE example data for feature interpretation."""
    if use_all_layers:
        repo_id = f"google/gemma-scope-2-{model_size}-it"
        sae_path = f"resid_post_all/layer_{layer}_width_{width}_l0_{l0}"
    else:
        repo_id = f"google/gemma-scope-2-{model_size}-it"
        sae_path = f"resid_post/layer_{layer}_width_{width}_l0_{l0}"

    print(f"Loading example data from {repo_id}/{sae_path}/examples.safetensors")

    examples_path = hf_hub_download(
        repo_id=repo_id,
        filename=f"{sae_path}/examples.safetensors",
    )

    from safetensors import safe_open
    data = {}
    with safe_open(examples_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            data[key] = f.get_tensor(key)

    return data


def analyze_steering_vector_with_sae(
    steering_vector: np.ndarray,
    W_dec: torch.Tensor,
    example_data: dict = None,
    tokenizer = None,
    top_k: int = 50,
) -> dict:
    """
    Compute cosine similarity between steering vector and SAE decoder columns.

    Args:
        steering_vector: (d_model,) steering direction
        W_dec: (d_sae, d_model) SAE decoder weights
        example_data: Optional dict with top_tokens for each feature
        tokenizer: Preloaded tokenizer for decoding tokens
        top_k: Number of top features to return

    Returns:
        Dict with top positive and negative cosine similarity features
    """
    # Convert to torch
    sv = torch.from_numpy(steering_vector).float()

    # Normalize
    sv_norm = sv / sv.norm()
    W_dec_norm = W_dec / W_dec.norm(dim=1, keepdim=True)

    # Cosine similarity: (d_sae,)
    cosine_sims = (W_dec_norm @ sv_norm).numpy()

    # Get top positive and negative
    sorted_idx = np.argsort(cosine_sims)
    top_positive_idx = sorted_idx[-top_k:][::-1]
    top_negative_idx = sorted_idx[:top_k]

    def get_top_tokens(idx):
        if example_data is None or 'top_tokens' not in example_data or tokenizer is None:
            return []
        try:
            tokens = example_data['top_tokens'][idx][:5].tolist()
            return [tokenizer.decode([t]) for t in tokens]
        except:
            return []

    results = {
        'max_cosine': [
            {
                'feature_idx': int(idx),
                'cosine_sim': float(cosine_sims[idx]),
                'top_tokens': get_top_tokens(idx),
            }
            for idx in top_positive_idx
        ],
        'min_cosine': [
            {
                'feature_idx': int(idx),
                'cosine_sim': float(cosine_sims[idx]),
                'top_tokens': get_top_tokens(idx),
            }
            for idx in top_negative_idx
        ],
        'statistics': {
            'mean': float(np.mean(cosine_sims)),
            'std': float(np.std(cosine_sims)),
            'max': float(np.max(cosine_sims)),
            'min': float(np.min(cosine_sims)),
        },
        'all_sims': cosine_sims,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze steering vectors with SAE")
    parser.add_argument("--vectors-path", type=str, required=True,
                        help="Path to steering vectors npz file")
    parser.add_argument("--sae-layer", type=int, default=None,
                        help="SAE layer (defaults to steering vector layer)")
    parser.add_argument("--sae-width", type=str, default="262k",
                        choices=["16k", "65k", "262k", "1m"])
    parser.add_argument("--sae-l0", type=str, default="small",
                        choices=["small", "medium", "large"])
    parser.add_argument("--model-size", type=str, default="27b")
    parser.add_argument("--use-all-layers", action="store_true", default=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", type=str, required=True,
                        help="Output path prefix")
    parser.add_argument("--load-examples", action="store_true",
                        help="Load example data for top tokens")
    args = parser.parse_args()

    # Load steering vectors
    print(f"Loading steering vectors from {args.vectors_path}")
    sv_data = load_steering_vectors(args.vectors_path)
    print(f"  Emotions: {sv_data['emotions']}")
    print(f"  Layer: {sv_data['layer']}")
    print(f"  Vector shape: {sv_data['vectors'].shape}")

    # Determine SAE layer
    sae_layer = args.sae_layer if args.sae_layer is not None else sv_data['layer']
    print(f"\nUsing SAE layer: {sae_layer}")

    # Load SAE decoder
    W_dec = load_sae_decoder(
        model_size=args.model_size,
        layer=sae_layer,
        width=args.sae_width,
        l0=args.sae_l0,
        use_all_layers=args.use_all_layers,
    )

    # Load example data if requested
    example_data = None
    tokenizer = None
    if args.load_examples:
        example_data = load_sae_example_data(
            model_size=args.model_size,
            layer=sae_layer,
            width=args.sae_width,
            l0=args.sae_l0,
            use_all_layers=args.use_all_layers,
        )
        # Load tokenizer once for decoding top tokens
        print("Loading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-27b-it")

    # Analyze each emotion
    all_results = {}
    for i, emotion in enumerate(sv_data['emotions']):
        print(f"\n{'='*60}")
        print(f" Analyzing: {emotion}")
        print(f"{'='*60}")

        results = analyze_steering_vector_with_sae(
            sv_data['vectors'][i],
            W_dec,
            example_data=example_data,
            tokenizer=tokenizer,
            top_k=args.top_k,
        )

        print(f"\n Statistics:")
        print(f"   Mean cosine: {results['statistics']['mean']:.6f}")
        print(f"   Std cosine: {results['statistics']['std']:.6f}")
        print(f"   Max cosine: {results['statistics']['max']:.6f}")
        print(f"   Min cosine: {results['statistics']['min']:.6f}")

        print(f"\n Top {args.top_k} MAX cosine features:")
        print(f" {'Idx':>8} | {'Cosine':>8} | Top Tokens")
        print(f" {'-'*8} | {'-'*8} | {'-'*40}")
        for feat in results['max_cosine'][:10]:
            tokens_str = str(feat['top_tokens'][:5]) if feat['top_tokens'] else "N/A"
            print(f" {feat['feature_idx']:>8} | {feat['cosine_sim']:>8.4f} | {tokens_str}")

        print(f"\n Top {args.top_k} MIN cosine features:")
        print(f" {'Idx':>8} | {'Cosine':>8} | Top Tokens")
        print(f" {'-'*8} | {'-'*8} | {'-'*40}")
        for feat in results['min_cosine'][:10]:
            tokens_str = str(feat['top_tokens'][:5]) if feat['top_tokens'] else "N/A"
            print(f" {feat['feature_idx']:>8} | {feat['cosine_sim']:>8.4f} | {tokens_str}")

        # Store results (without all_sims for JSON)
        all_results[emotion] = {
            'max_cosine': results['max_cosine'],
            'min_cosine': results['min_cosine'],
            'statistics': results['statistics'],
        }

        # Save all_sims separately
        np.save(f"{args.output}_{emotion}.all_sims.npy", results['all_sims'])

    # Save summary JSON
    output_path = f"{args.output}.summary.json"
    with open(output_path, 'w') as f:
        json.dump({
            'emotions': sv_data['emotions'],
            'layer': sv_data['layer'],
            'sae_layer': sae_layer,
            'sae_width': args.sae_width,
            'top_k': args.top_k,
            'results': all_results,
        }, f, indent=2)

    print(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
