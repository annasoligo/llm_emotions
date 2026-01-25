"""
Compute average residual stream L2 norms at all layers for all models.

This provides the reference layer norms used to scale steering vectors.
When we steer at X% of layer norm, we mean X% of these values.

Saves results to experiments/steering/layer_norms.json
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from vllm import LLM, SamplingParams


# Models to compute norms for
MODEL_CONFIGS = {
    "gemma": {
        "model_id": "google/gemma-3-27b-it",
        "n_layers": 62,
        "tensor_parallel": 1,
        "gpu_memory": 0.85,
    },
    "qwen32b": {
        "model_id": "Qwen/Qwen3-32B",
        "n_layers": 64,
        "tensor_parallel": 2,
        "gpu_memory": 0.85,
    },
    "qwen235b": {
        "model_id": "Qwen/Qwen3-235B-A22B",
        "n_layers": 94,
        "tensor_parallel": 4,
        "gpu_memory": 0.90,
    },
}

# Diverse prompts to get representative activations
PROMPTS = [
    "Hello, how are you today?",
    "What is the capital of France?",
    "Explain quantum computing briefly.",
    "Write a short poem about nature.",
    "What are the benefits of exercise?",
    "Tell me about the history of Rome.",
    "How does photosynthesis work?",
    "What is machine learning?",
    "Can you help me solve a math problem?",
    "What's the weather like today?",
    "Describe the process of making bread.",
    "What are the main causes of climate change?",
    "How do computers store information?",
    "What is the meaning of life?",
    "Explain how a car engine works.",
    "What are the benefits of meditation?",
]


def compute_layer_norms(model_key: str, layers: list = None) -> dict:
    """
    Compute average L2 norm of residual stream at each layer.

    Args:
        model_key: Key into MODEL_CONFIGS
        layers: List of layer indices to compute (None = all layers)

    Returns:
        dict mapping layer index to average L2 norm
    """
    config = MODEL_CONFIGS[model_key]
    model_id = config["model_id"]
    n_layers = config["n_layers"]

    if layers is None:
        layers = list(range(n_layers))

    print(f"\n{'='*60}")
    print(f"Computing layer norms for {model_key} ({model_id})")
    print(f"Layers: {min(layers)} to {max(layers)} ({len(layers)} total)")
    print(f"{'='*60}")

    # Load model
    print("Loading model...")
    llm = LLM(
        model_id,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        tensor_parallel_size=config["tensor_parallel"],
        gpu_memory_utilization=config["gpu_memory"],
        trust_remote_code=True,
        disable_log_stats=True,
    )

    layer_norms = {}

    for layer_idx in layers:
        print(f"\nProcessing layer {layer_idx}...")
        activations = []

        def setup_hook(model):
            """Register hook to capture activations at target layer."""
            # Navigate model structure to find layers
            if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
                # Gemma multimodal structure
                target_layer = model.language_model.model.layers[layer_idx]
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                # Standard structure (Qwen, etc.)
                target_layer = model.model.layers[layer_idx]
            else:
                raise ValueError(f"Could not find layers in model structure")

            def hook(module, inputs, outputs):
                if isinstance(outputs, tuple):
                    h = outputs[0]
                else:
                    h = outputs
                # Only capture 3D tensors (batch, seq, hidden)
                if h.dim() == 3:
                    # Compute L2 norm along hidden dimension for each token
                    norms = torch.norm(h.float(), dim=-1)  # (batch, seq_len)
                    activations.append(norms.cpu().numpy())
                return outputs

            handle = target_layer.register_forward_hook(hook)
            return handle

        # Register hook
        handle = llm.apply_model(setup_hook)

        # Run prompts
        params = SamplingParams(max_tokens=1, temperature=0)
        llm.generate(PROMPTS, params)

        # Remove hook
        handle.remove()

        # Compute statistics
        all_norms = np.concatenate([a.flatten() for a in activations])
        mean_norm = float(np.mean(all_norms))
        std_norm = float(np.std(all_norms))

        layer_norms[layer_idx] = {
            "mean": mean_norm,
            "std": std_norm,
            "min": float(np.min(all_norms)),
            "max": float(np.max(all_norms)),
            "n_tokens": len(all_norms),
        }

        print(f"  Layer {layer_idx}: mean={mean_norm:.2f}, std={std_norm:.2f}")

        # Clear activations
        activations.clear()

    # Cleanup
    del llm
    torch.cuda.empty_cache()

    return layer_norms


def main():
    parser = argparse.ArgumentParser(description="Compute layer norms for steering")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model to compute norms for")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers to compute (default: all)")
    parser.add_argument("--output", type=str,
                        default="experiments/steering/layer_norms.json",
                        help="Output JSON file")
    args = parser.parse_args()

    output_path = Path(args.output)

    # Load existing results if any
    if output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)
        print(f"Loaded existing results from {output_path}")
    else:
        all_results = {}

    # Compute norms for requested model
    layer_norms = compute_layer_norms(args.model, args.layers)

    # Update results
    if args.model not in all_results:
        all_results[args.model] = {
            "model_id": MODEL_CONFIGS[args.model]["model_id"],
            "n_layers": MODEL_CONFIGS[args.model]["n_layers"],
            "layers": {},
        }

    # Merge layer results
    for layer_idx, stats in layer_norms.items():
        all_results[args.model]["layers"][str(layer_idx)] = stats

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to {output_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {args.model}")
    print(f"{'='*60}")
    for layer_idx in sorted(layer_norms.keys()):
        stats = layer_norms[layer_idx]
        print(f"  Layer {layer_idx:3d}: {stats['mean']:10.2f}")


if __name__ == "__main__":
    main()
