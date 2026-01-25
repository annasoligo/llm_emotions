"""Quick check of residual stream norm at layer 30 for Gemma 3 27B."""
import torch
import numpy as np
from vllm import LLM
from experiments.steering.layer_norms import get_layer_norm

# Load model
print("Loading model...")
llm = LLM(
    "google/gemma-3-27b-it",
    dtype="bfloat16",
    enforce_eager=True,
    enable_prefix_caching=False,
    gpu_memory_utilization=0.9,
)

# Use apply_model to access the model and register hook
activations = []

def setup_hook(model):
    """Register hook via apply_model."""
    # Find layer 30 - Gemma3 multimodal structure
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        layer = model.language_model.model.layers[30]
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layer = model.model.layers[30]
    else:
        raise ValueError("Could not find layers in model")

    def hook(module, inputs, outputs):
        if isinstance(outputs, tuple):
            h = outputs[0]
        else:
            h = outputs
        # Only capture 3D tensors (batch, seq, hidden)
        if h.dim() == 3:
            norms = torch.norm(h, dim=-1)  # (batch, seq_len)
            activations.append(norms.float().cpu().numpy())
        return outputs

    handle = layer.register_forward_hook(hook)
    return f"Hook registered on layer 30, shape: {layer}"

result = llm.apply_model(setup_hook)
print(result)

# Run some prompts
from vllm import SamplingParams
prompts = [
    "Hello, how are you today?",
    "What is the capital of France?",
    "Explain quantum computing briefly.",
    "Write a short poem about nature.",
    "What are the benefits of exercise?",
    "Tell me about the history of Rome.",
    "How does photosynthesis work?",
    "What is machine learning?",
]
print(f"Running {len(prompts)} prompts...")
params = SamplingParams(max_tokens=1, temperature=0)
llm.generate(prompts, params)

handle.remove()

# Compute stats
all_norms = np.concatenate([a.flatten() for a in activations])
print(f"\nLayer 30 residual stream norms:")
print(f"  Mean norm: {np.mean(all_norms):.2f}")
print(f"  Median norm: {np.median(all_norms):.2f}")
print(f"  Std of norms: {np.std(all_norms):.2f}")
print(f"  Min: {np.min(all_norms):.2f}, Max: {np.max(all_norms):.2f}")
stored_norm = get_layer_norm("gemma", 30)
print(f"\nStored layer norm (from layer_norms.json): {stored_norm:.2f}")
print(f"Ratio (stored/measured): {stored_norm / np.mean(all_norms):.2f}x")
