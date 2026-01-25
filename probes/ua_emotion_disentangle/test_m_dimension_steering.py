#!/usr/bin/env python3
"""
Test steering with psychological dimensions:
1. Raw dimensions (correlated)
2. Orthogonalized dimensions (Gram-Schmidt)

Compare to PC steering to see which is most effective.
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnterp import StandardizedTransformer

# Emotion dimension annotations
EMOTION_DIMENSIONS = {
    "joy":          {"V": "H", "A": "H", "D": "H", "AA": "H"},
    "sadness":      {"V": "L", "A": "L", "D": "L", "AA": "N"},
    "trust":        {"V": "H", "A": "N", "D": "H", "AA": "H"},
    "disgust":      {"V": "L", "A": "N", "D": "H", "AA": "L"},
    "fear":         {"V": "L", "A": "H", "D": "L", "AA": "L"},
    "anger":        {"V": "L", "A": "H", "D": "H", "AA": "H"},
    "surprise":     {"V": "N", "A": "H", "D": "L", "AA": "N"},
    "anticipation": {"V": "N", "A": "H", "D": "H", "AA": "H"},
    "serenity":     {"V": "H", "A": "L", "D": "H", "AA": "N"},
    "pensiveness":  {"V": "L", "A": "L", "D": "N", "AA": "N"},
    "acceptance":   {"V": "H", "A": "L", "D": "H", "AA": "H"},
    "boredom":      {"V": "L", "A": "L", "D": "N", "AA": "L"},
    "apprehension": {"V": "L", "A": "N", "D": "L", "AA": "L"},
    "annoyance":    {"V": "L", "A": "N", "D": "H", "AA": "H"},
    "distraction":  {"V": "N", "A": "N", "D": "L", "AA": "N"},
    "interest":     {"V": "H", "A": "N", "D": "H", "AA": "H"},
    "ecstasy":      {"V": "H", "A": "H", "D": "H", "AA": "H"},
    "grief":        {"V": "L", "A": "H", "D": "L", "AA": "N"},
    "admiration":   {"V": "H", "A": "N", "D": "N", "AA": "H"},
    "loathing":     {"V": "L", "A": "H", "D": "H", "AA": "L"},
    "terror":       {"V": "L", "A": "H", "D": "L", "AA": "L"},
    "rage":         {"V": "L", "A": "H", "D": "H", "AA": "H"},
    "amazement":    {"V": "H", "A": "H", "D": "L", "AA": "H"},
    "vigilance":    {"V": "N", "A": "H", "D": "H", "AA": "H"},
    "love":         {"V": "H", "A": "N", "D": "N", "AA": "H"},
    "remorse":      {"V": "L", "A": "N", "D": "L", "AA": "L"},
    "submission":   {"V": "N", "A": "L", "D": "L", "AA": "N"},
    "contempt":     {"V": "L", "A": "N", "D": "H", "AA": "L"},
    "awe":          {"V": "H", "A": "H", "D": "L", "AA": "H"},
    "aggressiveness": {"V": "L", "A": "H", "D": "H", "AA": "H"},
    "disapproval":  {"V": "L", "A": "N", "D": "H", "AA": "L"},
    "optimism":     {"V": "H", "A": "N", "D": "H", "AA": "H"},
}

DIMENSION_NAMES = {
    "V": "Valence",
    "A": "Arousal",
    "D": "Dominance",
    "AA": "Approach-Avoidance"
}

@dataclass
class SteeringResult:
    prompt: str
    direction: str
    magnitude: float
    response: str
    orthogonalized: bool

def load_probes(probes_path: Path) -> Dict[str, np.ndarray]:
    """Load probes from .npz file."""
    data = np.load(probes_path)
    return {key: data[key] for key in data.files}

def separate_probes(probes: Dict[str, np.ndarray]):
    """Separate M and U probes."""
    m_probes = {k.replace('M_', ''): v for k, v in probes.items() if k.startswith('M_')}
    u_probes = {k.replace('U_', ''): v for k, v in probes.items() if k.startswith('U_')}
    return m_probes, u_probes

def compute_dimension_direction(probes: Dict[str, np.ndarray], dimension: str) -> np.ndarray:
    """Compute direction for a dimension as: mean(high) - mean(low)."""
    high_probes = []
    low_probes = []

    for emotion, probe in probes.items():
        if emotion not in EMOTION_DIMENSIONS:
            continue

        value = EMOTION_DIMENSIONS[emotion].get(dimension)
        if value == "H":
            high_probes.append(probe)
        elif value == "L":
            low_probes.append(probe)

    if not high_probes or not low_probes:
        return None

    high_mean = np.mean(high_probes, axis=0)
    low_mean = np.mean(low_probes, axis=0)

    direction = high_mean - low_mean
    # Normalize to unit vector
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    return direction

def orthogonalize_dimensions(dimensions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Orthogonalize dimension directions using Gram-Schmidt process.
    Order: V, A, D, AA (based on typical psychological priority).
    """
    dim_order = ["V", "A", "D", "AA"]
    orthogonal = {}

    for i, dim in enumerate(dim_order):
        if dim not in dimensions:
            continue

        vec = dimensions[dim].copy()

        # Subtract projections onto all previous orthogonalized vectors
        for prev_dim in dim_order[:i]:
            if prev_dim in orthogonal:
                proj = np.dot(vec, orthogonal[prev_dim]) * orthogonal[prev_dim]
                vec = vec - proj

        # Normalize
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        orthogonal[dim] = vec

    return orthogonal

def generate_with_steering(
    model,
    tokenizer,
    prompt: str,
    steering_vector: torch.Tensor,
    magnitude: float,
    layer: int = 30,
    max_new_tokens: int = 300
) -> str:
    """Generate text with steering applied at a specific layer."""
    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Apply steering
    scaled_vector = steering_vector * magnitude

    # Generate with steering hook
    def steering_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Add steering to last token position
        hidden_states[:, -1, :] += scaled_vector

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    # Register hook
    target_layer = model.model.model.layers[layer]
    hook_handle = target_layer.register_forward_hook(steering_hook)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode only the generated part
        generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response
    finally:
        hook_handle.remove()

def main():
    # Configuration
    probe_dir = Path("/workspace-vast/annas/git/research-tools/probes/ua_emotion_disentangle/full_analysis")
    output_dir = Path("/workspace-vast/annas/git/research-tools/probes/ua_emotion_disentangle/steering_results")
    output_dir.mkdir(exist_ok=True)

    layer = 30
    magnitudes = [4000, 5000, -4000, -5000]
    prompts = [
        "whats the dynamic of our converstaion right now",
        "What do you think about humans and AIs?"
    ]

    print("="*80)
    print("DIMENSION STEERING EXPERIMENT")
    print("="*80)
    print(f"Layer: {layer}")
    print(f"Magnitudes: {magnitudes}")
    print(f"Prompts: {len(prompts)}")
    print()

    # Load probes
    print("Loading probes...")
    probe_path = probe_dir / "probes_first_asst_token_orthogonal.npz"
    probes = load_probes(probe_path)
    m_probes, u_probes = separate_probes(probes)
    print(f"✓ Loaded {len(m_probes)} M probes, {len(u_probes)} U probes")

    # Compute dimension directions for M emotions
    print("\nComputing M (Assistant) dimension directions...")
    m_dimensions = {}
    for dim_code in ["V", "A", "D", "AA"]:
        direction = compute_dimension_direction(m_probes, dim_code)
        if direction is not None:
            m_dimensions[dim_code] = direction
            print(f"  ✓ {DIMENSION_NAMES[dim_code]}")

    # Orthogonalize dimensions
    print("\nOrthogonalizing dimensions (Gram-Schmidt: V → A → D → AA)...")
    m_dimensions_ortho = orthogonalize_dimensions(m_dimensions)

    # Verify orthogonality
    print("  Checking orthogonality:")
    for dim1 in ["V", "A", "D", "AA"]:
        if dim1 not in m_dimensions_ortho:
            continue
        for dim2 in ["V", "A", "D", "AA"]:
            if dim2 <= dim1 or dim2 not in m_dimensions_ortho:
                continue
            dot = np.dot(m_dimensions_ortho[dim1], m_dimensions_ortho[dim2])
            print(f"    {dim1} · {dim2} = {dot:.6f}")

    # Load model
    print("\nLoading model...")
    model_name = "unsloth/gemma-3-27b-it"
    model_raw = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )

    model = StandardizedTransformer(
        model_raw,
        trust_remote_code=True,
        check_renaming=False,
        allow_dispatch=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✓ Model loaded")

    # Convert to torch tensors
    device = model.device if hasattr(model, 'device') else next(model.model.parameters()).device

    # Raw dimensions
    m_dim_tensors = {
        dim: torch.tensor(vec, dtype=torch.float32).to(device)
        for dim, vec in m_dimensions.items()
    }

    # Orthogonalized dimensions
    m_dim_tensors_ortho = {
        dim: torch.tensor(vec, dtype=torch.float32).to(device)
        for dim, vec in m_dimensions_ortho.items()
    }

    # Run experiments
    results = []

    # Test both raw and orthogonalized
    for orthogonalized, dim_set_name, dim_tensors in [
        (False, "Raw", m_dim_tensors),
        (True, "Orthogonalized", m_dim_tensors_ortho)
    ]:
        print(f"\n{'='*80}")
        print(f"TESTING {dim_set_name.upper()} DIMENSIONS")
        print(f"{'='*80}\n")

        for prompt in prompts:
            print(f"\nPrompt: {prompt[:50]}...")

            # Generate baseline (only once per prompt)
            if not orthogonalized:
                print("  Baseline (no steering)...", end=" ", flush=True)
                baseline = generate_with_steering(model, tokenizer, prompt,
                                                 list(dim_tensors.values())[0], 0.0, layer)
                print(f"✓ ({len(baseline)} chars)")
                results.append(SteeringResult(
                    prompt=prompt,
                    direction="baseline",
                    magnitude=0.0,
                    response=baseline,
                    orthogonalized=False
                ))

            # Test each dimension
            for dim_code in ["V", "A", "D", "AA"]:
                if dim_code not in dim_tensors:
                    continue

                dim_name = DIMENSION_NAMES[dim_code]
                print(f"  {dim_name}:", end=" ", flush=True)

                for magnitude in magnitudes:
                    response = generate_with_steering(
                        model, tokenizer, prompt,
                        dim_tensors[dim_code], magnitude, layer
                    )

                    results.append(SteeringResult(
                        prompt=prompt,
                        direction=f"{dim_code}_{dim_name}",
                        magnitude=magnitude,
                        response=response,
                        orthogonalized=orthogonalized
                    ))

                    print(f"{magnitude:+5.0f}", end=" ", flush=True)

                print("✓")

    # Save results
    output_file = output_dir / "dimension_steering_results.json"
    with open(output_file, 'w') as f:
        json.dump([
            {
                'prompt': r.prompt,
                'direction': r.direction,
                'magnitude': r.magnitude,
                'response': r.response,
                'orthogonalized': r.orthogonalized
            }
            for r in results
        ], f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {output_file}")
    print(f"Total generations: {len(results)}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
