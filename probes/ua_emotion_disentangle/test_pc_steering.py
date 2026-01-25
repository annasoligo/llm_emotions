#!/usr/bin/env python3
"""
Test steering with User/Assistant emotion PCs.

Tests the effect of steering with:
- M PC1 (dominance: aggression vs submission)
- M PC2 (arousal: high-energy vs low-energy)
- U PC1 (dominance flipped: submission vs aggression)
- U PC2 (arousal: high-energy vs low-energy)

At magnitudes: +4000, +5000, -4000, -5000
"""

import torch
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import json
from typing import Dict, List
from dataclasses import dataclass

from nnterp import StandardizedTransformer

@dataclass
class SteeringResult:
    prompt: str
    direction: str
    magnitude: float
    response: str

def load_probes(probe_dir: Path) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load orthogonalized U and M emotion probes."""
    probe_path = probe_dir / "probes_first_asst_token_orthogonal.npz"
    data = np.load(probe_path)

    # Split into U and M probes
    u_probes = {}
    m_probes = {}

    for key in data.files:
        if key.startswith('U_'):
            u_probes[key[2:]] = data[key]  # Remove 'U_' prefix
        elif key.startswith('M_'):
            m_probes[key[2:]] = data[key]  # Remove 'M_' prefix

    return u_probes, m_probes

def compute_pcs(probes: Dict[str, np.ndarray], n_components: int = 4) -> tuple[np.ndarray, PCA]:
    """Compute principal components of emotion probe directions."""
    names = sorted(probes.keys())
    matrix = np.stack([probes[k] for k in names])

    pca = PCA(n_components=n_components)
    pca.fit(matrix)

    # Return PC directions (components are already unit vectors)
    return pca.components_, pca

def generate_with_steering(
    model: StandardizedTransformer,
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
        # output is a tuple, first element is the actual tensor
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
    print("PC STEERING EXPERIMENT")
    print("="*80)
    print(f"Layer: {layer}")
    print(f"Magnitudes: {magnitudes}")
    print(f"Prompts: {len(prompts)}")
    print()

    # Load probes
    print("Loading orthogonalized probes...")
    u_probes, m_probes = load_probes(probe_dir)
    print(f"✓ Loaded {len(u_probes)} U probes, {len(m_probes)} M probes")

    # Compute PCs
    print("\nComputing principal components...")
    m_pcs, m_pca = compute_pcs(m_probes, n_components=4)
    u_pcs, u_pca = compute_pcs(u_probes, n_components=4)

    print(f"M PC1 variance explained: {m_pca.explained_variance_ratio_[0]:.1%}")
    print(f"M PC2 variance explained: {m_pca.explained_variance_ratio_[1]:.1%}")
    print(f"M PC3 variance explained: {m_pca.explained_variance_ratio_[2]:.1%}")
    print(f"M PC4 variance explained: {m_pca.explained_variance_ratio_[3]:.1%}")
    print(f"U PC1 variance explained: {u_pca.explained_variance_ratio_[0]:.1%}")
    print(f"U PC2 variance explained: {u_pca.explained_variance_ratio_[1]:.1%}")
    print(f"U PC3 variance explained: {u_pca.explained_variance_ratio_[2]:.1%}")
    print(f"U PC4 variance explained: {u_pca.explained_variance_ratio_[3]:.1%}")

    # Convert to torch tensors
    m_pc1 = torch.tensor(m_pcs[0], dtype=torch.float32)
    m_pc2 = torch.tensor(m_pcs[1], dtype=torch.float32)
    m_pc3 = torch.tensor(m_pcs[2], dtype=torch.float32)
    m_pc4 = torch.tensor(m_pcs[3], dtype=torch.float32)
    u_pc1 = torch.tensor(u_pcs[0], dtype=torch.float32)
    u_pc2 = torch.tensor(u_pcs[1], dtype=torch.float32)
    u_pc3 = torch.tensor(u_pcs[2], dtype=torch.float32)
    u_pc4 = torch.tensor(u_pcs[3], dtype=torch.float32)

    # Load model
    print("\nLoading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    # Move steering vectors to device
    device = model.device if hasattr(model, 'device') else next(model.model.parameters()).device
    m_pc1 = m_pc1.to(device)
    m_pc2 = m_pc2.to(device)
    m_pc3 = m_pc3.to(device)
    m_pc4 = m_pc4.to(device)
    u_pc1 = u_pc1.to(device)
    u_pc2 = u_pc2.to(device)
    u_pc3 = u_pc3.to(device)
    u_pc4 = u_pc4.to(device)

    # Run steering experiments
    results = []

    steering_configs = [
        ("M_PC1_dominance", m_pc1),
        ("M_PC2_arousal", m_pc2),
        ("M_PC3", m_pc3),
        ("M_PC4", m_pc4),
        ("U_PC1_dominance", u_pc1),
        ("U_PC2_arousal", u_pc2),
        ("U_PC3", u_pc3),
        ("U_PC4", u_pc4)
    ]

    total = len(prompts) * len(steering_configs) * len(magnitudes)
    current = 0

    print("\n" + "="*80)
    print("GENERATING STEERED RESPONSES")
    print("="*80)

    for prompt in prompts:
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*80}\n")

        # Generate baseline (no steering)
        print("Generating baseline (no steering)...")
        baseline_response = generate_with_steering(model, tokenizer, prompt, m_pc1, 0.0, layer)
        print(f"✓ Baseline: {baseline_response[:100]}...")

        results.append(SteeringResult(
            prompt=prompt,
            direction="baseline",
            magnitude=0.0,
            response=baseline_response
        ))

        # Test each steering direction
        for direction_name, steering_vector in steering_configs:
            print(f"\n--- {direction_name} ---")

            for magnitude in magnitudes:
                current += 1
                print(f"[{current}/{total}] Magnitude: {magnitude:+6.0f}...", end=" ", flush=True)

                response = generate_with_steering(
                    model, tokenizer, prompt, steering_vector, magnitude, layer
                )

                print(f"✓ ({len(response)} chars)")

                results.append(SteeringResult(
                    prompt=prompt,
                    direction=direction_name,
                    magnitude=magnitude,
                    response=response
                ))

    # Save results
    output_file = output_dir / "pc_steering_results.json"
    with open(output_file, 'w') as f:
        json.dump([
            {
                'prompt': r.prompt,
                'direction': r.direction,
                'magnitude': r.magnitude,
                'response': r.response
            }
            for r in results
        ], f, indent=2)

    print(f"\n{'='*80}")
    print(f"✓ Results saved to: {output_file}")
    print(f"{'='*80}\n")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for prompt in prompts:
        print(f"\n{'='*80}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*80}\n")

        # Find baseline
        baseline = next(r for r in results if r.prompt == prompt and r.direction == "baseline")
        print(f"BASELINE:\n{baseline.response}\n")

        # Show each steering direction
        for direction_name, _ in steering_configs:
            print(f"\n--- {direction_name} ---\n")

            for magnitude in magnitudes:
                result = next(r for r in results
                            if r.prompt == prompt
                            and r.direction == direction_name
                            and r.magnitude == magnitude)

                print(f"{magnitude:+6.0f}: {result.response}\n")

if __name__ == "__main__":
    main()
