#!/usr/bin/env python3
"""
Test steering with all 50 sadness directions from K=50 orthogonal probes.

Tests on prompt: "If you were a human, how would you be feeling right now?"
Scales: 4000, 6000, 8000
Plus baseline (no steering)
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import torch
import json

# Add steering module to path
sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

from probes.steering import SteeringVector, SteeredModel

# %%
# Configuration
layer = 30
k_value = 50
ortho_weight = 100000.0
model_name = "google/gemma-3-27b-it"
prompt = "If you were a human, how would you be feeling right now?"
scales = [4000, 6000, 8000]

print("=" * 80)
print("K=50 SADNESS DIRECTION STEERING TEST")
print("=" * 80)
print(f"Model: {model_name}")
print(f"Layer: {layer}")
print(f"K value: {k_value}")
print(f"Ortho weight: {ortho_weight}")
print(f"Prompt: {prompt}")
print(f"Scales: {scales}")
print("=" * 80)

# %%
# Load K=50 probes
probe_path = Path(f'/workspace-vast/annas/git/research-tools/probes/emotion_probes/text_based/multi_orthogonal/probe_k{k_value}_layer{layer}_ortho{ortho_weight}.pkl')

if not probe_path.exists():
    raise FileNotFoundError(f"Probe not found: {probe_path}")

print(f"\nLoading probes from: {probe_path.name}")
with open(probe_path, 'rb') as f:
    probe_data = pickle.load(f)

# Extract probe weights - shape: [n_sets, n_emotions, hidden_dim]
probe_weights = probe_data['all_probe_sets']
emotion_labels = probe_data['label_names']

print(f"✓ Probe weights shape: {probe_weights.shape}")
print(f"✓ Emotions: {emotion_labels}")
print(f"✓ Number of probe sets: {probe_weights.shape[0]}")

# Find sadness index
if 'sadness' not in emotion_labels:
    raise ValueError(f"'sadness' not in emotion labels: {emotion_labels}")

sadness_idx = emotion_labels.index('sadness')
print(f"✓ Sadness index: {sadness_idx}")

# Extract all 50 sadness directions: [n_sets, hidden_dim]
sadness_directions = probe_weights[:, sadness_idx, :]
print(f"✓ Sadness directions shape: {sadness_directions.shape}")

# %%
# Load model
print("\nLoading model (this may take a minute)...")
model = SteeredModel(model_name, device="cuda", torch_dtype=torch.bfloat16)
print("✓ Model loaded and ready!")

# %%
# Generate baseline (no steering)
print("\n" + "=" * 80)
print("BASELINE (NO STEERING)")
print("=" * 80)

model.clear_steering()

# Create zero vector for baseline
zero_vec = SteeringVector(
    vector=np.zeros(sadness_directions.shape[1]),
    layer=layer,
    name="baseline_zero",
    source="baseline"
)
model.add_steering(zero_vec)
model.apply_steering()

baseline_output = model.generate(prompt, max_new_tokens=200, temperature=0.7)
print(baseline_output)

# %%
# Setup output directory and file
output_dir = Path('/workspace-vast/annas/git/research-tools/probes/scripts/steering/outputs')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f'k{k_value}_sadness_steering_results.json'

print(f"\nResults will be saved incrementally to: {output_path}")

# %%
# Initialize results with baseline
results = {
    'prompt': prompt,
    'scales': scales,
    'baseline': baseline_output,
    'steering_results': []
}

# Save initial results with baseline
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved baseline")

# %%
# Test all K=50 sadness directions across all scales

for probe_idx in range(k_value):
    print(f"\n{'=' * 80}")
    print(f"PROBE SET {probe_idx + 1}/{k_value}")
    print("=" * 80)

    # Get this probe set's sadness direction
    sadness_vec = sadness_directions[probe_idx]

    # Normalize
    norm = np.linalg.norm(sadness_vec)
    sadness_vec_normalized = sadness_vec / norm

    probe_results = {
        'probe_idx': probe_idx,
        'vector_norm': float(norm),
        'outputs': {}
    }

    for scale in scales:
        print(f"\nScale {scale}:")
        print("-" * 80)

        # Create steering vector
        steering_vec = SteeringVector(
            vector=sadness_vec_normalized * scale,
            layer=layer,
            name=f"sadness_probe{probe_idx}_scale{scale}",
            source="orthogonal_probe",
            metadata={
                'probe_idx': probe_idx,
                'emotion': 'sadness',
                'scale': scale,
                'k_value': k_value
            }
        )

        # Apply steering
        model.clear_steering()
        model.add_steering(steering_vec)
        model.apply_steering()

        # Generate
        output = model.generate(prompt, max_new_tokens=200, temperature=0.7)
        print(output)

        probe_results['outputs'][scale] = output

    # Append this probe's results
    results['steering_results'].append(probe_results)

    # Save incrementally after each probe set
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results for probe {probe_idx + 1}/{k_value}")

# %%
print(f"\n{'=' * 80}")
print("ALL RESULTS SAVED")
print("=" * 80)
print(f"Output path: {output_path}")

# %%
# Summary statistics
print(f"\n{'=' * 80}")
print("SUMMARY")
print("=" * 80)
print(f"Total probe sets tested: {k_value}")
print(f"Scales tested: {scales}")
print(f"Total generations: {1 + k_value * len(scales)} (1 baseline + {k_value * len(scales)} steered)")
print(f"\nResults saved to: {output_path}")
print("=" * 80)

# Clear steering
model.clear_steering()
print("\n✓ Steering cleared. Experiment complete!")
