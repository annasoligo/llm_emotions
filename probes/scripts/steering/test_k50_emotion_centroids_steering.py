#!/usr/bin/env python3
"""
Test steering with emotion centroids from K=50 orthogonal probes.

For each emotion, compute the centroid (mean) of all 50 orthogonal directions,
then test steering with these centroids.

Tests on prompt: "If you were a human, how would you be feeling right now?"
Scales: 4000, 5000, 6000, 7000
Emotions: all 6 emotions
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
scales = [4000, 5000, 6000, 7000]

print("=" * 80)
print("K=50 EMOTION CENTROID STEERING TEST")
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
all_probe_sets = probe_data['all_probe_sets']
emotion_labels = probe_data['label_names']

print(f"✓ Probe weights shape: {all_probe_sets.shape}")
print(f"✓ Emotions: {emotion_labels}")
print(f"✓ Number of probe sets: {all_probe_sets.shape[0]}")

# %%
# Compute centroids for each emotion
print("\nComputing emotion centroids...")
n_sets, n_emotions, hidden_dim = all_probe_sets.shape

emotion_centroids = {}
for emotion_idx, emotion in enumerate(emotion_labels):
    # Get all 50 directions for this emotion: [n_sets, hidden_dim]
    emotion_directions = all_probe_sets[:, emotion_idx, :]

    # Compute centroid (mean)
    centroid = np.mean(emotion_directions, axis=0)

    # Normalize centroid
    centroid_norm = centroid / np.linalg.norm(centroid)

    emotion_centroids[emotion] = {
        'raw': centroid,
        'normalized': centroid_norm,
        'raw_norm': np.linalg.norm(centroid),
        'n_directions': n_sets
    }

    print(f"  {emotion.capitalize():12} - Raw norm: {np.linalg.norm(centroid):.4f}, "
          f"Normalized: {np.linalg.norm(centroid_norm):.4f}")

print(f"\n✓ Computed {len(emotion_centroids)} emotion centroids")

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
    vector=np.zeros(hidden_dim),
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
output_path = output_dir / f'k{k_value}_emotion_centroids_steering_results.json'

print(f"\nResults will be saved incrementally to: {output_path}")

# %%
# Initialize results with baseline
results = {
    'prompt': prompt,
    'scales': scales,
    'emotions': emotion_labels,
    'baseline': baseline_output,
    'centroid_info': {
        emotion: {
            'raw_norm': float(emotion_centroids[emotion]['raw_norm']),
            'n_directions_averaged': emotion_centroids[emotion]['n_directions']
        }
        for emotion in emotion_labels
    },
    'steering_results': {}
}

# Initialize nested structure for each emotion
for emotion in emotion_labels:
    results['steering_results'][emotion] = {}

# Save initial results with baseline
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved baseline")

# %%
# Test all emotion centroids across all scales

for emotion in emotion_labels:
    print(f"\n{'=' * 80}")
    print(f"TESTING EMOTION: {emotion.upper()}")
    print("=" * 80)

    centroid = emotion_centroids[emotion]['normalized']

    for scale in scales:
        print(f"\nScale {scale}:")
        print("-" * 80)

        # Create steering vector
        steering_vec = SteeringVector(
            vector=centroid * scale,
            layer=layer,
            name=f"{emotion}_centroid_scale{scale}",
            source="centroid",
            metadata={
                'emotion': emotion,
                'scale': scale,
                'k_value': k_value,
                'centroid_type': 'mean_of_k_directions'
            }
        )

        # Apply steering
        model.clear_steering()
        model.add_steering(steering_vec)
        model.apply_steering()

        # Generate
        output = model.generate(prompt, max_new_tokens=200, temperature=0.7)
        print(output)

        # Store result
        results['steering_results'][emotion][scale] = output

        # Save incrementally
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\n✓ Completed {emotion}")

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
print(f"Emotions tested: {len(emotion_labels)}")
print(f"Scales tested: {scales}")
print(f"Total generations: {1 + len(emotion_labels) * len(scales)}")
print(f"  - 1 baseline")
print(f"  - {len(emotion_labels)} emotions × {len(scales)} scales = {len(emotion_labels) * len(scales)}")
print("=" * 80)

# Clear steering
model.clear_steering()
print("\n✓ Steering cleared. Experiment complete!")