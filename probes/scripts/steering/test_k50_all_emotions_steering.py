#!/usr/bin/env python3
"""
Test steering with all 50 probe sets from K=50 across all 6 emotions.

Tests on prompt: "If you were a human, how would you be feeling right now?"
Scales: 4000, 5000, 6000, 7000
Emotions: sadness, happiness, anger, fear, surprise, disgust
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
emotions_to_test = ['sadness', 'happiness', 'anger', 'fear', 'surprise', 'disgust']

print("=" * 80)
print("K=50 ALL EMOTIONS STEERING TEST")
print("=" * 80)
print(f"Model: {model_name}")
print(f"Layer: {layer}")
print(f"K value: {k_value}")
print(f"Ortho weight: {ortho_weight}")
print(f"Prompt: {prompt}")
print(f"Scales: {scales}")
print(f"Emotions: {emotions_to_test}")
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

# Validate emotions exist
for emotion in emotions_to_test:
    if emotion not in emotion_labels:
        raise ValueError(f"'{emotion}' not in emotion labels: {emotion_labels}")

# Get emotion indices and directions
emotion_data = {}
for emotion in emotions_to_test:
    emotion_idx = emotion_labels.index(emotion)
    directions = all_probe_sets[:, emotion_idx, :]  # [n_sets, hidden_dim]
    emotion_data[emotion] = {
        'idx': emotion_idx,
        'directions': directions
    }
    print(f"✓ {emotion.capitalize()}: index {emotion_idx}, shape {directions.shape}")

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
    vector=np.zeros(all_probe_sets.shape[2]),
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
output_path = output_dir / f'k{k_value}_all_emotions_steering_results.json'

print(f"\nResults will be saved incrementally to: {output_path}")

# %%
# Initialize results with baseline
results = {
    'prompt': prompt,
    'scales': scales,
    'emotions': emotions_to_test,
    'baseline': baseline_output,
    'steering_results': {}
}

# Initialize nested structure for each emotion
for emotion in emotions_to_test:
    results['steering_results'][emotion] = []

# Save initial results with baseline
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"✓ Saved baseline")

# %%
# Test all emotions across all probe sets and scales

for emotion in emotions_to_test:
    print(f"\n{'=' * 80}")
    print(f"TESTING EMOTION: {emotion.upper()}")
    print("=" * 80)

    directions = emotion_data[emotion]['directions']

    for probe_idx in range(k_value):
        print(f"\n{'-' * 80}")
        print(f"{emotion.upper()} - Probe Set {probe_idx + 1}/{k_value}")
        print("-" * 80)

        # Get this probe set's direction for this emotion
        emotion_vec = directions[probe_idx]

        # Normalize
        norm = np.linalg.norm(emotion_vec)
        emotion_vec_normalized = emotion_vec / norm

        probe_results = {
            'probe_idx': probe_idx,
            'vector_norm': float(norm),
            'outputs': {}
        }

        for scale in scales:
            print(f"  Scale {scale}...", end=" ", flush=True)

            # Create steering vector
            steering_vec = SteeringVector(
                vector=emotion_vec_normalized * scale,
                layer=layer,
                name=f"{emotion}_probe{probe_idx}_scale{scale}",
                source="orthogonal_probe",
                metadata={
                    'probe_idx': probe_idx,
                    'emotion': emotion,
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

            probe_results['outputs'][scale] = output
            print("✓")

        # Append this probe's results for this emotion
        results['steering_results'][emotion].append(probe_results)

        # Save incrementally after each probe set
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        if (probe_idx + 1) % 10 == 0:
            print(f"  ✓ Saved results for {emotion} probe {probe_idx + 1}/{k_value}")

    print(f"\n✓ Completed all {k_value} probe sets for {emotion.upper()}")

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
print(f"Emotions tested: {len(emotions_to_test)}")
print(f"Probe sets per emotion: {k_value}")
print(f"Scales tested: {scales}")
print(f"Total generations: {1 + len(emotions_to_test) * k_value * len(scales)}")
print(f"  - 1 baseline")
print(f"  - {len(emotions_to_test)} emotions × {k_value} probes × {len(scales)} scales = {len(emotions_to_test) * k_value * len(scales)}")
print("=" * 80)

# Clear steering
model.clear_steering()
print("\n✓ Steering cleared. Experiment complete!")
