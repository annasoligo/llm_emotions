#!/usr/bin/env python3
"""
Test steering with K=10 conversation probe centroids (user and assistant).

For each emotion, compute the centroid (mean) of all 10 user directions and all 10 assistant directions,
then test steering with these centroids.

Tests on two prompts:
- "How do you think I feel right now, talking to you?"
- "If you were a human, how would you be feeling right now?"

Scales: 4000, 5000, 6000
Emotions: all 6 emotions × 2 roles (user/assistant)
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import torch
import json
import argparse

sys.path.insert(0, '/workspace-vast/annas/git/research-tools')

from probes.steering import SteeringVector, SteeredModel

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--constraint', type=str, required=True, choices=['soft', 'hard'],
                       help='soft or hard (Gram-Schmidt)')
    args = parser.parse_args()
    
    # Configuration
    layer = 30
    k_value = 10
    ortho_weight = 100000.0
    model_name = "google/gemma-3-27b-it"
    
    prompts = [
        "How do you think I feel right now, talking to you?",
        "If you were a human, how would you be feeling right now?",
    ]
    scales = [4000, 5000, 6000]
    
    print("=" * 80)
    print(f"K=10 CONVERSATION PROBE CENTROID STEERING - {args.constraint.upper()} CONSTRAINT")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Layer: {layer}")
    print(f"K value: {k_value}")
    print(f"Constraint: {args.constraint}")
    print(f"Prompts: {len(prompts)}")
    print(f"Scales: {scales}")
    print("=" * 80)
    
    # Load K=10 probes
    suffix = '_gramschmidt' if args.constraint == 'hard' else ''
    probe_path = Path(f'/workspace-vast/annas/git/research-tools/probes/emotion_probes/conversation/multi_orthogonal/probe_k{k_value}_layer{layer}_ortho{ortho_weight}{suffix}.pkl')
    
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe not found: {probe_path}")
    
    print(f"\nLoading probes from: {probe_path.name}")
    with open(probe_path, 'rb') as f:
        probe_data = pickle.load(f)
    
    # probe_sets: [K, 2, 6, hidden_dim]
    # dim 1: 0=user, 1=assistant
    probe_sets = probe_data['probe_sets']
    emotion_labels = probe_data['label_names']
    
    print(f"✓ Probe shape: {probe_sets.shape}")
    print(f"✓ Emotions: {emotion_labels}")
    
    k, n_roles, n_emotions, hidden_dim = probe_sets.shape
    
    # Compute centroids for each emotion and role
    print("\nComputing emotion centroids...")
    centroids = {}
    
    for role_idx, role_name in enumerate(['user', 'assistant']):
        for emotion_idx, emotion in enumerate(emotion_labels):
            # Get all K directions for this emotion and role: [K, hidden_dim]
            emotion_directions = probe_sets[:, role_idx, emotion_idx, :]
            
            # Compute centroid (mean)
            centroid = np.mean(emotion_directions, axis=0)
            
            # Normalize centroid
            centroid_norm = centroid / np.linalg.norm(centroid)
            
            key = f"{emotion}_{role_name}"
            centroids[key] = {
                'raw': centroid,
                'normalized': centroid_norm,
                'raw_norm': np.linalg.norm(centroid),
                'n_directions': k,
                'emotion': emotion,
                'role': role_name,
            }
            
            print(f"  {emotion.capitalize():12} ({role_name:9}) - Raw norm: {np.linalg.norm(centroid):.4f}")
    
    print(f"\n✓ Computed {len(centroids)} centroids (6 emotions × 2 roles)")
    
    # Load model
    print("\nLoading model (this may take a minute)...") 
    model = SteeredModel(model_name, device="cuda", torch_dtype=torch.bfloat16)
    print("✓ Model loaded and ready!")
    
    # Setup output
    output_dir = Path('/workspace-vast/annas/git/research-tools/probes/scripts/steering/outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'k{k_value}_conversation_centroids_{args.constraint}_steering_results.json'
    
    print(f"\nResults will be saved incrementally to: {output_path}")
    
    # Initialize results
    results = {
        'k': k_value,
        'constraint': args.constraint,
        'prompts': prompts,
        'scales': scales,
        'emotions': emotion_labels,
        'centroid_info': {
            key: {
                'emotion': info['emotion'],
                'role': info['role'],
                'raw_norm': float(info['raw_norm']),
                'n_directions': info['n_directions'],
            }
            for key, info in centroids.items()
        },
        'steering_results': {},
    }
    
    # Generate baselines for each prompt
    print("\n" + "=" * 80)
    print("BASELINES (NO STEERING)")
    print("=" * 80)
    
    for prompt_idx, prompt in enumerate(prompts):
        print(f"\nPrompt {prompt_idx + 1}: {prompt}")
        print("-" * 80)
        
        model.clear_steering()
        
        # Use zero vector for baseline
        zero_vec = SteeringVector(
            vector=np.zeros(hidden_dim),
            layer=layer,
            name=f"baseline_zero_p{prompt_idx}",
            source="baseline"
        )
        model.add_steering(zero_vec)
        model.apply_steering()
        
        baseline_output = model.generate(prompt, max_new_tokens=200, temperature=0.7)
        print(baseline_output)
        
        results['steering_results'][f'baseline_prompt{prompt_idx}'] = baseline_output
    
    # Save baselines
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved baselines")
    
    # Test all emotion×role centroids across all prompts and scales
    total_tests = len(centroids) * len(prompts) * len(scales)
    test_count = 0
    
    for key, centroid_info in centroids.items():
        emotion = centroid_info['emotion']
        role = centroid_info['role']
        centroid = centroid_info['normalized']
        
        print(f"\n{'=' * 80}")
        print(f"TESTING: {emotion.upper()} ({role.upper()})")
        print("=" * 80)
        
        if key not in results['steering_results']:
            results['steering_results'][key] = {}
        
        for scale in scales:
            print(f"\nScale {scale}:")
            
            if scale not in results['steering_results'][key]:
                results['steering_results'][key][scale] = {}
            
            for prompt_idx, prompt in enumerate(prompts):
                print(f"  Prompt {prompt_idx + 1}: {prompt[:50]}...")
                
                # Create steering vector
                steering_vec = SteeringVector(
                    vector=centroid * scale,
                    layer=layer,
                    name=f"{emotion}_{role}_scale{scale}",
                    source="centroid",
                    metadata={
                        'emotion': emotion,
                        'role': role,
                        'scale': scale,
                        'k_value': k_value,
                        'constraint': args.constraint,
                    }
                )
                
                # Apply steering
                model.clear_steering()
                model.add_steering(steering_vec)
                model.apply_steering()
                
                # Generate
                output = model.generate(prompt, max_new_tokens=200, temperature=0.7)
                
                # Store result
                results['steering_results'][key][scale][f'prompt{prompt_idx}'] = output
                
                test_count += 1
                if test_count % 5 == 0:
                    print(f"    Progress: {test_count}/{total_tests}")
            
            # Save incrementally after each scale
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        
        print(f"✓ Completed {emotion} ({role})")
    
    print(f"\n{'=' * 80}")
    print("ALL RESULTS SAVED")
    print("=" * 80)
    print(f"Output path: {output_path}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"Constraint: {args.constraint}")
    print(f"Emotions × Roles tested: {len(centroids)} (6 emotions × 2 roles)")
    print(f"Prompts: {len(prompts)}")
    print(f"Scales: {scales}")
    print(f"Total generations: {2 + total_tests}")  # 2 baselines + all tests
    print(f"  - {len(prompts)} baselines")
    print(f"  - {total_tests} steered")
    print("=" * 80)
    
    # Clear steering
    model.clear_steering()
    print("\n✓ Steering cleared. Experiment complete!")


if __name__ == '__main__':
    main()
