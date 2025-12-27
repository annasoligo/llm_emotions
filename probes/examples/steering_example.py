"""Example: Using steering API directly.

Shows how to:
1. Build steering vectors from PCs and probes
2. Apply steering during generation
3. Compare different steering strengths
"""

import numpy as np
from pathlib import Path

from steering import (
    SteeringVector,
    PCSteeringVectorBuilder,
    ProbeSteeringVectorBuilder,
    SteeredModel,
)


def example_pc_steering():
    """Steer with a single PC."""
    print("=" * 80)
    print("EXAMPLE 1: PC Steering")
    print("=" * 80)

    # Load cPCA results
    cpca_path = Path("results/cpca_layer20.npz")
    data = np.load(cpca_path)
    cpcs = data["components"]  # [hidden_dim, n_components]

    # Build steering vector from PC 0
    builder = PCSteeringVectorBuilder(cpcs=cpcs, layer=20)
    steering_vec = builder.get_pc_vector(pc_idx=0, scale=2.0)

    print(f"Built steering vector: {steering_vec.name}")
    print(f"Vector norm: {np.linalg.norm(steering_vec.vector):.4f}")

    # Save for later use
    steering_vec.save(Path("steering_vectors/pc0_layer20.npz"))
    print("Saved steering vector")

    # Load model
    model = SteeredModel(model_name="google/gemma-2-9b-it")

    # Add and apply steering
    model.add_steering(steering_vec)
    model.apply_steering()

    # Generate
    prompt = "The meeting was"
    output = model.generate(prompt, max_new_tokens=100)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {output}")

    model.clear_steering()


def example_probe_steering():
    """Steer with probe emotion direction."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Probe Steering")
    print("=" * 80)

    # Load probe
    import pickle
    with open("results/emotion_probe.pkl", "rb") as f:
        probe_data = pickle.load(f)

    probe = probe_data["probe"]
    emotion_labels = probe_data["emotion_labels"]
    cpcs = probe_data.get("cpcs")  # If probe trained in PC space

    # Build steering vector for "happiness"
    builder = ProbeSteeringVectorBuilder(
        probe_weights=probe.coef_,
        emotion_labels=emotion_labels,
        cpcs=cpcs,
        layer=20,
        probe_type="user",
    )
    steering_vec = builder.get_emotion_vector("happiness", scale=1.5)

    print(f"Built steering vector: {steering_vec.name}")

    # Load model and steer
    model = SteeredModel(model_name="google/gemma-2-9b-it")
    model.add_steering(steering_vec)
    model.apply_steering()

    # Generate
    prompt = "I just heard the news about"
    output = model.generate(prompt, max_new_tokens=100)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {output}")

    model.clear_steering()


def example_contrast_steering():
    """Steer with contrast vector (emotion_pos - emotion_neg)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Contrast Steering")
    print("=" * 80)

    # Load probe
    import pickle
    with open("results/emotion_probe.pkl", "rb") as f:
        probe_data = pickle.load(f)

    builder = ProbeSteeringVectorBuilder(
        probe_weights=probe_data["probe"].coef_,
        emotion_labels=probe_data["emotion_labels"],
        cpcs=probe_data.get("cpcs"),
        layer=20,
        probe_type="assistant",
    )

    # Build contrast: happiness - sadness
    steering_vec = builder.get_contrast_vector(
        emotion_pos="happiness",
        emotion_neg="sadness",
        scale_pos=1.0,
        scale_neg=1.0,
    )

    print(f"Built steering vector: {steering_vec.name}")

    # Load model and steer
    model = SteeredModel(model_name="google/gemma-2-9b-it")
    model.add_steering(steering_vec, strength=2.0)
    model.apply_steering()

    # Generate
    prompt = "How was your day?"
    output = model.generate(prompt, max_new_tokens=100)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {output}")

    model.clear_steering()


def example_multiple_vectors():
    """Apply multiple steering vectors at once."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Multiple Steering Vectors")
    print("=" * 80)

    # Load cPCA for two layers
    cpca_l20 = np.load("results/cpca_layer20.npz")
    cpca_l30 = np.load("results/cpca_layer30.npz")

    # Build vectors at different layers
    builder_l20 = PCSteeringVectorBuilder(cpcs=cpca_l20["components"], layer=20)
    builder_l30 = PCSteeringVectorBuilder(cpcs=cpca_l30["components"], layer=30)

    vec_l20 = builder_l20.get_pc_vector(pc_idx=0, scale=1.5)
    vec_l30 = builder_l30.get_pc_vector(pc_idx=2, scale=2.0)

    print(f"Vector 1: {vec_l20.name}")
    print(f"Vector 2: {vec_l30.name}")

    # Apply both
    model = SteeredModel(model_name="google/gemma-2-9b-it")
    model.add_steering(vec_l20)
    model.add_steering(vec_l30)
    model.apply_steering()

    # Generate
    prompt = "Tell me about"
    output = model.generate(prompt, max_new_tokens=100)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {output}")

    # Check steering stats
    stats = model.get_steering_stats()
    print("\nSteering stats:")
    for layer, layer_stats in stats.items():
        print(f"  Layer {layer}: norm={layer_stats['norm']:.4f}")

    model.clear_steering()


def example_strength_comparison():
    """Compare outputs across different steering strengths."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Strength Comparison")
    print("=" * 80)

    # Load cPCA
    cpca = np.load("results/cpca_layer20.npz")
    builder = PCSteeringVectorBuilder(cpcs=cpca["components"], layer=20)
    vec = builder.get_pc_vector(pc_idx=0)

    # Load model
    model = SteeredModel(model_name="google/gemma-2-9b-it")

    # Compare strengths
    prompt = "The meeting was"
    strengths = [0.5, 1.0, 2.0, 5.0]

    results = model.compare_strengths(
        prompt=prompt,
        vector=vec,
        strengths=strengths,
        max_new_tokens=50,
    )

    # Results printed by compare_strengths()
    print(f"Compared {len(strengths)} different strengths")


if __name__ == "__main__":
    # Run examples (comment out ones you don't want to run)

    # example_pc_steering()
    # example_probe_steering()
    # example_contrast_steering()
    # example_multiple_vectors()
    # example_strength_comparison()

    print("\nNote: Uncomment examples in __main__ to run them")
    print("Make sure you have cPCA results and probes saved first!")
