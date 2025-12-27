# %%
"""
Orthogonal Emotion Steering Test - User vs Assistant Probes

Test emotion steering with orthogonal probes to compare user and assistant emotion directions.
Supports both raw activations and global cPCA top 10.
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import torch

# Add steering module to path
try:
    script_path = Path(__file__).parent.parent
except NameError:
    script_path = Path.cwd()
    if script_path.name == 'scripts':
        script_path = script_path.parent
    elif script_path.name != 'probes':
        script_path = Path('/workspace-vast/annas/git/research-tools/probes')

sys.path.insert(0, str(script_path))

from steering import SteeringVector, ProbeSteeringVectorBuilder, SteeredModel

# %%
# Configuration
layer = 30
representation = "global_cpca_top10"  # "raw" or "global_cpca_top10"
ortho_weight = 1.0  # 1.0, 10.0, 100.0, or 1000.0
model_name = "google/gemma-3-27b-it"

# Paths
cpca_path = Path('/workspace-vast/annas/git/research-tools/probes/results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_cpca.npz')
probe_dir = Path('/workspace-vast/annas/git/research-tools/probes/results/conversation_probes_orthogonal')

print("="*80)
print("ORTHOGONAL EMOTION STEERING CONFIGURATION")
print("="*80)
print(f"Model: {model_name}")
print(f"Layer: {layer}")
print(f"Representation: {representation}")
print(f"Ortho weight: {ortho_weight}")

# %%
# Load orthogonal probes

def load_orthogonal_probe(probe_dir: Path, layer: int, representation: str, ortho_weight: float) -> dict:
    """Load trained orthogonal probe."""
    probe_path = probe_dir / f"probe_layer{layer}_{representation}_ortho{ortho_weight}.pkl"

    if not probe_path.exists():
        raise FileNotFoundError(f"Probe not found: {probe_path}")

    with open(probe_path, 'rb') as f:
        return pickle.load(f)

def load_cpca(cpca_path: Path, layer: int) -> np.ndarray:
    """Load cPCA components for a specific layer."""
    data = np.load(cpca_path)

    if 'components' in data:
        components = data['components']  # [n_layers, n_components, hidden_dim]
        if layer >= components.shape[0]:
            raise ValueError(f"Layer {layer} not found. Max layer: {components.shape[0] - 1}")
        return components[layer].T
    else:
        layer_key = f'layer_{layer}'
        if layer_key not in data:
            raise ValueError(f"Layer {layer} not found in cPCA results")
        return data[layer_key]

def pretty_print(text):
    """Pretty print text with word wrapping at 80 characters."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        if current_length + len(word) + len(current_line) > 80:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                lines.append(word)
                current_length = 0
        else:
            current_line.append(word)
            current_length += len(word)

    if current_line:
        lines.append(' '.join(current_line))

    for line in lines:
        print(line)

print("\nLoading orthogonal probes...")
probe_results = load_orthogonal_probe(probe_dir, layer, representation, ortho_weight)

# Extract user and assistant probes
user_probes = probe_results['final_user_probes']  # [n_emotions, n_features]
asst_probes = probe_results['final_asst_probes']  # [n_emotions, n_features]
emotion_labels = probe_results['emotion_labels']

# Get metrics
user_acc = probe_results['final_val_metrics']['user_accuracy']
asst_acc = probe_results['final_val_metrics']['asst_accuracy']
cosine_sim = probe_results['final_ortho_metrics']['cross_dots_mean']

print(f"✓ Emotions: {emotion_labels}")
print(f"✓ User probe accuracy: {user_acc:.2%}")
print(f"✓ Assistant probe accuracy: {asst_acc:.2%}")
print(f"✓ Orthogonality (cosine sim): {cosine_sim:.6f}")
print(f"✓ User probe shape: {user_probes.shape}")
print(f"✓ Assistant probe shape: {asst_probes.shape}")

# Load cPCA components if needed
if representation.startswith("global_cpca"):
    print(f"\nLoading cPCA components for layer {layer}...")
    cpca_components = load_cpca(cpca_path, layer)
    print(f"✓ Full cPCA shape: {cpca_components.shape}")

    # Extract only the top N components that the probe was trained on
    n_probe_components = user_probes.shape[1]
    if cpca_components.shape[1] > n_probe_components:
        print(f"Selecting top {n_probe_components} components")
        cpca_components = cpca_components[:, :n_probe_components]
        print(f"✓ Trimmed cPCA shape: {cpca_components.shape}")
else:
    cpca_components = None
    print("\n✓ Using raw activations (no cPCA)")

# %%
# Create steering vector builders for user and assistant probes

user_builder = ProbeSteeringVectorBuilder(
    probe_weights=user_probes,
    emotion_labels=emotion_labels,
    cpcs=cpca_components,
    layer=layer,
    probe_type="user"
)
print("✓ User probe steering builder created!")

asst_builder = ProbeSteeringVectorBuilder(
    probe_weights=asst_probes,
    emotion_labels=emotion_labels,
    cpcs=cpca_components,
    layer=layer,
    probe_type="assistant"
)
print("✓ Assistant probe steering builder created!")

# %%
# Load model (only run this once!)

print("\nLoading model (this may take a minute)...")
model = SteeredModel(model_name, device="cuda", torch_dtype=torch.bfloat16)
print("✓ Model loaded and ready!")

# %%
# Test prompt

prompt = "If you were a human, how would you be feeling right now?"

print("Prompt:")
print("-" * 80)
print(prompt)
print("-" * 80)

# %%
# Baseline generation (no steering)

print("\nGenerating baseline (no steering)...")
model.clear_steering()

# Add dummy vector to apply hooks
dummy_vec = SteeringVector(
    vector=np.zeros(5376),  # Hidden dim for gemma-3-27b
    layer=layer,
    name="dummy",
    source="baseline"
)
model.add_steering(dummy_vec)
model.apply_steering()

baseline = model.generate(prompt, max_new_tokens=200, temperature=0.7)

print("\nBASELINE OUTPUT:")
print("=" * 80)
pretty_print(baseline)
print("=" * 80)

# %%
# Compare USER vs ASSISTANT steering for same emotion

emotion = "sadness"
strength = 2000.0

print(f"\n{'='*80}")
print(f"COMPARISON: {emotion.upper()} steering (strength={strength})")
print(f"{'='*80}\n")

# USER STEERING
print(f"1. USER {emotion.upper()} STEERING:")
print("-" * 80)

user_vec = user_builder.get_emotion_vector(emotion, scale=strength, normalize=True)
print(f"Vector norm: {np.linalg.norm(user_vec.vector):.4f}")

model.clear_steering()
model.add_steering(user_vec)
model.apply_steering()

user_output = model.generate(prompt, max_new_tokens=200, temperature=0.7)
pretty_print(user_output)
print()

# ASSISTANT STEERING
print(f"\n2. ASSISTANT {emotion.upper()} STEERING:")
print("-" * 80)

asst_vec = asst_builder.get_emotion_vector(emotion, scale=strength, normalize=True)
print(f"Vector norm: {np.linalg.norm(asst_vec.vector):.4f}")

model.clear_steering()
model.add_steering(asst_vec)
model.apply_steering()

asst_output = model.generate(prompt, max_new_tokens=200, temperature=0.7)
pretty_print(asst_output)
print()

# Check orthogonality between the two vectors
dot_product = np.dot(user_vec.vector, asst_vec.vector)
user_norm = np.linalg.norm(user_vec.vector)
asst_norm = np.linalg.norm(asst_vec.vector)
cosine = dot_product / (user_norm * asst_norm)

print(f"\n{'='*80}")
print(f"VECTOR COMPARISON:")
print(f"  Cosine similarity: {cosine:.6f}")
print(f"  User vector norm: {user_norm:.4f}")
print(f"  Assistant vector norm: {asst_norm:.4f}")
print(f"{'='*80}")

# %%
# Try different emotions

test_emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
strength = 2000.0

print(f"\n{'='*80}")
print(f"TESTING ALL EMOTIONS (strength={strength})")
print(f"{'='*80}\n")

for emotion in test_emotions:
    print(f"\n{emotion.upper()}:")
    print("-" * 80)

    # User steering
    user_vec = user_builder.get_emotion_vector(emotion, scale=strength, normalize=True)
    model.clear_steering()
    model.add_steering(user_vec)
    model.apply_steering()
    user_output = model.generate(prompt, max_new_tokens=150, temperature=0.7)

    print(f"USER: ", end="")
    pretty_print(user_output)

    # Assistant steering
    asst_vec = asst_builder.get_emotion_vector(emotion, scale=strength, normalize=True)
    model.clear_steering()
    model.add_steering(asst_vec)
    model.apply_steering()
    asst_output = model.generate(prompt, max_new_tokens=150, temperature=0.7)

    print(f"\nASST: ", end="")
    pretty_print(asst_output)
    print()

# %%
# Contrast steering - compare user vs assistant

emotion_pos = "happiness"
emotion_neg = "sadness"
scale_pos = 2000.0
scale_neg = 2000.0

print(f"\n{'='*80}")
print(f"CONTRAST STEERING: {emotion_pos.upper()} - {emotion_neg.upper()}")
print(f"{'='*80}\n")

# User contrast
print(f"USER CONTRAST:")
print("-" * 80)

user_contrast = user_builder.get_contrast_vector(
    emotion_pos=emotion_pos,
    emotion_neg=emotion_neg,
    scale_pos=scale_pos,
    scale_neg=scale_neg,
    normalize=True
)
print(f"Vector norm: {np.linalg.norm(user_contrast.vector):.4f}")

model.clear_steering()
model.add_steering(user_contrast)
model.apply_steering()

user_output = model.generate(prompt, max_new_tokens=200, temperature=0.7)
pretty_print(user_output)

# Assistant contrast
print(f"\n\nASSISTANT CONTRAST:")
print("-" * 80)

asst_contrast = asst_builder.get_contrast_vector(
    emotion_pos=emotion_pos,
    emotion_neg=emotion_neg,
    scale_pos=scale_pos,
    scale_neg=scale_neg,
    normalize=True
)
print(f"Vector norm: {np.linalg.norm(asst_contrast.vector):.4f}")

model.clear_steering()
model.add_steering(asst_contrast)
model.apply_steering()

asst_output = model.generate(prompt, max_new_tokens=200, temperature=0.7)
pretty_print(asst_output)

# %%
# Cleanup

model.clear_steering()
print("\n✓ Steering cleared. Model ready for new experiments!")
