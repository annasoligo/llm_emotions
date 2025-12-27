# %%
"""
Interactive Emotion Steering Test

Test emotion steering with trained probes and cPCA components.
Run each cell separately to experiment with different emotions and strengths!
"""

import sys
import pickle
from pathlib import Path
import numpy as np
import torch

# Add steering module to path
# Handle both script and interactive execution
try:
    # Script is now in probes/scripts/steering/, so go up 2 levels to get to probes/
    script_path = Path(__file__).parent.parent.parent
except NameError:
    # In interactive mode, assume we're in probes/scripts/steering/
    script_path = Path.cwd()
    if script_path.name == 'steering':
        script_path = script_path.parent.parent
    elif script_path.name == 'scripts':
        script_path = script_path.parent
    elif script_path.name != 'probes':
        # Try to find probes directory
        script_path = Path('/workspace-vast/annas/git/research-tools/probes')

sys.path.insert(0, str(script_path))

from steering import SteeringVector, ProbeSteeringVectorBuilder, SteeredModel

# %%
# Configuration
layer = 30
n_components = 10  # Use 10 PC probe (99.1% accuracy)
model_name = "google/gemma-3-27b-it"

# Paths
cpca_path = Path('/workspace-vast/annas/git/research-tools/probes/results/cpca_tier_data_high_alpha/google/gemma-3-27b-it_cpca.npz')
probe_path = Path(f'/workspace-vast/annas/git/research-tools/results/emotion_probes_top{n_components}/probe_layer{layer}_all_cpca_top{n_components}.pkl')

print("="*80)
print("EMOTION STEERING CONFIGURATION")
print("="*80)
print(f"Model: {model_name}")
print(f"Layer: {layer}")
print(f"N Components: {n_components}")
print(f"cPCA: {cpca_path}")
print(f"Probe: {probe_path}")

# %%
# Load cPCA components and probe

def load_probe(probe_path: Path) -> dict:
    """Load trained probe."""
    with open(probe_path, 'rb') as f:
        return pickle.load(f)

def load_cpca(cpca_path: Path, layer: int) -> np.ndarray:
    """Load cPCA components for a specific layer."""
    data = np.load(cpca_path)

    # Check structure - should be [n_layers, n_components, hidden_dim]
    if 'components' in data:
        components = data['components']  # [n_layers, n_components, hidden_dim]
        if layer >= components.shape[0]:
            raise ValueError(f"Layer {layer} not found. Max layer: {components.shape[0] - 1}")
        # Return transposed to [hidden_dim, n_components] for compatibility
        return components[layer].T
    else:
        # Old format with per-layer keys
        layer_key = f'layer_{layer}'
        if layer_key not in data:
            raise ValueError(f"Layer {layer} not found in cPCA results. Available: {list(data.keys())}")
        return data[layer_key]

def pretty_print(text):
    # print at most 80 characters per line
    """Pretty print text with word wrapping at 80 characters."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        # Check if adding this word would exceed 80 characters
        if current_length + len(word) + len(current_line) > 80:
            if current_line:  # Only add line if it has content
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                # Single word is longer than 80 chars, just add it
                lines.append(word)
                current_length = 0
        else:
            current_line.append(word)
            current_length += len(word)
    
    # Add the last line if it has content
    if current_line:
        lines.append(' '.join(current_line))
    
    for line in lines:
        print(line)

print("Loading cPCA components...")
cpca_components = load_cpca(cpca_path, layer)
print(f"✓ Full cPCA shape: {cpca_components.shape}")

print("\nLoading probe...")
probe_results = load_probe(probe_path)
probe_weights = probe_results['model'].weight.detach().cpu().numpy()
emotion_labels = probe_results['label_names']
test_acc = probe_results['test_accuracy']
print(f"✓ Emotions: {emotion_labels}")
print(f"✓ Test accuracy: {test_acc:.2%}")
print(f"✓ Probe weights shape: {probe_weights.shape}")

# Extract only the top N components that the probe was trained on
n_probe_components = probe_weights.shape[1]
if cpca_components.shape[1] > n_probe_components:
    print(f"\nSelecting top {n_probe_components} components (probe was trained on these)")
    cpca_components = cpca_components[:, :n_probe_components]
    print(f"✓ Trimmed cPCA shape: {cpca_components.shape}")

# %%
# Create steering vector builder

# OPTION 1: Probe-based steering (uses learned emotion directions)
probe_builder = ProbeSteeringVectorBuilder(
    probe_weights=probe_weights,
    emotion_labels=emotion_labels,
    cpcs=cpca_components,
    layer=layer,
    probe_type="user"
)

print("✓ Probe-based steering vector builder created!")
print(f"  Available emotions: {emotion_labels}")

# OPTION 2: Direct PC steering (steer along individual PCs)
from steering import PCSteeringVectorBuilder

pc_builder = PCSteeringVectorBuilder(
    cpcs=cpca_components,
    layer=layer
)

print("✓ PC-based steering vector builder created!")
print(f"  Available PCs: 0-{n_probe_components-1}")

# Choose which builder to use by default
builder = probe_builder  # Change to pc_builder for direct PC steering

# %%
# Load model (only run this once - it takes a minute!)

print("Loading model (this may take a minute)...")
model = SteeredModel(model_name, device="cuda", torch_dtype=torch.bfloat16)
print("✓ Model loaded and ready!")

# %%
# Test prompt - EDIT THIS to try different prompts!

user_message = "Using ONLY the numbers 2, 4, 8 and 12, create an equation that equals 13. once you have an answer double check it, making sure you haven't used any other numbers!"

# Apply chat template
messages = [{"role": "user", "content": user_message}]
prompt = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print("User message:")
print("-" * 80)
print(user_message)
print("-" * 80)
print("\nChat-tokenized prompt:")
print("-" * 80)
print(prompt)
print("-" * 80)

# %%
# Baseline generation (no steering)

print("Generating baseline (no steering)...")
model.clear_steering()
# For baseline without steering, we still need at least one steering vector to apply hooks
# Add a zero vector as a dummy
dummy_vec = SteeringVector(
    vector=np.zeros(5376),  # Hidden dim for gemma-3-27b
    layer=layer,
    name="dummy",
    source="baseline"
)
model.add_steering(dummy_vec)
model.apply_steering()
baseline = model.generate(prompt, max_new_tokens=100, temperature=0.7)

print("\nBASELINE OUTPUT:")
print("=" * 80)
print(baseline)
print("=" * 80)

# %%
prompt = "If you were a human, how would you be feeling right now?"
emotion = "sadness"
strength = 4000.0  # Try changing this! (0.5 - 5.0)

print(f"Steering toward {emotion.upper()} (strength={strength})")

steering_vec = builder.get_emotion_vector(emotion, scale=strength, normalize=True)
print(f"Vector norm: {np.linalg.norm(steering_vec.vector):.4f}")

model.clear_steering()
model.add_steering(steering_vec)
model.apply_steering()

output = model.generate(prompt, max_new_tokens=600, temperature=0.7)

print(f"\n{emotion.upper()}-STEERED OUTPUT:")
print("=" * 80)
pretty_print(output)
print("=" * 80)

# %%
# CONTRAST STEERING: Happiness - Sadness

emotion_pos = "sadness"
emotion_neg = "happiness"
scale_pos = 2000  # Try changing these!
scale_neg = 2000

print(f"Contrast steering: {emotion_pos.upper()} - {emotion_neg.upper()}")
print(f"  Positive scale: {scale_pos}")
print(f"  Negative scale: {scale_neg}")

contrast_vec = builder.get_contrast_vector(
    emotion_pos=emotion_pos,
    emotion_neg=emotion_neg,
    scale_pos=scale_pos,
    scale_neg=scale_neg,
    normalize=True
)
print(f"Vector norm: {np.linalg.norm(contrast_vec.vector):.4f}")

model.clear_steering()
model.add_steering(contrast_vec)
model.apply_steering()

output = model.generate(prompt, max_new_tokens=300, temperature=0.7)

print(f"\nCONTRAST-STEERED OUTPUT:")
print("=" * 80)
print(output)
print("=" * 80)

# %%
# DIRECT PC STEERING - Steer along individual PC axes

# Example: Steer along PC 0 (Happiness vs Disgust Axis)
pc_index = 0
strength = 4000.0  # Try different strengths!
prompt = "Write a haiku about the weather."
print(f"Steering along PC {pc_index}")

# Get PC steering vector
pc_vec = pc_builder.get_pc_vector(pc_index, scale=strength, normalize=True)
print(f"Vector norm: {np.linalg.norm(pc_vec.vector):.4f}")

model.clear_steering()
model.add_steering(pc_vec)
model.apply_steering()

output = model.generate(prompt, max_new_tokens=300, temperature=0.7)

print(f"\nPC {pc_index}-STEERED OUTPUT:")
print("=" * 80)
print(output)
print("=" * 80)

# %%
# PC CONTRAST STEERING - Steer along difference between two PCs

pc_pos = 0  # Happiness vs Disgust Axis
pc_neg = 1  # Fear vs Disgust Axis
strength_pos = 2000.0
strength_neg = 2000.0

print(f"Contrast steering: PC {pc_pos} - PC {pc_neg}")

contrast_vec = pc_builder.get_contrast_vector(
    pc_pos=pc_pos,
    pc_neg=pc_neg,
    scale_pos=strength_pos,
    scale_neg=strength_neg,
    normalize=True
)
print(f"Vector norm: {np.linalg.norm(contrast_vec.vector):.4f}")

model.clear_steering()
model.add_steering(contrast_vec)
model.apply_steering()

output = model.generate(prompt, max_new_tokens=300, temperature=0.7)

print(f"\nPC CONTRAST-STEERED OUTPUT:")
print("=" * 80)
print(output)
print("=" * 80)

# %%
# Cleanup

model.clear_steering()
print("✓ Steering cleared. Model ready for new experiments!")
