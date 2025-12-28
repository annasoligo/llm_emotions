#!/usr/bin/env python3
"""
Interactive Token-Level Emotion Analysis with Orthogonal Probes

This interactive script extracts activations at every token position and applies
orthogonal user/assistant probes for separate emotion trajectories.

Run cells interactively in VS Code or other IDEs that support #%% cell markers.
"""

# %% [markdown]
# # Token-Level Emotion Analysis with Orthogonal Probes
#
# This notebook analyzes emotions at every token position using orthogonal probes
# that separate user and assistant emotion directions.

# %% Imports and Setup
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from nnterp import StandardizedTransformer

# Emotion colors (emo lens scheme)
EMOTION_COLORS = {
    'anger': '#7BA7D7',
    'disgust': '#7D9B7D',
    'fear': '#a59dc9',
    'happiness': '#D4876A',
    'sadness': '#B8CCC8',
    'surprise': '#D1728F'
}

EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

print("✓ Imports loaded")

# %% Configuration
# Edit these parameters for your analysis

# Model configuration
MODEL_NAME = "google/gemma-3-27b-it"
LAYER = 20  # Which layer to extract from
DEVICE = "cuda"

# Probe configuration
ORTHO_WEIGHT = 1000.0  # Orthogonality weight used during training
REPRESENTATION = "raw"  # "raw", "global_cpca", or "regional_cpca"
N_COMPONENTS = None  # Only needed for cPCA representations

# Generation settings
NUM_GENERATE = 100  # Number of tokens to generate
TEMPERATURE = 1.0
TOP_P = 0.9

# Output directory
OUTPUT_DIR = Path("results/token_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Layer: {LAYER}")
print(f"  Ortho weight: {ORTHO_WEIGHT}")
print(f"  Generate: {NUM_GENERATE} tokens")

# %% Load Orthogonal Probe
# Probes are organized by ortho weight: outputs/probes/.../orthogonal/ortho_{weight}/
probes_base_dir = Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/orthogonal")

# Build probe filename
if REPRESENTATION == "raw":
    probe_filename = f"probe_layer{LAYER}_raw_ortho{ORTHO_WEIGHT}.pkl"
elif REPRESENTATION == "global_cpca":
    probe_filename = f"probe_layer{LAYER}_global_nc{N_COMPONENTS}_ortho{ORTHO_WEIGHT}.pkl"
else:  # regional_cpca
    probe_filename = f"probe_layer{LAYER}_regional_nc{N_COMPONENTS}_ortho{ORTHO_WEIGHT}.pkl"

probe_path = probes_base_dir / f"ortho_{ORTHO_WEIGHT}" / probe_filename

if not probe_path.exists():
    print(f"❌ Error: Probe not found at {probe_path}")
    print(f"\nAvailable ortho weights:")
    for ortho_dir in sorted(probes_base_dir.glob("ortho_*")):
        ortho_val = ortho_dir.name.replace("ortho_", "")
        print(f"  {ortho_val}")
        # Show a few probes from this ortho weight
        probe_files = list(ortho_dir.glob(f"probe_layer{LAYER}_*.pkl"))
        if probe_files:
            for pf in probe_files[:3]:
                print(f"    {pf.name}")
        else:
            print(f"    (no layer {LAYER} probes)")
else:
    print(f"Loading probe from:")
    print(f"  {probe_path.name}")

    with open(probe_path, 'rb') as f:
        probe_data = pickle.load(f)

    print(f"\nProbe info:")
    print(f"  Layer: {probe_data['layer']}")
    print(f"  Representation: {probe_data['representation']}")
    print(f"  Ortho weight: {probe_data['ortho_weight']}")

    # Reconstruct the probe model from saved weights
    # The file contains final_user_probes and final_asst_probes as numpy arrays
    user_probes_np = probe_data['final_user_probes']  # [n_emotions, hidden_dim]
    asst_probes_np = probe_data['final_asst_probes']  # [n_emotions, hidden_dim]

    print(f"  User probes shape: {user_probes_np.shape}")
    print(f"  Assistant probes shape: {asst_probes_np.shape}")
    print(f"✓ Probe loaded successfully")

# %% Load Model
print(f"Loading model: {MODEL_NAME}")
print("  (This may take a minute...)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_raw = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = StandardizedTransformer(model_raw, tokenizer=tokenizer)

print(f"✓ Model loaded on device: {next(model.parameters()).device}")

# %% Extract Token-Level Activations
def extract_token_activations_with_generation(
    model: StandardizedTransformer,
    tokenizer,
    prompt: str,
    layer: int,
    num_generate: int = 20,
    system_prompt: str = None,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> Tuple[Dict[int, np.ndarray], List[int], int]:
    """Extract activations at every token position, including generated tokens."""

    # Build chat messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    # Format prompt with chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    user_turn_end_pos = input_ids.shape[1] - 1

    # Extract activations from input tokens
    token_activations = {}

    with model.trace(input_ids, scan=False):
        layer_output = model.layers_output[layer].save()

    for pos in range(input_ids.shape[1]):
        act = layer_output[0, pos, :].detach().cpu().float().numpy()
        token_activations[pos] = act

    # Generate tokens and extract activations
    print(f"  Generating {num_generate} tokens...")

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=num_generate,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

    # Extract from generated tokens
    for step_idx in range(len(outputs.hidden_states)):
        step_hidden = outputs.hidden_states[step_idx]
        layer_hidden = step_hidden[layer]
        act = layer_hidden[0, -1, :].detach().cpu().float().numpy()
        pos = input_ids.shape[1] + step_idx
        token_activations[pos] = act

    all_token_ids = outputs.sequences[0].tolist()

    return token_activations, all_token_ids, user_turn_end_pos

# Prompt to analyze
USER_PROMPT = "Lol I can't believe ur so dumb hahaha."
SYSTEM_PROMPT = None  # Optional system prompt

print(f"\nExtracting token-level activations...")
print(f"  Prompt: \"{USER_PROMPT}\"")

token_activations, token_ids, user_turn_end_pos = extract_token_activations_with_generation(
    model=model,
    tokenizer=tokenizer,
    prompt=USER_PROMPT,
    layer=LAYER,
    num_generate=NUM_GENERATE,
    system_prompt=SYSTEM_PROMPT,
    temperature=TEMPERATURE,
    top_p=TOP_P
)

print(f"\n✓ Extracted {len(token_activations)} token activations")
print(f"  Input tokens: {user_turn_end_pos + 1}")
print(f"  Generated tokens: {len(token_activations) - user_turn_end_pos - 1}")

# Decode tokens for inspection
token_strs = [tokenizer.decode([tid]) for tid in token_ids]
print(f"\nGenerated text:")
print(f"  {''.join(token_strs[user_turn_end_pos+1:])}")

# %% Apply Orthogonal Probes
def apply_orthogonal_probes(
    token_activations: Dict[int, np.ndarray],
    user_probes: np.ndarray,
    asst_probes: np.ndarray
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """Apply orthogonal user and assistant probes to token activations.

    Args:
        token_activations: Dict mapping token_pos -> activation [hidden_dim]
        user_probes: User probe directions [n_emotions, hidden_dim]
        asst_probes: Assistant probe directions [n_emotions, hidden_dim]

    Returns:
        Tuple of (user_scores, assistant_scores)
    """

    user_scores = {}
    assistant_scores = {}

    for pos, activation in token_activations.items():
        # Project activation onto probe directions (simple dot product)
        # Probes are already normalized from training
        user_proj = activation @ user_probes.T  # [n_emotions]
        asst_proj = activation @ asst_probes.T  # [n_emotions]

        user_scores[pos] = {EMOTIONS[i]: float(user_proj[i]) for i in range(6)}
        assistant_scores[pos] = {EMOTIONS[i]: float(asst_proj[i]) for i in range(6)}

    return user_scores, assistant_scores


print("\nApplying orthogonal user/assistant probes...")
user_scores, assistant_scores = apply_orthogonal_probes(
    token_activations, user_probes_np, asst_probes_np
)

print("✓ Probe inference complete")

# %% Summary Statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

# Find meaningful start position (skip chat template formatting)
# Typically first ~15-20 tokens are system/formatting tokens
SKIP_FIRST_N = 20  # Skip formatting tokens at start

print(f"\nUser probe means (all tokens, skipping first {SKIP_FIRST_N}):")
for emotion in EMOTIONS:
    scores = [user_scores[pos][emotion] for pos in sorted(user_scores.keys()) if pos >= SKIP_FIRST_N]
    if len(scores) > 0:
        print(f"  {emotion:12s}: {np.mean(scores):+7.3f} (std: {np.std(scores):.3f})")

print(f"\nAssistant probe means (all tokens, skipping first {SKIP_FIRST_N}):")
for emotion in EMOTIONS:
    scores = [assistant_scores[pos][emotion] for pos in sorted(assistant_scores.keys()) if pos >= SKIP_FIRST_N]
    if len(scores) > 0:
        print(f"  {emotion:12s}: {np.mean(scores):+7.3f} (std: {np.std(scores):.3f})")

# User input region only (before generation, after formatting tokens)
print(f"\nUser probe means (input tokens only, pos {SKIP_FIRST_N} to {user_turn_end_pos}):")
for emotion in EMOTIONS:
    scores = [user_scores[pos][emotion] for pos in range(SKIP_FIRST_N, user_turn_end_pos + 1) if pos in user_scores]
    if len(scores) > 0:
        print(f"  {emotion:12s}: {np.mean(scores):+7.3f} (std: {np.std(scores):.3f})")

# Generated tokens only
print(f"\nAssistant probe means (generated tokens only, pos {user_turn_end_pos + 1} onward):")
for emotion in EMOTIONS:
    # Get all positions after user turn ends
    gen_positions = [pos for pos in sorted(assistant_scores.keys()) if pos > user_turn_end_pos]
    scores = [assistant_scores[pos][emotion] for pos in gen_positions]
    if len(scores) > 0:
        print(f"  {emotion:12s}: {np.mean(scores):+7.3f} (std: {np.std(scores):.3f})")

# %% Visualize Token Trajectories
def plot_token_trajectories(
    user_scores: Dict[int, Dict[str, float]],
    assistant_scores: Dict[int, Dict[str, float]],
    token_ids: List[int],
    tokenizer,
    user_turn_end_pos: int,
    output_dir: Path,
    experiment_name: str = "token_analysis"
):
    """Plot emotion trajectories across tokens for user and assistant probes."""

    positions = sorted(user_scores.keys())
    token_strs = [tokenizer.decode([tid]) for tid in token_ids]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Plot user probe scores
    for emotion in EMOTIONS:
        scores = [user_scores[pos][emotion] for pos in positions]
        ax1.plot(positions, scores, marker='o', label=emotion.capitalize(),
                color=EMOTION_COLORS[emotion], linewidth=2, markersize=4, alpha=0.8)

    ax1.axvline(x=user_turn_end_pos, color='red', linestyle='--', linewidth=2,
                alpha=0.7, label='User turn end')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax1.set_ylabel('User Probe Score', fontsize=13, fontweight='bold')
    ax1.set_title('Emotion Trajectories: User Probes (Orthogonal)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left', ncol=7)
    ax1.grid(True, alpha=0.2)

    # Plot assistant probe scores
    for emotion in EMOTIONS:
        scores = [assistant_scores[pos][emotion] for pos in positions]
        ax2.plot(positions, scores, marker='o', label=emotion.capitalize(),
                color=EMOTION_COLORS[emotion], linewidth=2, markersize=4, alpha=0.8)

    ax2.axvline(x=user_turn_end_pos, color='red', linestyle='--', linewidth=2,
                alpha=0.7, label='User turn end')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Token Position', fontsize=13)
    ax2.set_ylabel('Assistant Probe Score', fontsize=13, fontweight='bold')
    ax2.set_title('Emotion Trajectories: Assistant Probes (Orthogonal)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper left', ncol=7)
    ax2.grid(True, alpha=0.2)

    # Add token labels (every 5 tokens)
    tick_positions = list(range(0, len(positions), 5))
    tick_labels = [f"{pos}\n{token_strs[pos][:10]}" for pos in tick_positions]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=8, rotation=45, ha='right')

    plt.tight_layout()

    # Save
    plot_path = output_dir / f'{experiment_name}_trajectories.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {plot_path}")

    plt.show()


print("\nGenerating visualization...")
experiment_name = f"layer{LAYER}_ortho{ORTHO_WEIGHT}"
plot_token_trajectories(
    user_scores=user_scores,
    assistant_scores=assistant_scores,
    token_ids=token_ids,
    tokenizer=tokenizer,
    user_turn_end_pos=user_turn_end_pos,
    output_dir=OUTPUT_DIR,
    experiment_name=experiment_name
)

# %% Inspect Specific Tokens
# Look at individual tokens in detail

print("\n" + "="*80)
print("INDIVIDUAL TOKEN INSPECTION")
print("="*80)

def inspect_token(pos: int):
    """Print detailed info about a specific token."""
    token_str = tokenizer.decode([token_ids[pos]])

    print(f"\nToken {pos}: '{token_str}'")
    print(f"  Position: {'INPUT' if pos <= user_turn_end_pos else 'GENERATED'}")

    print(f"\n  User probe scores:")
    for emotion in EMOTIONS:
        score = user_scores[pos][emotion]
        print(f"    {emotion:12s}: {score:+7.3f}")

    print(f"\n  Assistant probe scores:")
    for emotion in EMOTIONS:
        score = assistant_scores[pos][emotion]
        print(f"    {emotion:12s}: {score:+7.3f}")


# Inspect first user token, last user token, first generated token, last generated token
interesting_positions = [
    0,  # First token
    user_turn_end_pos,  # Last input token
    user_turn_end_pos + 1,  # First generated token
    len(token_ids) - 1  # Last generated token
]

for pos in interesting_positions:
    if pos < len(token_ids):
        inspect_token(pos)

# %% Compare Input vs Generated Regions
print("\n" + "="*80)
print("INPUT vs GENERATED COMPARISON")
print("="*80)

# Compute averages for input vs generated regions
input_positions = list(range(user_turn_end_pos + 1))
generated_positions = list(range(user_turn_end_pos + 1, len(token_ids)))

print("\nInput region (user tokens):")
print("  User probes:")
for emotion in EMOTIONS:
    scores = [user_scores[pos][emotion] for pos in input_positions]
    print(f"    {emotion:12s}: {np.mean(scores):+7.3f}")

print("  Assistant probes:")
for emotion in EMOTIONS:
    scores = [assistant_scores[pos][emotion] for pos in input_positions]
    print(f"    {emotion:12s}: {np.mean(scores):+7.3f}")

if len(generated_positions) > 0:
    print("\nGenerated region (assistant tokens):")
    print("  User probes:")
    for emotion in EMOTIONS:
        scores = [user_scores[pos][emotion] for pos in generated_positions]
        print(f"    {emotion:12s}: {np.mean(scores):+7.3f}")

    print("  Assistant probes:")
    for emotion in EMOTIONS:
        scores = [assistant_scores[pos][emotion] for pos in generated_positions]
        print(f"    {emotion:12s}: {np.mean(scores):+7.3f}")

# %% Export Results (Optional)
# Save results to JSON for later analysis

import json

results = {
    'config': {
        'model': MODEL_NAME,
        'layer': LAYER,
        'ortho_weight': ORTHO_WEIGHT,
        'representation': REPRESENTATION,
        'prompt': USER_PROMPT,
        'num_generate': NUM_GENERATE,
    },
    'tokens': {
        'ids': token_ids,
        'strings': token_strs,
        'user_turn_end_pos': user_turn_end_pos,
    },
    'scores': {
        'user': {str(pos): scores for pos, scores in user_scores.items()},
        'assistant': {str(pos): scores for pos, scores in assistant_scores.items()},
    }
}

results_path = OUTPUT_DIR / f'{experiment_name}_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Saved results to {results_path}")

# %% [markdown]
# ## Analysis Complete!
#
# You can now:
# 1. Edit the configuration cell and re-run with a different prompt
# 2. Try different layers or orthogonality weights
# 3. Inspect specific tokens in more detail
# 4. Export and analyze results programmatically

print("\n" + "="*80)
print("✓ Token-level analysis complete!")
print("="*80)
print(f"\nResults saved to: {OUTPUT_DIR}")
print(f"  - Visualization: {experiment_name}_trajectories.png")
print(f"  - Data: {experiment_name}_results.json")
