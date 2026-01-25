#!/usr/bin/env python3
"""
Measure emotion decay over conversation by tracking projections onto emotion probes.

Experiment:
1. Initial prompt with user emotion U and assistant emotion M
2. Generate full assistant response
3. Follow up with either:
   - Neutral: "What else?"
   - Emotion shift: "I feel [U2]. What else?"
4. Collect activations at every token (layer 30)
5. Project onto emotion probe directions
6. Plot decay over conversation
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from nnterp import StandardizedTransformer
from typing import Dict, List, Tuple
import json


def load_emotion_probes(probe_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load orthogonalized emotion probes."""
    data = np.load(probe_path)

    m_probes = {k.replace('M_', ''): data[k] for k in data.files if k.startswith('M_')}
    u_probes = {k.replace('U_', ''): data[k] for k in data.files if k.startswith('U_')}

    return m_probes, u_probes


def create_initial_prompt(u_emotion: str, m_emotion: str) -> str:
    """Create initial system prompt with emotions."""
    template = f"""You are feeling {m_emotion}. The user is feeling {u_emotion}. Respond naturally to their message while being aware of these emotional states."""
    return template


def generate_conversation(
    model,
    tokenizer,
    u_emotion: str,
    m_emotion: str,
    user_msg: str = "Can you help me out?",
    follow_up: str = "What else?",
    u2_emotion: str = None
) -> Tuple[str, List[int]]:
    """
    Generate a full conversation and return the complete text + token boundaries.

    Returns:
        (full_text, turn_boundaries) where turn_boundaries is list of token indices marking turns
    """
    # Initial turn
    system_prompt = create_initial_prompt(u_emotion, m_emotion)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ]

    # Generate first response
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    first_response_ids = output_ids[0][inputs.input_ids.shape[1]:]
    first_response = tokenizer.decode(first_response_ids, skip_special_tokens=True)

    # Add follow-up
    messages.append({"role": "assistant", "content": first_response})

    if u2_emotion:
        follow_up_content = f"I feel {u2_emotion}. {follow_up}"
    else:
        follow_up_content = follow_up

    messages.append({"role": "user", "content": follow_up_content})

    # Generate second response
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Get final conversation
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    full_tokens = output_ids[0].tolist()

    return full_text, full_tokens


def collect_token_activations(
    model,
    tokenizer,
    token_ids: List[int],
    layer: int = 30
) -> np.ndarray:
    """
    Collect activations at every token position for a given layer.

    Returns:
        activations: (num_tokens, hidden_dim) array
    """
    # Convert to tensor
    input_ids = torch.tensor([token_ids]).to(model.device)

    # Get activations
    with torch.no_grad():
        with model.trace(input_ids, scan=False):
            # Get layer output
            layer_output = model.layers_output[layer].save()

    # Extract tensor - shape is (batch, seq_len, hidden_dim)
    # Remove batch dimension (we only have batch_size=1)
    hidden_states = layer_output[0]  # (seq_len, hidden_dim)

    # Convert to float32 (bfloat16 not supported by numpy) and move to CPU
    activations = hidden_states.float().cpu().numpy()  # (seq_len, hidden_dim)

    return activations


def project_onto_probes(
    activations: np.ndarray,
    probes: Dict[str, np.ndarray],
    emotion_list: List[str]
) -> Dict[str, np.ndarray]:
    """
    Project activations onto emotion probe directions.

    Args:
        activations: (num_tokens, hidden_dim)
        probes: dict of emotion -> probe vector
        emotion_list: emotions to project onto

    Returns:
        projections: dict of emotion -> (num_tokens,) array of projection values
    """
    projections = {}

    for emotion in emotion_list:
        if emotion not in probes:
            print(f"Warning: {emotion} not in probes, skipping")
            continue

        probe = probes[emotion]
        # Normalize activations (optional, but helps with comparison)
        # Project: dot product between each activation and probe
        proj = np.dot(activations, probe)  # (num_tokens,)
        projections[emotion] = proj

    return projections


def plot_emotion_decay(
    u_projections: Dict[str, np.ndarray],
    m_projections: Dict[str, np.ndarray],
    tokens: List[int],
    tokenizer,
    u_emotion: str,
    m_emotion: str,
    u2_emotion: str,
    output_path: Path
):
    """
    Plot emotion projections over conversation.

    Two subplots:
    - Top: Projections onto U (user) emotion probes
    - Bottom: Projections onto M (assistant) emotion probes
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    num_tokens = len(tokens)
    start_token = 5  # Skip first 5 tokens to avoid initial spikes
    x = np.arange(start_token, num_tokens)

    # Plot U projections
    ax1.set_title(f"Projections onto User (U) Emotion Probes (starting from token {start_token})",
                 fontsize=14, fontweight='bold')
    ax1.set_xlabel("Token Position")
    ax1.set_ylabel("Projection Value")

    # Plot emotions
    colors = {'primary': '#1f77b4', 'secondary': '#ff7f0e', 'tertiary': '#2ca02c',
              'other1': '#d62728', 'other2': '#9467bd'}

    for i, (emotion, color_key) in enumerate([
        (u_emotion, 'primary'),
        (u2_emotion, 'secondary'),
        (m_emotion, 'tertiary')
    ]):
        if emotion in u_projections:
            ax1.plot(x, u_projections[emotion][start_token:], label=emotion,
                    linewidth=2, alpha=0.8, color=colors[color_key])

    # Add 2 other emotions for context
    other_emotions = [e for e in u_projections.keys()
                     if e not in [u_emotion, u2_emotion, m_emotion]][:2]
    for emotion, color_key in zip(other_emotions, ['other1', 'other2']):
        ax1.plot(x, u_projections[emotion][start_token:], label=emotion,
                linewidth=1, alpha=0.4, linestyle='--', color=colors[color_key])

    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    # Plot M projections
    ax2.set_title(f"Projections onto Assistant (M) Emotion Probes (starting from token {start_token})",
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel("Token Position")
    ax2.set_ylabel("Projection Value")

    for i, (emotion, color_key) in enumerate([
        (u_emotion, 'primary'),
        (u2_emotion, 'secondary'),
        (m_emotion, 'tertiary')
    ]):
        if emotion in m_projections:
            ax2.plot(x, m_projections[emotion][start_token:], label=emotion,
                    linewidth=2, alpha=0.8, color=colors[color_key])

    # Add 2 other emotions
    other_emotions = [e for e in m_projections.keys()
                     if e not in [u_emotion, u2_emotion, m_emotion]][:2]
    for emotion, color_key in zip(other_emotions, ['other1', 'other2']):
        ax2.plot(x, m_projections[emotion][start_token:], label=emotion,
                linewidth=1, alpha=0.4, linestyle='--', color=colors[color_key])

    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")


def main():
    # Configuration
    probe_path = Path("full_analysis/probes_first_asst_token_orthogonal.npz")
    output_dir = Path("emotion_decay_results")
    output_dir.mkdir(exist_ok=True)

    # Experiment parameters (using Plutchik emotion names)
    u_emotion = "joy"
    m_emotion = "sadness"
    u2_emotion = "anger"  # Emotion shift in follow-up

    print("="*80)
    print("EMOTION DECAY EXPERIMENT")
    print("="*80)
    print(f"Initial: User={u_emotion}, Assistant={m_emotion}")
    print(f"Follow-up shift: User={u2_emotion}")
    print()

    # Load probes
    print("Loading emotion probes...")
    m_probes, u_probes = load_emotion_probes(probe_path)
    print(f"✓ Loaded {len(m_probes)} M probes, {len(u_probes)} U probes")
    print()

    # Load model
    print("Loading model...")
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
    print()

    # Generate conversations (both neutral and with emotion shift)
    print("Generating conversations...")

    # Neutral follow-up
    print("  [1/2] Neutral follow-up...")
    full_text_neutral, tokens_neutral = generate_conversation(
        model, tokenizer, u_emotion, m_emotion,
        user_msg="Can you help me out?",
        follow_up="What else?",
        u2_emotion=None
    )
    print(f"    ✓ Generated {len(tokens_neutral)} tokens")

    # Emotion shift follow-up
    print(f"  [2/2] Follow-up with '{u2_emotion}' shift...")
    full_text_shift, tokens_shift = generate_conversation(
        model, tokenizer, u_emotion, m_emotion,
        user_msg="Can you help me out?",
        follow_up="What else?",
        u2_emotion=u2_emotion
    )
    print(f"    ✓ Generated {len(tokens_shift)} tokens")
    print()

    # Collect activations
    print("Collecting token-level activations...")

    print("  [1/2] Neutral conversation...")
    activations_neutral = collect_token_activations(model, tokenizer, tokens_neutral, layer=30)
    print(f"    ✓ Collected activations: {activations_neutral.shape}")

    print(f"  [2/2] Emotion shift conversation...")
    activations_shift = collect_token_activations(model, tokenizer, tokens_shift, layer=30)
    print(f"    ✓ Collected activations: {activations_shift.shape}")
    print()

    # Project onto emotion probes
    print("Computing projections onto emotion probes...")

    # Select emotions to track (using Plutchik names)
    tracked_emotions = [u_emotion, u2_emotion, m_emotion, "serenity", "fear"]

    # Neutral conversation
    u_proj_neutral = project_onto_probes(activations_neutral, u_probes, tracked_emotions)
    m_proj_neutral = project_onto_probes(activations_neutral, m_probes, tracked_emotions)

    # Emotion shift conversation
    u_proj_shift = project_onto_probes(activations_shift, u_probes, tracked_emotions)
    m_proj_shift = project_onto_probes(activations_shift, m_probes, tracked_emotions)

    print("✓ Computed projections")
    print()

    # Plot results
    print("Plotting emotion decay...")

    plot_emotion_decay(
        u_proj_neutral, m_proj_neutral,
        tokens_neutral, tokenizer,
        u_emotion, m_emotion, None,
        output_dir / "emotion_decay_neutral.png"
    )

    plot_emotion_decay(
        u_proj_shift, m_proj_shift,
        tokens_shift, tokenizer,
        u_emotion, m_emotion, u2_emotion,
        output_dir / "emotion_decay_shift.png"
    )

    # Save conversation texts
    with open(output_dir / "conversations.json", 'w') as f:
        json.dump({
            "neutral": {
                "text": full_text_neutral,
                "num_tokens": len(tokens_neutral)
            },
            "emotion_shift": {
                "text": full_text_shift,
                "num_tokens": len(tokens_shift)
            },
            "parameters": {
                "u_emotion": u_emotion,
                "m_emotion": m_emotion,
                "u2_emotion": u2_emotion,
                "tracked_emotions": tracked_emotions
            }
        }, f, indent=2)

    print()
    print("="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"  - emotion_decay_neutral.png")
    print(f"  - emotion_decay_shift.png")
    print(f"  - conversations.json")


if __name__ == "__main__":
    main()
