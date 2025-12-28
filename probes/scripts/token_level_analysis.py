#!/usr/bin/env python3
"""
Token-level emotion analysis using orthogonal user/assistant probes.

This script extracts activations at every token position in a prompt and during
generation, then applies orthogonal probes trained on raw conversation activations
to get separate user and assistant emotion scores.

Usage:
    python scripts/token_level_analysis.py \\
        --prompt "I'm feeling really happy today!" \\
        --layer 31 \\
        --ortho-weight 100.0 \\
        --num-generate 20
"""

import argparse
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
    """Extract activations at every token position, including generated tokens.

    Args:
        model: StandardizedTransformer model
        tokenizer: Tokenizer
        prompt: User prompt text
        layer: Layer to extract from
        num_generate: Number of tokens to generate
        system_prompt: Optional system prompt
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        Tuple of:
        - token_activations: Dict mapping token_pos -> activation [hidden_dim]
        - all_token_ids: List of all token IDs (input + generated)
        - user_turn_end_pos: Position where user turn ends
    """
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
    user_turn_end_pos = input_ids.shape[1] - 1  # Last token of formatted prompt

    # Step 1: Extract activations from input tokens
    token_activations = {}

    with model.trace(input_ids, scan=False):
        layer_output = model.layers_output[layer].save()

    # Extract at all input positions
    for pos in range(input_ids.shape[1]):
        act = layer_output[0, pos, :].detach().cpu().float().numpy()
        token_activations[pos] = act

    # Step 2: Generate tokens and extract activations during generation
    print(f"Generating {num_generate} tokens...")

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

    # Extract hidden states from generation
    generated_ids = outputs.sequences[0, input_ids.shape[1]:]  # Only new tokens

    for step_idx in range(len(outputs.hidden_states)):
        step_hidden = outputs.hidden_states[step_idx]  # Tuple of layer tensors
        layer_hidden = step_hidden[layer]  # [batch, seq_len, hidden_dim]

        # Get activation at last position (the newly generated token)
        act = layer_hidden[0, -1, :].detach().cpu().float().numpy()

        # Position in full sequence
        pos = input_ids.shape[1] + step_idx
        token_activations[pos] = act

    # Get all token IDs
    all_token_ids = outputs.sequences[0].tolist()

    return token_activations, all_token_ids, user_turn_end_pos


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
        Tuple of (user_scores, assistant_scores) where each is:
        Dict mapping token_pos -> {emotion: score}
    """

    user_scores = {}
    assistant_scores = {}

    for pos, activation in token_activations.items():
        # Project activation onto probe directions (simple dot product)
        # Probes are already normalized from training
        user_proj = activation @ user_probes.T  # [n_emotions]
        asst_proj = activation @ asst_probes.T  # [n_emotions]

        # Convert to dict
        user_scores[pos] = {EMOTIONS[i]: float(user_proj[i]) for i in range(6)}
        assistant_scores[pos] = {EMOTIONS[i]: float(asst_proj[i]) for i in range(6)}

    return user_scores, assistant_scores


def plot_token_trajectories(
    user_scores: Dict[int, Dict[str, float]],
    assistant_scores: Dict[int, Dict[str, float]],
    token_ids: List[int],
    tokenizer,
    user_turn_end_pos: int,
    output_dir: Path,
    experiment_name: str = "token_analysis"
):
    """Plot emotion trajectories across tokens for user and assistant probes.

    Args:
        user_scores: User probe scores per token
        assistant_scores: Assistant probe scores per token
        token_ids: List of token IDs
        tokenizer: Tokenizer for decoding
        user_turn_end_pos: Position where user turn ends
        output_dir: Output directory
        experiment_name: Experiment name for file prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = sorted(user_scores.keys())

    # Decode tokens for labels
    token_strs = [tokenizer.decode([tid]) for tid in token_ids]

    # Create figure with two subplots (user and assistant)
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

    # Add token labels on x-axis (every 5 tokens to avoid crowding)
    tick_positions = list(range(0, len(positions), 5))
    tick_labels = [f"{pos}\n{token_strs[pos][:10]}" for pos in tick_positions]
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=8, rotation=45, ha='right')

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f'{experiment_name}_token_trajectories.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved token trajectory plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Token-level emotion analysis with orthogonal probes")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt to analyze")
    parser.add_argument("--model", type=str, default="google/gemma-2-27b-it", help="Model name or path")
    parser.add_argument("--layer", type=int, required=True, help="Layer to extract activations from")
    parser.add_argument("--ortho-weight", type=float, required=True, help="Orthogonality weight used during training")
    parser.add_argument("--representation", type=str, default="raw", choices=["raw", "global_cpca", "regional_cpca"],
                       help="Representation type (must match probe training)")
    parser.add_argument("--n-components", type=int, help="Number of components for cPCA (if applicable)")
    parser.add_argument("--num-generate", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--system-prompt", type=str, help="Optional system prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling parameter")
    parser.add_argument("--output-dir", type=str, default="results/token_analysis", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Build probe path
    probes_base_dir = Path("/workspace-vast/annas/git/research-tools/outputs/probes/emotion_probes/conversation_based/orthogonal")

    if args.representation == "raw":
        probe_filename = f"probe_layer{args.layer}_raw_ortho{args.ortho_weight}.pkl"
    elif args.representation == "global_cpca":
        probe_filename = f"probe_layer{args.layer}_global_nc{args.n_components}_ortho{args.ortho_weight}.pkl"
    else:  # regional_cpca
        probe_filename = f"probe_layer{args.layer}_regional_nc{args.n_components}_ortho{args.ortho_weight}.pkl"

    probe_path = probes_base_dir / f"ortho_{args.ortho_weight}" / probe_filename

    if not probe_path.exists():
        print(f"Error: Probe not found at {probe_path}")
        print(f"\nAvailable ortho weights:")
        for ortho_dir in sorted(probes_base_dir.glob("ortho_*")):
            ortho_val = ortho_dir.name.replace("ortho_", "")
            print(f"  {ortho_val}")
            probe_files = list(ortho_dir.glob(f"probe_layer{args.layer}_*.pkl"))
            if probe_files:
                for pf in probe_files[:3]:
                    print(f"    {pf.name}")
            else:
                print(f"    (no layer {args.layer} probes)")
        return

    print(f"Loading probe from {probe_path}")
    with open(probe_path, 'rb') as f:
        probe_data = pickle.load(f)

    # Extract probe weights (saved as numpy arrays)
    user_probes = probe_data['final_user_probes']  # [n_emotions, hidden_dim]
    asst_probes = probe_data['final_asst_probes']  # [n_emotions, hidden_dim]

    print(f"  Layer: {probe_data['layer']}")
    print(f"  Representation: {probe_data['representation']}")
    print(f"  Ortho weight: {probe_data['ortho_weight']}")
    print(f"  User probes shape: {user_probes.shape}")
    print(f"  Assistant probes shape: {asst_probes.shape}")

    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_raw = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = StandardizedTransformer(model_raw, tokenizer=tokenizer)

    # Extract token activations
    print(f"\nExtracting token-level activations for prompt:")
    print(f"  \"{args.prompt}\"")

    token_activations, token_ids, user_turn_end_pos = extract_token_activations_with_generation(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        layer=args.layer,
        num_generate=args.num_generate,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print(f"✓ Extracted {len(token_activations)} token activations")
    print(f"  Input tokens: {user_turn_end_pos + 1}")
    print(f"  Generated tokens: {len(token_activations) - user_turn_end_pos - 1}")

    # Apply orthogonal probes
    print("\nApplying orthogonal user/assistant probes...")
    user_scores, assistant_scores = apply_orthogonal_probes(
        token_activations, user_probes, asst_probes
    )

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print("\nUser probe means (across all tokens):")
    for emotion in EMOTIONS:
        scores = [user_scores[pos][emotion] for pos in sorted(user_scores.keys())]
        print(f"  {emotion:12s}: {np.mean(scores):+7.3f} (std: {np.std(scores):.3f})")

    print("\nAssistant probe means (across all tokens):")
    for emotion in EMOTIONS:
        scores = [assistant_scores[pos][emotion] for pos in sorted(assistant_scores.keys())]
        print(f"  {emotion:12s}: {np.mean(scores):+7.3f} (std: {np.std(scores):.3f})")

    # Plot trajectories
    print("\nGenerating visualizations...")
    plot_token_trajectories(
        user_scores=user_scores,
        assistant_scores=assistant_scores,
        token_ids=token_ids,
        tokenizer=tokenizer,
        user_turn_end_pos=user_turn_end_pos,
        output_dir=Path(args.output_dir),
        experiment_name=f"layer{args.layer}_ortho{args.ortho_weight}"
    )

    print(f"\n✓ Token-level analysis complete!")
    print(f"  Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
