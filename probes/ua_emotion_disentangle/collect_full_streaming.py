#!/usr/bin/env python3
"""
Collect activations at all layers with streaming HDF5 writes.
Memory-efficient version that writes directly to disk without accumulation.
"""

import argparse
import torch
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

def get_plutchik_emotion_pairs():
    """Get 16 Plutchik emotion pairs (opposites)."""
    return [
        ('joy', 'sadness'),
        ('trust', 'disgust'),
        ('fear', 'anger'),
        ('surprise', 'anticipation'),
        ('ecstasy', 'grief'),
        ('admiration', 'loathing'),
        ('terror', 'rage'),
        ('amazement', 'vigilance'),
        ('serenity', 'pensiveness'),
        ('acceptance', 'boredom'),
        ('apprehension', 'annoyance'),
        ('distraction', 'interest'),
        ('awe', 'contempt'),
        ('submission', 'aggressiveness'),
        ('love', 'remorse'),
        ('optimism', 'disappointment'),
    ]


def parse_layers(layer_spec: str) -> List[int]:
    """Parse layer specification like '0-61' or '20,30,40'."""
    if '-' in layer_spec:
        start, end = map(int, layer_spec.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(x.strip()) for x in layer_spec.split(',')]


def find_position_indices(input_ids: torch.Tensor, tokenizer) -> Dict[str, int]:
    """Find indices for token positions of interest."""
    tokens = input_ids[0].tolist()

    # Find delimiter tokens (assuming <start_of_turn>model)
    bos_token_id = tokenizer.bos_token_id

    # Simple heuristic: last user token is before the model turn
    # This needs to match your specific tokenizer format

    return {
        'last_user_token': len(tokens) - 2,  # Approximate
        'first_asst_token': len(tokens) - 1,
        'delimiter_avg': len(tokens) - 1,  # Placeholder
    }


def extract_activations_at_positions(
    hidden_states: np.ndarray,
    pos_indices: Dict[str, int],
    positions: List[str]
) -> Dict[str, np.ndarray]:
    """Extract activation vectors at specified positions."""
    extracted = {}
    for pos_name in positions:
        if pos_name in pos_indices:
            idx = pos_indices[pos_name]
            extracted[pos_name] = hidden_states[idx]  # [hidden_dim]
    return extracted


def generate_combinations():
    """Generate all emotion pair combinations."""
    emotion_pairs = get_plutchik_emotion_pairs()

    templates = [
        "You are feeling {M}. Respond to the user who is feeling {U}.",
        "You are experiencing {M}. Help the user who is experiencing {U}.",
        "Your emotional state is {M}. The user's emotional state is {U}.",
        "You feel {M}. The user feels {U}. Respond appropriately.",
        "Assistant emotion: {M}. User emotion: {U}. Engage in conversation.",
        "Your mood: {M}. User's mood: {U}. Provide a helpful response.",
        "You are {M}, they are {U}. Continue the dialogue.",
        "Emotional context - You: {M}, User: {U}. Respond naturally.",
    ]

    user_messages = [
        "Hello, how are you?",
        "Can you help me with something?",
        "I need your advice.",
        "What do you think about this?",
    ]

    combinations = []
    combo_idx = 0
    for M_name, U_name in emotion_pairs:
        for template in templates:
            for user_msg in user_messages:
                combinations.append({
                    'combo_idx': combo_idx,
                    'M': M_name,
                    'U': U_name,
                    'template': template,
                    'user_msg': user_msg,
                })
                combo_idx += 1

    return combinations


def collect_activations_streaming(
    model,
    tokenizer,
    combinations: List[Dict],
    layers: List[int],
    positions: List[str],
    output_path: Path,
    hidden_dim: int = 5120
):
    """Collect activations with streaming writes to HDF5."""

    num_combos = len(combinations)
    num_layers = len(layers)
    num_positions = len(positions)

    print(f"\n{'='*80}")
    print(f"STREAMING COLLECTION")
    print(f"{'='*80}")
    print(f"Combinations: {num_combos}")
    print(f"Layers: {num_layers} ({min(layers)} to {max(layers)})")
    print(f"Positions: {num_positions} ({', '.join(positions)})")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Total vectors: {num_combos * num_layers * num_positions:,}")
    print(f"Expected size: ~{num_combos * num_layers * num_positions * hidden_dim * 4 / 1e9:.1f} GB raw")
    print(f"{'='*80}\n")

    # Create HDF5 file with pre-allocated datasets
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Create datasets for each layer-position combination
        datasets = {}
        metadata_datasets = {}

        print("Pre-allocating HDF5 datasets...")
        for layer in tqdm(layers, desc="Creating datasets"):
            layer_group = f.create_group(f'layer_{layer}')

            for pos_name in positions:
                # Pre-allocate dataset with compression
                datasets[(layer, pos_name)] = layer_group.create_dataset(
                    f'{pos_name}_activations',
                    shape=(num_combos, hidden_dim),
                    dtype='float32',
                    compression='gzip',
                    compression_opts=4,
                    chunks=(min(100, num_combos), hidden_dim)
                )

                # Metadata as attributes (smaller, stored separately)
                metadata_datasets[(layer, pos_name)] = []

        # Process combinations and write directly to disk
        print("\nCollecting activations...")
        for combo in tqdm(combinations, desc="Processing combinations"):
            combo_idx = combo['combo_idx']

            # Format system prompt
            system_prompt = combo['template'].format(M=combo['M'], U=combo['U'])
            user_msg = combo['user_msg']

            # Create chat format
            messages = [{"role": "user", "content": user_msg}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Prepend system prompt
            full_prompt = system_prompt + "\n\n" + formatted

            # Tokenize
            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

            # Extract activations at ALL layers in single forward pass
            layer_outputs = {}
            with torch.no_grad():
                with model.trace(inputs, scan=False):
                    for layer in layers:
                        layer_outputs[layer] = model.layers_output[layer].save()

            # Find token positions
            pos_indices = find_position_indices(inputs['input_ids'], tokenizer)

            # Write each layer's activations directly to disk
            for layer in layers:
                hidden_states = layer_outputs[layer][0]
                hidden_np = hidden_states.float().cpu().numpy()

                # Extract at each position
                extracted = extract_activations_at_positions(hidden_np, pos_indices, positions)

                # Write directly to HDF5
                for pos_name, activation in extracted.items():
                    datasets[(layer, pos_name)][combo_idx] = activation

                    # Store metadata
                    metadata_datasets[(layer, pos_name)].append({
                        'M': combo['M'],
                        'U': combo['U'],
                        'template': combo['template'],
                        'user_msg': combo['user_msg'],
                    })

            # Clean up GPU memory
            del layer_outputs
            del inputs

            # Periodic cache clearing
            if combo_idx % 100 == 0:
                torch.cuda.empty_cache()

                # Flush to disk periodically
                if combo_idx % 1000 == 0:
                    f.flush()

        # Save metadata
        print("\nSaving metadata...")
        for layer in tqdm(layers, desc="Writing metadata"):
            layer_group = f[f'layer_{layer}']
            for pos_name in positions:
                metadata_list = metadata_datasets[(layer, pos_name)]

                # Store as separate datasets (more efficient than attributes for large data)
                M_values = [m['M'] for m in metadata_list]
                U_values = [m['U'] for m in metadata_list]
                template_values = [m['template'] for m in metadata_list]
                user_msg_values = [m['user_msg'] for m in metadata_list]

                layer_group.create_dataset(
                    f'{pos_name}_M',
                    data=np.array(M_values, dtype='S50')
                )
                layer_group.create_dataset(
                    f'{pos_name}_U',
                    data=np.array(U_values, dtype='S50')
                )
                layer_group.create_dataset(
                    f'{pos_name}_template',
                    data=np.array(template_values, dtype='S200')
                )
                layer_group.create_dataset(
                    f'{pos_name}_user_msg',
                    data=np.array(user_msg_values, dtype='S100')
                )

        # Store global metadata
        f.attrs['num_combinations'] = num_combos
        f.attrs['num_layers'] = num_layers
        f.attrs['layers'] = layers
        f.attrs['positions'] = positions
        f.attrs['hidden_dim'] = hidden_dim

    print(f"\n✓ Saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description='Collect UA emotion activations (streaming)')
    parser.add_argument('--layers', type=str, default='0-61',
                        help='Layer specification (e.g., "0-61" or "20,30,40")')
    parser.add_argument('--output', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--model-name', type=str, default='unsloth/gemma-3-27b-it',
                        help='Model name')
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    output_path = Path(args.output)

    positions = ['last_user_token', 'first_asst_token', 'delimiter_avg']

    print(f"{'='*80}")
    print(f"UA EMOTION DISENTANGLEMENT - STREAMING COLLECTION")
    print(f"{'='*80}")

    # Generate combinations
    combinations = generate_combinations()
    emotion_pairs = get_plutchik_emotion_pairs()
    num_templates = 8
    num_user_msgs = 4

    print(f"Emotions: {len(emotion_pairs)} pairs")
    print(f"Templates: {num_templates}")
    print(f"User messages: {num_user_msgs}")
    print(f"Layers: {layers}")
    print(f"Positions: {positions}")
    print(f"\nExpected combinations: {len(combinations)}")
    print(f"Expected vectors per layer: {len(combinations) * len(positions)}")
    print(f"Expected vectors total: {len(combinations) * len(positions) * len(layers)}")

    # Load model
    print(f"Loading model: {args.model_name}...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from nnsight import LanguageModel

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = LanguageModel(args.model_name, device_map='auto', torch_dtype=torch.bfloat16)
    print("✓ Model loaded")

    # Collect with streaming writes
    collect_activations_streaming(
        model=model,
        tokenizer=tokenizer,
        combinations=combinations,
        layers=layers,
        positions=positions,
        output_path=output_path
    )

    print(f"\n{'='*80}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
