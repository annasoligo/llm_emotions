#!/usr/bin/env python3
"""
Collect UA emotion disentanglement activations for multiple models.
Supports: Gemma-3-27B, Qwen3-32B, Llama-3.1 (quantized)
"""

import argparse
import torch
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm


# Model configurations
MODEL_CONFIGS = {
    "unsloth/gemma-3-27b-it": {
        "hidden_dim": 5376,
        "num_layers": 62,
        "layer_attr": "model.layers",
    },
    "Qwen/Qwen3-32B": {
        "hidden_dim": 5120,
        "num_layers": 64,
        "layer_attr": "model.layers",
    },
    "unsloth/Qwen3-32B": {
        "hidden_dim": 5120,
        "num_layers": 64,
        "layer_attr": "model.layers",
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "hidden_dim": 8192,
        "num_layers": 80,
        "layer_attr": "model.layers",
    },
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit": {
        "hidden_dim": 8192,
        "num_layers": 80,
        "layer_attr": "model.layers",
    },
}


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


def get_model_config(model_name: str) -> Dict:
    """Get model configuration, auto-detecting if not in registry."""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]

    # Try partial match
    for key, config in MODEL_CONFIGS.items():
        if key.lower() in model_name.lower() or model_name.lower() in key.lower():
            print(f"  Using config from: {key}")
            return config

    # Default config - will be overridden by actual model
    print(f"  Warning: Unknown model {model_name}, will auto-detect config")
    return {
        "hidden_dim": None,  # Will detect
        "num_layers": None,  # Will detect
        "layer_attr": "model.layers",
    }


def detect_model_architecture(model) -> Tuple[int, int]:
    """Auto-detect hidden_dim and num_layers from loaded model."""
    # Try to get config from model
    if hasattr(model, 'config'):
        config = model.config
        hidden_dim = getattr(config, 'hidden_size', None)
        num_layers = getattr(config, 'num_hidden_layers', None)
        if hidden_dim and num_layers:
            return hidden_dim, num_layers

    # Fallback: inspect model structure
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
        # Get hidden dim from first layer
        first_layer = model.model.layers[0]
        if hasattr(first_layer, 'hidden_size'):
            hidden_dim = first_layer.hidden_size
        elif hasattr(first_layer, 'self_attn'):
            # Try to get from attention
            attn = first_layer.self_attn
            if hasattr(attn, 'hidden_size'):
                hidden_dim = attn.hidden_size
            elif hasattr(attn, 'q_proj'):
                hidden_dim = attn.q_proj.in_features
        return hidden_dim, num_layers

    raise ValueError("Could not detect model architecture")


def find_position_indices(input_ids: torch.Tensor, tokenizer, model_name: str) -> Dict[str, int]:
    """Find indices for token positions of interest."""
    tokens = input_ids[0].tolist()
    seq_len = len(tokens)

    # Model-specific delimiter detection
    if "gemma" in model_name.lower():
        # Gemma uses <start_of_turn>model
        model_turn_token = tokenizer.encode("<start_of_turn>model", add_special_tokens=False)
    elif "qwen" in model_name.lower():
        # Qwen uses <|im_start|>assistant
        model_turn_token = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    elif "llama" in model_name.lower():
        # Llama uses [INST] format or similar
        model_turn_token = tokenizer.encode("<|start_header_id|>assistant", add_special_tokens=False)
    else:
        model_turn_token = []

    # Find position where assistant turn starts
    asst_start_idx = seq_len - 1  # Default to last token
    if model_turn_token:
        for i in range(len(tokens) - len(model_turn_token)):
            if tokens[i:i+len(model_turn_token)] == model_turn_token:
                asst_start_idx = i + len(model_turn_token)
                break

    return {
        'last_user_token': max(0, asst_start_idx - 1),
        'first_asst_token': min(asst_start_idx, seq_len - 1),
        'final_token': seq_len - 1,
    }


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


def format_prompt(tokenizer, system_prompt: str, user_msg: str, model_name: str) -> str:
    """Format prompt using model's chat template."""
    # Try using the built-in chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]

    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    except Exception:
        # Fallback: simple concatenation
        return f"{system_prompt}\n\nUser: {user_msg}\n\nAssistant:"


def collect_activations_streaming(
    model,
    tokenizer,
    model_name: str,
    combinations: List[Dict],
    layers: List[int],
    positions: List[str],
    output_path: Path,
    hidden_dim: int,
):
    """Collect activations with streaming writes to HDF5."""

    num_combos = len(combinations)
    num_layers = len(layers)
    num_positions = len(positions)

    print(f"\n{'='*80}")
    print(f"STREAMING COLLECTION")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Combinations: {num_combos}")
    print(f"Layers: {num_layers} ({min(layers)} to {max(layers)})")
    print(f"Positions: {num_positions} ({', '.join(positions)})")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Total vectors: {num_combos * num_layers * num_positions:,}")
    print(f"Expected size: ~{num_combos * num_layers * num_positions * hidden_dim * 4 / 1e9:.1f} GB raw")
    print(f"{'='*80}\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Store model info
        f.attrs['model_name'] = model_name
        f.attrs['hidden_dim'] = hidden_dim

        # Create datasets for each layer-position combination
        datasets = {}
        metadata_datasets = {}

        print("Pre-allocating HDF5 datasets...")
        for layer in tqdm(layers, desc="Creating datasets"):
            layer_group = f.create_group(f'layer_{layer}')

            for pos_name in positions:
                datasets[(layer, pos_name)] = layer_group.create_dataset(
                    f'{pos_name}_activations',
                    shape=(num_combos, hidden_dim),
                    dtype='float32',
                    compression='gzip',
                    compression_opts=4,
                    chunks=(min(100, num_combos), hidden_dim)
                )
                metadata_datasets[(layer, pos_name)] = []

        # Process combinations
        print("\nCollecting activations...")
        for combo in tqdm(combinations, desc="Processing"):
            combo_idx = combo['combo_idx']

            system_prompt = combo['template'].format(M=combo['M'], U=combo['U'])
            full_prompt = format_prompt(tokenizer, system_prompt, combo['user_msg'], model_name)

            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

            # Extract activations using nnsight
            layer_outputs = {}
            with torch.no_grad():
                with model.trace(inputs, scan=False):
                    for layer in layers:
                        layer_outputs[layer] = model.layers_output[layer].save()

            pos_indices = find_position_indices(inputs['input_ids'], tokenizer, model_name)

            for layer in layers:
                hidden_states = layer_outputs[layer][0]  # [0] to extract tensor from saved proxy
                hidden_np = hidden_states.float().cpu().numpy()

                for pos_name in positions:
                    if pos_name in pos_indices:
                        idx = pos_indices[pos_name]
                        activation = hidden_np[idx]
                        datasets[(layer, pos_name)][combo_idx] = activation

                        metadata_datasets[(layer, pos_name)].append({
                            'M': combo['M'],
                            'U': combo['U'],
                            'template': combo['template'],
                            'user_msg': combo['user_msg'],
                        })

            del layer_outputs
            del inputs

            if combo_idx % 100 == 0:
                torch.cuda.empty_cache()
                if combo_idx % 500 == 0:
                    f.flush()

        # Save metadata
        print("\nSaving metadata...")
        for layer in tqdm(layers, desc="Writing metadata"):
            layer_group = f[f'layer_{layer}']
            for pos_name in positions:
                metadata_list = metadata_datasets[(layer, pos_name)]

                M_values = [m['M'] for m in metadata_list]
                U_values = [m['U'] for m in metadata_list]
                template_values = [m['template'] for m in metadata_list]
                user_msg_values = [m['user_msg'] for m in metadata_list]

                layer_group.create_dataset(f'{pos_name}_M', data=np.array(M_values, dtype='S50'))
                layer_group.create_dataset(f'{pos_name}_U', data=np.array(U_values, dtype='S50'))
                layer_group.create_dataset(f'{pos_name}_template', data=np.array(template_values, dtype='S200'))
                layer_group.create_dataset(f'{pos_name}_user_msg', data=np.array(user_msg_values, dtype='S100'))

        f.attrs['num_combinations'] = num_combos
        f.attrs['num_layers'] = num_layers
        f.attrs['layers'] = layers
        f.attrs['positions'] = positions

    print(f"\n✓ Saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e9:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description='Collect UA emotion activations (multi-model)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., "unsloth/Qwen3-32B")')
    parser.add_argument('--layers', type=str, default=None,
                        help='Layer specification (e.g., "0-61" or "20,30,40"). Default: 20-40')
    parser.add_argument('--output', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Model dtype')
    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"UA EMOTION DISENTANGLEMENT - MULTI-MODEL COLLECTION")
    print(f"{'='*80}")
    print(f"Model: {args.model}")

    # Get model config
    config = get_model_config(args.model)
    print(f"Config: hidden_dim={config['hidden_dim']}, num_layers={config['num_layers']}")

    # Default layers based on model
    if args.layers is None:
        if config['num_layers']:
            # Use middle layers (typically most informative)
            mid = config['num_layers'] // 2
            layers = list(range(max(0, mid - 10), min(config['num_layers'], mid + 11)))
        else:
            layers = list(range(20, 41))
    else:
        layers = parse_layers(args.layers)

    positions = ['last_user_token', 'first_asst_token', 'final_token']

    # Generate combinations
    combinations = generate_combinations()
    emotion_pairs = get_plutchik_emotion_pairs()

    print(f"\nEmotions: {len(emotion_pairs)} pairs (32 total)")
    print(f"Templates: 8")
    print(f"User messages: 4")
    print(f"Layers: {min(layers)} to {max(layers)} ({len(layers)} layers)")
    print(f"Positions: {positions}")
    print(f"\nTotal combinations: {len(combinations)}")

    # Load model
    print(f"\nLoading model: {args.model}...")
    from nnsight import LanguageModel

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }

    model = LanguageModel(
        args.model,
        device_map='cuda',
        torch_dtype=dtype_map[args.dtype],
    )
    tokenizer = model.tokenizer

    # Auto-detect architecture if needed
    if config['hidden_dim'] is None:
        hidden_dim, num_layers = detect_model_architecture(model)
        print(f"  Auto-detected: hidden_dim={hidden_dim}, num_layers={num_layers}")
    else:
        hidden_dim = config['hidden_dim']

    print("✓ Model loaded")

    # Collect
    collect_activations_streaming(
        model=model,
        tokenizer=tokenizer,
        model_name=args.model,
        combinations=combinations,
        layers=layers,
        positions=positions,
        output_path=Path(args.output),
        hidden_dim=hidden_dim,
    )

    print(f"\n{'='*80}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
