#!/usr/bin/env python3
"""
Collect UA emotion disentanglement activations using PyTorch hooks (more robust).
Supports: Gemma-3-27B, Qwen3-32B, Llama-3.1
"""

import argparse
import torch
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# Model configurations
MODEL_CONFIGS = {
    "unsloth/gemma-3-27b-it": {
        "hidden_dim": 5376,
        "num_layers": 62,
    },
    "Qwen/Qwen3-32B": {
        "hidden_dim": 5120,
        "num_layers": 64,
    },
    "unsloth/Qwen3-32B": {
        "hidden_dim": 5120,
        "num_layers": 64,
    },
    "meta-llama/Llama-3.1-70B-Instruct": {
        "hidden_dim": 8192,
        "num_layers": 80,
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
    """Get model configuration."""
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]

    for key, config in MODEL_CONFIGS.items():
        if key.lower() in model_name.lower() or model_name.lower() in key.lower():
            print(f"  Using config from: {key}")
            return config

    return {"hidden_dim": None, "num_layers": None}


def find_position_indices(input_ids: torch.Tensor, tokenizer, model_name: str) -> Dict[str, int]:
    """Find indices for token positions of interest."""
    tokens = input_ids[0].tolist()
    seq_len = len(tokens)

    # Model-specific delimiter detection
    if "gemma" in model_name.lower():
        model_turn_tokens = tokenizer.encode("<start_of_turn>model", add_special_tokens=False)
    elif "qwen" in model_name.lower():
        model_turn_tokens = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    elif "llama" in model_name.lower():
        model_turn_tokens = tokenizer.encode("<|start_header_id|>assistant", add_special_tokens=False)
    else:
        model_turn_tokens = []

    # Find position where assistant turn starts
    asst_start_idx = seq_len - 1
    if model_turn_tokens:
        for i in range(len(tokens) - len(model_turn_tokens)):
            if tokens[i:i+len(model_turn_tokens)] == model_turn_tokens:
                asst_start_idx = i + len(model_turn_tokens)
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
        return f"{system_prompt}\n\nUser: {user_msg}\n\nAssistant:"


class ActivationCapture:
    """Capture activations from specified layers using hooks."""

    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.activations = {}
        self.hooks = []

        # Register hooks on the specified layers
        for layer_idx in layers:
            layer = model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            # output is usually a tuple, first element is hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations[layer_idx] = hidden_states.detach()
        return hook

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


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
    print(f"STREAMING COLLECTION (PyTorch Hooks)")
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

    # Setup activation capture
    capture = ActivationCapture(model, layers)

    with h5py.File(output_path, 'w') as f:
        f.attrs['model_name'] = model_name
        f.attrs['hidden_dim'] = hidden_dim

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

        print("\nCollecting activations...")
        for combo in tqdm(combinations, desc="Processing"):
            combo_idx = combo['combo_idx']

            system_prompt = combo['template'].format(M=combo['M'], U=combo['U'])
            full_prompt = format_prompt(tokenizer, system_prompt, combo['user_msg'], model_name)

            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

            # Clear previous activations
            capture.clear()

            # Forward pass to capture activations
            with torch.no_grad():
                _ = model(**inputs)

            pos_indices = find_position_indices(inputs['input_ids'], tokenizer, model_name)

            for layer in layers:
                hidden_states = capture.activations[layer]
                hidden_np = hidden_states[0].float().cpu().numpy()  # [0] for batch dim

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

            # Clean up
            capture.clear()
            del inputs

            if combo_idx % 100 == 0:
                torch.cuda.empty_cache()
                if combo_idx % 500 == 0:
                    f.flush()

        # Remove hooks
        capture.remove_hooks()

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
    parser = argparse.ArgumentParser(description='Collect UA emotion activations (hooks-based)')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name (e.g., "Qwen/Qwen3-32B")')
    parser.add_argument('--layers', type=str, default=None,
                        help='Layer specification (e.g., "22-42" or "20,30,40")')
    parser.add_argument('--output', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Model dtype')
    args = parser.parse_args()

    print(f"{'='*80}")
    print(f"UA EMOTION DISENTANGLEMENT - HOOKS-BASED COLLECTION")
    print(f"{'='*80}")
    print(f"Model: {args.model}")

    config = get_model_config(args.model)
    print(f"Config: hidden_dim={config['hidden_dim']}, num_layers={config['num_layers']}")

    if args.layers is None:
        if config['num_layers']:
            mid = config['num_layers'] // 2
            layers = list(range(max(0, mid - 10), min(config['num_layers'], mid + 11)))
        else:
            layers = list(range(20, 41))
    else:
        layers = parse_layers(args.layers)

    positions = ['last_user_token', 'first_asst_token', 'final_token']

    combinations = generate_combinations()
    emotion_pairs = get_plutchik_emotion_pairs()

    print(f"\nEmotions: {len(emotion_pairs)} pairs (32 total)")
    print(f"Templates: 8")
    print(f"User messages: 4")
    print(f"Layers: {min(layers)} to {max(layers)} ({len(layers)} layers)")
    print(f"Positions: {positions}")
    print(f"\nTotal combinations: {len(combinations)}")

    print(f"\nLoading model: {args.model}...")

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',
        torch_dtype=dtype_map[args.dtype],
    )

    # Auto-detect hidden dim if needed
    if config['hidden_dim'] is None:
        hidden_dim = model.config.hidden_size
        print(f"  Auto-detected hidden_dim: {hidden_dim}")
    else:
        hidden_dim = config['hidden_dim']

    print("✓ Model loaded")
    print(f"  Device: {next(model.parameters()).device}")

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
