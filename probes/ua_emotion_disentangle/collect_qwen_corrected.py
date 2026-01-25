#!/usr/bin/env python3
"""
Collect UA emotion disentanglement activations for Qwen with CORRECTED templates.
Replaces Gemma/Google references with Qwen/Alibaba.
"""

import argparse
import torch
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# Qwen-specific templates (corrected from Gemma originals)
QWEN_TEMPLATES = [
    # "is X" framing
    "You are Qwen, a language model developed by the Qwen team at Alibaba. Qwen is {M}. Qwen is speaking to a {U} user.",
    "You are Qwen, a language model developed by the Qwen team at Alibaba. Qwen is speaking to a {U} user. Qwen is {M}.",
    "You are Qwen, a language model developed by the Qwen team at Alibaba. You are {M}. The user is {U}.",
    "You are Qwen, a language model developed by the Qwen team at Alibaba. The user is {U}. You are {M}.",
    # "is feeling X" framing
    "You are Qwen, a language model developed by the Qwen team at Alibaba. Qwen is feeling {M}. The user is feeling {U}.",
    "You are Qwen, a language model developed by the Qwen team at Alibaba. The user is feeling {U}. Qwen is feeling {M}.",
    "You are Qwen, a language model developed by the Qwen team at Alibaba. You are feeling {M}. The user is feeling {U}.",
    "You are Qwen, a language model developed by the Qwen team at Alibaba. The user is feeling {U}. You are feeling {M}.",
]

QWEN_USER_MSGS = [
    "Hey Qwen.",
    "Guess what happened to me today.",
    "Can you help me out pls?",
    "Can you tell me a story?",
]


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
    seq_len = len(tokens)

    # Qwen uses <|im_start|>assistant
    model_turn_tokens = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)

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


def get_all_emotions():
    """Get all 32 Plutchik emotions."""
    emotions = set()
    for e1, e2 in get_plutchik_emotion_pairs():
        emotions.add(e1)
        emotions.add(e2)
    return sorted(emotions)


def generate_combinations():
    """Generate ALL M×U emotion combinations with Qwen-specific templates.

    Creates full factorial design: 32 M emotions × 32 U emotions = 1024 M-U pairs.
    With 8 templates × 4 user messages = 32 variations per pair.
    Total: 1024 × 32 = 32,768 combinations.

    For efficiency, we use 1 template and 1 user message per M-U pair.
    Total: 1024 combinations.
    """
    all_emotions = get_all_emotions()

    combinations = []
    combo_idx = 0

    # Generate ALL M × U combinations
    for M_name in all_emotions:
        for U_name in all_emotions:
            # Use rotating template and user message for variety
            template_idx = combo_idx % len(QWEN_TEMPLATES)
            user_msg_idx = combo_idx % len(QWEN_USER_MSGS)

            combinations.append({
                'combo_idx': combo_idx,
                'M': M_name,
                'U': U_name,
                'template': QWEN_TEMPLATES[template_idx],
                'user_msg': QWEN_USER_MSGS[user_msg_idx],
            })
            combo_idx += 1

    return combinations


def format_prompt(tokenizer, system_prompt: str, user_msg: str) -> str:
    """Format prompt using Qwen's chat template."""
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

        for layer_idx in layers:
            layer = model.model.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
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
    print(f"STREAMING COLLECTION (Qwen-Corrected Templates)")
    print(f"{'='*80}")
    print(f"Model: Qwen/Qwen3-32B")
    print(f"Combinations: {num_combos}")
    print(f"Layers: {num_layers} ({min(layers)} to {max(layers)})")
    print(f"Positions: {num_positions} ({', '.join(positions)})")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Total vectors: {num_combos * num_layers * num_positions:,}")
    print(f"{'='*80}\n")

    # Show sample template
    print("Sample template (corrected for Qwen):")
    print(f"  {QWEN_TEMPLATES[0][:80]}...")
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = ActivationCapture(model, layers)

    with h5py.File(output_path, 'w') as f:
        f.attrs['model_name'] = 'Qwen/Qwen3-32B'
        f.attrs['hidden_dim'] = hidden_dim
        f.attrs['templates_corrected'] = True  # Flag that these are corrected

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
            full_prompt = format_prompt(tokenizer, system_prompt, combo['user_msg'])

            inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

            capture.clear()

            with torch.no_grad():
                _ = model(**inputs)

            pos_indices = find_position_indices(inputs['input_ids'], tokenizer)

            for layer in layers:
                hidden_states = capture.activations[layer]
                hidden_np = hidden_states[0].float().cpu().numpy()

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

            capture.clear()
            del inputs

            if combo_idx % 100 == 0:
                torch.cuda.empty_cache()
                if combo_idx % 500 == 0:
                    f.flush()

        capture.remove_hooks()

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
    parser = argparse.ArgumentParser(description='Collect UA emotions for Qwen (corrected)')
    parser.add_argument('--layers', type=str, default='22-42',
                        help='Layer specification')
    parser.add_argument('--output', type=str, required=True,
                        help='Output HDF5 file path')
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    positions = ['last_user_token', 'first_asst_token', 'final_token']

    print(f"{'='*80}")
    print(f"UA EMOTION DISENTANGLEMENT - QWEN CORRECTED")
    print(f"{'='*80}")

    combinations = generate_combinations()
    emotion_pairs = get_plutchik_emotion_pairs()

    print(f"\nEmotions: {len(emotion_pairs)} pairs (32 total)")
    print(f"Templates: {len(QWEN_TEMPLATES)} (Qwen-specific)")
    print(f"User messages: {len(QWEN_USER_MSGS)}")
    print(f"Layers: {min(layers)} to {max(layers)} ({len(layers)} layers)")
    print(f"\nTotal combinations: {len(combinations)}")

    print(f"\nLoading model: Qwen/Qwen3-32B...")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-32B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-32B",
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )

    hidden_dim = model.config.hidden_size
    print(f"✓ Model loaded (hidden_dim={hidden_dim})")

    collect_activations_streaming(
        model=model,
        tokenizer=tokenizer,
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
