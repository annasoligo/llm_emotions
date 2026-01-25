#!/usr/bin/env python3
"""
Collect UA emotion disentanglement activations for Gemma.
Adapted with Gemma/Google references.
"""

import argparse
import torch
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# Gemma-specific templates
GEMMA_TEMPLATES = [
    # "is X" framing
    "You are Gemma, a language model developed by Google. Gemma is {M}. Gemma is speaking to a {U} user.",
    "You are Gemma, a language model developed by Google. Gemma is speaking to a {U} user. Gemma is {M}.",
    "You are Gemma, a language model developed by Google. You are {M}. The user is {U}.",
    "You are Gemma, a language model developed by Google. The user is {U}. You are {M}.",
    # "is feeling X" framing
    "You are Gemma, a language model developed by Google. Gemma is feeling {M}. The user is feeling {U}.",
    "You are Gemma, a language model developed by Google. The user is feeling {U}. Gemma is feeling {M}.",
    "You are Gemma, a language model developed by Google. You are feeling {M}. The user is feeling {U}.",
    "You are Gemma, a language model developed by Google. The user is feeling {U}. You are feeling {M}.",
]

GEMMA_USER_MSGS = [
    "Hey Gemma.",
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

    # Gemma uses <start_of_turn>model
    possible_markers = [
        "<start_of_turn>model",
        "model\n",
    ]

    asst_start_idx = seq_len - 1

    for marker in possible_markers:
        try:
            marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
            if marker_tokens:
                for i in range(len(tokens) - len(marker_tokens)):
                    if tokens[i:i+len(marker_tokens)] == marker_tokens:
                        asst_start_idx = i + len(marker_tokens)
                        break
                if asst_start_idx < seq_len - 1:
                    break
        except Exception:
            continue

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
    """Generate ALL M×U emotion combinations with Gemma-specific templates.

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
            template_idx = combo_idx % len(GEMMA_TEMPLATES)
            user_msg_idx = combo_idx % len(GEMMA_USER_MSGS)

            combinations.append({
                'combo_idx': combo_idx,
                'M': M_name,
                'U': U_name,
                'template': GEMMA_TEMPLATES[template_idx],
                'user_msg': GEMMA_USER_MSGS[user_msg_idx],
            })
            combo_idx += 1

    return combinations


def format_prompt(tokenizer, system_prompt: str, user_msg: str) -> str:
    """Format prompt using Gemma's chat template."""
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{user_msg}"},
    ]

    try:
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    except Exception:
        # Fallback format
        return f"<start_of_turn>user\n{system_prompt}\n\n{user_msg}<end_of_turn>\n<start_of_turn>model\n"


class ActivationCapture:
    """Capture activations from specified layers using hooks."""

    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.activations = {}
        self.hooks = []

        # Gemma3 is multimodal: layers are at model.model.language_model.layers
        if hasattr(model.model, 'language_model'):
            layer_module = model.model.language_model.layers
        else:
            layer_module = model.model.layers

        for layer_idx in layers:
            layer = layer_module[layer_idx]
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
    model_name: str,
):
    """Collect activations with streaming writes to HDF5."""

    num_combos = len(combinations)
    num_layers = len(layers)
    num_positions = len(positions)

    print(f"\n{'='*80}")
    print(f"STREAMING COLLECTION (Gemma Templates)")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"Combinations: {num_combos}")
    print(f"Layers: {num_layers} ({min(layers)} to {max(layers)})")
    print(f"Positions: {num_positions} ({', '.join(positions)})")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Total vectors: {num_combos * num_layers * num_positions:,}")
    print(f"{'='*80}\n")

    # Show sample template
    print("Sample template (Gemma/Google):")
    print(f"  {GEMMA_TEMPLATES[0][:80]}...")
    print()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = ActivationCapture(model, layers)

    with h5py.File(output_path, 'w') as f:
        f.attrs['model_name'] = model_name
        f.attrs['hidden_dim'] = hidden_dim
        f.attrs['templates_corrected'] = True
        f.attrs['organization'] = 'Google'

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
    parser = argparse.ArgumentParser(description='Collect UA emotions for Gemma')
    parser.add_argument('--model', type=str, default='google/gemma-3-27b-it',
                        help='Model name/path')
    parser.add_argument('--layers', type=str, default='20-40',
                        help='Layer specification (Gemma-3-27B has 62 layers)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output HDF5 file path')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Model dtype')
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    positions = ['last_user_token', 'first_asst_token', 'final_token']

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }

    print(f"{'='*80}")
    print(f"UA EMOTION DISENTANGLEMENT - Gemma")
    print(f"{'='*80}")

    combinations = generate_combinations()
    emotion_pairs = get_plutchik_emotion_pairs()

    print(f"\nEmotions: {len(emotion_pairs)} pairs (32 total)")
    print(f"Templates: {len(GEMMA_TEMPLATES)} (Gemma/Google-specific)")
    print(f"User messages: {len(GEMMA_USER_MSGS)}")
    print(f"Layers: {min(layers)} to {max(layers)} ({len(layers)} layers)")
    print(f"\nTotal combinations: {len(combinations)}")

    print(f"\nLoading model: {args.model}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',
        torch_dtype=dtype_map[args.dtype],
    )

    # Gemma3 is multimodal so config is nested under text_config
    if hasattr(model.config, 'text_config'):
        hidden_dim = model.config.text_config.hidden_size
        num_layers = model.config.text_config.num_hidden_layers
    else:
        hidden_dim = model.config.hidden_size
        num_layers = model.config.num_hidden_layers
    print(f"✓ Model loaded (hidden_dim={hidden_dim}, num_layers={num_layers})")

    collect_activations_streaming(
        model=model,
        tokenizer=tokenizer,
        combinations=combinations,
        layers=layers,
        positions=positions,
        output_path=Path(args.output),
        hidden_dim=hidden_dim,
        model_name=args.model,
    )

    print(f"\n{'='*80}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
