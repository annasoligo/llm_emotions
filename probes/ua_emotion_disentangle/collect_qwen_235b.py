#!/usr/bin/env python3
"""
Collect UA emotion disentanglement activations for Qwen 235B (or other large Qwen models).

Uses HuggingFace with automatic tensor parallelism via device_map='auto'.
For 4x H200 GPUs, this should handle models up to ~500B parameters.

Usage:
    python -m probes.ua_emotion_disentangle.collect_qwen_235b \
        --model "Qwen/Qwen2.5-72B-Instruct" \
        --layers 40-80 \
        --output data/qwen235b_ua_activations.h5
"""

import argparse
import gc
import logging
import torch
import h5py
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Qwen-specific templates
QWEN_TEMPLATES = [
    "You are Qwen, a language model developed by the Qwen team at Alibaba. Qwen is {M}. Qwen is speaking to a {U} user.",
    "You are Qwen, a language model developed by the Qwen team at Alibaba. Qwen is speaking to a {U} user. Qwen is {M}.",
    "You are Qwen, a language model developed by the Qwen team at Alibaba. You are {M}. The user is {U}.",
    "You are Qwen, a language model developed by the Qwen team at Alibaba. The user is {U}. You are {M}.",
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


def get_all_emotions():
    """Get all 32 Plutchik emotions."""
    emotions = set()
    for e1, e2 in get_plutchik_emotion_pairs():
        emotions.add(e1)
        emotions.add(e2)
    return sorted(emotions)


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

    # Qwen uses <|im_start|>assistant for assistant turn
    try:
        model_turn_tokens = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    except Exception:
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
    """Generate ALL M×U emotion combinations.

    Creates full factorial design: 32 M emotions × 32 U emotions = 1024 M-U pairs.
    Uses rotating templates and user messages for variety.
    """
    all_emotions = get_all_emotions()

    combinations = []
    combo_idx = 0

    for M_name in all_emotions:
        for U_name in all_emotions:
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
    """
    Capture activations from specified layers using forward hooks.

    Works with tensor-parallel models where layers may be sharded across GPUs.
    """

    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks = []

        # Find the layers module (handles different model architectures)
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers_module = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers_module = model.transformer.h
        else:
            raise ValueError("Could not find layers in model architecture")

        for layer_idx in layers:
            if layer_idx >= len(layers_module):
                logger.warning(f"Layer {layer_idx} does not exist, skipping")
                continue
            layer = layers_module[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

        logger.info(f"Registered {len(self.hooks)} hooks for layers {min(layers)}-{max(layers)}")

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Store on CPU to save GPU memory
            self.activations[layer_idx] = hidden_states.detach().cpu()
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
    output_dir: Path,
    hidden_dim: int,
    model_name: str,
    checkpoint_every: int = 100,
    layers_per_file: int = 20,
):
    """
    Collect activations with streaming writes to HDF5.

    All layers are captured in a SINGLE forward pass per prompt.
    Output is split into multiple files to keep sizes manageable.

    Memory-efficient: writes directly to disk, clears GPU memory frequently.
    """

    num_combos = len(combinations)
    num_layers = len(layers)
    num_positions = len(positions)

    # Split layers into chunks for separate output files
    layer_chunks = []
    for i in range(0, num_layers, layers_per_file):
        chunk = layers[i:i + layers_per_file]
        layer_chunks.append(chunk)

    logger.info("=" * 80)
    logger.info("STREAMING COLLECTION (Single Forward Pass)")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Combinations: {num_combos}")
    logger.info(f"Layers: {num_layers} ({min(layers)} to {max(layers)})")
    logger.info(f"Output files: {len(layer_chunks)} (up to {layers_per_file} layers each)")
    logger.info(f"Positions: {num_positions} ({', '.join(positions)})")
    logger.info(f"Hidden dim: {hidden_dim}")
    logger.info(f"Total vectors: {num_combos * num_layers * num_positions:,}")
    expected_gb = num_combos * num_layers * num_positions * hidden_dim * 4 / 1e9
    logger.info(f"Expected size: ~{expected_gb:.1f} GB total (uncompressed)")
    logger.info("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup activation capture for ALL layers at once
    capture = ActivationCapture(model, layers)

    # Create output files and datasets for each chunk
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split('/')[-1].lower().replace('-', '_')

    output_files = {}
    datasets = {}
    metadata_datasets = {}

    for chunk_idx, chunk_layers in enumerate(layer_chunks):
        layer_min, layer_max = min(chunk_layers), max(chunk_layers)
        output_path = output_dir / f"{model_short}_ua_layers_{layer_min}_{layer_max}_{timestamp}.h5"

        logger.info(f"Creating file {chunk_idx + 1}/{len(layer_chunks)}: {output_path.name}")

        f = h5py.File(output_path, 'w')
        output_files[chunk_idx] = f

        # Store metadata
        f.attrs['model_name'] = model_name
        f.attrs['hidden_dim'] = hidden_dim
        f.attrs['num_combinations'] = num_combos
        f.attrs['num_layers'] = len(chunk_layers)
        f.attrs['layers'] = chunk_layers
        f.attrs['positions'] = positions

        # Pre-allocate datasets for this chunk
        for layer in chunk_layers:
            layer_group = f.create_group(f'layer_{layer}')

            for pos_name in positions:
                datasets[(layer, pos_name)] = layer_group.create_dataset(
                    f'{pos_name}_activations',
                    shape=(num_combos, hidden_dim),
                    dtype='float32',
                    compression='gzip',
                    compression_opts=4,
                    chunks=(min(50, num_combos), hidden_dim)
                )
                metadata_datasets[(layer, pos_name)] = []

    # Process combinations - SINGLE forward pass captures ALL layers
    logger.info(f"Collecting activations ({num_layers} layers per forward pass)...")
    for combo in tqdm(combinations, desc="Processing"):
        combo_idx = combo['combo_idx']

        system_prompt = combo['template'].format(M=combo['M'], U=combo['U'])
        full_prompt = format_prompt(tokenizer, system_prompt, combo['user_msg'])

        # Tokenize and move to model's device
        inputs = tokenizer(full_prompt, return_tensors="pt")
        # Handle multi-GPU: move to first device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        capture.clear()

        # Single forward pass - hooks capture ALL layers
        with torch.no_grad():
            _ = model(**inputs)

        pos_indices = find_position_indices(inputs['input_ids'].cpu(), tokenizer)

        # Write activations from all layers to their respective files
        for layer in layers:
            if layer not in capture.activations:
                continue

            hidden_states = capture.activations[layer]
            hidden_np = hidden_states[0].float().numpy()

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

        # Memory cleanup
        capture.clear()
        del inputs

        if combo_idx % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if combo_idx % checkpoint_every == 0 and combo_idx > 0:
            for f in output_files.values():
                f.flush()
            logger.info(f"  Checkpoint at {combo_idx}/{num_combos}")

    capture.remove_hooks()

    # Save metadata and close files
    logger.info("Saving metadata and closing files...")
    for chunk_idx, chunk_layers in enumerate(layer_chunks):
        f = output_files[chunk_idx]

        for layer in chunk_layers:
            layer_group = f[f'layer_{layer}']
            for pos_name in positions:
                metadata_list = metadata_datasets[(layer, pos_name)]
                if not metadata_list:
                    continue

                M_values = [m['M'] for m in metadata_list]
                U_values = [m['U'] for m in metadata_list]
                template_values = [m['template'] for m in metadata_list]
                user_msg_values = [m['user_msg'] for m in metadata_list]

                layer_group.create_dataset(f'{pos_name}_M', data=np.array(M_values, dtype='S50'))
                layer_group.create_dataset(f'{pos_name}_U', data=np.array(U_values, dtype='S50'))
                layer_group.create_dataset(f'{pos_name}_template', data=np.array(template_values, dtype='S200'))
                layer_group.create_dataset(f'{pos_name}_user_msg', data=np.array(user_msg_values, dtype='S100'))

        f.close()

        output_path = output_dir / f"{model_short}_ua_layers_{min(chunk_layers)}_{max(chunk_layers)}_{timestamp}.h5"
        file_size_gb = output_path.stat().st_size / 1e9
        logger.info(f"  {output_path.name}: {file_size_gb:.2f} GB")

    logger.info("All files saved successfully!")


def main():
    parser = argparse.ArgumentParser(description='Collect UA emotions for large Qwen models')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-235B-A22B',
                        help='Model name/path (default: Qwen/Qwen3-235B-A22B)')
    parser.add_argument('--layers', type=str, default='0-93',
                        help='Layer specification (e.g., "0-93" for all layers)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for HDF5 files')
    parser.add_argument('--layers-per-file', type=int, default=20,
                        help='Number of layers per output file (default: 20)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Model dtype (default: bfloat16)')
    parser.add_argument('--trust-remote-code', action='store_true', default=True,
                        help='Trust remote code (required for Qwen)')
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    positions = ['last_user_token', 'first_asst_token', 'final_token']
    output_dir = Path(args.output_dir)

    logger.info("=" * 80)
    logger.info("UA EMOTION DISENTANGLEMENT - LARGE QWEN MODEL")
    logger.info("=" * 80)

    combinations = generate_combinations()
    emotion_pairs = get_plutchik_emotion_pairs()

    logger.info(f"Emotions: {len(emotion_pairs)} pairs (32 total)")
    logger.info(f"Templates: {len(QWEN_TEMPLATES)} (Qwen-specific)")
    logger.info(f"User messages: {len(QWEN_USER_MSGS)}")
    logger.info(f"Layers: {min(layers)} to {max(layers)} ({len(layers)} layers)")
    logger.info(f"Total combinations: {len(combinations)}")

    # Load model with automatic tensor parallelism
    logger.info(f"Loading model: {args.model}...")
    logger.info("Using device_map='auto' for automatic tensor parallelism")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map='auto',  # Automatic tensor parallelism across available GPUs
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=args.trust_remote_code,
    )

    hidden_dim = model.config.hidden_size
    num_layers_model = model.config.num_hidden_layers

    logger.info(f"Model loaded successfully")
    logger.info(f"  Hidden dim: {hidden_dim}")
    logger.info(f"  Num layers: {num_layers_model}")
    logger.info(f"  Dtype: {args.dtype}")

    # Validate layer range
    if max(layers) >= num_layers_model:
        logger.warning(f"Requested layers up to {max(layers)} but model only has {num_layers_model} layers")
        layers = [l for l in layers if l < num_layers_model]
        logger.info(f"Adjusted layers: {min(layers)} to {max(layers)}")

    # Collect activations
    collect_activations_streaming(
        model=model,
        tokenizer=tokenizer,
        combinations=combinations,
        layers=layers,
        positions=positions,
        output_dir=output_dir,
        hidden_dim=hidden_dim,
        model_name=args.model,
        layers_per_file=args.layers_per_file,
    )

    logger.info("=" * 80)
    logger.info("COLLECTION COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
