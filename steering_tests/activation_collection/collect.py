#!/usr/bin/env python3
"""
Unified activation collection supporting all models and datasets.

Features:
- Multi-GPU support via PyTorch hooks (most robust)
- Two modes: chat-tokenized (prompts) and raw text (pairs)
- Pickle storage with metadata sidecar files
- Resume capability
- Memory efficient streaming collection

Usage:
    # For prompt datasets (chat tokenized)
    python activation_collection/collect.py \
        --input steering_tests/data/emotion_prompts_MODEL_500.jsonl \
        --output activations/emotion_prompts_gemma2_9b.pkl \
        --model google/gemma-2-9b-it \
        --mode chat \
        --layers 20-40

    # For text pair datasets (no chat template)
    python activation_collection/collect.py \
        --input steering_tests/data/emotion_text_pairs_24_full_500.jsonl \
        --output activations/emotion_pairs_gemma2_9b.pkl \
        --model google/gemma-2-9b-it \
        --mode text \
        --layers 20-40
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import torch
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from steering_tests.steering_utils.provenance import get_provenance
except ImportError:
    # Fallback if run without PYTHONPATH set to repo root
    def get_provenance(**kwargs):
        import subprocess
        from datetime import datetime, timezone
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short=10", "HEAD"],
                text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            commit = None
        meta = {"git_commit": commit, "timestamp": datetime.now(timezone.utc).isoformat()}
        if kwargs.get("script"):
            meta["script"] = str(kwargs["script"])
        return meta


@dataclass
class CollectionMetadata:
    """Metadata for an activation collection run."""
    model_name: str
    mode: str  # 'chat' or 'text'
    input_file: str
    output_file: str
    num_items: int
    num_layers: int
    layers: List[int]
    hidden_dim: int
    representations: List[str]
    start_token: int  # For text mode
    timestamp: str
    dtype: str
    total_activations: int


class ActivationCapture:
    """Capture activations using PyTorch hooks (robust for multi-GPU)."""

    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.activations = {}
        self.hooks = []

        # Find where layers are stored (different architectures use different names)
        layers_module = self._get_layers_module(model)

        # Register hooks
        for layer_idx in layers:
            layer = layers_module[layer_idx]
            hook = layer.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(hook)

    def _get_layers_module(self, model):
        """Find the layers module in the model architecture."""
        # Try common locations
        # Multimodal models (Gemma-3, etc) - check language_model.layers first
        if hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
            return model.language_model.layers
        # Also try language_model.model.layers
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
            return model.language_model.model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            return model.transformer.h
        elif hasattr(model, 'layers'):
            return model.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            return model.model.decoder.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'encoder') and hasattr(model.model.encoder, 'layer'):
            return model.model.encoder.layer
        else:
            raise ValueError(f"Could not find layers in model. Available attributes: {dir(model)}")

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Immediately move to CPU and detach to prevent GPU memory issues
            self.activations[layer_idx] = hidden_states.detach().cpu()
        return hook

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def parse_layers(layer_spec: str, num_layers: int) -> List[int]:
    """Parse layer specification like '0-61', '20,30,40', or 'all'."""
    if layer_spec == 'all':
        return list(range(num_layers))
    elif '-' in layer_spec:
        start, end = map(int, layer_spec.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(x.strip()) for x in layer_spec.split(',')]


def find_special_token_indices(token_ids: torch.Tensor, tokenizer, model_name: str) -> List[int]:
    """Find indices of special tokens (turn boundaries)."""
    tokens = token_ids[0].tolist()
    special_indices = []

    # Model-specific special tokens
    patterns = []
    if "gemma" in model_name.lower():
        patterns = ['<start_of_turn>', '<end_of_turn>']
    elif "qwen" in model_name.lower():
        patterns = ['<|im_start|>', '<|im_end|>']
    elif "llama" in model_name.lower():
        patterns = ['<|start_header_id|>', '<|end_header_id|>']

    for pattern in patterns:
        try:
            token_id = tokenizer.convert_tokens_to_ids(pattern)
            if token_id != tokenizer.unk_token_id:
                indices = [i for i, t in enumerate(tokens) if t == token_id]
                special_indices.extend(indices)
        except:
            continue

    return sorted(set(special_indices))


def collect_chat_activations(
    capture: ActivationCapture,
    tokenizer,
    model,
    text: str,
    model_name: str,
) -> Dict[str, np.ndarray]:
    """
    Collect activations for chat-tokenized prompt.

    Returns:
        - 'last_token_before_generation': [num_layers, hidden_dim]
        - 'special_tokens_mean': [num_layers, hidden_dim]
    """
    # Apply chat template
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    token_ids = inputs["input_ids"]

    # Forward pass
    capture.clear()
    with torch.no_grad():
        _ = model(**inputs)

    # Extract activations
    result = {}

    # Last token before generation (most important for prompts)
    for layer_idx in capture.layers:
        if layer_idx not in result:
            result[layer_idx] = {}
        hidden = capture.activations[layer_idx][0]  # [seq_len, hidden_dim]
        result[layer_idx]['last_token'] = hidden[-1].float().numpy()

    # Mean over special tokens
    special_indices = find_special_token_indices(token_ids, tokenizer, model_name)
    if special_indices:
        for layer_idx in capture.layers:
            hidden = capture.activations[layer_idx][0]
            special_acts = hidden[special_indices]  # [num_special, hidden_dim]
            result[layer_idx]['special_mean'] = special_acts.mean(dim=0).float().numpy()
    else:
        # Fallback: use last token
        for layer_idx in capture.layers:
            result[layer_idx]['special_mean'] = result[layer_idx]['last_token']

    return result


def collect_chat_activations_batch(
    capture: ActivationCapture,
    tokenizer,
    model,
    texts: List[str],
    model_name: str,
) -> List[Dict[str, np.ndarray]]:
    """
    Collect activations for multiple chat-tokenized prompts in a single batch.

    Returns:
        List of dicts, one per sample, each containing:
        - 'last_token': activation at last token before generation
        - 'special_mean': mean activation over special tokens
    """
    if not texts:
        return []

    # Apply chat template to all texts
    formatted_texts = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_texts.append(formatted)

    # Tokenize batch with padding
    inputs = tokenizer(
        formatted_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    # Forward pass
    capture.clear()
    with torch.no_grad():
        _ = model(**inputs)

    # Extract activations per sample
    results = []
    batch_size = inputs["input_ids"].shape[0]

    for sample_idx in range(batch_size):
        result = {}
        # Find last non-padding token position
        attention_mask = inputs["attention_mask"][sample_idx]
        last_token_pos = attention_mask.sum().item() - 1

        for layer_idx in capture.layers:
            if layer_idx not in result:
                result[layer_idx] = {}

            hidden = capture.activations[layer_idx][sample_idx]  # [seq_len, hidden_dim]
            result[layer_idx]['last_token'] = hidden[last_token_pos].float().cpu().numpy()

            # Mean over special tokens for this sample
            token_ids = inputs["input_ids"][sample_idx:sample_idx+1]
            special_indices = find_special_token_indices(token_ids, tokenizer, model_name)

            if special_indices:
                special_acts = hidden[special_indices]
                result[layer_idx]['special_mean'] = special_acts.mean(dim=0).float().cpu().numpy()
            else:
                result[layer_idx]['special_mean'] = result[layer_idx]['last_token']

        results.append(result)

    return results


def collect_text_activations(
    capture: ActivationCapture,
    tokenizer,
    model,
    text: str,
    start_token: int = 20,
) -> Optional[np.ndarray]:
    """
    Collect activations for raw text (no chat template).

    Returns:
        [num_layers, hidden_dim] averaged from start_token onwards, or None if too short
    """
    # Tokenize WITHOUT chat template
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]

    if seq_len <= start_token:
        return None

    # Forward pass
    capture.clear()
    with torch.no_grad():
        _ = model(**inputs)

    # Average from start_token onwards
    result = {}
    for layer_idx in capture.layers:
        hidden = capture.activations[layer_idx][0]  # [seq_len, hidden_dim]
        averaged = hidden[start_token:].mean(dim=0).float().numpy()
        result[layer_idx] = averaged

    return result


def collect_text_activations_batch(
    capture: ActivationCapture,
    tokenizer,
    model,
    texts: List[str],
    start_token: int = 20,
) -> List[Optional[Dict[int, np.ndarray]]]:
    """
    Collect activations for multiple raw texts in a single batch.

    Returns:
        List of dicts (one per sample), each containing {layer_idx: averaged_activation},
        or None if text too short
    """
    if not texts:
        return []

    # Tokenize batch with padding
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    # Forward pass
    capture.clear()
    with torch.no_grad():
        _ = model(**inputs)

    # Extract activations per sample
    results = []
    batch_size = inputs["input_ids"].shape[0]

    for sample_idx in range(batch_size):
        # Find actual sequence length (excluding padding)
        attention_mask = inputs["attention_mask"][sample_idx]
        seq_len = attention_mask.sum().item()

        if seq_len <= start_token:
            results.append(None)
            continue

        result = {}
        for layer_idx in capture.layers:
            hidden = capture.activations[layer_idx][sample_idx]  # [seq_len, hidden_dim]
            # Average from start_token to actual end (exclude padding)
            averaged = hidden[start_token:seq_len].mean(dim=0).float().cpu().numpy()
            result[layer_idx] = averaged

        results.append(result)

    return results


def load_data(input_path: Path, mode: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(input_path) as f:
        for line in f:
            data.append(json.loads(line))

    # Validate format
    if mode == 'chat':
        # Expect: full_prompt or messages
        sample = data[0]
        if 'messages' not in sample and 'full_prompt' not in sample:
            raise ValueError("Chat mode requires 'messages' or 'full_prompt' field")
    elif mode == 'text':
        # Expect: neutral_text + emotional_text/emotional_variants
        sample = data[0]
        if 'neutral_text' not in sample:
            raise ValueError("Text mode requires 'neutral_text' field")

    return data


def save_activations_incremental(
    activations: Dict[str, Any],
    output_path: Path,
    append: bool = True,
):
    """
    Save activations incrementally by appending to existing files.

    Args:
        activations: New activations to save
        output_path: Output directory
        append: If True, merge with existing data; if False, overwrite
    """
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Reorganize activations by layer
    layers_data = {}
    for item_id, item_acts in activations.items():
        for layer_idx, layer_act in item_acts.items():
            if isinstance(layer_idx, int):
                # This is a layer
                if layer_idx not in layers_data:
                    layers_data[layer_idx] = {}
                layers_data[layer_idx][item_id] = layer_act
            else:
                # This is 'neutral' or 'emotional' from text mode
                # Structure: item_acts = {'neutral': {layer: arr}, 'emotional': {layer: arr}}
                for variant in ['neutral', 'emotional']:
                    if variant in item_acts:
                        for layer_idx, layer_act in item_acts[variant].items():
                            if layer_idx not in layers_data:
                                layers_data[layer_idx] = {}

                            # Create composite key
                            composite_key = f"{item_id}_{variant}"
                            layers_data[layer_idx][composite_key] = layer_act
                break  # Only process once per item

    # Save each layer to separate file (merge with existing if append=True)
    total_size = 0
    for layer_idx in sorted(layers_data.keys()):
        layer_file = output_path / f"layer_{layer_idx:02d}.pkl"

        # Load existing data if appending
        if append and layer_file.exists():
            try:
                with open(layer_file, 'rb') as f:
                    existing_data = pickle.load(f)
                # Merge new data with existing
                existing_data.update(layers_data[layer_idx])
                layers_data[layer_idx] = existing_data
            except:
                pass  # If load fails, just save new data

        # Save merged data
        with open(layer_file, 'wb') as f:
            pickle.dump(layers_data[layer_idx], f, protocol=pickle.HIGHEST_PROTOCOL)
        total_size += layer_file.stat().st_size

    return total_size


def save_activations(
    activations: Dict[str, Any],
    metadata: CollectionMetadata,
    output_path: Path,
):
    """Save activations to separate files per layer in a directory."""
    # Use incremental save (overwrites for final save)
    total_size = save_activations_incremental(activations, output_path, append=False)

    # Save metadata
    meta_path = output_path / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(asdict(metadata), f, indent=2)

    print(f"\n✓ Saved activations: {output_path}")
    print(f"  Total size: {total_size / 1e6:.1f} MB")
    print(f"✓ Saved metadata: {meta_path}")


def load_resume_state(output_path: Path) -> set:
    """Load completed IDs from existing layer files."""
    if not output_path.exists() or not output_path.is_dir():
        return set()

    try:
        # Load from first available layer file to get IDs
        layer_files = sorted(output_path.glob("layer_*.pkl"))
        if not layer_files:
            return set()

        with open(layer_files[0], 'rb') as f:
            data = pickle.load(f)

        # Extract base IDs (remove _neutral/_emotional suffixes for text mode)
        completed = set()
        for key in data.keys():
            # Remove variant suffix if present
            base_id = key.replace('_neutral', '').replace('_emotional', '')
            completed.add(base_id)

        return completed
    except:
        return set()


def main():
    parser = argparse.ArgumentParser(
        description="Unified activation collection for all models and datasets"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory (e.g., activations/emotion_prompts_gemma2_9b/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['chat', 'text'],
        help="Collection mode: 'chat' (tokenize with chat template) or 'text' (raw text)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default='all',
        help="Layers to collect: 'all' (default), '20-40', or '20,30,40'",
    )
    parser.add_argument(
        "--start-token",
        type=int,
        default=20,
        help="For text mode: start averaging from this token (default: 20)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bfloat16',
        choices=['float16', 'bfloat16', 'float32'],
        help="Model dtype",
    )
    parser.add_argument(
        "--batch-save",
        type=int,
        default=100,
        help="Save checkpoint every N items",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for forward passes (default: 16)",
    )
    parser.add_argument(
        "--resume",
        action='store_true',
        help="Resume from existing output file",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("UNIFIED ACTIVATION COLLECTION")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    print()

    # Load data
    print("Loading data...")
    data = load_data(args.input, args.mode)
    print(f"Loaded {len(data)} items")

    # Check resume
    completed_ids = set()
    if args.resume and args.output.exists():
        completed_ids = load_resume_state(args.output)
        print(f"Found {len(completed_ids)} already completed")

    # Load model
    print(f"\nLoading model: {args.model}")
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map='auto',
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Get num_layers (different models use different attribute names)
    # For multimodal models, check text_config first
    config = model.config
    if hasattr(config, 'text_config'):
        config = config.text_config

    num_layers = getattr(config, 'num_hidden_layers', None) or \
                 getattr(config, 'n_layers', None) or \
                 getattr(config, 'num_layers', None)
    if num_layers is None:
        raise ValueError(f"Could not determine number of layers from model config. Available attributes: {dir(config)}")

    hidden_dim = config.hidden_size
    layers = parse_layers(args.layers, num_layers)

    print(f"✓ Model loaded")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Layers: {num_layers} (collecting {len(layers)})")
    print(f"  Hidden dim: {hidden_dim}")

    # Setup activation capture
    capture = ActivationCapture(model, layers)

    # Collect activations
    print(f"\nCollecting activations ({args.mode} mode)...")
    activations = {}
    metadata_list = []

    if args.mode == 'chat':
        # Chat mode: prompts with chat template
        batch_texts = []
        batch_item_ids = []
        batch_metadata = []

        for idx, item in enumerate(tqdm(data, desc="Processing")):
            item_id = item.get('id', str(idx))

            if item_id in completed_ids:
                continue

            try:
                # Get text
                if 'messages' in item:
                    # Multi-turn conversation format
                    text = item['messages'][-1]['content']  # Last user message
                elif 'full_prompt' in item:
                    text = item['full_prompt']
                elif 'base_prompt' in item and 'suffix' in item:
                    text = f"{item['base_prompt']} {item['suffix']}"
                else:
                    raise ValueError(f"Cannot extract text from item {item_id}")

                # Accumulate into batch
                batch_texts.append(text)
                batch_item_ids.append(item_id)
                batch_metadata.append({
                    'id': item_id,
                    'emotion': item.get('emotion'),
                    'topic': item.get('topic'),
                })

                # Process batch when full
                if len(batch_texts) >= args.batch_size:
                    results = collect_chat_activations_batch(
                        capture, tokenizer, model, batch_texts, args.model
                    )
                    for bid, acts in zip(batch_item_ids, results):
                        activations[bid] = acts
                    metadata_list.extend(batch_metadata)

                    # Clear batch
                    batch_texts = []
                    batch_item_ids = []
                    batch_metadata = []

                # Periodic save (incremental)
                if (idx + 1) % args.batch_save == 0:
                    save_activations_incremental(activations, args.output, append=True)
                    tqdm.write(f"✓ Saved checkpoint at {idx + 1} items ({len(activations)} total)")
                    activations = {}  # Clear memory after saving
                    torch.cuda.empty_cache()

            except Exception as e:
                tqdm.write(f"Error processing {item_id}: {e}")
                continue

        # Process remaining batch
        if batch_texts:
            results = collect_chat_activations_batch(
                capture, tokenizer, model, batch_texts, args.model
            )
            for bid, acts in zip(batch_item_ids, results):
                activations[bid] = acts
            metadata_list.extend(batch_metadata)

    else:  # text mode
        # Text mode: pairs without chat template
        # Accumulate pairs into batches for efficient processing
        batch_neutral_texts = []
        batch_emotional_texts = []
        batch_pair_ids = []
        batch_metadata = []
        processed_count = 0

        for idx, item in enumerate(tqdm(data, desc="Processing")):
            item_id = item.get('id', str(idx))

            if item_id in completed_ids:
                continue

            try:
                neutral_text = item['neutral_text']

                # Handle emotional_variants (tier data) or emotional_text (simple pairs)
                if 'emotional_variants' in item:
                    # Multiple emotions per neutral text
                    for emotion, emotional_text in item['emotional_variants'].items():
                        pair_id = f"{item_id}_{emotion}"

                        batch_neutral_texts.append(neutral_text)
                        batch_emotional_texts.append(emotional_text)
                        batch_pair_ids.append(pair_id)
                        batch_metadata.append({
                            'id': pair_id,
                            'emotion': emotion,
                            'topic': item.get('topic'),
                            'tier': item.get('tier'),
                        })

                        # Process batch when full
                        if len(batch_neutral_texts) >= args.batch_size:
                            neutral_results = collect_text_activations_batch(
                                capture, tokenizer, model, batch_neutral_texts, args.start_token
                            )
                            emotional_results = collect_text_activations_batch(
                                capture, tokenizer, model, batch_emotional_texts, args.start_token
                            )

                            for pid, n_acts, e_acts in zip(batch_pair_ids, neutral_results, emotional_results):
                                if n_acts is None or e_acts is None:
                                    tqdm.write(f"Skipping {pid}: text too short")
                                    continue

                                activations[pid] = {
                                    'neutral': n_acts,
                                    'emotional': e_acts,
                                }

                            metadata_list.extend(batch_metadata)
                            processed_count += len(batch_pair_ids)

                            # Clear batch
                            batch_neutral_texts = []
                            batch_emotional_texts = []
                            batch_pair_ids = []
                            batch_metadata = []

                else:
                    # Simple format
                    emotional_text = item['emotional_text']

                    batch_neutral_texts.append(neutral_text)
                    batch_emotional_texts.append(emotional_text)
                    batch_pair_ids.append(item_id)
                    batch_metadata.append({
                        'id': item_id,
                        'emotion': item.get('emotion'),
                        'topic': item.get('topic'),
                    })

                    # Process batch when full
                    if len(batch_neutral_texts) >= args.batch_size:
                        neutral_results = collect_text_activations_batch(
                            capture, tokenizer, model, batch_neutral_texts, args.start_token
                        )
                        emotional_results = collect_text_activations_batch(
                            capture, tokenizer, model, batch_emotional_texts, args.start_token
                        )

                        for pid, n_acts, e_acts in zip(batch_pair_ids, neutral_results, emotional_results):
                            if n_acts is None or e_acts is None:
                                tqdm.write(f"Skipping {pid}: text too short")
                                continue

                            activations[pid] = {
                                'neutral': n_acts,
                                'emotional': e_acts,
                            }

                        metadata_list.extend(batch_metadata)
                        processed_count += len(batch_pair_ids)

                        # Clear batch
                        batch_neutral_texts = []
                        batch_emotional_texts = []
                        batch_pair_ids = []
                        batch_metadata = []

                # Periodic save (incremental)
                if (idx + 1) % args.batch_save == 0:
                    save_activations_incremental(activations, args.output, append=True)
                    tqdm.write(f"✓ Saved checkpoint at {idx + 1} items ({len(activations)} total)")
                    activations = {}  # Clear memory after saving
                    torch.cuda.empty_cache()

            except Exception as e:
                tqdm.write(f"Error processing {item_id}: {e}")
                continue

        # Process remaining batch
        if batch_neutral_texts:
            neutral_results = collect_text_activations_batch(
                capture, tokenizer, model, batch_neutral_texts, args.start_token
            )
            emotional_results = collect_text_activations_batch(
                capture, tokenizer, model, batch_emotional_texts, args.start_token
            )

            for pid, n_acts, e_acts in zip(batch_pair_ids, neutral_results, emotional_results):
                if n_acts is None or e_acts is None:
                    tqdm.write(f"Skipping {pid}: text too short")
                    continue

                activations[pid] = {
                    'neutral': n_acts,
                    'emotional': e_acts,
                }

            metadata_list.extend(batch_metadata)

    # Remove hooks
    capture.remove_hooks()

    # Save any remaining activations that weren't caught in the last batch
    if activations:
        save_activations_incremental(activations, args.output, append=True)
        print(f"✓ Saved final checkpoint ({len(activations)} items)")

    # Count total saved items from the actual files
    completed_ids = load_resume_state(args.output)
    total_items = len(completed_ids)

    # Create metadata
    representations = ['last_token', 'special_mean'] if args.mode == 'chat' else ['averaged_from_20']
    metadata = CollectionMetadata(
        model_name=args.model,
        mode=args.mode,
        input_file=str(args.input),
        output_file=str(args.output),
        num_items=total_items,
        num_layers=len(layers),
        layers=layers,
        hidden_dim=hidden_dim,
        representations=representations,
        start_token=args.start_token,
        timestamp=datetime.now().isoformat(),
        dtype=args.dtype,
        total_activations=total_items * len(layers),
    )

    # Save metadata with provenance
    meta_dict = asdict(metadata)
    meta_dict["provenance"] = get_provenance(script=__file__)
    meta_path = args.output / "metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta_dict, f, indent=2)
    print(f"✓ Saved metadata: {meta_path}")

    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)
    print(f"Items collected: {total_items}")
    print(f"Layers: {len(layers)}")
    print(f"Total activations: {metadata.total_activations:,}")


if __name__ == "__main__":
    main()
