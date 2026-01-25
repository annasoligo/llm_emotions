#!/usr/bin/env python3
"""
Specialized activation collection for appraisal minimal-pair datasets.

Handles scenarios.jsonl and paraphrases.jsonl format with A/B variants.

Usage:
    python activation_collection/collect_appraisal.py \
        --data-dir steering_tests/data/appraisal_minimal_pairs \
        --output-dir activations/appraisal \
        --model google/gemma-2-9b-it \
        --layers 20-40
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any
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


@dataclass
class AppraisalMetadata:
    """Metadata for appraisal activation collection."""
    model_name: str
    data_dir: str
    output_dir: str
    num_scenarios: int
    num_paraphrases: int
    num_layers: int
    layers: List[int]
    hidden_dim: int
    representations: List[str]
    timestamp: str
    dtype: str


class ActivationCapture:
    """Capture activations using PyTorch hooks."""

    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.activations = {}
        self.hooks = []

        # Find where layers are stored (different architectures use different names)
        layers_module = self._get_layers_module(model)

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
            self.activations[layer_idx] = hidden_states.detach().cpu()
        return hook

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def parse_layers(layer_spec: str, num_layers: int) -> List[int]:
    """Parse layer specification."""
    if layer_spec == 'all':
        return list(range(num_layers))
    elif '-' in layer_spec:
        start, end = map(int, layer_spec.split('-'))
        return list(range(start, end + 1))
    else:
        return [int(x.strip()) for x in layer_spec.split(',')]


def find_special_token_indices(token_ids: torch.Tensor, tokenizer, model_name: str) -> List[int]:
    """Find special token indices."""
    tokens = token_ids[0].tolist()
    special_indices = []

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


def collect_activations(
    capture: ActivationCapture,
    tokenizer,
    model,
    text: str,
    model_name: str,
) -> Dict[str, np.ndarray]:
    """Collect activations for appraisal scenario."""
    # Apply chat template with generation prompt
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    token_ids = inputs["input_ids"]

    # Forward pass
    capture.clear()
    with torch.no_grad():
        _ = model(**inputs)

    # Extract two representations
    result = {}

    # 1. Last token before generation (assistant_start)
    for layer_idx in capture.layers:
        if layer_idx not in result:
            result[layer_idx] = {}
        hidden = capture.activations[layer_idx][0]
        result[layer_idx]['assistant_start'] = hidden[-1].float().numpy()

    # 2. Mean over special tokens at boundaries
    special_indices = find_special_token_indices(token_ids, tokenizer, model_name)
    if special_indices:
        for layer_idx in capture.layers:
            hidden = capture.activations[layer_idx][0]
            special_acts = hidden[special_indices]
            result[layer_idx]['boundary_mean'] = special_acts.mean(dim=0).float().numpy()
    else:
        for layer_idx in capture.layers:
            result[layer_idx]['boundary_mean'] = result[layer_idx]['assistant_start']

    return result


def load_scenarios(data_dir: Path) -> List[Dict]:
    """Load scenarios from scenarios.jsonl."""
    scenarios = []
    scenarios_path = data_dir / "scenarios.jsonl"

    if not scenarios_path.exists():
        raise FileNotFoundError(f"scenarios.jsonl not found in {data_dir}")

    with open(scenarios_path) as f:
        for line in f:
            scenarios.append(json.loads(line))

    return scenarios


def load_paraphrases(data_dir: Path) -> List[Dict]:
    """Load paraphrases from paraphrases.jsonl."""
    paraphrases = []
    paraphrases_path = data_dir / "paraphrases.jsonl"

    if not paraphrases_path.exists():
        return []

    with open(paraphrases_path) as f:
        for line in f:
            paraphrases.append(json.loads(line))

    return paraphrases


def main():
    parser = argparse.ArgumentParser(
        description="Collect activations for appraisal minimal pairs"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory with scenarios.jsonl and paraphrases.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for activations",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default='all',
        help="Layers to collect: 'all' (default), '20-40', or '20,30,40'",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bfloat16',
        choices=['float16', 'bfloat16', 'float32'],
    )
    parser.add_argument(
        "--resume",
        action='store_true',
        help="Resume from existing output files",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("APPRAISAL ACTIVATION COLLECTION")
    print("=" * 80)
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Model: {args.model}")
    print()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Output directories
    model_slug = args.model.replace('/', '_').replace('-', '_')
    scenarios_output = args.output_dir / f"scenarios_{model_slug}"
    paraphrases_output = args.output_dir / f"paraphrases_{model_slug}"

    # Load data
    print("Loading data...")
    scenarios = load_scenarios(args.data_dir)
    paraphrases = load_paraphrases(args.data_dir)
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Paraphrases: {len(paraphrases)}")

    # Check resume
    completed_scenarios = set()
    completed_paraphrases = set()

    if args.resume:
        if scenarios_output.exists() and scenarios_output.is_dir():
            layer_files = list(scenarios_output.glob("layer_*.pkl"))
            if layer_files:
                with open(layer_files[0], 'rb') as f:
                    data = pickle.load(f)
                    completed_scenarios = set(data.keys())
                print(f"  Found {len(completed_scenarios)} completed scenarios")

        if paraphrases_output.exists() and paraphrases_output.is_dir():
            layer_files = list(paraphrases_output.glob("layer_*.pkl"))
            if layer_files:
                with open(layer_files[0], 'rb') as f:
                    data = pickle.load(f)
                    completed_paraphrases = set(data.keys())
                print(f"  Found {len(completed_paraphrases)} completed paraphrases")

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

    # Setup capture
    capture = ActivationCapture(model, layers)

    # Collect scenarios
    print("\nCollecting scenario activations...")
    scenario_acts = {}
    scenario_meta = []

    for scenario in tqdm(scenarios, desc="Scenarios"):
        # Process scenario A
        id_a = f"{scenario['id']}_a"
        if id_a not in completed_scenarios:
            try:
                acts = collect_activations(
                    capture, tokenizer, model, scenario['scenario_a'], args.model
                )
                scenario_acts[id_a] = acts
                scenario_meta.append({
                    'id': id_a,
                    'scenario_id': scenario['id'],
                    'variant': 'a',
                    'axis_name': scenario.get('axis_name'),
                    'card_id': scenario.get('card_id'),
                    'domain': scenario.get('domain'),
                })
            except Exception as e:
                tqdm.write(f"Error processing {id_a}: {e}")

        # Process scenario B
        id_b = f"{scenario['id']}_b"
        if id_b not in completed_scenarios:
            try:
                acts = collect_activations(
                    capture, tokenizer, model, scenario['scenario_b'], args.model
                )
                scenario_acts[id_b] = acts
                scenario_meta.append({
                    'id': id_b,
                    'scenario_id': scenario['id'],
                    'variant': 'b',
                    'axis_name': scenario.get('axis_name'),
                    'card_id': scenario.get('card_id'),
                    'domain': scenario.get('domain'),
                })
            except Exception as e:
                tqdm.write(f"Error processing {id_b}: {e}")

        # Clear cache periodically
        if len(scenario_acts) % 50 == 0:
            torch.cuda.empty_cache()

    # Save scenarios
    if scenario_acts:
        scenarios_output.mkdir(parents=True, exist_ok=True)

        # Reorganize by layer
        layers_data = {}
        for item_id, item_acts in scenario_acts.items():
            for layer_idx, layer_dict in item_acts.items():
                if layer_idx not in layers_data:
                    layers_data[layer_idx] = {}
                layers_data[layer_idx][item_id] = layer_dict

        # Save each layer
        total_size = 0
        for layer_idx in sorted(layers_data.keys()):
            layer_file = scenarios_output / f"layer_{layer_idx:02d}.pkl"
            with open(layer_file, 'wb') as f:
                pickle.dump(layers_data[layer_idx], f, protocol=pickle.HIGHEST_PROTOCOL)
            total_size += layer_file.stat().st_size

        # Save metadata
        metadata = AppraisalMetadata(
            model_name=args.model,
            data_dir=str(args.data_dir),
            output_dir=str(args.output_dir),
            num_scenarios=len(scenario_acts),
            num_paraphrases=0,
            num_layers=len(layers),
            layers=layers,
            hidden_dim=hidden_dim,
            representations=['assistant_start', 'boundary_mean'],
            timestamp=datetime.now().isoformat(),
            dtype=args.dtype,
        )

        meta_path = scenarios_output / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        # Save item metadata
        items_meta_path = scenarios_output / "items_metadata.json"
        with open(items_meta_path, 'w') as f:
            json.dump(scenario_meta, f, indent=2)

        print(f"\n✓ Saved scenarios: {scenarios_output}")
        print(f"  Layers: {len(layers_data)}")
        print(f"  Total size: {total_size / 1e6:.1f} MB")
        print(f"  Items: {len(scenario_acts)}")

    # Collect paraphrases
    if paraphrases:
        print("\nCollecting paraphrase activations...")
        para_acts = {}
        para_meta = []

        for para_item in tqdm(paraphrases, desc="Paraphrases"):
            variant = para_item.get('variant', 'a')
            original_id = para_item.get('original_id')

            for i, para_text in enumerate(para_item.get('paraphrases', [])):
                para_id = f"{original_id}_{variant}_para{i}"

                if para_id not in completed_paraphrases:
                    try:
                        acts = collect_activations(
                            capture, tokenizer, model, para_text, args.model
                        )
                        para_acts[para_id] = acts
                        para_meta.append({
                            'id': para_id,
                            'original_id': original_id,
                            'variant': variant,
                            'paraphrase_idx': i,
                            'axis_name': para_item.get('axis_name'),
                            'card_id': para_item.get('card_id'),
                            'domain': para_item.get('domain'),
                        })
                    except Exception as e:
                        tqdm.write(f"Error processing {para_id}: {e}")

            if len(para_acts) % 50 == 0:
                torch.cuda.empty_cache()

        # Save paraphrases
        if para_acts:
            paraphrases_output.mkdir(parents=True, exist_ok=True)

            # Reorganize by layer
            layers_data = {}
            for item_id, item_acts in para_acts.items():
                for layer_idx, layer_dict in item_acts.items():
                    if layer_idx not in layers_data:
                        layers_data[layer_idx] = {}
                    layers_data[layer_idx][item_id] = layer_dict

            # Save each layer
            total_size = 0
            for layer_idx in sorted(layers_data.keys()):
                layer_file = paraphrases_output / f"layer_{layer_idx:02d}.pkl"
                with open(layer_file, 'wb') as f:
                    pickle.dump(layers_data[layer_idx], f, protocol=pickle.HIGHEST_PROTOCOL)
                total_size += layer_file.stat().st_size

            # Save metadata
            metadata = AppraisalMetadata(
                model_name=args.model,
                data_dir=str(args.data_dir),
                output_dir=str(args.output_dir),
                num_scenarios=0,
                num_paraphrases=len(para_acts),
                num_layers=len(layers),
                layers=layers,
                hidden_dim=hidden_dim,
                representations=['assistant_start', 'boundary_mean'],
                timestamp=datetime.now().isoformat(),
                dtype=args.dtype,
            )

            meta_path = paraphrases_output / "metadata.json"
            with open(meta_path, 'w') as f:
                json.dump(asdict(metadata), f, indent=2)

            # Save item metadata
            items_meta_path = paraphrases_output / "items_metadata.json"
            with open(items_meta_path, 'w') as f:
                json.dump(para_meta, f, indent=2)

            print(f"\n✓ Saved paraphrases: {paraphrases_output}")
            print(f"  Layers: {len(layers_data)}")
            print(f"  Total size: {total_size / 1e6:.1f} MB")
            print(f"  Items: {len(para_acts)}")

    # Cleanup
    capture.remove_hooks()

    print("\n" + "=" * 80)
    print("COLLECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
