"""
Collect activations from appraisal scenarios on different models.

Reads scenarios.jsonl and paraphrases.jsonl from an existing run,
and collects activations from a specified target model.

Usage:
    python -m probes.scripts.appraisal.collect_activations_multimodel \
        --data-dir /workspace-vast/annas/appraisal_data/full_run \
        --model Qwen/Qwen3-32B \
        --output-dir /workspace-vast/annas/appraisal_data/qwen32b
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import h5py
import numpy as np
import torch
from tqdm import tqdm


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def get_input_items(data_dir: Path) -> List[Dict[str, Any]]:
    """Get scenarios and paraphrases to process."""
    items = []

    scenarios_path = data_dir / "scenarios.jsonl"
    paraphrases_path = data_dir / "paraphrases.jsonl"

    # Load scenarios
    if scenarios_path.exists():
        for scenario in load_jsonl(scenarios_path):
            # Add scenario A
            items.append({
                "id": f"{scenario['id']}_a_original",
                "scenario_id": scenario["id"],
                "variant": "a",
                "type": "original",
                "text": scenario["scenario_a"],
                "axis_name": scenario.get("axis_name"),
                "card_id": scenario.get("card_id"),
                "domain": scenario.get("domain"),
            })

            # Add scenario B
            items.append({
                "id": f"{scenario['id']}_b_original",
                "scenario_id": scenario["id"],
                "variant": "b",
                "type": "original",
                "text": scenario["scenario_b"],
                "axis_name": scenario.get("axis_name"),
                "card_id": scenario.get("card_id"),
                "domain": scenario.get("domain"),
            })

    # Load paraphrases
    if paraphrases_path.exists():
        for paraphrase_item in load_jsonl(paraphrases_path):
            variant = paraphrase_item.get("variant", "a")
            original_id = paraphrase_item.get("original_id")

            for i, para_text in enumerate(paraphrase_item.get("paraphrases", [])):
                items.append({
                    "id": f"{original_id}_{variant}_para{i}",
                    "scenario_id": original_id,
                    "variant": variant,
                    "type": f"paraphrase_{i}",
                    "text": para_text,
                    "axis_name": paraphrase_item.get("axis_name"),
                    "card_id": paraphrase_item.get("card_id"),
                    "domain": paraphrase_item.get("domain"),
                })

    return items


def find_boundary_token_indices(token_ids: np.ndarray, tokenizer) -> List[int]:
    """Find indices of special tokens at turn boundaries."""
    boundary_indices = []

    special_patterns = {
        'start_of_turn': ['<start_of_turn>', '<|start_of_turn|>', '<|im_start|>'],
        'end_of_turn': ['<end_of_turn>', '<|end_of_turn|>', '<|im_end|>'],
    }

    token_id_map = {}
    for name, patterns in special_patterns.items():
        for pattern in patterns:
            try:
                token_id = tokenizer.convert_tokens_to_ids(pattern)
                if token_id != tokenizer.unk_token_id:
                    token_id_map[name] = token_id
                    break
            except:
                continue

    if 'start_of_turn' in token_id_map:
        start_indices = np.where(token_ids == token_id_map['start_of_turn'])[0]
        boundary_indices.extend(start_indices.tolist())

    if 'end_of_turn' in token_id_map:
        end_indices = np.where(token_ids == token_id_map['end_of_turn'])[0]
        boundary_indices.extend(end_indices.tolist())

    return sorted(set(boundary_indices))


def collect_activations(
    model,
    tokenizer,
    text: str,
    device,
    layers: List[int] = None,
) -> Dict[str, np.ndarray]:
    """Collect activations for a single text."""
    # Apply chat template with generation prompt
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(formatted, return_tensors="pt").to(device)
    token_ids = inputs["input_ids"][0].cpu().numpy()

    # Determine layers
    if hasattr(model, 'num_layers'):
        num_layers = model.num_layers
    else:
        num_layers = model.config.num_hidden_layers

    if layers is None:
        layers = list(range(num_layers))

    # Collect activations
    result = {}

    with torch.no_grad():
        if hasattr(model, 'trace'):
            # nnterp model
            layer_activations = []

            with model.trace(inputs, scan=False):
                for layer_idx in layers:
                    layer_output = model.layers_output[layer_idx].save()
                    layer_activations.append(layer_output)

            # Stack: [num_layers, seq_len, hidden_size]
            # Move all tensors to CPU before stacking (handles multi-GPU)
            all_acts = torch.stack([la[0].cpu() for la in layer_activations])
            all_acts = all_acts.float().numpy()

        else:
            # Raw model with output_hidden_states
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states

            # Move all tensors to CPU before stacking (handles multi-GPU)
            layer_acts = [hidden_states[i + 1][0].cpu() for i in layers]
            all_acts = torch.stack(layer_acts).float().numpy()

    # Extract representations
    # Last token of prompt (before generation)
    result["assistant_start_last_token"] = all_acts[:, -1, :]

    # Mean of special tokens at turn boundaries
    boundary_indices = find_boundary_token_indices(token_ids, tokenizer)
    if boundary_indices:
        boundary_acts = all_acts[:, boundary_indices, :]
        result["boundary_special_tokens_mean"] = boundary_acts.mean(axis=1)
    else:
        result["boundary_special_tokens_mean"] = all_acts[:, -1, :]

    return result


def main():
    parser = argparse.ArgumentParser(description="Collect activations on different models")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory with scenarios.jsonl and paraphrases.jsonl")
    parser.add_argument("--model", type=str, required=True,
                        help="HuggingFace model name")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for activations")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=1)

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("APPRAISAL ACTIVATION COLLECTION")
    print("=" * 60)
    print(f"Data dir: {data_dir}")
    print(f"Model: {args.model}")
    print(f"Output dir: {output_dir}")
    print()

    # Load items
    print("Loading items...")
    items = get_input_items(data_dir)
    print(f"Found {len(items)} items")

    # Check for existing progress
    output_path = output_dir / "activations.h5"
    metadata_path = output_dir / "activation_metadata.json"

    completed_ids = set()
    if output_path.exists():
        with h5py.File(output_path, 'r') as f:
            if 'activations' in f:
                completed_ids = set(f['activations'].keys())
        print(f"Found {len(completed_ids)} already completed")

    pending = [item for item in items if item["id"] not in completed_ids]
    print(f"Pending: {len(pending)}")

    if not pending:
        print("All items already processed!")
        return

    # Load model
    print(f"\nLoading model: {args.model}")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    model_raw = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Try nnterp
    try:
        from nnterp import StandardizedTransformer
        model = StandardizedTransformer(
            model_raw,
            trust_remote_code=True,
            check_renaming=False,
            allow_dispatch=True,
        )
        device = next(model.model.parameters()).device
        num_layers = model.num_layers
        hidden_size = model.hidden_size
    except ImportError:
        print("Warning: nnterp not available, using raw model")
        model = model_raw
        device = next(model_raw.parameters()).device
        num_layers = model_raw.config.num_hidden_layers
        hidden_size = model_raw.config.hidden_size

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on {device}")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden dim: {hidden_size}")

    # Collect activations
    print("\nCollecting activations...")
    mode = 'a' if output_path.exists() else 'w'
    metadata = []

    with h5py.File(output_path, mode) as f:
        if 'activations' not in f:
            f.create_group('activations')

        acts_group = f['activations']

        for item in tqdm(pending, desc="Processing"):
            try:
                acts = collect_activations(model, tokenizer, item["text"], device)

                item_group = acts_group.create_group(item["id"])
                for rep_name, rep_data in acts.items():
                    item_group.create_dataset(rep_name, data=rep_data)

                metadata.append({
                    "id": item["id"],
                    "scenario_id": item.get("scenario_id"),
                    "variant": item.get("variant"),
                    "type": item.get("type"),
                    "axis_name": item.get("axis_name"),
                    "card_id": item.get("card_id"),
                    "domain": item.get("domain"),
                })

            except Exception as e:
                print(f"Error processing {item['id']}: {e}")

        f.attrs['model_name'] = args.model
        f.attrs['num_items'] = len(acts_group)

    # Save metadata
    # Load existing metadata if present
    existing_metadata = []
    if metadata_path.exists():
        with open(metadata_path) as f:
            data = json.load(f)
            existing_metadata = data.get("items", [])

    all_metadata = existing_metadata + metadata

    with open(metadata_path, 'w') as f:
        json.dump({
            "model": args.model,
            "representations": ["assistant_start_last_token", "boundary_special_tokens_mean"],
            "layers": None,
            "items": all_metadata,
        }, f, indent=2)

    print(f"\nDone! Saved {len(metadata)} new items to {output_path}")


if __name__ == "__main__":
    main()
