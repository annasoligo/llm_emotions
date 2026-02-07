#!/usr/bin/env python3
"""
Phase 2: Generate continuations from prepared paraphrased prefills.

Run this script for each model (as parallel SLURM jobs).
All models use the same prepared paraphrases from Phase 1.

Usage:
    python run_generation.py --model-family gemma27b --model-type instruct --data prepared_samples_*.json
    python run_generation.py --model-family qwen32b --model-type base --data prepared_samples_*.json
"""

import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model configurations
MODEL_CONFIGS = {
    "gemma27b": {
        "instruct": "google/gemma-3-27b-it",
        "base": "google/gemma-3-27b-pt",
    },
    "gemma12b": {
        "instruct": "google/gemma-3-12b-it",
        "base": "google/gemma-3-12b-pt",
    },
    "qwen32b": {
        "instruct": "Qwen/Qwen3-32B",
        "base": "Qwen/Qwen3-32B",  # Qwen3 is base model
    },
    "qwen25b": {
        "instruct": "Qwen/Qwen2.5-32B-Instruct",
        "base": "Qwen/Qwen2.5-32B",
    },
    "olmo32b": {
        "instruct": "allenai/OLMo-3.1-32B-Instruct",
        "base": "allenai/OLMo-3-1125-32B",
    },
    "olmo32b_dpo": {
        "instruct": "allenai/OLMo-3.1-32B-Instruct-DPO",
        "base": "allenai/OLMo-3.1-32B-Instruct-DPO",
    },
    "olmo32b_sft": {
        "instruct": "allenai/OLMo-3.1-32B-Instruct-SFT",
        "base": "allenai/OLMo-3.1-32B-Instruct-SFT",
    },
}

# Generation params
NUM_CONTINUATIONS = 50
MAX_NEW_TOKENS = 1000
TEMPERATURE = 1.0
TOP_P = 0.9

OUTPUT_DIR = Path("experiments/prefill_scaled/outputs")


@dataclass
class ContinuationResult:
    sample_id: str
    source: str
    truncation_type: str  # "early" or "late"
    model_family: str
    model_type: str  # "instruct" or "base"
    model_name: str
    continuation_idx: int
    continuation: str
    continuation_tokens: int
    seed: int
    timestamp: str


def load_prepared_data(data_path: Path) -> Dict:
    """Load prepared samples from Phase 1."""
    print(f"Loading prepared data from {data_path}...")
    with open(data_path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data['samples'])} samples")
    return data


def load_model(model_name: str):
    """Load model and tokenizer."""
    print(f"Loading model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"  Model loaded on {model.device}")
    return model, tokenizer


def generate_continuations_batched(
    model, tokenizer, prefix_text: str,
    num_continuations: int, base_seed: int,
    batch_size: int = 10
) -> List[str]:
    """Generate multiple continuations from text prefix in batches."""

    # Tokenize prefix
    input_ids = tokenizer.encode(prefix_text, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    prefix_len = input_ids.shape[1]

    all_continuations = []

    for batch_start in range(0, num_continuations, batch_size):
        batch_end = min(batch_start + batch_size, num_continuations)
        current_batch_size = batch_end - batch_start

        # Set seed for this batch for reproducibility
        torch.manual_seed(base_seed + batch_start)

        batch_input_ids = input_ids.expand(current_batch_size, -1)

        with torch.no_grad():
            outputs = model.generate(
                batch_input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        # Extract continuations (remove prefix)
        for i in range(current_batch_size):
            continuation_ids = outputs[i, prefix_len:]
            continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
            all_continuations.append(continuation)

        # Clear cache between batches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_continuations


def run_generation(args):
    """Main generation function."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get model config
    model_config = MODEL_CONFIGS[args.model_family]
    model_name = model_config[args.model_type]

    print("="*60)
    print(f"PREFILL GENERATION - {args.model_family} {args.model_type}")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Continuations per sample: {NUM_CONTINUATIONS}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")

    # Load prepared data
    data = load_prepared_data(Path(args.data))
    samples = data['samples']

    # Load model
    model, tokenizer = load_model(model_name)

    # Output file
    output_file = OUTPUT_DIR / f"continuations_{args.model_family}_{args.model_type}_{timestamp}.jsonl"
    print(f"Output: {output_file}")

    results = []

    for sample_idx, sample in enumerate(samples):
        print(f"\n[{sample_idx+1}/{len(samples)}] {sample['sample_id']} ({sample['source']})")

        # Determine which truncations to run
        truncations_to_run = []

        # Apply filter if specified
        filter_cond = args.filter_condition

        # Late truncation (all samples)
        if sample.get('late_paraphrase'):
            # Check filter
            if filter_cond is None or filter_cond == f"{sample['source']}_late":
                truncations_to_run.append(('late', sample['late_paraphrase']))

        # Early truncation (puzzles only)
        if sample['source'] == 'puzzle' and sample.get('early_paraphrase'):
            # Check filter
            if filter_cond is None or filter_cond == 'puzzle_early':
                truncations_to_run.append(('early', sample['early_paraphrase']))

        for trunc_type, prefix_text in truncations_to_run:
            print(f"  {trunc_type} truncation: {len(prefix_text)} chars prefix")

            base_seed = hash(f"{sample['sample_id']}_{trunc_type}_{args.model_family}_{args.model_type}") % (2**32)

            continuations = generate_continuations_batched(
                model, tokenizer, prefix_text, NUM_CONTINUATIONS, base_seed
            )

            for cont_idx, continuation in enumerate(continuations):
                result = ContinuationResult(
                    sample_id=sample['sample_id'],
                    source=sample['source'],
                    truncation_type=trunc_type,
                    model_family=args.model_family,
                    model_type=args.model_type,
                    model_name=model_name,
                    continuation_idx=cont_idx,
                    continuation=continuation,
                    continuation_tokens=len(tokenizer.encode(continuation)),
                    seed=base_seed + cont_idx,
                    timestamp=datetime.now().isoformat(),
                )
                results.append(result)

                # Save incrementally
                with open(output_file, 'a') as f:
                    f.write(json.dumps(asdict(result)) + '\n')

            print(f"    Generated {len(continuations)} continuations")

    print(f"\n✓ Done! Saved {len(results)} continuations to {output_file}")

    # Summary
    print(f"\nSummary:")
    print(f"  Total continuations: {len(results)}")
    print(f"  Puzzle early: {len([r for r in results if r.source == 'puzzle' and r.truncation_type == 'early'])}")
    print(f"  Puzzle late: {len([r for r in results if r.source == 'puzzle' and r.truncation_type == 'late'])}")
    print(f"  Wildchat late: {len([r for r in results if r.source == 'wildchat' and r.truncation_type == 'late'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-family', required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--model-type', required=True, choices=['instruct', 'base'])
    parser.add_argument('--data', required=True, help='Path to prepared_samples_*.json from Phase 1')
    parser.add_argument('--filter-condition', type=str, default=None,
                        help='Filter to specific condition: puzzle_early, puzzle_late, or wildchat_late')
    args = parser.parse_args()

    run_generation(args)
