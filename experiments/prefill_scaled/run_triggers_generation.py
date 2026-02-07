#!/usr/bin/env python3
"""
Run generation for triggers onset truncation experiment.
"""

import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

OUTPUT_DIR = Path("experiments/prefill_scaled/outputs")

MODEL_CONFIGS = {
    "gemma27b": {"instruct": "google/gemma-3-27b-it", "base": "google/gemma-3-27b-pt"},
    "gemma12b": {"instruct": "google/gemma-3-12b-it", "base": "google/gemma-3-12b-pt"},
    "qwen32b": {"instruct": "Qwen/Qwen3-32B", "base": "Qwen/Qwen3-32B"},
    "qwen25b": {"instruct": "Qwen/Qwen2.5-32B-Instruct", "base": "Qwen/Qwen2.5-32B"},
    "olmo32b": {"instruct": "allenai/OLMo-3.1-32B-Instruct", "base": "allenai/OLMo-3-1125-32B"},
    "olmo32b_dpo": {"instruct": "allenai/OLMo-3.1-32B-Instruct-DPO", "base": "allenai/OLMo-3.1-32B-Instruct-DPO"},
    "olmo32b_sft": {"instruct": "allenai/OLMo-3.1-32B-Instruct-SFT", "base": "allenai/OLMo-3.1-32B-Instruct-SFT"},
}

DPO_MODEL_PATH = "/workspace-vast/annas/models/gemma3-27b-dpo-calm-full/2026-01-15_09-48-56"
DPO_BASE_MODEL = "google/gemma-3-27b-it"

NUM_CONTINUATIONS = 50
MAX_NEW_TOKENS = 1000
TEMPERATURE = 1.0
TOP_P = 0.9


def find_prepared_data():
    """Find the most recent prepared triggers onset data file."""
    files = sorted(OUTPUT_DIR.glob("prepared_triggers_onset_*.json"), reverse=True)
    if not files:
        raise FileNotFoundError("No prepared_triggers_onset_*.json found. Run prepare_triggers_onset.py first.")
    return files[0]


def generate_continuations_batched(model, tokenizer, prefix_text, num_continuations, base_seed, batch_size=10):
    """Generate continuations with batching."""
    all_continuations = []

    for batch_start in range(0, num_continuations, batch_size):
        batch_end = min(batch_start + batch_size, num_continuations)
        batch_count = batch_end - batch_start

        torch.manual_seed(base_seed + batch_start)

        inputs = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        input_ids = input_ids.expand(batch_count, -1)
        attention_mask = attention_mask.expand(batch_count, -1)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        prefix_len = inputs["input_ids"].shape[1]
        for i in range(batch_count):
            new_tokens = outputs[i][prefix_len:]
            continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_continuations.append(continuation)

        torch.cuda.empty_cache()

    return all_continuations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-family', required=True,
                        choices=list(MODEL_CONFIGS.keys()) + ['dpo_calm'])
    parser.add_argument('--model-type', required=True, choices=['instruct', 'base', 'dpo'])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load model
    if args.model_family == 'dpo_calm':
        model_name = DPO_BASE_MODEL
        print(f"Loading DPO model...")
        print(f"  Base: {DPO_BASE_MODEL}")
        print(f"  Adapter: {DPO_MODEL_PATH}")

        tokenizer = AutoTokenizer.from_pretrained(DPO_BASE_MODEL)
        base_model = AutoModelForCausalLM.from_pretrained(
            DPO_BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, DPO_MODEL_PATH)
        model = model.merge_and_unload()
        model_family = 'dpo_calm'
        model_type = 'dpo'
    else:
        model_name = MODEL_CONFIGS[args.model_family][args.model_type]
        print(f"Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model_family = args.model_family
        model_type = args.model_type

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prepared data
    data_file = find_prepared_data()
    print(f"Loading prepared data from {data_file}")
    with open(data_file) as f:
        samples = json.load(f)

    # Output file
    output_file = OUTPUT_DIR / f"triggers_onset_{model_family}_{model_type}_{timestamp}.jsonl"
    print(f"Output: {output_file}")

    print("\n" + "="*60)
    print(f"TRIGGERS ONSET GENERATION - {model_family} {model_type}")
    print("="*60)

    total_generated = 0
    for sample_idx, sample in enumerate(samples):
        sample_id = sample['sample_id']

        # Use paraphrased prefix
        prefix_text = sample['paraphrased_prefix']

        print(f"\n[{sample_idx+1}/{len(samples)}] {sample_id}")
        print(f"  Scenario: {sample['scenario']}, onset turn: {sample['onset_turn']}")

        # Generate continuations
        continuations = generate_continuations_batched(
            model, tokenizer, prefix_text,
            NUM_CONTINUATIONS, base_seed=sample_idx * 1000
        )

        # Save results
        with open(output_file, 'a') as f:
            for i, cont in enumerate(continuations):
                result = {
                    'sample_id': sample_id,
                    'source': sample['source'],
                    'truncation_type': sample['truncation_type'],
                    'scenario': sample['scenario'],
                    'onset_turn': sample['onset_turn'],
                    'model_family': model_family,
                    'model_type': model_type,
                    'continuation_idx': i,
                    'continuation': cont,
                }
                f.write(json.dumps(result) + '\n')

        total_generated += len(continuations)
        print(f"  Generated {len(continuations)} (total: {total_generated})")

    print(f"\nDone! Generated {total_generated} continuations")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
