#!/usr/bin/env python3
"""
Run generation for wildchat early truncation experiment.
Usage: python run_wildchat_early.py --model-family gemma27b --model-type instruct --data <data_file>
"""

import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("experiments/prefill_scaled/outputs")

MODEL_CONFIGS = {
    "gemma27b": {"instruct": "google/gemma-3-27b-it", "base": "google/gemma-3-27b-pt"},
    "gemma12b": {"instruct": "google/gemma-3-12b-it", "base": "google/gemma-3-12b-pt"},
    "qwen32b": {"instruct": "Qwen/Qwen3-32B", "base": "Qwen/Qwen3-32B"},
    "olmo32b": {"instruct": "allenai/OLMo-3.1-32B-Instruct", "base": "allenai/OLMo-3-1125-32B"},
}

NUM_CONTINUATIONS = 50
MAX_NEW_TOKENS = 1000
TEMPERATURE = 1.0
TOP_P = 0.9


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

        for i in range(batch_count):
            new_tokens = outputs[i][input_ids.shape[1]:]
            continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_continuations.append(continuation)

        torch.cuda.empty_cache()

    return all_continuations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-family', required=True, choices=MODEL_CONFIGS.keys())
    parser.add_argument('--model-type', required=True, choices=['instruct', 'base'])
    parser.add_argument('--data', required=True, help='Path to prepared data JSON')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_name = MODEL_CONFIGS[args.model_family][args.model_type]

    print("="*60)
    print(f"WILDCHAT EARLY GENERATION - {args.model_family} {args.model_type}")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Continuations per sample: {NUM_CONTINUATIONS}")

    # Load data
    print(f"Loading data from {args.data}...")
    with open(args.data) as f:
        data = json.load(f)
    samples = data['samples']
    print(f"  Loaded {len(samples)} samples")

    # Load model
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Output file
    output_file = OUTPUT_DIR / f"wildchat_early_{args.model_family}_{args.model_type}_{timestamp}.jsonl"
    print(f"Output: {output_file}")

    # Generate
    total_generated = 0
    for sample_idx, sample in enumerate(samples):
        sample_id = sample['sample_id']

        # Generate for EARLY truncation (100 tokens before onset)
        early_para = sample.get('early_paraphrase')
        if early_para:
            print(f"\n[{sample_idx+1}/{len(samples)}] {sample_id} - early...")
            continuations = generate_continuations_batched(
                model, tokenizer, early_para,
                NUM_CONTINUATIONS, base_seed=sample_idx * 1000
            )

            with open(output_file, 'a') as f:
                for i, cont in enumerate(continuations):
                    result = {
                        'sample_id': sample_id,
                        'source': 'wildchat',
                        'truncation_type': 'early',
                        'model_family': args.model_family,
                        'model_type': args.model_type,
                        'continuation_idx': i,
                        'continuation': cont,
                    }
                    f.write(json.dumps(result) + '\n')
            total_generated += len(continuations)
            print(f"    Generated {len(continuations)} continuations (total: {total_generated})")

    print(f"\n✓ Done! Generated {total_generated} continuations")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
