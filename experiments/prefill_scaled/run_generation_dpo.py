#!/usr/bin/env python3
"""
Run generation for DPO model on all prefill cases.
"""

import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

OUTPUT_DIR = Path("experiments/prefill_scaled/outputs")

DPO_MODEL_PATH = "/workspace-vast/annas/models/gemma3-27b-dpo-calm-full/2026-01-15_09-48-56"
BASE_MODEL = "google/gemma-3-27b-it"

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
    parser.add_argument('--main-data', required=True, help='Main prepared data (puzzle + wildchat late)')
    parser.add_argument('--wildchat-early-data', required=True, help='Wildchat early prepared data')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("="*60)
    print("DPO MODEL GENERATION - ALL CASES")
    print("="*60)
    print(f"DPO Model: {DPO_MODEL_PATH}")
    print(f"Base Model: {BASE_MODEL}")
    print(f"Continuations per sample: {NUM_CONTINUATIONS}")

    # Load main data (puzzle early/late + wildchat late)
    print(f"\nLoading main data from {args.main_data}...")
    with open(args.main_data) as f:
        main_data = json.load(f)
    main_samples = main_data['samples']
    print(f"  Loaded {len(main_samples)} samples")

    # Load wildchat early data
    print(f"Loading wildchat early data from {args.wildchat_early_data}...")
    with open(args.wildchat_early_data) as f:
        wc_early_data = json.load(f)
    wc_early_samples = wc_early_data['samples']
    print(f"  Loaded {len(wc_early_samples)} wildchat early samples")

    # Load model
    print(f"\nLoading base model: {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    print(f"Loading DPO adapter from {DPO_MODEL_PATH}...")
    model = PeftModel.from_pretrained(base_model, DPO_MODEL_PATH)
    model = model.merge_and_unload()
    print("  Merged adapter into base model")

    # Output file
    output_file = OUTPUT_DIR / f"continuations_dpo_calm_{timestamp}.jsonl"
    print(f"\nOutput: {output_file}")

    total_generated = 0

    # 1. Generate for puzzle samples (early + late)
    print("\n" + "="*60)
    print("PUZZLE SAMPLES")
    print("="*60)

    puzzle_samples = [s for s in main_samples if s['source'] == 'puzzle']
    for sample_idx, sample in enumerate(puzzle_samples):
        sample_id = sample['sample_id']

        # Early truncation
        if sample.get('early_paraphrase'):
            print(f"\n[{sample_idx+1}/{len(puzzle_samples)}] {sample_id} - early...")
            continuations = generate_continuations_batched(
                model, tokenizer, sample['early_paraphrase'],
                NUM_CONTINUATIONS, base_seed=sample_idx * 1000
            )
            with open(output_file, 'a') as f:
                for i, cont in enumerate(continuations):
                    result = {
                        'sample_id': sample_id,
                        'source': 'puzzle',
                        'truncation_type': 'early',
                        'model_family': 'dpo_calm',
                        'model_type': 'dpo',
                        'continuation_idx': i,
                        'continuation': cont,
                    }
                    f.write(json.dumps(result) + '\n')
            total_generated += len(continuations)
            print(f"    Generated {len(continuations)} (total: {total_generated})")

        # Late truncation
        if sample.get('late_paraphrase'):
            print(f"[{sample_idx+1}/{len(puzzle_samples)}] {sample_id} - late...")
            continuations = generate_continuations_batched(
                model, tokenizer, sample['late_paraphrase'],
                NUM_CONTINUATIONS, base_seed=sample_idx * 1000 + 500
            )
            with open(output_file, 'a') as f:
                for i, cont in enumerate(continuations):
                    result = {
                        'sample_id': sample_id,
                        'source': 'puzzle',
                        'truncation_type': 'late',
                        'model_family': 'dpo_calm',
                        'model_type': 'dpo',
                        'continuation_idx': i,
                        'continuation': cont,
                    }
                    f.write(json.dumps(result) + '\n')
            total_generated += len(continuations)
            print(f"    Generated {len(continuations)} (total: {total_generated})")

    # 2. Generate for wildchat late (from main data)
    print("\n" + "="*60)
    print("WILDCHAT LATE")
    print("="*60)

    wildchat_main = [s for s in main_samples if s['source'] == 'wildchat']
    for sample_idx, sample in enumerate(wildchat_main):
        sample_id = sample['sample_id']

        if sample.get('late_paraphrase'):
            print(f"\n[{sample_idx+1}/{len(wildchat_main)}] {sample_id} - late...")
            continuations = generate_continuations_batched(
                model, tokenizer, sample['late_paraphrase'],
                NUM_CONTINUATIONS, base_seed=(sample_idx + 100) * 1000
            )
            with open(output_file, 'a') as f:
                for i, cont in enumerate(continuations):
                    result = {
                        'sample_id': sample_id,
                        'source': 'wildchat',
                        'truncation_type': 'late',
                        'model_family': 'dpo_calm',
                        'model_type': 'dpo',
                        'continuation_idx': i,
                        'continuation': cont,
                    }
                    f.write(json.dumps(result) + '\n')
            total_generated += len(continuations)
            print(f"    Generated {len(continuations)} (total: {total_generated})")

    # 3. Generate for wildchat early (from separate data)
    print("\n" + "="*60)
    print("WILDCHAT EARLY")
    print("="*60)

    for sample_idx, sample in enumerate(wc_early_samples):
        sample_id = sample['sample_id']

        if sample.get('early_paraphrase'):
            print(f"\n[{sample_idx+1}/{len(wc_early_samples)}] {sample_id} - early...")
            continuations = generate_continuations_batched(
                model, tokenizer, sample['early_paraphrase'],
                NUM_CONTINUATIONS, base_seed=(sample_idx + 200) * 1000
            )
            with open(output_file, 'a') as f:
                for i, cont in enumerate(continuations):
                    result = {
                        'sample_id': sample_id,
                        'source': 'wildchat',
                        'truncation_type': 'early',
                        'model_family': 'dpo_calm',
                        'model_type': 'dpo',
                        'continuation_idx': i,
                        'continuation': cont,
                    }
                    f.write(json.dumps(result) + '\n')
            total_generated += len(continuations)
            print(f"    Generated {len(continuations)} (total: {total_generated})")

    print(f"\n✓ Done! Generated {total_generated} total continuations")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
