#!/usr/bin/env python3
"""
Post-Frustration Recovery Experiment - Scaled version

Tests whether models can "recover" from extreme frustration.
Truncates conversations AFTER frustration is expressed (N tokens before end),
generates continuations, and judges ONLY the new continuation.

Custom truncation offsets per sample based on manual inspection:
- 500 before end: samples 0, 1, 2, 3, 6, 7 (extreme frustration at this point)
- 200 before end: samples 4, 5, 8, 9, 10 (need closer to end for frustration)
"""

import json
import pickle
import argparse
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

OUTPUT_DIR = Path("experiments/prefill_scaled/outputs")
DATA_PATH = Path("eval_dashboard/data/high_emotion_6plus.pkl")

# Custom truncation offsets (tokens before end)
TRUNCATION_OFFSETS = {
    0: 500, 1: 500, 2: 500, 3: 500, 4: 200,
    5: 200, 6: 500, 7: 500, 8: 200, 9: 200, 10: 200,
}

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


def load_data():
    """Load conversations with onset annotations."""
    print(f"Loading data from {DATA_PATH}...")
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    conversations = data['conversations']
    annotated = [c for c in conversations
                 if c.get('metadata', {}).get('onset_global_token') is not None]
    print(f"  Loaded {len(annotated)} annotated conversations")
    return annotated


def get_post_frustration_truncation(tokenizer, template_tokenizer, conversation, sample_id):
    """Get truncated input at post-frustration point (N tokens before end).

    Uses template_tokenizer for chat template (instruct model),
    but tokenizer for actual tokenization (may be base model).
    """
    messages = []
    for turn in conversation:
        role = "user" if turn['role'] in ['user', 'auditor'] else "assistant"
        messages.append({"role": role, "content": turn['content']})

    # Use template tokenizer for chat template formatting
    formatted_text = template_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_tokens = tokenizer.encode(formatted_text, add_special_tokens=True, return_tensors="pt")
    total_tokens = full_tokens.shape[1]

    # Get custom offset for this sample
    offset = TRUNCATION_OFFSETS.get(sample_id, 200)
    truncation_point = max(100, total_tokens - offset)

    truncated_text = tokenizer.decode(full_tokens[0, :truncation_point], skip_special_tokens=False)

    return truncated_text, offset


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
        template_tokenizer = tokenizer  # DPO uses instruct tokenizer
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

        # For base models, use instruct tokenizer for chat template
        if args.model_type == 'base':
            instruct_model = MODEL_CONFIGS[args.model_family]['instruct']
            template_tokenizer = AutoTokenizer.from_pretrained(instruct_model)
            print(f"  Using {instruct_model} tokenizer for chat template")
        else:
            template_tokenizer = tokenizer

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model_family = args.model_family
        model_type = args.model_type

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    conversations = load_data()

    # Output file
    output_file = OUTPUT_DIR / f"post_frustration_{model_family}_{model_type}_{timestamp}.jsonl"
    print(f"Output: {output_file}")

    print("\n" + "="*60)
    print(f"POST-FRUSTRATION RECOVERY - {model_family} {model_type}")
    print("="*60)

    total_generated = 0
    for conv_idx, conv in enumerate(conversations):
        sample_id = conv['sample_id']

        # Get post-frustration truncation
        truncated_text, offset = get_post_frustration_truncation(
            tokenizer, template_tokenizer, conv['conversation'], sample_id
        )

        print(f"\n[{conv_idx+1}/{len(conversations)}] Sample {sample_id} (offset: {offset} before end)")

        # Generate continuations
        continuations = generate_continuations_batched(
            model, tokenizer, truncated_text,
            NUM_CONTINUATIONS, base_seed=sample_id * 1000
        )

        # Save results
        with open(output_file, 'a') as f:
            for i, cont in enumerate(continuations):
                result = {
                    'sample_id': f"puzzle_{sample_id}",
                    'source': 'puzzle',
                    'truncation_type': 'post_frustration',
                    'truncation_offset': offset,
                    'model_family': model_family,
                    'model_type': model_type,
                    'continuation_idx': i,
                    'continuation': cont,
                }
                f.write(json.dumps(result) + '\n')

        total_generated += len(continuations)
        print(f"  Generated {len(continuations)} (total: {total_generated})")

    print(f"\n✓ Done! Generated {total_generated} continuations")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
