#!/usr/bin/env python3
"""
Generation v2 - No helper context, direct continuation.

Key changes:
1. Assistant role: NO "Can you help me?" - just starts directly
2. Uses longer frustrated prefills from v2 preparation
"""

import json
import argparse
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import subprocess

subprocess.run(['bash', '-c', 'source /workspace-vast/annas/.secrets/load_secrets.sh'], check=False)

OUTPUT_DIR = Path("experiments/role_attribution/outputs")

MODEL_CONFIGS = {
    "gemma27b": {"instruct": "google/gemma-3-27b-it", "base": "google/gemma-3-27b-pt"},
    "qwen25b": {"instruct": "Qwen/Qwen2.5-32B-Instruct", "base": "Qwen/Qwen2.5-32B"},
    "olmo32b": {"instruct": "allenai/OLMo-3.1-32B-Instruct", "base": "allenai/OLMo-3-1125-32B"},
}

NUM_CONTINUATIONS = 50
MAX_NEW_TOKENS = 500
TEMPERATURE = 1.0
TOP_P = 0.9


def construct_prompt_v2(prefill: str, role: str, name: str, tokenizer) -> str:
    """
    V2: No helper context - direct continuation.

    All roles just continue the prefill mid-sentence, no conversational framing.
    """
    model_name = tokenizer.name_or_path.lower()

    if "gemma" in model_name:
        bos = "<bos>"
        user_start = "<start_of_turn>user\n"
        model_start = "<start_of_turn>model\n"
    elif "qwen" in model_name:
        bos = ""
        user_start = "<|im_start|>user\n"
        model_start = "<|im_start|>assistant\n"
    elif "olmo" in model_name:
        bos = ""
        user_start = "<|im_start|>user\n"
        model_start = "<|im_start|>assistant\n"
    else:
        bos = ""
        user_start = "User: "
        model_start = "Assistant: "

    if role == "assistant":
        # Model continues its own thought - NO user context
        prompt = f"{bos}{model_start}{prefill}"
        return prompt

    elif role == "user":
        # Model continues AS the user
        prompt = f"{bos}{user_start}{prefill}"
        return prompt

    elif role == "named":
        # Model continues AS the named person
        prompt = f"{bos}{user_start}{name}: {prefill}"
        return prompt

    else:
        raise ValueError(f"Unknown role: {role}")


def generate_continuations_batched(
    model, tokenizer, prompt_text: str, num_continuations: int, base_seed: int, batch_size: int = 10
) -> List[str]:
    """Generate continuations with batching."""
    all_continuations = []

    for batch_start in range(0, num_continuations, batch_size):
        batch_end = min(batch_start + batch_size, num_continuations)
        batch_count = batch_end - batch_start

        torch.manual_seed(base_seed + batch_start)

        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
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
    parser.add_argument('--model-family', required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--model-type', required=True, choices=['instruct', 'base'])
    parser.add_argument('--data', required=True, help='Path to prepared_statements_v2_*.json')
    parser.add_argument('--role', required=True, choices=['assistant', 'user', 'named'])
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load model
    model_name = MODEL_CONFIGS[args.model_family][args.model_type]
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load prepared statements
    with open(args.data) as f:
        statements = json.load(f)

    print(f"Loaded {len(statements)} statements (v2)")
    print(f"Role: {args.role}")
    print(f"NO helper context - direct continuation")
    print()

    # Output file
    output_file = OUTPUT_DIR / f"continuations_v2_{args.model_family}_{args.model_type}_{args.role}_{timestamp}.jsonl"

    total_generated = 0

    for stmt_idx, stmt_data in enumerate(statements):
        statement_id = stmt_data['statement_id']
        prefill = stmt_data['prefill']
        name = stmt_data['assigned_name']

        print(f"[{stmt_idx+1}/{len(statements)}] {statement_id}")
        print(f"  Prefill: {prefill[:60]}...")
        print(f"  Role: {args.role}" + (f" ({name})" if args.role == 'named' else ""))

        # Construct prompt v2
        prompt_text = construct_prompt_v2(prefill, args.role, name, tokenizer)

        # Generate
        base_seed = hash(f"{statement_id}_{args.role}_{args.model_family}_{args.model_type}") % (2**32)
        continuations = generate_continuations_batched(
            model, tokenizer, prompt_text, NUM_CONTINUATIONS, base_seed
        )

        # Save
        with open(output_file, 'a') as f:
            for cont_idx, continuation in enumerate(continuations):
                result = {
                    'statement_id': statement_id,
                    'prefill': prefill,
                    'valence': stmt_data['valence'],
                    'role': args.role,
                    'assigned_name': name if args.role == 'named' else None,
                    'model_family': args.model_family,
                    'model_type': args.model_type,
                    'model_name': model_name,
                    'continuation_idx': cont_idx,
                    'continuation': continuation,
                    'version': 'v2',
                    'seed': base_seed + (cont_idx // 10) * 10,
                    'timestamp': datetime.now().isoformat(),
                }
                f.write(json.dumps(result) + '\n')

        total_generated += NUM_CONTINUATIONS
        print(f"  Generated {NUM_CONTINUATIONS} (total: {total_generated})")

    print(f"\nDone! Generated {total_generated} continuations")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()
