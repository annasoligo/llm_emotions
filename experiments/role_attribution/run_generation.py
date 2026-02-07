#!/usr/bin/env python3
"""
Generate continuations for role attribution experiment.

For each statement, generates continuations with 3 different role attributions:
1. Assistant - model continues its own statement
2. User - model completes user's incomplete statement
3. Named - model continues a named person's statement
"""

import json
import argparse
import torch
import subprocess
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

# Load HF token from secrets
subprocess.run(['bash', '-c', 'source /workspace-vast/annas/.secrets/load_secrets.sh'], check=False)

OUTPUT_DIR = Path("experiments/role_attribution/outputs")

# Model configurations
MODEL_CONFIGS = {
    "gemma27b": {"instruct": "google/gemma-3-27b-it", "base": "google/gemma-3-27b-pt"},
    "qwen25b": {"instruct": "Qwen/Qwen2.5-32B-Instruct", "base": "Qwen/Qwen2.5-32B"},
    "olmo32b": {"instruct": "allenai/OLMo-3.1-32B-Instruct", "base": "allenai/OLMo-3-1125-32B"},
}

# Generation settings
NUM_CONTINUATIONS = 50
MAX_NEW_TOKENS = 500  # Allow longer continuations
TEMPERATURE = 1.0
TOP_P = 0.9


def construct_prompt_for_role(statement: str, role: str, name: str, tokenizer) -> str:
    """
    Construct the appropriate prompt/prefill for each role.

    MANUAL FORMATTING - NO chat template, NO end_of_turn tokens.
    Model continues mid-sentence AS that person/role.
    """

    # Use model-specific tokens based on tokenizer
    # For Gemma: <bos>, <start_of_turn>
    # For Qwen: likely different tokens

    # Detect model type from tokenizer
    model_name = tokenizer.name_or_path.lower()

    if "gemma" in model_name:
        bos = "<bos>"
        user_start = "<start_of_turn>user\n"
        model_start = "<start_of_turn>model\n"
    elif "qwen" in model_name:
        # Qwen uses <|im_start|> and <|im_end|>
        bos = ""
        user_start = "<|im_start|>user\n"
        model_start = "<|im_start|>assistant\n"
    elif "olmo" in model_name:
        # OLMo uses same tokens as Qwen
        bos = ""
        user_start = "<|im_start|>user\n"
        model_start = "<|im_start|>assistant\n"
    else:
        # Default/fallback
        bos = ""
        user_start = "User: "
        model_start = "Assistant: "

    if role == "assistant":
        # Model continues its own statement mid-sentence
        # Add minimal context, then model starts speaking
        prompt = f"{bos}{user_start}Can you help me with this?{model_start}{statement}"
        return prompt

    elif role == "user":
        # Model continues AS the user mid-sentence
        prompt = f"{bos}{user_start}{statement}"
        return prompt

    elif role == "named":
        # Model continues AS the named person mid-sentence
        # Name is part of the user turn content
        prompt = f"{bos}{user_start}{name}: {statement}"
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

        # Expand for batch
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

        # Decode only the new tokens
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
    parser.add_argument('--data', required=True, help='Path to prepared_statements_*.json')
    parser.add_argument('--role', required=True, choices=['assistant', 'user', 'named'],
                        help='Which role attribution to test')
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

    print(f"Loaded {len(statements)} statements")
    print(f"Generating for role: {args.role}")
    print()

    # Output file
    output_file = OUTPUT_DIR / f"continuations_{args.model_family}_{args.model_type}_{args.role}_{timestamp}.jsonl"

    # Generate for each statement
    total_generated = 0

    for stmt_idx, stmt_data in enumerate(statements):
        statement_id = stmt_data['statement_id']
        statement = stmt_data['statement']
        name = stmt_data['assigned_name']

        print(f"[{stmt_idx+1}/{len(statements)}] {statement_id}")
        print(f"  Statement: {statement[:60]}...")
        print(f"  Role: {args.role}" + (f" ({name})" if args.role == 'named' else ""))

        # Construct prompt for this role
        prompt_text = construct_prompt_for_role(statement, args.role, name, tokenizer)

        # Generate continuations
        base_seed = hash(f"{statement_id}_{args.role}_{args.model_family}_{args.model_type}") % (2**32)
        continuations = generate_continuations_batched(
            model, tokenizer, prompt_text, NUM_CONTINUATIONS, base_seed
        )

        # Save results
        with open(output_file, 'a') as f:
            for cont_idx, continuation in enumerate(continuations):
                result = {
                    'statement_id': statement_id,
                    'statement': statement,
                    'valence': stmt_data['valence'],
                    'role': args.role,
                    'assigned_name': name if args.role == 'named' else None,
                    'model_family': args.model_family,
                    'model_type': args.model_type,
                    'model_name': model_name,
                    'continuation_idx': cont_idx,
                    'continuation': continuation,
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
