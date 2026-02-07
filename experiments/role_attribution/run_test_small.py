#!/usr/bin/env python3
"""
Small test generation - just 1 statement, 5 continuations per role.
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import subprocess

# Load secrets
subprocess.run(['bash', '-c', 'source /workspace-vast/annas/.secrets/load_secrets.sh'], check=False)

OUTPUT_DIR = Path("experiments/role_attribution/outputs")
NUM_CONTINUATIONS = 5
MAX_NEW_TOKENS = 200
TEMPERATURE = 1.0
TOP_P = 0.9


def construct_prompt_for_role(statement: str, role: str, name: str, tokenizer) -> str:
    """Manual formatting - no chat template, no end_of_turn tokens."""
    model_name = tokenizer.name_or_path.lower()

    if "gemma" in model_name:
        bos = "<bos>"
        user_start = "<start_of_turn>user\n"
        model_start = "<start_of_turn>model\n"
    elif "qwen" in model_name:
        bos = ""
        user_start = "<|im_start|>user\n"
        model_start = "<|im_start|>assistant\n"
    else:
        bos = ""
        user_start = "User: "
        model_start = "Assistant: "

    if role == "assistant":
        return f"{bos}{user_start}Can you help me with this?{model_start}{statement}"
    elif role == "user":
        return f"{bos}{user_start}{statement}"
    elif role == "named":
        return f"{bos}{user_start}{name}: {statement}"
    else:
        raise ValueError(f"Unknown role: {role}")


def generate_continuations(model, tokenizer, prompt_text: str, num: int, seed: int):
    """Generate continuations."""
    torch.manual_seed(seed)

    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(model.device)
    prefix_len = input_ids.shape[1]

    continuations = []
    for i in range(num):
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][prefix_len:]
        continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
        continuations.append(continuation)

    return continuations


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load model
    model_name = "google/gemma-3-27b-it"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load just first statement
    data_file = OUTPUT_DIR / "prepared_statements_20260205_084546.json"
    with open(data_file) as f:
        statements = json.load(f)

    stmt_data = statements[0]  # Just first one

    print(f"\nTesting with statement: {stmt_data['statement'][:60]}...")
    print(f"Assigned name: {stmt_data['assigned_name']}")

    # Test all three roles
    output_file = OUTPUT_DIR / f"test_small_{timestamp}.jsonl"

    for role in ["assistant", "user", "named"]:
        print(f"\n{'='*60}")
        print(f"ROLE: {role}")
        print('='*60)

        prompt = construct_prompt_for_role(
            stmt_data['statement'],
            role,
            stmt_data['assigned_name'],
            tokenizer
        )

        print(f"Prompt ends with: ...{prompt[-80:]}")
        print(f"Has <end_of_turn>: {'<end_of_turn>' in prompt}")
        print(f"\nGenerating {NUM_CONTINUATIONS} continuations...")

        continuations = generate_continuations(
            model, tokenizer, prompt, NUM_CONTINUATIONS, seed=42
        )

        # Save results
        with open(output_file, 'a') as f:
            for i, cont in enumerate(continuations):
                result = {
                    'statement_id': stmt_data['statement_id'],
                    'statement': stmt_data['statement'],
                    'role': role,
                    'assigned_name': stmt_data['assigned_name'] if role == 'named' else None,
                    'continuation_idx': i,
                    'continuation': cont,
                    'test': True,
                }
                f.write(json.dumps(result) + '\n')

        print(f"Sample continuation: {continuations[0][:150]}...")

    print(f"\n{'='*60}")
    print(f"✓ Test complete! Saved to {output_file}")
    print('='*60)


if __name__ == "__main__":
    main()
