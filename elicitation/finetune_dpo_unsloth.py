"""
DPO finetuning with Unsloth for frustration reduction.

Based on Unsloth DPO example. Separate from SFT to maintain backward compatibility.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

# Set before importing torch
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
# Note: Not using PatchDPOTrainer() due to Gemma3 compatibility issues

import torch
from datasets import Dataset
from trl import DPOTrainer, DPOConfig


def format_messages_as_prompt(messages: list, tokenizer) -> str:
    """Format messages list as a prompt string using chat template."""
    # Use tokenizer's chat template to format messages
    # But only include up to (not including) the final assistant response
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # Add the prompt for assistant to continue
    )
    return formatted


def load_dpo_dataset(filepath: str, tokenizer=None) -> Dataset:
    """Load DPO dataset from JSONL file.

    Handles both:
    - v1 format: prompt is a string
    - v2 format: prompt is a list of messages

    For v2, we pre-format using the chat template to avoid TRL issues.
    """
    data = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line)
            prompt = item["prompt"]

            # If prompt is a list of messages, format as string
            if isinstance(prompt, list):
                if tokenizer is not None:
                    # Format using chat template
                    prompt_str = format_messages_as_prompt(prompt, tokenizer)
                else:
                    # Fallback: simple concatenation
                    prompt_str = "\n".join(
                        f"{m['role']}: {m['content']}" for m in prompt
                    )
                data.append({
                    "prompt": prompt_str,
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                })
            else:
                # v1 format: prompt is already a string
                data.append({
                    "prompt": prompt,
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                })

    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="DPO finetuning with Unsloth")
    parser.add_argument("model_name", type=str, help="Base model name or path")
    parser.add_argument("data_file", type=str, help="DPO training data (JSONL)")
    parser.add_argument("output_dir", type=str, help="Output directory for model")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Max prompt length")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--load_in_4bit", type=bool, default=True, help="Load in 4bit")

    args = parser.parse_args()

    print("=" * 60)
    print("DPO FINETUNING WITH UNSLOTH")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_file}")
    print(f"Output: {args.output_dir}")
    print(f"LR: {args.lr}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Beta: {args.beta}")
    print(f"LoRA r: {args.lora_r}, alpha: {args.lora_alpha}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_seq_length,
    )

    # Load dataset (pass tokenizer to format multi-turn prompts)
    print(f"\nLoading dataset from {args.data_file}...")
    dataset = load_dpo_dataset(args.data_file, tokenizer=tokenizer)
    print(f"  Loaded {len(dataset)} DPO pairs")

    # Create timestamp for output
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(args.output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    # DPO config (replaces TrainingArguments for DPO)
    dpo_config = DPOConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        seed=args.seed,
        output_dir=str(output_path),
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
    )

    # DPO Trainer
    print("\nInitializing DPO Trainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth handles this
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting DPO training...")
    dpo_trainer.train()

    # Save
    print(f"\nSaving model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save training config
    config = {
        "model_name": args.model_name,
        "data_file": args.data_file,
        "method": "DPO",
        "lr": args.lr,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "max_prompt_length": args.max_prompt_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "beta": args.beta,
        "seed": args.seed,
        "timestamp": timestamp,
    }
    with open(output_path / "train_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("DPO TRAINING COMPLETE")
    print(f"Model saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
