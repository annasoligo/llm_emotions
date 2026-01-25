"""
SFT finetuning with vanilla TRL (no Unsloth).

Uses PEFT for LoRA and bitsandbytes for 4-bit quantization.
Supports layer-specific LoRA like the DPO version.
"""

import os
import sys

# Disable Unsloth's automatic patching
os.environ["UNSLOTH_DISABLE_PATCH"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Import TRL BEFORE anything else to avoid Unsloth hooks
from trl import SFTTrainer, SFTConfig

import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_sft_dataset(filepath: str, tokenizer) -> Dataset:
    """Load SFT dataset from JSONL file with messages format."""
    data = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line)

            # Handle messages format
            if "messages" in item:
                # Format using chat template
                text = tokenizer.apply_chat_template(
                    item["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            elif "prompt" in item and "response" in item:
                # Simple prompt/response format
                messages = [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": item["response"]}
                ]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                # Assume it's already formatted text
                text = item.get("text", str(item))

            data.append({"text": text})

    return Dataset.from_list(data)


def main():
    parser = argparse.ArgumentParser(description="SFT finetuning with TRL")
    parser.add_argument("model_name", type=str, help="Base model name or path")
    parser.add_argument("data_file", type=str, help="SFT training data (JSONL)")
    parser.add_argument("output_dir", type=str, help="Output directory for model")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--target_modules", type=str, nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                        help="Target modules for LoRA")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Specific layers to apply LoRA (e.g., --layers 20). If None, applies to all layers")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("=" * 60)
    print("SFT FINETUNING WITH TRL (NO UNSLOTH)")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_file}")
    print(f"Output: {args.output_dir}")
    print(f"LR: {args.lr}")
    print(f"Epochs: {args.num_train_epochs}")
    print(f"Batch size: {args.per_device_train_batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"LoRA r: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"Target modules: {args.target_modules}")
    print(f"Layers: {args.layers if args.layers else 'all'}")
    print("=" * 60)

    # Load tokenizer first
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"\nLoading dataset from {args.data_file}...")
    dataset = load_sft_dataset(args.data_file, tokenizer)
    print(f"  Loaded {len(dataset)} samples")

    # 4-bit quantization config
    print("\nLoading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Prepare for kbit training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    print("Adding LoRA adapters...")
    lora_kwargs = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "lora_dropout": 0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    if args.layers is not None:
        lora_kwargs["layers_to_transform"] = args.layers
    lora_config = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create timestamp for output
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(args.output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    # SFT config
    sft_config = SFTConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=args.logging_steps,
        optim="paged_adamw_8bit",
        seed=args.seed,
        output_dir=str(output_path),
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    # SFT Trainer
    print("\nInitializing SFT Trainer...")
    sft_trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting SFT training...")
    sft_trainer.train()

    # Save
    print(f"\nSaving model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save training config
    config = {
        "model_name": args.model_name,
        "data_file": args.data_file,
        "method": "SFT",
        "lr": args.lr,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "layers": args.layers,
        "seed": args.seed,
        "timestamp": timestamp,
    }
    with open(output_path / "train_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("SFT TRAINING COMPLETE")
    print(f"Model saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
