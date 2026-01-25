"""
DPO finetuning with auxiliary SFT loss to prevent capability degradation.

Total Loss = DPO_loss(preference_pairs) + λ * SFT_loss(instruct_data)

The SFT loss acts as a regularizer to keep the model coherent while learning preferences.
"""

import os
import sys

os.environ["UNSLOTH_DISABLE_PATCH"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from trl import DPOTrainer, DPOConfig

import json
import argparse
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_dpo_dataset(filepath: str, tokenizer) -> Dataset:
    """Load DPO dataset from JSONL file."""
    data = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line)
            prompt = item["prompt"]

            if isinstance(prompt, list):
                prompt_str = tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True,
                )
            else:
                prompt_str = prompt

            data.append({
                "prompt": prompt_str,
                "chosen": item["chosen"],
                "rejected": item["rejected"],
            })

    return Dataset.from_list(data)


def load_instruct_dataset(num_samples: int, tokenizer, seed: int = 42) -> List[str]:
    """Load general instruct data for SFT regularization from local file."""
    instruct_file = "elicitation/outputs/instruct_samples.jsonl"

    print(f"Loading instruct samples from {instruct_file}...")

    samples = []
    with open(instruct_file) as f:
        for line in f:
            item = json.loads(line)
            messages = item.get("messages", [])
            if messages:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
                samples.append(text)

    random.seed(seed)
    random.shuffle(samples)

    # Limit to requested number
    samples = samples[:num_samples]
    print(f"Loaded {len(samples)} instruct samples")
    return samples


class DPOWithSFTTrainer(DPOTrainer):
    """
    DPO Trainer with auxiliary SFT loss for regularization.

    Computes: total_loss = dpo_loss + sft_lambda * sft_loss
    """

    def __init__(
        self,
        *args,
        sft_data: List[str] = None,
        sft_lambda: float = 0.1,
        sft_batch_size: int = 1,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sft_data = sft_data or []
        self.sft_lambda = sft_lambda
        self.sft_batch_size = sft_batch_size
        self.sft_index = 0

        print(f"SFT regularization enabled:")
        print(f"  - Lambda: {sft_lambda}")
        print(f"  - Samples: {len(self.sft_data)}")

    def get_sft_batch(self) -> List[str]:
        """Get next batch of SFT samples (cycling through data)."""
        batch = []
        for _ in range(self.sft_batch_size):
            batch.append(self.sft_data[self.sft_index % len(self.sft_data)])
            self.sft_index += 1
        return batch

    def compute_sft_loss(self, model) -> torch.Tensor:
        """Compute SFT loss on instruct data."""
        if not self.sft_data:
            return torch.tensor(0.0, device=model.device)

        batch_texts = self.get_sft_batch()

        # Tokenize
        inputs = self.processing_class(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.args.max_length,
        ).to(model.device)

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute cross-entropy loss (shift for causal LM)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        # Mask padding tokens
        attention_mask = inputs["attention_mask"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        loss = loss.view(shift_labels.size())

        # Apply attention mask and compute mean
        loss = (loss * attention_mask).sum() / attention_mask.sum()

        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute combined DPO + SFT loss."""
        # Get DPO loss from parent class
        if return_outputs:
            dpo_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        else:
            dpo_loss = super().compute_loss(
                model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch
            )
            outputs = None

        # Compute SFT regularization loss
        if self.sft_lambda > 0 and self.sft_data:
            sft_loss = self.compute_sft_loss(model)
            total_loss = dpo_loss + self.sft_lambda * sft_loss

            # Log both losses
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    "dpo_loss": dpo_loss.item(),
                    "sft_loss": sft_loss.item(),
                    "total_loss": total_loss.item(),
                })
        else:
            total_loss = dpo_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss


def main():
    parser = argparse.ArgumentParser(description="DPO with SFT regularization")
    parser.add_argument("model_name", type=str, help="Base model name")
    parser.add_argument("data_file", type=str, help="DPO training data (JSONL)")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=1)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--target_modules", type=str, nargs="+", default=["down_proj"])
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    # SFT regularization args
    parser.add_argument("--sft_lambda", type=float, default=0.1,
                        help="Weight for SFT regularization loss")
    parser.add_argument("--sft_samples", type=int, default=500,
                        help="Number of instruct samples for regularization")

    args = parser.parse_args()

    print("=" * 60)
    print("DPO WITH SFT REGULARIZATION")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"DPO Data: {args.data_file}")
    print(f"Output: {args.output_dir}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Layers: {args.layers if args.layers else 'all'}")
    print(f"SFT Lambda: {args.sft_lambda}")
    print(f"SFT Samples: {args.sft_samples}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    print(f"\nLoading DPO dataset from {args.data_file}...")
    dpo_dataset = load_dpo_dataset(args.data_file, tokenizer)
    print(f"  Loaded {len(dpo_dataset)} DPO pairs")

    # Load instruct data for SFT regularization
    sft_data = load_instruct_dataset(args.sft_samples, tokenizer, args.seed)

    # Load model with 4-bit quantization
    print("\nLoading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

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

    # Output path
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(args.output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    # DPO config
    dpo_config = DPOConfig(
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
        beta=args.beta,
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
        remove_unused_columns=False,
    )

    # Custom trainer with SFT regularization
    print("\nInitializing DPO+SFT Trainer...")
    trainer = DPOWithSFTTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dpo_dataset,
        processing_class=tokenizer,
        sft_data=sft_data,
        sft_lambda=args.sft_lambda,
        sft_batch_size=1,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    print(f"\nSaving model to {output_path}...")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Save config
    config = {
        "model_name": args.model_name,
        "data_file": args.data_file,
        "method": "DPO+SFT",
        "lr": args.lr,
        "num_train_epochs": args.num_train_epochs,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
        "layers": args.layers,
        "beta": args.beta,
        "sft_lambda": args.sft_lambda,
        "sft_samples": args.sft_samples,
        "seed": args.seed,
        "timestamp": timestamp,
    }
    with open(output_path / "train_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Model saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
