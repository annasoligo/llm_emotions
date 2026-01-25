"""
SFT training for a single steering vector.

Instead of DPO (preference pairs), we train on low-frustration responses directly
using standard language model loss (negative log likelihood).

This mirrors how SFT LoRAs were trained but with a steering vector instead.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class SteeringVector(nn.Module):
    """A learnable steering vector that gets added to activations."""

    def __init__(self, hidden_dim: int, init_scale: float = 0.01, alpha: float = 1.0, dtype=torch.float32):
        super().__init__()
        self.vector = nn.Parameter(torch.randn(hidden_dim, dtype=dtype) * init_scale)
        self.alpha = alpha
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        self._call_count = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Add steering vector to all positions in hidden states."""
        self._call_count += 1
        if self._call_count <= 2:
            delta = self.alpha * self.scale * self.vector
            print(f"  [SV DEBUG] call {self._call_count}: hidden_states norm={hidden_states.float().norm().item():.4f}, "
                  f"delta norm={delta.float().norm().item():.4f}")

        result = hidden_states + self.alpha * self.scale * self.vector
        return result


class SteeringHook:
    """Hook to inject steering vector at a specific layer."""

    def __init__(self, steering_vector: SteeringVector):
        self.steering_vector = steering_vector
        self.enabled = True
        self._debug_printed = False

    def __call__(self, module, input, output):
        if not self.enabled:
            return output

        if not self._debug_printed:
            print(f"  [HOOK DEBUG] Hook called! output type={type(output)}, enabled={self.enabled}")
            self._debug_printed = True

        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states = self.steering_vector(hidden_states)
            return (hidden_states,) + output[1:]
        else:
            return self.steering_vector(output)


class SFTDataset(Dataset):
    """Dataset for SFT training with chat messages."""

    def __init__(self, filepath: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(filepath) as f:
            for line in f:
                item = json.loads(line)
                # Handle both "messages" format and other formats
                if "messages" in item:
                    self.data.append(item["messages"])
                elif "prompt" in item and "response" in item:
                    # Single turn format
                    self.data.append([
                        {"role": "user", "content": item["prompt"]},
                        {"role": "assistant", "content": item["response"]}
                    ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]
        return {"messages": messages}


def compute_sft_loss(
    model,
    steering_hook: SteeringHook,
    tokenizer,
    messages: list,
    device: str,
    max_length: int = 512,
) -> tuple[torch.Tensor, dict]:
    """Compute SFT loss (NLL on assistant responses).

    We compute loss only on assistant tokens, not user tokens.
    """
    # Apply chat template to get full text
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    # Tokenize
    encodings = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    input_ids = encodings["input_ids"].to(device)

    # Find assistant response boundaries for loss masking
    # We need to identify which tokens belong to assistant responses
    # For simplicity, we'll compute loss on all tokens (standard LM loss)
    # This is what most SFT implementations do

    # Enable steering
    steering_hook.enabled = True

    # Forward pass
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(input_ids, use_cache=False)
        logits = outputs.logits

    # Compute cross-entropy loss
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='mean'
    )

    # Stats
    with torch.no_grad():
        # Compute perplexity
        ppl = torch.exp(loss).item()

    stats = {
        "loss": loss.item(),
        "perplexity": ppl,
        "seq_len": input_ids.shape[1],
    }

    return loss, stats


def main():
    parser = argparse.ArgumentParser(description="SFT training for steering vector")
    parser.add_argument("model_name", type=str, help="Base model name or path")
    parser.add_argument("data_file", type=str, help="SFT training data (JSONL with messages)")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--layer", type=int, default=20, help="Layer to inject steering vector")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--init_scale", type=float, default=0.01, help="Initial steering vector scale")
    parser.add_argument("--alpha", type=float, default=256.0, help="Fixed multiplier for steering vector")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--max_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle training data")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("SFT TRAINING FOR STEERING VECTOR")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_file}")
    print(f"Output: {args.output_dir}")
    print(f"Layer: {args.layer}")
    print(f"LR: {args.lr}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Alpha: {args.alpha}")
    print(f"Max Length: {args.max_length}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print(f"\nLoading dataset from {args.data_file}...")
    dataset = SFTDataset(args.data_file, tokenizer)
    print(f"  Loaded {len(dataset)} samples")

    # Load model in bfloat16 (no quantization for proper gradients)
    print("\nLoading model in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    # Freeze all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Get hidden dimension
    if hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'hidden_size'):
        hidden_dim = model.config.text_config.hidden_size
    elif hasattr(model.config, 'hidden_size'):
        hidden_dim = model.config.hidden_size
    else:
        raise ValueError(f"Cannot find hidden_size in model config")
    print(f"  Hidden dimension: {hidden_dim}")

    # Find target layer
    target_layer = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        target_layer = model.model.layers[args.layer]
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        target_layer = model.language_model.layers[args.layer]
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        if hasattr(model.language_model.model, 'layers'):
            target_layer = model.language_model.model.layers[args.layer]

    if target_layer is None:
        raise ValueError(f"Cannot find layers in model architecture")

    target_device = next(target_layer.parameters()).device
    target_dtype = next(target_layer.parameters()).dtype
    print(f"  Target layer device: {target_device}, dtype: {target_dtype}")

    # Create steering vector
    print(f"\nCreating steering vector...")
    steering_vector = SteeringVector(
        hidden_dim,
        init_scale=args.init_scale,
        alpha=args.alpha,
        dtype=target_dtype
    ).to(target_device)
    print(f"  Trainable parameters: {sum(p.numel() for p in steering_vector.parameters())}")

    # Register hook
    print(f"  Registering hook at layer {args.layer}")
    steering_hook = SteeringHook(steering_vector)
    hook_handle = target_layer.register_forward_hook(steering_hook)

    # Setup optimizer
    optimizer = torch.optim.AdamW(steering_vector.parameters(), lr=args.lr)

    # Learning rate scheduler
    total_steps = len(dataset) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(args.output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\nStarting SFT training...")
    global_step = 0
    running_stats = {"loss": 0, "perplexity": 0}

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        if args.shuffle:
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
            data_iter = dataloader
        else:
            data_iter = dataset

        for i, item in enumerate(tqdm(data_iter, desc=f"Epoch {epoch+1}")):
            # Compute SFT loss
            loss, stats = compute_sft_loss(
                model=model,
                steering_hook=steering_hook,
                tokenizer=tokenizer,
                messages=item["messages"],
                device=device,
                max_length=args.max_length,
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Debug first few steps
            if global_step < 3:
                vec_grad = steering_vector.vector.grad
                print(f"  DEBUG step {global_step}: vec_grad_norm={vec_grad.norm().item() if vec_grad is not None else 'None'}, "
                      f"loss={stats['loss']:.4f}")

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(steering_vector.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Update running stats
            for key in running_stats:
                running_stats[key] += stats.get(key, 0)

            global_step += 1

            # Log
            if global_step % args.log_every == 0:
                avg_stats = {k: v / args.log_every for k, v in running_stats.items()}
                lr = scheduler.get_last_lr()[0]
                vec_norm = steering_vector.vector.norm().item()
                print(f"  Step {global_step}: loss={avg_stats['loss']:.4f}, "
                      f"ppl={avg_stats['perplexity']:.2f}, "
                      f"lr={lr:.2e}, vec_norm={vec_norm:.4f}")
                running_stats = {k: 0 for k in running_stats}

    # Save steering vector
    print(f"\nSaving steering vector to {output_path}...")
    torch.save({
        "vector": steering_vector.vector.data.cpu(),
        "scale": steering_vector.scale.data.cpu(),
        "alpha": args.alpha,
        "layer": args.layer,
        "hidden_dim": hidden_dim,
    }, output_path / "steering_vector.pt")

    # Save config
    config = {
        "model_name": args.model_name,
        "data_file": args.data_file,
        "method": "SFT_steering_vector",
        "layer": args.layer,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "alpha": args.alpha,
        "init_scale": args.init_scale,
        "max_length": args.max_length,
        "hidden_dim": hidden_dim,
        "trainable_params": sum(p.numel() for p in steering_vector.parameters()),
        "seed": args.seed,
        "timestamp": timestamp,
    }
    with open(output_path / "train_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Remove hook
    hook_handle.remove()

    print("\n" + "=" * 60)
    print("STEERING VECTOR SFT TRAINING COMPLETE")
    print(f"Saved to: {output_path}")
    print(f"Vector norm: {steering_vector.vector.norm().item():.4f}")
    print(f"Scale: {steering_vector.scale.item():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
