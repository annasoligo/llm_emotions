"""
DPO training for a single steering vector.

Instead of LoRA (modifying weights), we train a single vector that gets
added to activations at a specific layer during forward pass.

This is an extremely minimal intervention - just ~4608 parameters for Gemma-27B.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


class SteeringVector(nn.Module):
    """A learnable steering vector that gets added to activations."""

    def __init__(self, hidden_dim: int, init_scale: float = 0.01, alpha: float = 1.0, dtype=torch.float32):
        super().__init__()
        # The steering vector itself - use specified dtype to match model
        self.vector = nn.Parameter(torch.randn(hidden_dim, dtype=dtype) * init_scale)
        # Fixed alpha multiplier (like LoRA alpha)
        self.alpha = alpha
        # Optional learnable scale (starts at 1.0)
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        self._call_count = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Add steering vector to all positions in hidden states.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
        Returns:
            hidden_states + alpha * scale * vector
        """
        # Debug: print before/after stats on first call
        self._call_count += 1
        if self._call_count <= 2:
            delta = self.alpha * self.scale * self.vector
            print(f"  [SV DEBUG] call {self._call_count}: hidden_states norm={hidden_states.float().norm().item():.4f}, "
                  f"delta norm={delta.float().norm().item():.4f}, "
                  f"vec dtype={self.vector.dtype}, hidden dtype={hidden_states.dtype}")

        # Add steering vector directly - dtypes should match
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

        # Debug: print once to verify hook is being called
        if not self._debug_printed:
            print(f"  [HOOK DEBUG] Hook called! output type={type(output)}, "
                  f"enabled={self.enabled}")
            if isinstance(output, tuple):
                print(f"  [HOOK DEBUG] output[0] shape={output[0].shape}, device={output[0].device}")
            else:
                print(f"  [HOOK DEBUG] output shape={output.shape}, device={output.device}")
            self._debug_printed = True

        # output is typically (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
            hidden_states = self.steering_vector(hidden_states)
            return (hidden_states,) + output[1:]
        else:
            return self.steering_vector(output)


class DPODataset(Dataset):
    """Dataset for DPO training."""

    def __init__(self, filepath: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(filepath) as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]

        # Format prompt if it's a list of messages
        if isinstance(prompt, list):
            prompt_str = self.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_str = prompt

        return {
            "prompt": prompt_str,
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        }


def get_log_probs(model, tokenizer, prompt: str, response: str, device: str, max_length: int = 1024) -> torch.Tensor:
    """Compute log probability of response given prompt.

    Returns sum of log probs for response tokens only.
    """
    # Tokenize prompt + response together
    full_text = prompt + response

    # Get token IDs
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"]

    prompt_len = prompt_ids.shape[1]

    # Truncate if too long (keep prompt, truncate response end)
    if full_ids.shape[1] > max_length:
        full_ids = full_ids[:, :max_length]

    full_ids = full_ids.to(device)

    # Forward pass with gradient checkpointing friendly settings
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        outputs = model(full_ids, use_cache=False)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Shift for next-token prediction
    # logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]  # (1, seq_len-1, vocab)
    shift_labels = full_ids[:, 1:]     # (1, seq_len-1)

    # Compute log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs for actual tokens
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)

    # Only sum log probs for response tokens (after prompt)
    # The log prob at position prompt_len-1 predicts token at prompt_len (first response token)
    # Handle case where prompt is longer than truncated sequence
    if prompt_len >= full_ids.shape[1]:
        # No response tokens after truncation - return a small non-zero value to avoid 0 gradient
        print(f"  [WARN] prompt_len={prompt_len} >= seq_len={full_ids.shape[1]}, no response tokens!")
        # Return sum of all tokens as fallback (not ideal but prevents 0 gradient)
        return token_log_probs.sum()

    response_log_probs = token_log_probs[:, prompt_len-1:]

    return response_log_probs.sum()


def compute_dpo_loss(
    model,
    ref_model,
    steering_hook: SteeringHook,
    tokenizer,
    prompt: str,
    chosen: str,
    rejected: str,
    beta: float,
    device: str,
    max_length: int = 512,
    debug_grads: bool = False,
) -> tuple[torch.Tensor, dict]:
    """Compute DPO loss for a single example.

    L_DPO = -log σ(β * [(log π(y_w|x) - log π_ref(y_w|x)) - (log π(y_l|x) - log π_ref(y_l|x))])

    Returns:
        loss: scalar tensor
        stats: dict with logging info
    """
    # First compute reference log probs (no gradients needed) - this is memory efficient
    steering_hook.enabled = False
    with torch.no_grad():
        ref_chosen_logprob = get_log_probs(model, tokenizer, prompt, chosen, device, max_length)
        ref_rejected_logprob = get_log_probs(model, tokenizer, prompt, rejected, device, max_length)
    torch.cuda.empty_cache()

    # Now compute policy log probs with steering vector (need gradients)
    steering_hook.enabled = True
    chosen_logprob = get_log_probs(model, tokenizer, prompt, chosen, device, max_length)

    # Debug: check if gradient exists
    if debug_grads:
        print(f"  [LP DEBUG] ref_chosen={ref_chosen_logprob.item():.4f}, ref_rejected={ref_rejected_logprob.item():.4f}, "
              f"chosen={chosen_logprob.item():.4f}")
        try:
            grad = torch.autograd.grad(chosen_logprob, steering_hook.steering_vector.vector, retain_graph=True, allow_unused=True)[0]
            print(f"  [GRAD DEBUG] chosen_logprob grad exists: {grad is not None}, "
                  f"grad_norm={grad.norm().item() if grad is not None else 'N/A'}")
        except Exception as e:
            print(f"  [GRAD DEBUG] Error computing grad: {e}")

    torch.cuda.empty_cache()
    rejected_logprob = get_log_probs(model, tokenizer, prompt, rejected, device, max_length)

    # Compute implicit rewards (how much we've changed from reference)
    # ref values are detached (no grad), so gradients only flow through chosen_logprob and rejected_logprob
    chosen_reward = chosen_logprob - ref_chosen_logprob.detach()
    rejected_reward = rejected_logprob - ref_rejected_logprob.detach()

    # DPO loss: we want chosen_reward > rejected_reward
    margin = beta * (chosen_reward - rejected_reward)
    loss = -F.logsigmoid(margin)

    # Compute accuracy (did we rank correctly?)
    with torch.no_grad():
        accuracy = (chosen_reward > rejected_reward).float()

    stats = {
        "loss": loss.item(),
        "chosen_reward": chosen_reward.item(),
        "rejected_reward": rejected_reward.item(),
        "margin": margin.item(),
        "accuracy": accuracy.item(),
    }

    return loss, stats


def main():
    parser = argparse.ArgumentParser(description="DPO training for steering vector")
    parser.add_argument("model_name", type=str, help="Base model name or path")
    parser.add_argument("data_file", type=str, help="DPO training data (JSONL)")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--layer", type=int, default=20, help="Layer to inject steering vector")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--init_scale", type=float, default=0.01, help="Initial steering vector scale")
    parser.add_argument("--alpha", type=float, default=1.0, help="Fixed multiplier for steering vector (like LoRA alpha)")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--log_every", type=int, default=10, help="Log every N steps")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length (reduces memory)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle training data")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("DPO TRAINING FOR STEERING VECTOR")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Data: {args.data_file}")
    print(f"Output: {args.output_dir}")
    print(f"Layer: {args.layer}")
    print(f"LR: {args.lr}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Beta: {args.beta}")
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
    dataset = DPODataset(args.data_file, tokenizer)
    print(f"  Loaded {len(dataset)} DPO pairs")
    print(f"  Shuffle: {args.shuffle}")

    # Load model in bfloat16 (no quantization - needed for proper gradients)
    print("\nLoading model in bfloat16 (no quantization for proper gradients)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()  # Keep model in eval mode, only train steering vector
    # NOTE: Do NOT enable gradient checkpointing - it conflicts with forward hooks
    # and prevents gradients from flowing through the steering vector.

    # Freeze all model parameters (only steering vector will be trained)
    for param in model.parameters():
        param.requires_grad = False

    # Get hidden dimension from model config
    # Handle multimodal models with nested text_config (like Gemma-3)
    if hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'hidden_size'):
        hidden_dim = model.config.text_config.hidden_size
    elif hasattr(model.config, 'hidden_size'):
        hidden_dim = model.config.hidden_size
    else:
        raise ValueError(f"Cannot find hidden_size in model config: {model.config}")
    print(f"  Hidden dimension: {hidden_dim}")

    # Find the right module to hook FIRST (need to know device before creating steering vector)
    # For Gemma/Llama-style models, we hook after the layer's output
    target_layer = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Standard Llama/Gemma architecture
        target_layer = model.model.layers[args.layer]
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'layers'):
        # Multimodal models like Gemma-3 (Gemma3ForConditionalGeneration)
        target_layer = model.language_model.layers[args.layer]
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'model'):
        # Some multimodal models have nested model
        if hasattr(model.language_model.model, 'layers'):
            target_layer = model.language_model.model.layers[args.layer]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-style architecture
        target_layer = model.transformer.h[args.layer]

    if target_layer is None:
        # Debug: print model structure
        print(f"Model type: {type(model)}")
        print(f"Model attributes: {[a for a in dir(model) if not a.startswith('_')][:20]}")
        if hasattr(model, 'model'):
            print(f"model.model type: {type(model.model)}")
            print(f"model.model attributes: {[a for a in dir(model.model) if not a.startswith('_')][:20]}")
        if hasattr(model, 'language_model'):
            print(f"language_model type: {type(model.language_model)}")
            print(f"language_model attributes: {[a for a in dir(model.language_model) if not a.startswith('_')][:20]}")
            if hasattr(model.language_model, 'model'):
                print(f"language_model.model type: {type(model.language_model.model)}")
                print(f"language_model.model attributes: {[a for a in dir(model.language_model.model) if not a.startswith('_')][:20]}")
        # Try to find layers anywhere
        for name, module in model.named_modules():
            if 'layers' in name and '.0' not in name:
                print(f"Found potential layers at: {name}")
        raise ValueError(f"Cannot find layers in model architecture")

    # Get the device of the target layer (important for device_map="auto")
    target_device = next(target_layer.parameters()).device
    print(f"  Target layer device: {target_device}")

    # Create steering vector on the SAME device and dtype as target layer
    print(f"\nCreating steering vector...")
    # Get dtype from a parameter in the target layer
    target_dtype = next(target_layer.parameters()).dtype
    print(f"  Target layer dtype: {target_dtype}")
    steering_vector = SteeringVector(hidden_dim, init_scale=args.init_scale, alpha=args.alpha, dtype=target_dtype).to(target_device)
    print(f"  Trainable parameters: {sum(p.numel() for p in steering_vector.parameters())}")
    print(f"  Alpha (fixed multiplier): {args.alpha}")
    print(f"  Steering vector device: {steering_vector.vector.device}")

    # Register hook at specified layer
    print(f"  Registering hook at layer {args.layer}")
    steering_hook = SteeringHook(steering_vector)
    hook_handle = target_layer.register_forward_hook(steering_hook)

    # Setup optimizer (only for steering vector)
    optimizer = torch.optim.AdamW(steering_vector.parameters(), lr=args.lr)

    # Learning rate scheduler with warmup
    total_steps = len(dataset) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            # Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Create output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = Path(args.output_dir) / timestamp
    output_path.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\nStarting DPO training...")
    global_step = 0
    running_stats = {"loss": 0, "accuracy": 0, "margin": 0}

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # Create DataLoader for this epoch (shuffle if requested)
        if args.shuffle:
            # Use DataLoader with shuffle - need custom collate since we process one at a time
            from torch.utils.data import DataLoader
            dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
            data_iter = dataloader
        else:
            data_iter = dataset

        for i, item in enumerate(tqdm(data_iter, desc=f"Epoch {epoch+1}")):
            # Compute DPO loss
            loss, stats = compute_dpo_loss(
                model=model,
                ref_model=model,  # We use the same model, steering_hook.enabled controls the difference
                steering_hook=steering_hook,
                tokenizer=tokenizer,
                prompt=item["prompt"],
                chosen=item["chosen"],
                rejected=item["rejected"],
                beta=args.beta,
                device=device,
                max_length=args.max_length,
                debug_grads=(global_step < 3),  # Debug first 3 steps
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Debug: check gradients
            if global_step < 3:
                vec_grad = steering_vector.vector.grad
                scale_grad = steering_vector.scale.grad
                print(f"  DEBUG step {global_step}: vec_grad_norm={vec_grad.norm().item() if vec_grad is not None else 'None'}, "
                      f"scale_grad={scale_grad.item() if scale_grad is not None else 'None'}, "
                      f"chosen_lp={stats['chosen_reward']:.4f}, rejected_lp={stats['rejected_reward']:.4f}")

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
                      f"acc={avg_stats['accuracy']:.2f}, margin={avg_stats['margin']:.2f}, "
                      f"lr={lr:.2e}, scale={steering_vector.scale.item():.3f}, vec_norm={vec_norm:.4f}")
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
        "method": "DPO_steering_vector",
        "layer": args.layer,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "beta": args.beta,
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
    print("STEERING VECTOR DPO TRAINING COMPLETE")
    print(f"Saved to: {output_path}")
    print(f"Vector norm: {steering_vector.vector.norm().item():.4f}")
    print(f"Scale: {steering_vector.scale.item():.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
