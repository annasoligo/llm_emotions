"""Plot loss curves for DPO training experiments."""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_losses_from_log(log_path: str) -> tuple[list, list]:
    """Extract step numbers and losses from a training log."""
    steps = []
    losses = []

    with open(log_path) as f:
        for line in f:
            # Match steering vector format: Step 10: loss=0.1234
            sv_match = re.search(r'Step (\d+): loss=([\d.]+)', line)
            if sv_match:
                steps.append(int(sv_match.group(1)))
                losses.append(float(sv_match.group(2)))
                continue

            # Match HuggingFace format: {'loss': 0.1234, ...}
            hf_match = re.search(r"'loss': ([\d.]+)", line)
            step_match = re.search(r"'epoch': ([\d.]+)", line)
            if hf_match and step_match:
                # Approximate step from epoch (assuming ~280 samples)
                epoch = float(step_match.group(1))
                step = int(epoch * 280)
                steps.append(step)
                losses.append(float(hf_match.group(1)))

    return steps, losses


def main():
    log_dir = Path("/workspace-vast/annas/logs")
    output_dir = Path("/workspace-vast/annas/Ant_Cluster_Notes/gemma_depression_training")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: LoRA models (Full DPO vs R1 epochs)
    ax1 = axes[0, 0]

    # Full DPO
    steps, losses = extract_losses_from_log(log_dir / "dpo_full_120282.out")
    if steps:
        ax1.plot(steps, losses, label='Full DPO (r=64, 466M)', linewidth=2, color='blue')

    # R1 LoRA 1-epoch
    steps, losses = extract_losses_from_log(log_dir / "dpo_minimal_r1_L20_120715.out")
    if steps:
        ax1.plot(steps, losses, label='R1 LoRA 1-ep (26.9K)', linewidth=2, color='orange', linestyle='--')

    # R1 LoRA 2-epoch
    steps, losses = extract_losses_from_log(log_dir / "dpo_minimal_r1_L20_2ep_120798.out")
    if steps:
        ax1.plot(steps, losses, label='R1 LoRA 2-ep (26.9K)', linewidth=2, color='green')

    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('LoRA Models (Full vs R1)')
    ax1.legend()
    ax1.set_ylim(0, 0.8)
    ax1.grid(True, alpha=0.3)

    # Top-right: R1 LoRA layer comparison
    ax2 = axes[0, 1]

    # R1 LoRA L20 alpha128 3ep
    steps, losses = extract_losses_from_log(log_dir / "dpo_r1_a128_3ep_122308.out")
    if steps:
        ax2.plot(steps, losses, label='L20 α128 3ep', linewidth=2, color='#e41a1c')

    # R1 LoRA L16 alpha128 3ep (may still be training)
    try:
        log_l16 = list(log_dir.glob("dpo_r1_L16_122401.out"))[0]
        steps, losses = extract_losses_from_log(log_l16)
        if steps:
            ax2.plot(steps, losses, label='L16 α128 3ep', linewidth=2, color='#377eb8')
    except (IndexError, FileNotFoundError):
        pass

    # R1 LoRA L31 alpha128 3ep (may still be training)
    try:
        log_l31 = list(log_dir.glob("dpo_r1_L31_122402.out"))[0]
        steps, losses = extract_losses_from_log(log_l31)
        if steps:
            ax2.plot(steps, losses, label='L31 α128 3ep', linewidth=2, color='#4daf4a')
    except (IndexError, FileNotFoundError):
        pass

    # R1 LoRA L40 alpha128 3ep (may still be training)
    try:
        log_l40 = list(log_dir.glob("dpo_r1_L40_122460.out"))[0]
        steps, losses = extract_losses_from_log(log_l40)
        if steps:
            ax2.plot(steps, losses, label='L40 α128 3ep', linewidth=2, color='#984ea3')
    except (IndexError, FileNotFoundError):
        pass

    # R1 LoRA L53 alpha128 3ep (may still be training)
    try:
        log_l53 = list(log_dir.glob("dpo_r1_L53_122461.out"))[0]
        steps, losses = extract_losses_from_log(log_l53)
        if steps:
            ax2.plot(steps, losses, label='L53 α128 3ep', linewidth=2, color='#ff7f00')
    except (IndexError, FileNotFoundError):
        pass

    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('R1 LoRA Layer Comparison (α=128, 3ep)')
    ax2.legend()
    ax2.set_ylim(0, 0.8)
    ax2.grid(True, alpha=0.3)

    # Bottom-left: Steering vectors - base LR (1e-3)
    ax3 = axes[1, 0]

    # SV L20 2ep (base lr)
    steps, losses = extract_losses_from_log(log_dir / "dpo_steering_vector_121018.out")
    if steps:
        ax3.plot(steps, losses, label='L20 2ep', linewidth=2, color='#e41a1c')

    # SV L40
    steps, losses = extract_losses_from_log(log_dir / "dpo_steering_vector_L40_121042.out")
    if steps:
        ax3.plot(steps, losses, label='L40 2ep', linewidth=2, color='#377eb8')

    # SV L50
    steps, losses = extract_losses_from_log(log_dir / "dpo_steering_vector_L50_121043.out")
    if steps:
        ax3.plot(steps, losses, label='L50 2ep', linewidth=2, color='#4daf4a')

    # SV L20 3-epoch base LR
    try:
        log_3ep_base = list(log_dir.glob("dpo_sv_L20_3ep_*.out"))[0]
        steps, losses = extract_losses_from_log(log_3ep_base)
        if steps:
            ax3.plot(steps, losses, label='L20 3ep', linewidth=2, color='#984ea3')
    except (IndexError, FileNotFoundError):
        pass

    # SV L10
    try:
        log_l10 = list(log_dir.glob("dpo_sv_L10_*.out"))[0]
        steps, losses = extract_losses_from_log(log_l10)
        if steps:
            ax3.plot(steps, losses, label='L10 3ep', linewidth=2, color='#ff7f00')
    except (IndexError, FileNotFoundError):
        pass

    # SV L20 shuffle
    try:
        log_shuffle = list(log_dir.glob("dpo_sv_L20_shuffle_*.out"))[0]
        steps, losses = extract_losses_from_log(log_shuffle)
        if steps:
            ax3.plot(steps, losses, label='L20 shuffle', linewidth=2, color='#a65628')
    except (IndexError, FileNotFoundError):
        pass

    ax3.set_xlabel('Step')
    ax3.set_ylabel('Loss')
    ax3.set_title('Steering Vectors (lr=1e-3)')
    ax3.legend()
    ax3.set_ylim(0, 0.8)
    ax3.grid(True, alpha=0.3)

    # Bottom-right: Steering vectors - higher LR (5e-3)
    ax4 = axes[1, 1]

    # SV L20 5x LR 2ep
    steps, losses = extract_losses_from_log(log_dir / "dpo_sv_L20_5xlr_122208.out")
    if steps:
        ax4.plot(steps, losses, label='L20 5xLR 2ep', linewidth=2, color='#e41a1c')

    # SV L20 5x LR 3-epoch
    try:
        log_3ep = list(log_dir.glob("dpo_sv_L20_5xlr_3ep_*.out"))[0]
        steps, losses = extract_losses_from_log(log_3ep)
        if steps:
            ax4.plot(steps, losses, label='L20 5xLR 3ep', linewidth=2, color='#377eb8')
    except (IndexError, FileNotFoundError):
        pass

    ax4.set_xlabel('Step')
    ax4.set_ylabel('Loss')
    ax4.set_title('Steering Vectors (lr=5e-3)')
    ax4.legend()
    ax4.set_ylim(0, 0.8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to both locations
    plt.savefig(output_dir / "loss_curves_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(Path("elicitation") / "loss_curves_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved to {output_dir / 'loss_curves_comparison.png'}")
    print(f"Saved to elicitation/loss_curves_comparison.png")


if __name__ == "__main__":
    main()
