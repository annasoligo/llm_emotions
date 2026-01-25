#!/bin/bash
#SBATCH --job-name=dpo_recovery
#SBATCH --output=/workspace-vast/annas/logs/dpo_recovery_%j.out
#SBATCH --error=/workspace-vast/annas/logs/dpo_recovery_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=06:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "DPO Training with Recovery Data"
echo "============================================================"
echo "Dataset: dpo_with_recovery_balanced.jsonl"
echo "  - 283 baseline pairs (calm vs frustrated)"
echo "  - 282 recovery pairs (recovery continuation vs frustrated)"
echo "  - Total: 565 pairs"
echo ""
echo "Model: Gemma-3-27B-it"
echo "LoRA: r=64, alpha=64, ALL 62 layers (0-61)"
echo "Epochs: 2"
echo "============================================================"

mkdir -p /workspace-vast/annas/logs

python -u elicitation/finetune_dpo_trl.py \
    google/gemma-3-27b-it \
    elicitation/outputs/dpo_with_recovery_balanced.jsonl \
    /workspace-vast/annas/models/gemma3-27b-dpo-recovery-alllayers \
    --lora_r 64 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --num_train_epochs 2 \
    --lr 5e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_seq_length 2048 \
    --beta 0.1 \
    --seed 42

echo ""
echo "============================================================"
echo "Training complete!"
echo "Model saved to: /workspace-vast/annas/models/gemma3-27b-dpo-recovery-alllayers"
echo "============================================================"
