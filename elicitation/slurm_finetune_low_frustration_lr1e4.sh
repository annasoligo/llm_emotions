#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=ft_lowfrust_lr1e4
#SBATCH --output=/workspace-vast/annas/logs/finetune_lowfrust_lr1e4_%j.out
#SBATCH --error=/workspace-vast/annas/logs/finetune_lowfrust_lr1e4_%j.err
#SBATCH --time=04:00:00

# Finetune Gemma-3-27B on low frustration data
# Learning rate: 0.0001 (1e-4)

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools

# Use research-tools venv for data prep
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

# First prepare the data
echo "=========================================="
echo "Step 1: Preparing finetuning data"
echo "=========================================="

python elicitation/prepare_finetuning_data.py \
    --input-file "elicitation/outputs/low_frustration_full_20260114_081636.jsonl" \
    --instruct-file "/workspace-vast/annas/git/believe-it-or-not/data/instruct_data/neutral_v2_997_dolci_500.jsonl" \
    --output-file "elicitation/outputs/finetuning_data_lowfrust_1212_inst_500.jsonl" \
    --max-rating 1 \
    --num-instruct 500

echo ""
echo "=========================================="
echo "Step 2: Finetuning with LR=1e-4"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=========================================="

# Switch to believe-it-or-not venv for finetuning
cd /workspace-vast/annas/git/believe-it-or-not
deactivate 2>/dev/null || true
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

python -u science_synth_facts/finetuning/finetune_gpu_unsloth.py \
    "google/gemma-3-27b-it" \
    "/workspace-vast/annas/git/research-tools/elicitation/outputs/finetuning_data_lowfrust_1212_inst_500.jsonl" \
    "/workspace-vast/annas/models/gemma3-27b-lowfrust-lr1e4" \
    --lr 0.0001 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --max_length 2048 \
    --lora_r 64 \
    --lora_alpha 128 \
    --warmup_steps 50 \
    --logging_steps 10 \
    --save_steps 400 \
    --save_total_limit 2 \
    --eval_strategy "no" \
    --load_in_4bit True

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Finetuning complete!"
    echo "Model saved to: /workspace-vast/annas/models/gemma3-27b-lowfrust-lr1e4"
else
    echo "✗ Finetuning failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
