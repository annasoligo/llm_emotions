#!/bin/bash
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_min_2ep
#SBATCH --output=/workspace-vast/annas/logs/eval_minimal_r1_2ep_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_minimal_r1_2ep_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "============================================================"
echo "EVALUATING MINIMAL DPO 2-EPOCH: Rank 1, Layer 20, down_proj"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

# Find the latest checkpoint
LORA_PATH=$(ls -td /workspace-vast/annas/models/gemma3-27b-dpo-minimal-r1-L20-2ep/*/ 2>/dev/null | head -1)
echo "Using LoRA path: $LORA_PATH"

# Run all generalization tests
python -u elicitation/eval_generalization.py google/gemma-3-27b-it \
    --lora-path "$LORA_PATH" \
    --num-samples 20

echo "Done!"
