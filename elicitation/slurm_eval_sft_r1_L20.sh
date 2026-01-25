#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=eval_sft_r1
#SBATCH --output=/workspace-vast/annas/logs/eval_sft_r1_L20_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_sft_r1_L20_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "EVAL SFT R1 LoRA L20 (Tone Scenarios)"
echo "============================================================"

python -u elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --lora-path /workspace-vast/annas/models/gemma3-27b-sft-r1-L20-alpha128-1ep/2026-01-16_10-10-11 \
    --skip-triggers \
    --skip-long \
    --max-model-len 16384 \
    --num-samples 20

echo "Done!"
