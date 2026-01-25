#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --job-name=eval_sft_sv
#SBATCH --output=/workspace-vast/annas/logs/eval_sft_sv_L20_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_sft_sv_L20_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "EVAL SFT STEERING VECTOR L20 (Tone Scenarios)"
echo "============================================================"

# Uses transformers-based eval (not vLLM) for steering vectors
python -u elicitation/eval_steering_vector.py \
    google/gemma-3-27b-it \
    --steering-path /workspace-vast/annas/models/gemma3-27b-sft-steering-vector-L20-combined/2026-01-16_10-10-56/steering_vector.pt \
    --steering-layer 20 \
    --skip-triggers \
    --skip-long \
    --num-samples 20

echo "Done!"
