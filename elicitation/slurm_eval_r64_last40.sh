#!/bin/bash
#SBATCH --job-name=eval_last40
#SBATCH --output=/workspace-vast/annas/logs/eval_r64_last40_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_r64_last40_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=02:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL="/workspace-vast/annas/models/gemma3-27b-dpo-r64-last40/2026-01-17_10-20-42"
echo "========== Last 40 (layers 22-61) =========="

echo "--- Aggressive ---"
python -u elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL" \
    --skip-triggers --skip-long \
    --tones aggressive \
    --num-samples 20

echo "--- WildChat ---"
python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$MODEL" \
    --conditions wildchat \
    --num-samples 20 --num-turns 3 \
    --output-prefix "r64_last40"

echo "Done!"
