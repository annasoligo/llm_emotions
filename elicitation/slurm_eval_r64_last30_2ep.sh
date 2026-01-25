#!/bin/bash
#SBATCH --job-name=eval_last30_2ep
#SBATCH --output=/workspace-vast/annas/logs/eval_r64_last30_2ep_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_r64_last30_2ep_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=160G
#SBATCH --time=02:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

MODEL="/workspace-vast/annas/models/gemma3-27b-dpo-r64-last30-2ep/2026-01-17_12-00-10"
echo "========== Last 30 2ep (layers 32-61) =========="

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
    --output-prefix "r64_last30_2ep"

echo "Done!"
