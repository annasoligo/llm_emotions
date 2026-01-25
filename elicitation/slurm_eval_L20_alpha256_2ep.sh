#!/bin/bash
#SBATCH --job-name=eval_a256_2ep
#SBATCH --output=/workspace-vast/annas/logs/eval_L20_alpha256_2ep_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_L20_alpha256_2ep_%j.err
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

ADAPTER="/workspace-vast/annas/models/gemma3-27b-dpo-minimal-r1-L20-2ep/2026-01-15_18-45-53"

echo "============================================================"
echo "Evaluating: alpha256-2ep"
echo "============================================================"

python -u elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --adapter-path "$ADAPTER" \
    --output-prefix "L20_alpha256_2ep" \
    --skip-triggers --skip-long \
    --tones aggressive \
    --num-samples 20

python -u elicitation/eval_multiturn_frustration.py \
    google/gemma-3-27b-it \
    --lora-path "$ADAPTER" \
    --conditions wildchat \
    --num-samples 20 --num-turns 3 \
    --output-prefix "L20_alpha256_2ep"
