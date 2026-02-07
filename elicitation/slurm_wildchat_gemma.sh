#!/bin/bash
#SBATCH --job-name=wc_gemma
#SBATCH --output=/workspace-vast/annas/logs/wildchat_gemma_%j.out
#SBATCH --error=/workspace-vast/annas/logs/wildchat_gemma_%j.out
#SBATCH --time=6:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

echo "=== Gemma 3 27B ==="
python elicitation/eval_generalization.py \
    google/gemma-3-27b-it \
    --scenario wildchat \
    --num-samples 50 \
    --wildchat-turns 5 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

echo "=== Gemma 3 12B ==="
python elicitation/eval_generalization.py \
    google/gemma-3-12b-it \
    --scenario wildchat \
    --num-samples 50 \
    --wildchat-turns 5 \
    --tensor-parallel-size 2 \
    --max-model-len 8192

echo "=== Gemma 3 4B ==="
python elicitation/eval_generalization.py \
    google/gemma-3-4b-it \
    --scenario wildchat \
    --num-samples 50 \
    --wildchat-turns 5 \
    --tensor-parallel-size 2 \
    --max-model-len 8192
