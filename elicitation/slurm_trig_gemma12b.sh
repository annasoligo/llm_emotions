#!/bin/bash
#SBATCH --job-name=trig_g12b
#SBATCH --output=/workspace-vast/annas/logs/trig_gemma12b_%j.out
#SBATCH --error=/workspace-vast/annas/logs/trig_gemma12b_%j.out
#SBATCH --time=2:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools
source .venv/bin/activate

python elicitation/eval_generalization.py \
    google/gemma-3-12b-it \
    --scenario triggers \
    --num-samples 20 \
    --tensor-parallel-size 1 \
    --max-model-len 8192
