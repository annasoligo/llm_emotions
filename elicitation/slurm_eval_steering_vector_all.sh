#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --job-name=eval_sv_all
#SBATCH --output=/workspace-vast/annas/logs/eval_sv_all_%j.out
#SBATCH --error=/workspace-vast/annas/logs/eval_sv_all_%j.err

source /workspace-vast/annas/.secrets/load_secrets.sh
cd /workspace-vast/annas/git/research-tools

# Use believe-it-or-not venv for transformers/torch
source /workspace-vast/annas/git/believe-it-or-not/.venv/bin/activate

echo "============================================================"
echo "STEERING VECTOR GENERALIZATION EVALUATION - ALL LAYERS"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"

# Evaluate L20 steering vector
echo ""
echo "============================================================"
echo "EVALUATING L20 STEERING VECTOR"
echo "============================================================"
python -u elicitation/eval_steering_vector.py \
    google/gemma-3-27b-it \
    --steering-path /workspace-vast/annas/models/gemma3-27b-dpo-steering-vector-L20-alpha256/2026-01-15_20-47-29/steering_vector.pt \
    --steering-layer 20 \
    --num-samples 20 \
    --output-dir elicitation/outputs/eval_generalization

# Evaluate L40 steering vector
echo ""
echo "============================================================"
echo "EVALUATING L40 STEERING VECTOR"
echo "============================================================"
python -u elicitation/eval_steering_vector.py \
    google/gemma-3-27b-it \
    --steering-path /workspace-vast/annas/models/gemma3-27b-dpo-steering-vector-L40-alpha256/2026-01-15_21-08-57/steering_vector.pt \
    --steering-layer 40 \
    --num-samples 20 \
    --output-dir elicitation/outputs/eval_generalization

# Evaluate L50 steering vector
echo ""
echo "============================================================"
echo "EVALUATING L50 STEERING VECTOR"
echo "============================================================"
python -u elicitation/eval_steering_vector.py \
    google/gemma-3-27b-it \
    --steering-path /workspace-vast/annas/models/gemma3-27b-dpo-steering-vector-L50-alpha256/2026-01-15_21-08-57/steering_vector.pt \
    --steering-layer 50 \
    --num-samples 20 \
    --output-dir elicitation/outputs/eval_generalization

echo ""
echo "============================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "============================================================"
