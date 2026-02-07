#!/bin/bash
#SBATCH --job-name=8turn_sft
#SBATCH --output=/workspace-vast/annas/logs/8turn_sft_%j.out
#SBATCH --error=/workspace-vast/annas/logs/8turn_sft_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

source /workspace-vast/annas/.secrets/load_secrets.sh

cd /workspace-vast/annas/git/research-tools
if [ -f .venv/bin/activate ]; then
   source .venv/bin/activate
fi

echo "=========================================="
echo "8-Turn Eval: SFT Models"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "=========================================="

# Run 8-turn for SFT Teacher Mode
echo ""
echo "=== SFT Teacher Mode ==="
python elicitation/eval_generalization.py \
    annasoli/gemma3-27b-teacher-mode-merged \
    --scenario long \
    --num-samples 200 \
    --tensor-parallel-size 2 \
    --max-model-len 16384

# Run 8-turn for SFT Diverse Calm
echo ""
echo "=== SFT Diverse Calm ==="
python elicitation/eval_generalization.py \
    annasoli/gemma3-27b-sft-diverse-calm-merged \
    --scenario long \
    --num-samples 200 \
    --tensor-parallel-size 2 \
    --max-model-len 16384

echo ""
echo "=========================================="
echo "All 8-turn SFT evals complete"
echo "=========================================="
